// finalblock_study.cu
//
// CUDA SHA-1 vanity-search study for the exact special case:
//   * all complete blocks before the final SHA-1 compression block are fixed;
//   * a 7-byte base64url nonce is wholly inside that final compression block;
//   * the final block also contains SHA-1 padding/length;
//   * the target is a prefix of the final SHA-1 digest (hot path tests H0).
//
// This intentionally models a multi-block Git object. PREFIX_BLOCKS defaults
// to 3, so the nonce is after >192 bytes of already-hashed input. Performance
// is independent of how many fixed blocks precede the final block: their
// chaining state is supplied as C_HIN.
//
// Build examples:
//   nvcc -O3 -arch=sm_89 -DNONCE_OFF=32 finalblock_study.cu -o study
//   nvcc -O3 -arch=sm_89 -DNONCE_OFF=47 finalblock_study.cu -o study47
//
// NONCE_OFF may be 0..48. DATA_LEN defaults to 55, so padding begins at byte
// 55 and the 64-bit bit length occupies bytes 56..63.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#ifndef NONCE_OFF
#define NONCE_OFF 32
#endif
#ifndef DATA_LEN
#define DATA_LEN 55
#endif
#ifndef PREFIX_BLOCKS
#define PREFIX_BLOCKS 3
#endif
#ifndef BENCH_REPEATS
#define BENCH_REPEATS 12
#endif
#ifndef NONCE_LEN
#define NONCE_LEN 7
#endif

static_assert(NONCE_OFF >= 0, "NONCE_OFF must be non-negative");
static_assert(NONCE_LEN >= 2 && NONCE_LEN <= 10, "nonce length 2..10 supported by 64-bit candidate id");
static_assert(NONCE_OFF + NONCE_LEN <= DATA_LEN, "nonce must fit before padding");
static_assert(DATA_LEN <= 55, "specialization requires padding+length in same final block");
static_assert(PREFIX_BLOCKS >= 0, "PREFIX_BLOCKS must be non-negative");

static constexpr int OUTER_LEN   = NONCE_LEN - 1;
static constexpr int FIRST_WORD  = NONCE_OFF / 4;
static constexpr int INNER_POS   = NONCE_OFF + NONCE_LEN - 1;
static constexpr int INNER_WORD  = INNER_POS / 4;
static constexpr int INNER_SHIFT = 8 * (3 - (INNER_POS & 3));
static constexpr int DELTA_ROWS  = 65; // row0 = raw inner D; rows1..64 => W16..W79 delta
static constexpr int DELTA_WORDS = DELTA_ROWS * 64;
static constexpr int PAIR_OUTER_LEN = NONCE_LEN - 2;
static constexpr int PAIR_A_POS = NONCE_OFF + NONCE_LEN - 2;
static constexpr int PAIR_B_POS = NONCE_OFF + NONCE_LEN - 1;
static constexpr int PAIR_FIRST_WORD = PAIR_A_POS / 4;
static constexpr int PAIR_ROWS = 80 - PAIR_FIRST_WORD;
static constexpr int PAIR_SEP_WORDS = PAIR_ROWS * 64;
static constexpr int PAIR_TABLE_WORDS = PAIR_ROWS * 4096;

// A G=4 warp contains eight independent lane-groups.  Shared-memory accesses
// at a fixed schedule round therefore touch eight group bases.  If the compact
// schedule stride repeats in fewer than eight of the 32 banks, add one word of
// padding per group.  This is notably needed when FIRST_WORD==8 (stride 72).
static constexpr int ct_gcd(int a,int b) { return b ? ct_gcd(b,a%b) : a; }
static constexpr int COMPACT_WORDS = 80-FIRST_WORD;
static constexpr int G4_GROUPS_PER_WARP = 32/4;
static constexpr int COMPACT_BANK_PERIOD = 32/ct_gcd(32,COMPACT_WORDS);
static constexpr int G4_COMPACT_AUTO_PAD = COMPACT_BANK_PERIOD < G4_GROUPS_PER_WARP ? 1 : 0;

static constexpr uint32_t K0 = 0x5a827999u;
static constexpr uint32_t K1 = 0x6ed9eba1u;
static constexpr uint32_t K2 = 0x8f1bbcdcu;
static constexpr uint32_t K3 = 0xca62c1d6u;
static constexpr uint64_t NO_WINNER = ~uint64_t(0);
static constexpr char HOST_CHARSET[65] =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

#ifdef JOB_CONSTANTS_HEADER
#include JOB_CONSTANTS_HEADER
#endif

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

__constant__ uint32_t C_BASE16[16];
__constant__ uint32_t C_BASE80[80];
__constant__ uint32_t C_PRE[5];
__constant__ uint32_t C_HIN[5];
__constant__ unsigned char C_CHARSET[64];
// Three storage experiments for the affine inner-character schedule deltas.
// Constant memory is especially attractive for G=1 because every warp lane
// requests the same j at a given ILP slot. Wider groups create progressively
// more distinct addresses per warp and therefore test the broadcast crossover.
__constant__ __align__(16) uint32_t C_DELTA[DELTA_WORDS];
__device__ __align__(128) uint32_t G_DELTA[DELTA_WORDS];
// 5+2 decomposition experiments.  The separable representation keeps two
// byte-delta schedules (~40 KiB total at early offsets).  The pair table folds
// those XORs on the host and trades a ~1.25 MiB read-mostly table for one
// vector load per SHA-1 word.
__device__ __align__(128) uint32_t G_PAIR_A[PAIR_SEP_WORDS];
__device__ __align__(128) uint32_t G_PAIR_B[PAIR_SEP_WORDS];
__device__ __align__(128) uint32_t G_PAIR_TABLE[PAIR_TABLE_WORDS];

template<int DM>
__device__ __forceinline__ uint32_t delta_get(const uint32_t *sd, int idx) {
  if constexpr (DM == 1) return sd[idx];
  else if constexpr (DM == 2) return C_DELTA[idx];
  else if constexpr (DM == 3) return __ldg(&G_DELTA[idx]);
  else return G_DELTA[idx];
}

__host__ __device__ static inline uint32_t hrol32(uint32_t x, unsigned n) {
  return (x << n) | (x >> (32 - n));
}
__device__ __forceinline__ uint32_t rol32(uint32_t x, unsigned n) {
  return __funnelshift_l(x, x, n);
}

__device__ __forceinline__ uint32_t f_ch(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}
__device__ __forceinline__ uint32_t f_par(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}
__device__ __forceinline__ uint32_t f_maj(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}

struct DState { uint32_t a,b,c,d,e; };

__device__ __forceinline__ void step_ch(DState &s, uint32_t w) {
  const uint32_t z = rol32(s.a,5) + f_ch(s.b,s.c,s.d) + s.e + K0 + w;
  s.e=s.d; s.d=s.c; s.c=rol32(s.b,30); s.b=s.a; s.a=z;
}
__device__ __forceinline__ void step_par1(DState &s, uint32_t w) {
  const uint32_t z = rol32(s.a,5) + f_par(s.b,s.c,s.d) + s.e + K1 + w;
  s.e=s.d; s.d=s.c; s.c=rol32(s.b,30); s.b=s.a; s.a=z;
}
__device__ __forceinline__ void step_maj(DState &s, uint32_t w) {
  const uint32_t z = rol32(s.a,5) + f_maj(s.b,s.c,s.d) + s.e + K2 + w;
  s.e=s.d; s.d=s.c; s.c=rol32(s.b,30); s.b=s.a; s.a=z;
}
__device__ __forceinline__ void step_par3(DState &s, uint32_t w) {
  const uint32_t z = rol32(s.a,5) + f_par(s.b,s.c,s.d) + s.e + K3 + w;
  s.e=s.d; s.d=s.c; s.c=rol32(s.b,30); s.b=s.a; s.a=z;
}

__device__ __forceinline__ uint32_t final_a79(const DState &s, uint32_t w79) {
  return rol32(s.a,5) + f_par(s.b,s.c,s.d) + s.e + K3 + w79;
}

// Role-rotated form used by mature SHA-1 GPU implementations.  It does not
// perform five C assignments per round; the caller rotates the *names* passed
// to the helper.  Modern ptxas already SSA-eliminates most of those moves in
// DState, so this is primarily a register-lifetime/code-generation experiment.
template<int PHASE>
__device__ __forceinline__ void rr_step(uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d,uint32_t &e,uint32_t w) {
  uint32_t f,k;
  if constexpr(PHASE==0){f=f_ch(b,c,d);k=K0;}
  else if constexpr(PHASE==1){f=f_par(b,c,d);k=K1;}
  else if constexpr(PHASE==2){f=f_maj(b,c,d);k=K2;}
  else {f=f_par(b,c,d);k=K3;}
  e += rol32(a,5)+f+k+w;
  b = rol32(b,30);
}

struct RRState { uint32_t a,b,c,d,e; };
template<int T,int START,int PHASE>
__device__ __forceinline__ void rr_round(RRState &s,uint32_t w) {
  constexpr int r=(T-START)%5;
  if constexpr(r==0) rr_step<PHASE>(s.a,s.b,s.c,s.d,s.e,w);
  else if constexpr(r==1) rr_step<PHASE>(s.e,s.a,s.b,s.c,s.d,w);
  else if constexpr(r==2) rr_step<PHASE>(s.d,s.e,s.a,s.b,s.c,w);
  else if constexpr(r==3) rr_step<PHASE>(s.c,s.d,s.e,s.a,s.b,w);
  else rr_step<PHASE>(s.b,s.c,s.d,s.e,s.a,w);
}
template<int NROUNDS>
__device__ __forceinline__ uint32_t rr_semantic_a(const RRState &s) {
  constexpr int r=NROUNDS%5;
  if constexpr(r==0) return s.a;
  else if constexpr(r==1) return s.e;
  else if constexpr(r==2) return s.d;
  else if constexpr(r==3) return s.c;
  else return s.b;
}

// ---------------- CPU reference / setup ----------------

static inline uint32_t cpu_f(int t, uint32_t b, uint32_t c, uint32_t d) {
  if (t < 20) return d ^ (b & (c ^ d));
  if (t < 40) return b ^ c ^ d;
  if (t < 60) return (b & c) | (d & (b | c));
  return b ^ c ^ d;
}
static inline uint32_t cpu_k(int t) {
  return t < 20 ? K0 : t < 40 ? K1 : t < 60 ? K2 : K3;
}
static void cpu_expand_classic(const uint32_t in[16], uint32_t w[80]) {
  for(int i=0;i<16;i++) w[i]=in[i];
  for(int t=16;t<80;t++) w[t]=hrol32(w[t-3]^w[t-8]^w[t-14]^w[t-16],1);
}
static void cpu_expand_alt(const uint32_t in[16], uint32_t w[80]) {
  for(int i=0;i<16;i++) w[i]=in[i];
  for(int t=16;t<32;t++) w[t]=hrol32(w[t-3]^w[t-8]^w[t-14]^w[t-16],1);
  for(int t=32;t<80;t++) w[t]=hrol32(w[t-6]^w[t-16]^w[t-28]^w[t-32],2);
}
static void cpu_rounds(uint32_t s[5], const uint32_t w[80], int begin, int end) {
  uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4];
  for(int t=begin;t<end;t++) {
    const uint32_t z=hrol32(a,5)+cpu_f(t,b,c,d)+e+cpu_k(t)+w[t];
    e=d; d=c; c=hrol32(b,30); b=a; a=z;
  }
  s[0]=a;s[1]=b;s[2]=c;s[3]=d;s[4]=e;
}
static void cpu_compress(const uint32_t hin[5], const uint32_t in[16], uint32_t out[5]) {
  uint32_t w[80]; cpu_expand_classic(in,w);
  memcpy(out,hin,20); cpu_rounds(out,w,0,80);
  for(int i=0;i<5;i++) out[i]+=hin[i];
}
static uint32_t load_be32(const unsigned char *p) {
  return (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|uint32_t(p[3]);
}
static void store_be64(unsigned char *p, uint64_t x) {
  for(int i=7;i>=0;i--) { p[i]=(unsigned char)x; x>>=8; }
}
static void make_prefix_midstate(uint32_t hin[5]) {
  const uint32_t iv[5]={0x67452301u,0xefcdab89u,0x98badcfeu,0x10325476u,0xc3d2e1f0u};
  memcpy(hin,iv,20);
  for(int blk=0; blk<PREFIX_BLOCKS; ++blk) {
    unsigned char b[64];
    for(int i=0;i<64;i++) b[i]=(unsigned char)((17 + 73*i + 29*blk) & 255);
    uint32_t w16[16],out[5];
    for(int i=0;i<16;i++) w16[i]=load_be32(b+4*i);
    cpu_compress(hin,w16,out); memcpy(hin,out,20);
  }
}
static void make_base_block(unsigned char b[64]) {
  memset(b,0,64);
  for(int i=0;i<DATA_LEN;i++) b[i]=(unsigned char)((31 + 41*i) & 255);
  for(int i=0;i<NONCE_LEN;i++) b[NONCE_OFF+i]=0;
  b[DATA_LEN]=0x80;
  const uint64_t total_bytes=uint64_t(PREFIX_BLOCKS)*64u+uint64_t(DATA_LEN);
  store_be64(b+56,total_bytes*8u);
}
static void make_words_for_id(uint64_t id, uint32_t w16[16]) {
#ifdef JOB_CONSTANTS_HEADER
  for(int i=0;i<16;i++) w16[i]=JOB_BASE16[i];
  const uint64_t outer=id>>6; const unsigned inner=(unsigned)(id&63u);
  for(int k=0;k<OUTER_LEN;k++) {
    const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
    w16[wi] |= uint32_t((unsigned char)HOST_CHARSET[(outer>>(6*k))&63u])<<sh;
  }
  w16[INNER_WORD] |= uint32_t((unsigned char)HOST_CHARSET[inner])<<INNER_SHIFT;
#else
  unsigned char b[64]; make_base_block(b);
  const uint64_t outer=id>>6; const unsigned inner=(unsigned)(id&63u);
  for(int k=0;k<OUTER_LEN;k++) b[NONCE_OFF+k]=(unsigned char)HOST_CHARSET[(outer>>(6*k))&63u];
  b[INNER_POS]=(unsigned char)HOST_CHARSET[inner];
  for(int i=0;i<16;i++) w16[i]=load_be32(b+4*i);
#endif
}
static void make_setup(uint32_t base16[16],uint32_t hin[5],uint32_t pre[5],uint32_t delta[DELTA_WORDS]) {
#ifdef JOB_CONSTANTS_HEADER
  for(int i=0;i<16;i++) base16[i]=JOB_BASE16[i];
  for(int i=0;i<5;i++) hin[i]=JOB_HIN[i];
#else
  unsigned char b[64]; make_base_block(b);
  for(int i=0;i<16;i++) base16[i]=load_be32(b+4*i);
  make_prefix_midstate(hin);
#endif
  uint32_t base80[80]; cpu_expand_classic(base16,base80);
  memcpy(pre,hin,20); cpu_rounds(pre,base80,0,FIRST_WORD);
  memset(delta,0,sizeof(uint32_t)*DELTA_WORDS);
  for(int j=0;j<64;j++) {
    uint32_t d16[16]={};
    d16[INNER_WORD]=uint32_t((unsigned char)HOST_CHARSET[j])<<INNER_SHIFT;
    uint32_t d80[80]; cpu_expand_classic(d16,d80);
    delta[j]=d16[INNER_WORD];
    for(int t=16;t<80;t++) delta[(1+t-16)*64+j]=d80[t];
  }
}

static void make_pair_setup(std::vector<uint32_t> &da,std::vector<uint32_t> &db,std::vector<uint32_t> &pair) {
  da.assign(PAIR_SEP_WORDS,0); db.assign(PAIR_SEP_WORDS,0); pair.assign(PAIR_TABLE_WORDS,0);
  auto one=[&](int pos,int j,std::vector<uint32_t>& dst) {
    uint32_t d16[16]={}; const int wi=pos>>2,sh=8*(3-(pos&3));
    d16[wi]=uint32_t((unsigned char)HOST_CHARSET[j])<<sh;
    uint32_t d80[80]; cpu_expand_classic(d16,d80);
    for(int t=PAIR_FIRST_WORD;t<80;t++) dst[(t-PAIR_FIRST_WORD)*64+j]=d80[t];
  };
  for(int j=0;j<64;j++) { one(PAIR_A_POS,j,da); one(PAIR_B_POS,j,db); }
  for(int row=0;row<PAIR_ROWS;row++)
    for(int a=0;a<64;a++) for(int b=0;b<64;b++)
      pair[row*4096+a*64+b]=da[row*64+a]^db[row*64+b];
}

// ---------------- Specialized kernels ----------------

// Best single-candidate/thread control: skip the invariant opening rounds and
// keep only a 16-word rolling schedule in registers. This is intentionally not
// a naive 80-round SHA-1 baseline.
__global__ void k_one_thread(uint64_t first_id,uint64_t count,uint32_t target,uint32_t mask,uint64_t *winner) {
  const uint64_t n=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(n>=count) return;
  const uint64_t id=first_id+n,outer=id>>6; const unsigned inner=(unsigned)(id&63u);

  uint32_t w[16];
  #pragma unroll
  for(int i=0;i<16;i++) w[i]=C_BASE16[i];
  #pragma unroll
  for(int k=0;k<OUTER_LEN;k++) {
    const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
    w[wi] |= uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;
  }
  w[INNER_WORD] |= uint32_t(C_CHARSET[inner])<<INNER_SHIFT;

  DState s{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<16;t++) step_ch(s,w[t]);
  #pragma unroll
  for(int t=16;t<20;t++) { const int i=t&15; w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1); step_ch(s,w[i]); }
  #pragma unroll
  for(int t=20;t<40;t++) { const int i=t&15; w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1); step_par1(s,w[i]); }
  #pragma unroll
  for(int t=40;t<60;t++) { const int i=t&15; w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1); step_maj(s,w[i]); }
  #pragma unroll
  for(int t=60;t<79;t++) { const int i=t&15; w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1); step_par3(s,w[i]); }
  { const int t=79,i=t&15; w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1);
    const uint32_t h0=final_a79(s,w[i])+C_HIN[0];
    if((h0&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)id);
  }
}

// High-register control inspired by optimized GPU SHA-1 implementations:
// materialize all 80 expanded schedule words as named scalars. This trades
// schedule dependency/reloads for register pressure and lets ptxas expose
// whether Ada prefers the hashcat-style strategy for this specialization.
template<int WI> __device__ __forceinline__ uint32_t candidate_word(uint64_t outer,unsigned inner) {
  uint32_t v=C_BASE16[WI];
  #pragma unroll
  for(int k=0;k<OUTER_LEN;k++) {
    if(((NONCE_OFF+k)/4)==WI) v |= uint32_t(C_CHARSET[(outer>>(6*k))&63u]) << (8*(3-((NONCE_OFF+k)&3)));
  }
  if((INNER_POS/4)==WI) v |= uint32_t(C_CHARSET[inner]) << INNER_SHIFT;
  return v;
}

__global__ void k_named80(uint64_t first_id,uint64_t count,uint32_t target,uint32_t mask,uint64_t *winner) {
  const uint64_t n=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(n>=count)return;
  const uint64_t id=first_id+n,outer=id>>6; const unsigned inner=(unsigned)(id&63u);
  uint32_t w00=candidate_word<0>(outer,inner);
  uint32_t w01=candidate_word<1>(outer,inner);
  uint32_t w02=candidate_word<2>(outer,inner);
  uint32_t w03=candidate_word<3>(outer,inner);
  uint32_t w04=candidate_word<4>(outer,inner);
  uint32_t w05=candidate_word<5>(outer,inner);
  uint32_t w06=candidate_word<6>(outer,inner);
  uint32_t w07=candidate_word<7>(outer,inner);
  uint32_t w08=candidate_word<8>(outer,inner);
  uint32_t w09=candidate_word<9>(outer,inner);
  uint32_t w10=candidate_word<10>(outer,inner);
  uint32_t w11=candidate_word<11>(outer,inner);
  uint32_t w12=candidate_word<12>(outer,inner);
  uint32_t w13=candidate_word<13>(outer,inner);
  uint32_t w14=candidate_word<14>(outer,inner);
  uint32_t w15=candidate_word<15>(outer,inner);
  uint32_t w16=rol32(w13^w08^w02^w00,1);
  uint32_t w17=rol32(w14^w09^w03^w01,1);
  uint32_t w18=rol32(w15^w10^w04^w02,1);
  uint32_t w19=rol32(w16^w11^w05^w03,1);
  uint32_t w20=rol32(w17^w12^w06^w04,1);
  uint32_t w21=rol32(w18^w13^w07^w05,1);
  uint32_t w22=rol32(w19^w14^w08^w06,1);
  uint32_t w23=rol32(w20^w15^w09^w07,1);
  uint32_t w24=rol32(w21^w16^w10^w08,1);
  uint32_t w25=rol32(w22^w17^w11^w09,1);
  uint32_t w26=rol32(w23^w18^w12^w10,1);
  uint32_t w27=rol32(w24^w19^w13^w11,1);
  uint32_t w28=rol32(w25^w20^w14^w12,1);
  uint32_t w29=rol32(w26^w21^w15^w13,1);
  uint32_t w30=rol32(w27^w22^w16^w14,1);
  uint32_t w31=rol32(w28^w23^w17^w15,1);
  uint32_t w32=rol32(w26^w16^w04^w00,2);
  uint32_t w33=rol32(w27^w17^w05^w01,2);
  uint32_t w34=rol32(w28^w18^w06^w02,2);
  uint32_t w35=rol32(w29^w19^w07^w03,2);
  uint32_t w36=rol32(w30^w20^w08^w04,2);
  uint32_t w37=rol32(w31^w21^w09^w05,2);
  uint32_t w38=rol32(w32^w22^w10^w06,2);
  uint32_t w39=rol32(w33^w23^w11^w07,2);
  uint32_t w40=rol32(w34^w24^w12^w08,2);
  uint32_t w41=rol32(w35^w25^w13^w09,2);
  uint32_t w42=rol32(w36^w26^w14^w10,2);
  uint32_t w43=rol32(w37^w27^w15^w11,2);
  uint32_t w44=rol32(w38^w28^w16^w12,2);
  uint32_t w45=rol32(w39^w29^w17^w13,2);
  uint32_t w46=rol32(w40^w30^w18^w14,2);
  uint32_t w47=rol32(w41^w31^w19^w15,2);
  uint32_t w48=rol32(w42^w32^w20^w16,2);
  uint32_t w49=rol32(w43^w33^w21^w17,2);
  uint32_t w50=rol32(w44^w34^w22^w18,2);
  uint32_t w51=rol32(w45^w35^w23^w19,2);
  uint32_t w52=rol32(w46^w36^w24^w20,2);
  uint32_t w53=rol32(w47^w37^w25^w21,2);
  uint32_t w54=rol32(w48^w38^w26^w22,2);
  uint32_t w55=rol32(w49^w39^w27^w23,2);
  uint32_t w56=rol32(w50^w40^w28^w24,2);
  uint32_t w57=rol32(w51^w41^w29^w25,2);
  uint32_t w58=rol32(w52^w42^w30^w26,2);
  uint32_t w59=rol32(w53^w43^w31^w27,2);
  uint32_t w60=rol32(w54^w44^w32^w28,2);
  uint32_t w61=rol32(w55^w45^w33^w29,2);
  uint32_t w62=rol32(w56^w46^w34^w30,2);
  uint32_t w63=rol32(w57^w47^w35^w31,2);
  uint32_t w64=rol32(w58^w48^w36^w32,2);
  uint32_t w65=rol32(w59^w49^w37^w33,2);
  uint32_t w66=rol32(w60^w50^w38^w34,2);
  uint32_t w67=rol32(w61^w51^w39^w35,2);
  uint32_t w68=rol32(w62^w52^w40^w36,2);
  uint32_t w69=rol32(w63^w53^w41^w37,2);
  uint32_t w70=rol32(w64^w54^w42^w38,2);
  uint32_t w71=rol32(w65^w55^w43^w39,2);
  uint32_t w72=rol32(w66^w56^w44^w40,2);
  uint32_t w73=rol32(w67^w57^w45^w41,2);
  uint32_t w74=rol32(w68^w58^w46^w42,2);
  uint32_t w75=rol32(w69^w59^w47^w43,2);
  uint32_t w76=rol32(w70^w60^w48^w44,2);
  uint32_t w77=rol32(w71^w61^w49^w45,2);
  uint32_t w78=rol32(w72^w62^w50^w46,2);
  uint32_t w79=rol32(w73^w63^w51^w47,2);
  DState s{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  if constexpr (FIRST_WORD <= 0) step_ch(s,w00);
  if constexpr (FIRST_WORD <= 1) step_ch(s,w01);
  if constexpr (FIRST_WORD <= 2) step_ch(s,w02);
  if constexpr (FIRST_WORD <= 3) step_ch(s,w03);
  if constexpr (FIRST_WORD <= 4) step_ch(s,w04);
  if constexpr (FIRST_WORD <= 5) step_ch(s,w05);
  if constexpr (FIRST_WORD <= 6) step_ch(s,w06);
  if constexpr (FIRST_WORD <= 7) step_ch(s,w07);
  if constexpr (FIRST_WORD <= 8) step_ch(s,w08);
  if constexpr (FIRST_WORD <= 9) step_ch(s,w09);
  if constexpr (FIRST_WORD <= 10) step_ch(s,w10);
  if constexpr (FIRST_WORD <= 11) step_ch(s,w11);
  if constexpr (FIRST_WORD <= 12) step_ch(s,w12);
  if constexpr (FIRST_WORD <= 13) step_ch(s,w13);
  if constexpr (FIRST_WORD <= 14) step_ch(s,w14);
  if constexpr (FIRST_WORD <= 15) step_ch(s,w15);
  step_ch(s,w16);
  step_ch(s,w17);
  step_ch(s,w18);
  step_ch(s,w19);
  step_par1(s,w20);
  step_par1(s,w21);
  step_par1(s,w22);
  step_par1(s,w23);
  step_par1(s,w24);
  step_par1(s,w25);
  step_par1(s,w26);
  step_par1(s,w27);
  step_par1(s,w28);
  step_par1(s,w29);
  step_par1(s,w30);
  step_par1(s,w31);
  step_par1(s,w32);
  step_par1(s,w33);
  step_par1(s,w34);
  step_par1(s,w35);
  step_par1(s,w36);
  step_par1(s,w37);
  step_par1(s,w38);
  step_par1(s,w39);
  step_maj(s,w40);
  step_maj(s,w41);
  step_maj(s,w42);
  step_maj(s,w43);
  step_maj(s,w44);
  step_maj(s,w45);
  step_maj(s,w46);
  step_maj(s,w47);
  step_maj(s,w48);
  step_maj(s,w49);
  step_maj(s,w50);
  step_maj(s,w51);
  step_maj(s,w52);
  step_maj(s,w53);
  step_maj(s,w54);
  step_maj(s,w55);
  step_maj(s,w56);
  step_maj(s,w57);
  step_maj(s,w58);
  step_maj(s,w59);
  step_par3(s,w60);
  step_par3(s,w61);
  step_par3(s,w62);
  step_par3(s,w63);
  step_par3(s,w64);
  step_par3(s,w65);
  step_par3(s,w66);
  step_par3(s,w67);
  step_par3(s,w68);
  step_par3(s,w69);
  step_par3(s,w70);
  step_par3(s,w71);
  step_par3(s,w72);
  step_par3(s,w73);
  step_par3(s,w74);
  step_par3(s,w75);
  step_par3(s,w76);
  step_par3(s,w77);
  step_par3(s,w78);
  const uint32_t h0=final_a79(s,w79)+C_HIN[0];
  if((h0&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)id);
}

// Optional round-4 compiler experiment.  The survey build leaves this empty;
// focus builds can define R4_LB_MIN_BLOCKS to let ptxas trade registers against
// a requested minimum resident-block count without maintaining a second kernel.
#ifdef R4_LB_MIN_BLOCKS
#ifndef R4_LB_THREADS
#define R4_LB_THREADS 256
#endif
#define R4_GROUP_LB __launch_bounds__(R4_LB_THREADS,R4_LB_MIN_BLOCKS)
#else
#define R4_GROUP_LB
#endif

// Lane grouping continuum:
//   G=1  : one outer candidate per lane; each lane serially/ILP-evaluates 64 inners
//   G=8  : 4 outer candidates per warp; 8 lanes cooperate on each; 8 inners/lane
//   G=16 : 2 outer candidates per warp; 4 inners/lane
//   G=32 : one outer candidate per warp; 2 inners/lane
//
// One expanded OUTER schedule is stored per lane-group, not per thread. The
// schedule expansion is executed by the group leaders and amortized 64x.
// N is the inner-candidate ILP width per lane and must divide 64/G.
template<int G,int N,int DELTA_MODE,int PAD=0>
__global__ R4_GROUP_LB void k_group(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=32,"G power of two 1..32");
  static_assert(64%G==0,"G divides 64");
  static_assert((64/G)%N==0,"N divides inners per lane");

  extern __shared__ uint32_t sm[];
  uint32_t *sd = sm;
  uint32_t *ss = sm + (DELTA_MODE == 1 ? DELTA_WORDS : 0);

  if constexpr (DELTA_MODE == 1) {
    for(int i=threadIdx.x;i<DELTA_WORDS;i+=blockDim.x) sd[i]=G_DELTA[i];
    __syncthreads();
  }

  const int group_local=threadIdx.x/G;
  const int lane_group=threadIdx.x&(G-1);
  const int groups_per_block=blockDim.x/G;
  const uint64_t group_global=uint64_t(blockIdx.x)*groups_per_block+group_local;
  const bool active=group_global<outer_count;
  const uint64_t outer=outer_base+group_global;
  constexpr int SCHED_STRIDE=80+PAD;
  uint32_t *sched=ss+group_local*SCHED_STRIDE;

  if(active && lane_group==0) {
    #pragma unroll
    for(int t=0;t<16;t++) sched[t]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++) {
      const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
      sched[wi] |= uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;
    }
    // Classic recurrence for 16..31, then the equivalent SHA-1 ROL2 form
    // used by optimized GPU implementations for the rest.
    #pragma unroll
    for(int t=16;t<32;t++) sched[t]=rol32(sched[t-3]^sched[t-8]^sched[t-14]^sched[t-16],1);
    #pragma unroll
    for(int t=32;t<80;t++) sched[t]=rol32(sched[t-6]^sched[t-16]^sched[t-28]^sched[t-32],2);
  }
  __syncwarp();
  if(!active) return;

  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t]);

  constexpr int PER_LANE=64/G;
  #pragma unroll 1
  for(int base=0;base<PER_LANE;base+=N) {
    DState s[N];
    unsigned jj[N];

    #pragma unroll
    for(int i=0;i<N;i++) {
      const unsigned j=(unsigned)(lane_group+(base+i)*G);
      jj[i]=j;
      const uint32_t d0 = delta_get<DELTA_MODE>(sd,j);
      s[i]=common;
      // sched[INNER_WORD] has zero bits in the inner byte, therefore +d0 is
      // identical to OR/XOR and can fold naturally into the round additions.
      step_ch(s[i],sched[INNER_WORD]+d0);
    }

    // Other original message words through W15 are inner-independent.
    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++) {
      const uint32_t bw=sched[t];
      #pragma unroll
      for(int i=0;i<N;i++) step_ch(s[i],bw);
    }

    #pragma unroll
    for(int t=16;t<20;t++) {
      const uint32_t bw=sched[t]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) { const uint32_t d=delta_get<DELTA_MODE>(sd,row*64+jj[i]); step_ch(s[i],bw^d); }
    }
    #pragma unroll
    for(int t=20;t<40;t++) {
      const uint32_t bw=sched[t]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) { const uint32_t d=delta_get<DELTA_MODE>(sd,row*64+jj[i]); step_par1(s[i],bw^d); }
    }
    #pragma unroll
    for(int t=40;t<60;t++) {
      const uint32_t bw=sched[t]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) { const uint32_t d=delta_get<DELTA_MODE>(sd,row*64+jj[i]); step_maj(s[i],bw^d); }
    }
    #pragma unroll
    for(int t=60;t<79;t++) {
      const uint32_t bw=sched[t]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) { const uint32_t d=delta_get<DELTA_MODE>(sd,row*64+jj[i]); step_par3(s[i],bw^d); }
    }
    {
      const int row=1+79-16; const uint32_t bw=sched[79];
      #pragma unroll
      for(int i=0;i<N;i++) {
        const uint32_t d=delta_get<DELTA_MODE>(sd,row*64+jj[i]);
        const uint32_t h0=final_a79(s[i],bw^d)+C_HIN[0];
        if((h0&mask)==target) {
          const uint64_t id=(outer<<6)|uint64_t(jj[i]);
          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)id);
        }
      }
    }
  }
}


// Compact-schedule variant: only shared-store W[FIRST_WORD..79]. Earlier
// expanded words are invariant and come from C_BASE80 while the group leader
// constructs later words. This reduces per-block shared schedule storage by
// FIRST_WORD/80 without changing candidate arithmetic.
template<int G,int N,int DELTA_MODE,int PAD=0>
__global__ R4_GROUP_LB void k_group_compact(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=32,"G power of two 1..32");
  static_assert(64%G==0,"G divides 64");
  static_assert((64/G)%N==0,"N divides inners per lane");
  static_assert(DELTA_MODE != 1,"compact kernel currently supports global/ldg/constant delta, not shared-delta staging");
  constexpr int SW = 80-FIRST_WORD;
  constexpr int SCHED_STRIDE = SW+PAD;

  extern __shared__ uint32_t sm[];
  uint32_t *ss=sm;
  const int group_local=threadIdx.x/G;
  const int lane_group=threadIdx.x&(G-1);
  const int groups_per_block=blockDim.x/G;
  const uint64_t group_global=uint64_t(blockIdx.x)*groups_per_block+group_local;
  const bool active=group_global<outer_count;
  const uint64_t outer=outer_base+group_global;
  uint32_t *sched=ss+group_local*SCHED_STRIDE;

  if(active && lane_group==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++) {
      const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
      sched[wi-FIRST_WORD] |= uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;
    }
    #pragma unroll
    for(int t=16;t<32;t++) {
      const uint32_t a=(t-3 <FIRST_WORD)?C_BASE80[t-3] :sched[t-3 -FIRST_WORD];
      const uint32_t b=(t-8 <FIRST_WORD)?C_BASE80[t-8] :sched[t-8 -FIRST_WORD];
      const uint32_t c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD];
      const uint32_t d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,1);
    }
    #pragma unroll
    for(int t=32;t<80;t++) {
      const uint32_t a=(t-6 <FIRST_WORD)?C_BASE80[t-6] :sched[t-6 -FIRST_WORD];
      const uint32_t b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      const uint32_t c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD];
      const uint32_t d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,2);
    }
  }
  __syncwarp();
  if(!active) return;

  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);

  constexpr int PER_LANE=64/G;
  #pragma unroll 1
  for(int base=0;base<PER_LANE;base+=N) {
    DState st[N]; unsigned jj[N];
    #pragma unroll
    for(int i=0;i<N;i++) {
      const unsigned j=(unsigned)(lane_group+(base+i)*G); jj[i]=j; st[i]=common;
      step_ch(st[i],sched[INNER_WORD-FIRST_WORD]+delta_get<DELTA_MODE>(nullptr,j));
    }
    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++) {
      const uint32_t bw=sched[t-FIRST_WORD];
      #pragma unroll
      for(int i=0;i<N;i++) step_ch(st[i],bw);
    }
    #pragma unroll
    for(int t=16;t<20;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) step_ch(st[i],bw^delta_get<DELTA_MODE>(nullptr,row*64+jj[i]));
    }
    #pragma unroll
    for(int t=20;t<40;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) step_par1(st[i],bw^delta_get<DELTA_MODE>(nullptr,row*64+jj[i]));
    }
    #pragma unroll
    for(int t=40;t<60;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) step_maj(st[i],bw^delta_get<DELTA_MODE>(nullptr,row*64+jj[i]));
    }
    #pragma unroll
    for(int t=60;t<79;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      #pragma unroll
      for(int i=0;i<N;i++) step_par3(st[i],bw^delta_get<DELTA_MODE>(nullptr,row*64+jj[i]));
    }
    {
      const int row=1+79-16; const uint32_t bw=sched[79-FIRST_WORD];
      #pragma unroll
      for(int i=0;i<N;i++) {
        const uint32_t h0=final_a79(st[i],bw^delta_get<DELTA_MODE>(nullptr,row*64+jj[i]))+C_HIN[0];
        if((h0&mask)==target) { const uint64_t id=(outer<<6)|uint64_t(jj[i]);
          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)id); }
      }
    }
  }
}


// G4/ILP4 specialization with a different inner-candidate mapping.  Each lane
// owns four consecutive inner values in each 16-value chunk, so the four affine
// deltas for a round are a naturally aligned uint4.  G_DELTA is 128-byte aligned,
// rows are 64 words (256 bytes), and jbase is a multiple of four, making every
// vector address 16-byte aligned.  CUDA's uint4 has 16-byte alignment; the point
// of this variant is to let ptxas use a wide global load instead of four scalar
// loads while preserving the same total cache footprint.
__device__ __forceinline__ uint4 delta4_global(int row,unsigned jbase) {
  const uint32_t *p=G_DELTA + row*64 + jbase;
  uint4 v;
  asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 delta4_nc(int row,unsigned jbase) {
  const uint32_t *p=G_DELTA + row*64 + jbase;
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  return v;
}

template<bool NC,int PAD=0>
__global__ R4_GROUP_LB void k_group_compact_g4_vec4(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  constexpr int G=4, N=4;
  constexpr int SW=80-FIRST_WORD;
  constexpr int SCHED_STRIDE=SW+PAD;
  extern __shared__ uint32_t sm[];

  const int group_local=threadIdx.x/G;
  const int lane_group=threadIdx.x&(G-1);
  const int groups_per_block=blockDim.x/G;
  const uint64_t group_global=uint64_t(blockIdx.x)*groups_per_block+group_local;
  const bool active=group_global<outer_count;
  const uint64_t outer=outer_base+group_global;
  uint32_t *sched=sm+group_local*SCHED_STRIDE;

  if(active && lane_group==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++) {
      const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
      sched[wi-FIRST_WORD] |= uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;
    }
    #pragma unroll
    for(int t=16;t<32;t++) {
      const uint32_t a=(t-3 <FIRST_WORD)?C_BASE80[t-3] :sched[t-3-FIRST_WORD];
      const uint32_t b=(t-8 <FIRST_WORD)?C_BASE80[t-8] :sched[t-8-FIRST_WORD];
      const uint32_t c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD];
      const uint32_t d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,1);
    }
    #pragma unroll
    for(int t=32;t<80;t++) {
      const uint32_t a=(t-6 <FIRST_WORD)?C_BASE80[t-6] :sched[t-6-FIRST_WORD];
      const uint32_t b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      const uint32_t c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD];
      const uint32_t d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,2);
    }
  }
  __syncwarp();
  if(!active) return;

  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);

  // Four 16-candidate chunks cover all 64 inner characters.  Within a chunk,
  // lane L owns j = chunk + 4*L + {0,1,2,3}.
  #pragma unroll 1
  for(unsigned chunk=0;chunk<64;chunk+=16) {
    const unsigned jbase=chunk+(unsigned)lane_group*4u;
    DState st0=common,st1=common,st2=common,st3=common;

    uint4 dv = NC ? delta4_nc(0,jbase) : delta4_global(0,jbase);
    const uint32_t w0=sched[INNER_WORD-FIRST_WORD];
    step_ch(st0,w0+dv.x); step_ch(st1,w0+dv.y); step_ch(st2,w0+dv.z); step_ch(st3,w0+dv.w);

    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++) {
      const uint32_t bw=sched[t-FIRST_WORD];
      step_ch(st0,bw); step_ch(st1,bw); step_ch(st2,bw); step_ch(st3,bw);
    }
    #pragma unroll
    for(int t=16;t<20;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      dv = NC ? delta4_nc(row,jbase) : delta4_global(row,jbase);
      step_ch(st0,bw^dv.x); step_ch(st1,bw^dv.y); step_ch(st2,bw^dv.z); step_ch(st3,bw^dv.w);
    }
    #pragma unroll
    for(int t=20;t<40;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      dv = NC ? delta4_nc(row,jbase) : delta4_global(row,jbase);
      step_par1(st0,bw^dv.x); step_par1(st1,bw^dv.y); step_par1(st2,bw^dv.z); step_par1(st3,bw^dv.w);
    }
    #pragma unroll
    for(int t=40;t<60;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      dv = NC ? delta4_nc(row,jbase) : delta4_global(row,jbase);
      step_maj(st0,bw^dv.x); step_maj(st1,bw^dv.y); step_maj(st2,bw^dv.z); step_maj(st3,bw^dv.w);
    }
    #pragma unroll
    for(int t=60;t<79;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; const int row=1+t-16;
      dv = NC ? delta4_nc(row,jbase) : delta4_global(row,jbase);
      step_par3(st0,bw^dv.x); step_par3(st1,bw^dv.y); step_par3(st2,bw^dv.z); step_par3(st3,bw^dv.w);
    }
    {
      const int row=1+79-16; const uint32_t bw=sched[79-FIRST_WORD];
      dv = NC ? delta4_nc(row,jbase) : delta4_global(row,jbase);
      const uint32_t h0=final_a79(st0,bw^dv.x)+C_HIN[0];
      const uint32_t h1=final_a79(st1,bw^dv.y)+C_HIN[0];
      const uint32_t h2=final_a79(st2,bw^dv.z)+C_HIN[0];
      const uint32_t h3=final_a79(st3,bw^dv.w)+C_HIN[0];
      if((h0&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+0u)));
      if((h1&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+1u)));
      if((h2&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+2u)));
      if((h3&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+3u)));
    }
  }
}


// ---------------- Round-5 contiguous-vector family ----------------
// Generalizes the successful G4/V4 mapping.  A lane owns V consecutive inner
// characters.  V=2 uses one 64-bit vector load; V=4/8/16 use 1/2/4 128-bit
// loads.  G controls how many lanes cooperate on one outer nonce.  The product
// G*V must divide 64, so each chunk covers exactly G*V inner characters.
//
// CACHE: 0=global.ca, 1=global.nc (read-only/texture path), 2=global.cg (L2),
//        3=constant-symbol vector load (experimental; can serialize on address
//        divergence, retained as a useful negative/positive control).
// COMBINE_MATCH: collapse V rare winner branches into one hot-path branch.

template<int CACHE>
__device__ __forceinline__ uint2 r5_delta2(int row,unsigned jbase) {
  const uint32_t *p=G_DELTA + row*64 + jbase; uint2 v;
  if constexpr (CACHE==0) {
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(v.x),"=r"(v.y) : "l"(p));
  } else if constexpr (CACHE==1) {
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x),"=r"(v.y) : "l"(p));
  } else if constexpr (CACHE==2) {
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x),"=r"(v.y) : "l"(p));
  } else {
    const uint2 *cp=reinterpret_cast<const uint2*>(C_DELTA + row*64 + jbase); v=*cp;
  }
  return v;
}

template<int CACHE>
__device__ __forceinline__ uint4 r5_delta4(int row,unsigned jbase) {
  const uint32_t *p=G_DELTA + row*64 + jbase; uint4 v;
  if constexpr (CACHE==0) {
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  } else if constexpr (CACHE==1) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  } else if constexpr (CACHE==2) {
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  } else {
    const uint4 *cp=reinterpret_cast<const uint4*>(C_DELTA + row*64 + jbase); v=*cp;
  }
  return v;
}

template<int V,int CACHE>
struct R5DeltaPack { uint32_t x[V]; };

template<int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r5_load_pack(int row,unsigned jbase) {
  static_assert(V==2 || V==4 || V==8 || V==16 || V==32,"round-5 vector width must be 2/4/8/16/32");
  R5DeltaPack<V,CACHE> out{};
  if constexpr (V==2) {
    uint2 a=r5_delta2<CACHE>(row,jbase); out.x[0]=a.x; out.x[1]=a.y;
  } else {
    #pragma unroll
    for(int q=0;q<V;q+=4) {
      uint4 a=r5_delta4<CACHE>(row,jbase+(unsigned)q);
      out.x[q+0]=a.x; out.x[q+1]=a.y; out.x[q+2]=a.z; out.x[q+3]=a.w;
    }
  }
  return out;
}

template<int G,int V,int CACHE,int PAD=0,bool COMBINE_MATCH=false,int MATCH_MODE=0>
__global__ R4_GROUP_LB void k_group_compact_vec(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=32,"G power of two 1..32");
  static_assert(V==2 || V==4 || V==8 || V==16 || V==32,"V=2/4/8/16/32");
  static_assert(64%(G*V)==0,"G*V must divide 64");
  constexpr int SW=80-FIRST_WORD;
  constexpr int SCHED_STRIDE=SW+PAD;
  extern __shared__ uint32_t sm[];

  const int group_local=threadIdx.x/G;
  const int lane_group=threadIdx.x&(G-1);
  const int groups_per_block=blockDim.x/G;
  const uint64_t group_global=uint64_t(blockIdx.x)*groups_per_block+group_local;
  const bool active=group_global<outer_count;
  const uint64_t outer=outer_base+group_global;
  uint32_t *sched=sm+group_local*SCHED_STRIDE;

  if(active && lane_group==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++) {
      const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));
      sched[wi-FIRST_WORD] |= uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;
    }
    #pragma unroll
    for(int t=16;t<32;t++) {
      const uint32_t a=(t-3 <FIRST_WORD)?C_BASE80[t-3] :sched[t-3-FIRST_WORD];
      const uint32_t b=(t-8 <FIRST_WORD)?C_BASE80[t-8] :sched[t-8-FIRST_WORD];
      const uint32_t c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD];
      const uint32_t d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,1);
    }
    #pragma unroll
    for(int t=32;t<80;t++) {
      const uint32_t a=(t-6 <FIRST_WORD)?C_BASE80[t-6] :sched[t-6-FIRST_WORD];
      const uint32_t b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      const uint32_t c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD];
      const uint32_t d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,2);
    }
  }
  __syncwarp();
  if(!active) return;

  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);

  constexpr unsigned CHUNK=G*V;
  #pragma unroll 1
  for(unsigned chunk=0;chunk<64;chunk+=CHUNK) {
    const unsigned jbase=chunk+(unsigned)lane_group*V;
    DState st[V];
    #pragma unroll
    for(int i=0;i<V;i++) st[i]=common;

    R5DeltaPack<V,CACHE> dv=r5_load_pack<V,CACHE>(0,jbase);
    const uint32_t w0=sched[INNER_WORD-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) step_ch(st[i],w0+dv.x[i]);

    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++) {
      const uint32_t bw=sched[t-FIRST_WORD];
      #pragma unroll
      for(int i=0;i<V;i++) step_ch(st[i],bw);
    }
    #pragma unroll
    for(int t=16;t<20;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) step_ch(st[i],bw^dv.x[i]);
    }
    #pragma unroll
    for(int t=20;t<40;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) step_par1(st[i],bw^dv.x[i]);
    }
    #pragma unroll
    for(int t=40;t<60;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) step_maj(st[i],bw^dv.x[i]);
    }
    #pragma unroll
    for(int t=60;t<79;t++) {
      const uint32_t bw=sched[t-FIRST_WORD]; dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) step_par3(st[i],bw^dv.x[i]);
    }
    {
      const uint32_t bw=sched[79-FIRST_WORD]; dv=r5_load_pack<V,CACHE>(64,jbase);
      if constexpr (COMBINE_MATCH) {
        unsigned hitmask=0;
        #pragma unroll
        for(int i=0;i<V;i++) {
          const uint32_t h=final_a79(st[i],bw^dv.x[i])+C_HIN[0];
          hitmask |= (unsigned)((h&mask)==target) << i;
        }
        if(hitmask) {
          int first=0;
          #pragma unroll
          for(int i=0;i<V;i++) if(hitmask&(1u<<i)) { first=i; break; }
          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+(unsigned)first)));
        }
      } else {
        #pragma unroll
        for(int i=0;i<V;i++) {
          const uint32_t a80=final_a79(st[i],bw^dv.x[i]);
          bool hit;
          if constexpr (MATCH_MODE==1) hit=(a80==(target-C_HIN[0]));
          else hit=(((a80+C_HIN[0])&mask)==target);
          if(hit)
            atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+(unsigned)i)));
        }
      }
    }
  }
}



// ---------------- Round-6 sparse-polynomial affine delta family ----------------
// SHA-1 expansion is linear over XOR/ROL1.  For a single mutable source word X,
// every expanded delta is XOR_k ROL(X,k).  The polynomial mask below is a
// compile-time property of INNER_WORD and round T.  This lets us avoid table
// loads for rows whose delta is identically zero, and optionally synthesize
// one-rotation rows directly from the already-loaded raw delta pack.
constexpr uint32_t r6_crol_mask(uint32_t x,int n) {
  n &= 31; return n ? ((x<<n)|(x>>(32-n))) : x;
}
constexpr uint32_t r6_poly_mask_const(int src,int t) {
  uint32_t p[80]{};
  p[src]=1u;
  for(int i=16;i<=t;i++) p[i]=r6_crol_mask(p[i-3]^p[i-8]^p[i-14]^p[i-16],1);
  return p[t];
}
constexpr int r6_popc_const(uint32_t x) {
  int n=0; while(x){n+=int(x&1u);x>>=1;} return n;
}
constexpr int r6_ctz_const(uint32_t x) {
  int n=0; while((x&1u)==0u){++n;x>>=1;} return n;
}
constexpr int r6_cache_exp0(int src) {
  return src<=1?8:(src<=7?2:(src<=12?4:7));
}
constexpr int r6_cache_exp1(int src) {
  return src<=1?4:(src<=7?4:(src<=12?2:3));
}

template<int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r6_rot_pack(const R5DeltaPack<V,CACHE>& x,int r) {
  R5DeltaPack<V,CACHE> z{};
  #pragma unroll
  for(int i=0;i<V;i++) z.x[i]=rol32(x.x[i],r);
  return z;
}

template<int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r6_zero_pack() {
  R5DeltaPack<V,CACHE> z{}; return z;
}

// MODE=0: skip only mathematically-zero expanded deltas.
// MODE=1: also synthesize popcount-1 polynomial rows with one rotate/candidate.
// MODE=2: MODE=1 plus cache the two most frequently reused single rotations.
template<int T,int MODE,int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r6_hybrid_delta(
    const R5DeltaPack<V,CACHE>& raw,unsigned jbase,
    const R5DeltaPack<V,CACHE>& cache0,const R5DeltaPack<V,CACHE>& cache1) {
  static_assert(T>=16 && T<80,"expanded row");
  constexpr uint32_t pm=r6_poly_mask_const(INNER_WORD,T);
  constexpr int pc=r6_popc_const(pm);
  if constexpr (pc==0) {
    return r6_zero_pack<V,CACHE>();
  } else if constexpr (MODE>=1 && pc==1) {
    constexpr int e=r6_ctz_const(pm);
    if constexpr (e==0) return raw;
    else if constexpr (MODE>=2 && e==r6_cache_exp0(INNER_WORD)) return cache0;
    else if constexpr (MODE>=2 && e==r6_cache_exp1(INNER_WORD)) return cache1;
    else return r6_rot_pack<V,CACHE>(raw,e);
  } else {
    return r5_load_pack<V,CACHE>(1+T-16,jbase);
  }
}

template<int G,int V,int CACHE,int MODE,int PAD=0>
__global__ void k_vec_hybrid(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=16,"G");
  static_assert(V==4 || V==8,"V");
  static_assert(64%(G*V)==0,"G*V");
  static_assert(MODE>=0 && MODE<=2,"MODE");
  constexpr int SW=80-FIRST_WORD,STRIDE=SW+PAD;
  extern __shared__ uint32_t sm[];
  const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;
  const uint64_t gi=uint64_t(blockIdx.x)*groups+gl;
  const bool active=gi<outer_count; const uint64_t outer=outer_base+gi;
  uint32_t *sched=sm+gl*STRIDE;
  if(active && lane==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++) { const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3)); sched[wi-FIRST_WORD]|=uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh; }
    #pragma unroll
    for(int t=16;t<32;t++) {
      const uint32_t a=(t-3<FIRST_WORD)?C_BASE80[t-3]:sched[t-3-FIRST_WORD];
      const uint32_t b=(t-8<FIRST_WORD)?C_BASE80[t-8]:sched[t-8-FIRST_WORD];
      const uint32_t c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD];
      const uint32_t d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,1);
    }
    #pragma unroll
    for(int t=32;t<80;t++) {
      const uint32_t a=(t-6<FIRST_WORD)?C_BASE80[t-6]:sched[t-6-FIRST_WORD];
      const uint32_t b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      const uint32_t c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD];
      const uint32_t d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,2);
    }
  }
  __syncwarp(); if(!active)return;
  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);
  constexpr unsigned CHUNK=G*V;
  #pragma unroll 1
  for(unsigned chunk=0;chunk<64;chunk+=CHUNK) {
    const unsigned jbase=chunk+unsigned(lane)*V;
    DState st[V];
    #pragma unroll
    for(int i=0;i<V;i++) st[i]=common;
    const R5DeltaPack<V,CACHE> raw=r5_load_pack<V,CACHE>(0,jbase);
    R5DeltaPack<V,CACHE> c0{},c1{};
    if constexpr(MODE>=2) {
      c0=r6_rot_pack<V,CACHE>(raw,r6_cache_exp0(INNER_WORD));
      c1=r6_rot_pack<V,CACHE>(raw,r6_cache_exp1(INNER_WORD));
    }
    const uint32_t w0=sched[INNER_WORD-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) step_ch(st[i],w0+raw.x[i]);
    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++) {
      const uint32_t bw=sched[t-FIRST_WORD];
      #pragma unroll
      for(int i=0;i<V;i++) step_ch(st[i],bw);
    }

    #define R6_HYB_ROUND(T,STEP) do { \
      const uint32_t bw=sched[(T)-FIRST_WORD]; \
      const auto dv=r6_hybrid_delta<(T),MODE,V,CACHE>(raw,jbase,c0,c1); \
      _Pragma("unroll") for(int i=0;i<V;i++) STEP(st[i],bw^dv.x[i]); \
    } while(0)
    R6_HYB_ROUND(16,step_ch); R6_HYB_ROUND(17,step_ch); R6_HYB_ROUND(18,step_ch); R6_HYB_ROUND(19,step_ch);
    R6_HYB_ROUND(20,step_par1); R6_HYB_ROUND(21,step_par1); R6_HYB_ROUND(22,step_par1); R6_HYB_ROUND(23,step_par1); R6_HYB_ROUND(24,step_par1);
    R6_HYB_ROUND(25,step_par1); R6_HYB_ROUND(26,step_par1); R6_HYB_ROUND(27,step_par1); R6_HYB_ROUND(28,step_par1); R6_HYB_ROUND(29,step_par1);
    R6_HYB_ROUND(30,step_par1); R6_HYB_ROUND(31,step_par1); R6_HYB_ROUND(32,step_par1); R6_HYB_ROUND(33,step_par1); R6_HYB_ROUND(34,step_par1);
    R6_HYB_ROUND(35,step_par1); R6_HYB_ROUND(36,step_par1); R6_HYB_ROUND(37,step_par1); R6_HYB_ROUND(38,step_par1); R6_HYB_ROUND(39,step_par1);
    R6_HYB_ROUND(40,step_maj); R6_HYB_ROUND(41,step_maj); R6_HYB_ROUND(42,step_maj); R6_HYB_ROUND(43,step_maj); R6_HYB_ROUND(44,step_maj);
    R6_HYB_ROUND(45,step_maj); R6_HYB_ROUND(46,step_maj); R6_HYB_ROUND(47,step_maj); R6_HYB_ROUND(48,step_maj); R6_HYB_ROUND(49,step_maj);
    R6_HYB_ROUND(50,step_maj); R6_HYB_ROUND(51,step_maj); R6_HYB_ROUND(52,step_maj); R6_HYB_ROUND(53,step_maj); R6_HYB_ROUND(54,step_maj);
    R6_HYB_ROUND(55,step_maj); R6_HYB_ROUND(56,step_maj); R6_HYB_ROUND(57,step_maj); R6_HYB_ROUND(58,step_maj); R6_HYB_ROUND(59,step_maj);
    R6_HYB_ROUND(60,step_par3); R6_HYB_ROUND(61,step_par3); R6_HYB_ROUND(62,step_par3); R6_HYB_ROUND(63,step_par3); R6_HYB_ROUND(64,step_par3);
    R6_HYB_ROUND(65,step_par3); R6_HYB_ROUND(66,step_par3); R6_HYB_ROUND(67,step_par3); R6_HYB_ROUND(68,step_par3); R6_HYB_ROUND(69,step_par3);
    R6_HYB_ROUND(70,step_par3); R6_HYB_ROUND(71,step_par3); R6_HYB_ROUND(72,step_par3); R6_HYB_ROUND(73,step_par3); R6_HYB_ROUND(74,step_par3);
    R6_HYB_ROUND(75,step_par3); R6_HYB_ROUND(76,step_par3); R6_HYB_ROUND(77,step_par3); R6_HYB_ROUND(78,step_par3);
    #undef R6_HYB_ROUND
    {
      constexpr int T=79; const uint32_t bw=sched[T-FIRST_WORD]; const auto dv=r6_hybrid_delta<T,MODE,V,CACHE>(raw,jbase,c0,c1);
      #pragma unroll
      for(int i=0;i<V;i++) { const uint32_t h=final_a79(st[i],bw^dv.x[i])+C_HIN[0]; if((h&mask)==target)
        atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+unsigned(i)))); }
    }
  }
}

// ---------------- Round-6 role-rotated 6+1 control ----------------
template<int G,int V,int CACHE,int PAD=0>
__global__ void k_vec_rr(uint64_t outer_base,uint64_t outer_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=16,"G"); static_assert(V==4 || V==8,"V"); static_assert(64%(G*V)==0,"G*V");
  constexpr int SW=80-FIRST_WORD,STRIDE=SW+PAD; extern __shared__ uint32_t sm[];
  const int group_local=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G; const uint64_t gi=uint64_t(blockIdx.x)*groups+group_local; const bool active=gi<outer_count; const uint64_t outer=outer_base+gi; uint32_t *sched=sm+group_local*STRIDE;
  if(active && lane==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++){const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));sched[wi-FIRST_WORD]|=uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;}
    #pragma unroll
    for(int t=16;t<32;t++){const uint32_t a=(t-3<FIRST_WORD)?C_BASE80[t-3]:sched[t-3-FIRST_WORD],b=(t-8<FIRST_WORD)?C_BASE80[t-8]:sched[t-8-FIRST_WORD],c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD],d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];sched[t-FIRST_WORD]=rol32(a^b^c^d,1);}
    #pragma unroll
    for(int t=32;t<80;t++){const uint32_t a=(t-6<FIRST_WORD)?C_BASE80[t-6]:sched[t-6-FIRST_WORD],b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD],c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD],d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];sched[t-FIRST_WORD]=rol32(a^b^c^d,2);}
  } __syncwarp(); if(!active)return;
  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);
  constexpr unsigned CHUNK=G*V;
  #pragma unroll 1
  for(unsigned chunk=0;chunk<64;chunk+=CHUNK) { const unsigned jbase=chunk+unsigned(lane)*V; RRState st[V];
    #pragma unroll
    for(int i=0;i<V;i++) st[i]={common.a,common.b,common.c,common.d,common.e}; R5DeltaPack<V,CACHE> dv{};
    if constexpr (INNER_WORD <= 0) {
      const uint32_t bw=sched[0-FIRST_WORD];
      if constexpr (INNER_WORD == 0) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<0,INNER_WORD,0>(st[i],(INNER_WORD==0 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 1) {
      const uint32_t bw=sched[1-FIRST_WORD];
      if constexpr (INNER_WORD == 1) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<1,INNER_WORD,0>(st[i],(INNER_WORD==1 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 2) {
      const uint32_t bw=sched[2-FIRST_WORD];
      if constexpr (INNER_WORD == 2) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<2,INNER_WORD,0>(st[i],(INNER_WORD==2 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 3) {
      const uint32_t bw=sched[3-FIRST_WORD];
      if constexpr (INNER_WORD == 3) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<3,INNER_WORD,0>(st[i],(INNER_WORD==3 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 4) {
      const uint32_t bw=sched[4-FIRST_WORD];
      if constexpr (INNER_WORD == 4) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<4,INNER_WORD,0>(st[i],(INNER_WORD==4 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 5) {
      const uint32_t bw=sched[5-FIRST_WORD];
      if constexpr (INNER_WORD == 5) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<5,INNER_WORD,0>(st[i],(INNER_WORD==5 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 6) {
      const uint32_t bw=sched[6-FIRST_WORD];
      if constexpr (INNER_WORD == 6) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<6,INNER_WORD,0>(st[i],(INNER_WORD==6 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 7) {
      const uint32_t bw=sched[7-FIRST_WORD];
      if constexpr (INNER_WORD == 7) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<7,INNER_WORD,0>(st[i],(INNER_WORD==7 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 8) {
      const uint32_t bw=sched[8-FIRST_WORD];
      if constexpr (INNER_WORD == 8) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<8,INNER_WORD,0>(st[i],(INNER_WORD==8 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 9) {
      const uint32_t bw=sched[9-FIRST_WORD];
      if constexpr (INNER_WORD == 9) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<9,INNER_WORD,0>(st[i],(INNER_WORD==9 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 10) {
      const uint32_t bw=sched[10-FIRST_WORD];
      if constexpr (INNER_WORD == 10) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<10,INNER_WORD,0>(st[i],(INNER_WORD==10 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 11) {
      const uint32_t bw=sched[11-FIRST_WORD];
      if constexpr (INNER_WORD == 11) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<11,INNER_WORD,0>(st[i],(INNER_WORD==11 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 12) {
      const uint32_t bw=sched[12-FIRST_WORD];
      if constexpr (INNER_WORD == 12) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<12,INNER_WORD,0>(st[i],(INNER_WORD==12 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 13) {
      const uint32_t bw=sched[13-FIRST_WORD];
      if constexpr (INNER_WORD == 13) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<13,INNER_WORD,0>(st[i],(INNER_WORD==13 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 14) {
      const uint32_t bw=sched[14-FIRST_WORD];
      if constexpr (INNER_WORD == 14) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<14,INNER_WORD,0>(st[i],(INNER_WORD==14 ? bw+dv.x[i] : bw));
    }
    if constexpr (INNER_WORD <= 15) {
      const uint32_t bw=sched[15-FIRST_WORD];
      if constexpr (INNER_WORD == 15) dv=r5_load_pack<V,CACHE>(0,jbase);
      #pragma unroll
      for(int i=0;i<V;i++) rr_round<15,INNER_WORD,0>(st[i],(INNER_WORD==15 ? bw+dv.x[i] : bw));
    }
    dv=r5_load_pack<V,CACHE>(1,jbase); const uint32_t bw16=sched[16-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<16,INNER_WORD,0>(st[i],bw16^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(2,jbase); const uint32_t bw17=sched[17-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<17,INNER_WORD,0>(st[i],bw17^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(3,jbase); const uint32_t bw18=sched[18-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<18,INNER_WORD,0>(st[i],bw18^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(4,jbase); const uint32_t bw19=sched[19-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<19,INNER_WORD,0>(st[i],bw19^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(5,jbase); const uint32_t bw20=sched[20-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<20,INNER_WORD,1>(st[i],bw20^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(6,jbase); const uint32_t bw21=sched[21-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<21,INNER_WORD,1>(st[i],bw21^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(7,jbase); const uint32_t bw22=sched[22-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<22,INNER_WORD,1>(st[i],bw22^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(8,jbase); const uint32_t bw23=sched[23-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<23,INNER_WORD,1>(st[i],bw23^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(9,jbase); const uint32_t bw24=sched[24-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<24,INNER_WORD,1>(st[i],bw24^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(10,jbase); const uint32_t bw25=sched[25-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<25,INNER_WORD,1>(st[i],bw25^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(11,jbase); const uint32_t bw26=sched[26-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<26,INNER_WORD,1>(st[i],bw26^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(12,jbase); const uint32_t bw27=sched[27-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<27,INNER_WORD,1>(st[i],bw27^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(13,jbase); const uint32_t bw28=sched[28-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<28,INNER_WORD,1>(st[i],bw28^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(14,jbase); const uint32_t bw29=sched[29-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<29,INNER_WORD,1>(st[i],bw29^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(15,jbase); const uint32_t bw30=sched[30-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<30,INNER_WORD,1>(st[i],bw30^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(16,jbase); const uint32_t bw31=sched[31-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<31,INNER_WORD,1>(st[i],bw31^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(17,jbase); const uint32_t bw32=sched[32-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<32,INNER_WORD,1>(st[i],bw32^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(18,jbase); const uint32_t bw33=sched[33-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<33,INNER_WORD,1>(st[i],bw33^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(19,jbase); const uint32_t bw34=sched[34-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<34,INNER_WORD,1>(st[i],bw34^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(20,jbase); const uint32_t bw35=sched[35-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<35,INNER_WORD,1>(st[i],bw35^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(21,jbase); const uint32_t bw36=sched[36-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<36,INNER_WORD,1>(st[i],bw36^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(22,jbase); const uint32_t bw37=sched[37-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<37,INNER_WORD,1>(st[i],bw37^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(23,jbase); const uint32_t bw38=sched[38-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<38,INNER_WORD,1>(st[i],bw38^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(24,jbase); const uint32_t bw39=sched[39-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<39,INNER_WORD,1>(st[i],bw39^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(25,jbase); const uint32_t bw40=sched[40-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<40,INNER_WORD,2>(st[i],bw40^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(26,jbase); const uint32_t bw41=sched[41-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<41,INNER_WORD,2>(st[i],bw41^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(27,jbase); const uint32_t bw42=sched[42-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<42,INNER_WORD,2>(st[i],bw42^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(28,jbase); const uint32_t bw43=sched[43-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<43,INNER_WORD,2>(st[i],bw43^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(29,jbase); const uint32_t bw44=sched[44-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<44,INNER_WORD,2>(st[i],bw44^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(30,jbase); const uint32_t bw45=sched[45-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<45,INNER_WORD,2>(st[i],bw45^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(31,jbase); const uint32_t bw46=sched[46-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<46,INNER_WORD,2>(st[i],bw46^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(32,jbase); const uint32_t bw47=sched[47-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<47,INNER_WORD,2>(st[i],bw47^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(33,jbase); const uint32_t bw48=sched[48-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<48,INNER_WORD,2>(st[i],bw48^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(34,jbase); const uint32_t bw49=sched[49-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<49,INNER_WORD,2>(st[i],bw49^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(35,jbase); const uint32_t bw50=sched[50-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<50,INNER_WORD,2>(st[i],bw50^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(36,jbase); const uint32_t bw51=sched[51-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<51,INNER_WORD,2>(st[i],bw51^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(37,jbase); const uint32_t bw52=sched[52-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<52,INNER_WORD,2>(st[i],bw52^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(38,jbase); const uint32_t bw53=sched[53-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<53,INNER_WORD,2>(st[i],bw53^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(39,jbase); const uint32_t bw54=sched[54-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<54,INNER_WORD,2>(st[i],bw54^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(40,jbase); const uint32_t bw55=sched[55-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<55,INNER_WORD,2>(st[i],bw55^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(41,jbase); const uint32_t bw56=sched[56-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<56,INNER_WORD,2>(st[i],bw56^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(42,jbase); const uint32_t bw57=sched[57-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<57,INNER_WORD,2>(st[i],bw57^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(43,jbase); const uint32_t bw58=sched[58-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<58,INNER_WORD,2>(st[i],bw58^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(44,jbase); const uint32_t bw59=sched[59-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<59,INNER_WORD,2>(st[i],bw59^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(45,jbase); const uint32_t bw60=sched[60-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<60,INNER_WORD,3>(st[i],bw60^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(46,jbase); const uint32_t bw61=sched[61-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<61,INNER_WORD,3>(st[i],bw61^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(47,jbase); const uint32_t bw62=sched[62-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<62,INNER_WORD,3>(st[i],bw62^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(48,jbase); const uint32_t bw63=sched[63-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<63,INNER_WORD,3>(st[i],bw63^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(49,jbase); const uint32_t bw64=sched[64-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<64,INNER_WORD,3>(st[i],bw64^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(50,jbase); const uint32_t bw65=sched[65-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<65,INNER_WORD,3>(st[i],bw65^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(51,jbase); const uint32_t bw66=sched[66-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<66,INNER_WORD,3>(st[i],bw66^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(52,jbase); const uint32_t bw67=sched[67-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<67,INNER_WORD,3>(st[i],bw67^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(53,jbase); const uint32_t bw68=sched[68-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<68,INNER_WORD,3>(st[i],bw68^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(54,jbase); const uint32_t bw69=sched[69-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<69,INNER_WORD,3>(st[i],bw69^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(55,jbase); const uint32_t bw70=sched[70-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<70,INNER_WORD,3>(st[i],bw70^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(56,jbase); const uint32_t bw71=sched[71-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<71,INNER_WORD,3>(st[i],bw71^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(57,jbase); const uint32_t bw72=sched[72-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<72,INNER_WORD,3>(st[i],bw72^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(58,jbase); const uint32_t bw73=sched[73-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<73,INNER_WORD,3>(st[i],bw73^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(59,jbase); const uint32_t bw74=sched[74-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<74,INNER_WORD,3>(st[i],bw74^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(60,jbase); const uint32_t bw75=sched[75-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<75,INNER_WORD,3>(st[i],bw75^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(61,jbase); const uint32_t bw76=sched[76-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<76,INNER_WORD,3>(st[i],bw76^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(62,jbase); const uint32_t bw77=sched[77-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<77,INNER_WORD,3>(st[i],bw77^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(63,jbase); const uint32_t bw78=sched[78-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<78,INNER_WORD,3>(st[i],bw78^dv.x[i]);
    dv=r5_load_pack<V,CACHE>(64,jbase); const uint32_t bw79=sched[79-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++) rr_round<79,INNER_WORD,3>(st[i],bw79^dv.x[i]);
    #pragma unroll
    for(int i=0;i<V;i++) { const uint32_t h=rr_semantic_a<80-INNER_WORD>(st[i])+C_HIN[0]; if((h&mask)==target) atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+i))); }
  }
}

// ---------------- Round-6 5+2 inner-pair family ----------------
// Five nonce characters define the outer schedule.  The last two are searched
// as a 4096-value inner space.  All message-schedule dependence is affine over
// XOR, so either two compact 64-entry byte tables can be combined per round or
// a precombined 4096-entry pair table can provide one delta vector load.
//
// REP=0: separable A/B tables (small footprint, scalar A + vector B loads)
// REP=1: precombined pair table (large L2-resident footprint, one vector load)
template<int CACHE>
__device__ __forceinline__ uint4 r6_load4(const uint32_t *p) {
  uint4 v;
  if constexpr (CACHE==0) asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  else if constexpr (CACHE==1) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  else asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
  return v;
}

template<int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r6_pair_pack(int row,unsigned a,unsigned bbase) {
  R5DeltaPack<V,CACHE> out{};
  #pragma unroll
  for(int q=0;q<V;q+=4) {
    uint4 z=r6_load4<CACHE>(G_PAIR_TABLE + row*4096 + a*64 + bbase + q);
    out.x[q]=z.x;out.x[q+1]=z.y;out.x[q+2]=z.z;out.x[q+3]=z.w;
  }
  return out;
}

template<int V,int CACHE>
__device__ __forceinline__ R5DeltaPack<V,CACHE> r6_sep_pack(int row,unsigned a,unsigned bbase) {
  R5DeltaPack<V,CACHE> out{};
  const uint32_t av = CACHE==1 ? __ldg(G_PAIR_A + row*64 + a) : G_PAIR_A[row*64+a];
  #pragma unroll
  for(int q=0;q<V;q+=4) {
    uint4 z=r6_load4<CACHE>(G_PAIR_B + row*64 + bbase + q);
    out.x[q]=av^z.x;out.x[q+1]=av^z.y;out.x[q+2]=av^z.z;out.x[q+3]=av^z.w;
  }
  return out;
}

template<int G,int V,int CACHE,int REP,int PAD=0>
__global__ void k_pair52(uint64_t outer5_base,uint64_t outer5_count,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=16,"G");
  static_assert(V==4 || V==8 || V==16,"V");
  static_assert(64%(G*V)==0,"G*V divides 64 B-values");
  static_assert(REP==0 || REP==1,"REP");
  constexpr int SW=80-FIRST_WORD, STRIDE=SW+PAD;
  extern __shared__ uint32_t sm[];
  const int group_local=threadIdx.x/G, lane=threadIdx.x&(G-1), groups=blockDim.x/G;
  const uint64_t gi=uint64_t(blockIdx.x)*groups+group_local;
  const bool active=gi<outer5_count; const uint64_t outer5=outer5_base+gi;
  uint32_t *sched=sm+group_local*STRIDE;
  if(active && lane==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<PAIR_OUTER_LEN;k++) { const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3)); sched[wi-FIRST_WORD]|=uint32_t(C_CHARSET[(outer5>>(6*k))&63u])<<sh; }
    #pragma unroll
    for(int t=16;t<32;t++) {
      const uint32_t a=(t-3<FIRST_WORD)?C_BASE80[t-3]:sched[t-3-FIRST_WORD];
      const uint32_t b=(t-8<FIRST_WORD)?C_BASE80[t-8]:sched[t-8-FIRST_WORD];
      const uint32_t c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD];
      const uint32_t d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,1);
    }
    #pragma unroll
    for(int t=32;t<80;t++) {
      const uint32_t a=(t-6<FIRST_WORD)?C_BASE80[t-6]:sched[t-6-FIRST_WORD];
      const uint32_t b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];
      const uint32_t c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD];
      const uint32_t d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];
      sched[t-FIRST_WORD]=rol32(a^b^c^d,2);
    }
  }
  __syncwarp(); if(!active)return;
  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<PAIR_FIRST_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);
  constexpr unsigned CHUNK=G*V;
  #pragma unroll 1
  for(unsigned a=0;a<64;a++) {
    #pragma unroll 1
    for(unsigned chunk=0;chunk<64;chunk+=CHUNK) {
      const unsigned bbase=chunk+unsigned(lane)*V;
      DState st[V];
      #pragma unroll
      for(int i=0;i<V;i++) st[i]=common;
      R5DeltaPack<V,CACHE> dv{};
      #pragma unroll
      for(int t=PAIR_FIRST_WORD;t<16;t++) {
        const int row=t-PAIR_FIRST_WORD;
        if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase);
        const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) step_ch(st[i],bw^dv.x[i]);
      }
      #pragma unroll
      for(int t=16;t<20;t++) {
        const int row=t-PAIR_FIRST_WORD;
        if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase);
        const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) step_ch(st[i],bw^dv.x[i]);
      }
      #pragma unroll
      for(int t=20;t<40;t++) {
        const int row=t-PAIR_FIRST_WORD;
        if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase);
        const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) step_par1(st[i],bw^dv.x[i]);
      }
      #pragma unroll
      for(int t=40;t<60;t++) {
        const int row=t-PAIR_FIRST_WORD;
        if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase);
        const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) step_maj(st[i],bw^dv.x[i]);
      }
      #pragma unroll
      for(int t=60;t<79;t++) {
        const int row=t-PAIR_FIRST_WORD;
        if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase);
        const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) step_par3(st[i],bw^dv.x[i]);
      }
      { const int t=79,row=t-PAIR_FIRST_WORD; if constexpr(REP==0) dv=r6_sep_pack<V,CACHE>(row,a,bbase); else dv=r6_pair_pack<V,CACHE>(row,a,bbase); const uint32_t bw=sched[t-FIRST_WORD];
        #pragma unroll
        for(int i=0;i<V;i++) { const uint32_t h=final_a79(st[i],bw^dv.x[i])+C_HIN[0]; if((h&mask)==target) {
          const uint64_t outer6=outer5 | (uint64_t(a)<<(6*PAIR_OUTER_LEN)); const uint64_t id=(outer6<<6)|uint64_t(bbase+i);
          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)id); } }
      }
    }
  }
}

// ---------------- Benchmark harness ----------------

struct Row {
  const char *name; double ghs; int block,regs,static_smem,dynamic_smem,local_bytes,active_blocks,warps; double occupancy;
};

template<typename K>
static void resources(Row &r,K kernel,int block,size_t dynamic_smem,const cudaDeviceProp &p) {
  cudaFuncAttributes a{}; CUDA_OK(cudaFuncGetAttributes(&a,kernel));
  int blocks=0; CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,block,dynamic_smem));
  r.block=block; r.regs=a.numRegs; r.static_smem=(int)a.sharedSizeBytes; r.dynamic_smem=(int)dynamic_smem;
  r.local_bytes=(int)a.localSizeBytes; r.active_blocks=blocks; r.warps=blocks*(block/32);
  const int maxwarps=p.maxThreadsPerMultiProcessor/32;
  r.occupancy=maxwarps?100.0*double(r.warps)/double(maxwarps):0.0;
}

static double elapsed_ghs(cudaEvent_t a,cudaEvent_t b,double hashes) {
  float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,a,b)); return hashes/(double(ms)*1e6);
}

static Row bench_one(uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  Row r{"one-thread-roll16"}; constexpr int B=256; const uint64_t count=outer_count*64u;
  const dim3 block(B),grid((unsigned)((count+B-1)/B));
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_one_thread<<<grid,block>>>(0,count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_one_thread<<<grid,block>>>(0,count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(count)*repeats); resources(r,k_one_thread,B,0,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

static Row bench_named80(uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  Row r{"one-thread-named80"}; constexpr int B=128; const uint64_t count=outer_count*64u;
  const dim3 block(B),grid((unsigned)((count+B-1)/B));
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_named80<<<grid,block>>>(0,count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_named80<<<grid,block>>>(0,count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(count)*repeats); resources(r,k_named80,B,0,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

template<int G,int N,int DM>
static Row bench_group(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; const int groups=B/G; const dim3 block(B),grid((unsigned)((outer_count+groups-1)/groups));
  const size_t smem=((DM==1?DELTA_WORDS:0)+(B/G)*80u)*sizeof(uint32_t);
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_group<G,N,DM><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_group<G,N,DM><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(outer_count)*64.0*repeats); resources(r,k_group<G,N,DM>,B,smem,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}


static bool cpu_verify_id(uint64_t id,uint32_t expected_h0,const uint32_t hin[5]);


template<int G,int N,int DM,int PAD>
static Row bench_group_pad(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; const int groups=B/G; const dim3 block(B),grid((unsigned)((outer_count+groups-1)/groups));
  const size_t smem=((DM==1?DELTA_WORDS:0)+(B/G)*(80+PAD))*sizeof(uint32_t);
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_group<G,N,DM,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_group<G,N,DM,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(outer_count)*64.0*repeats); resources(r,k_group<G,N,DM,PAD>,B,smem,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

template<int G,int N,int DM,int PAD>
static bool correctness_group_pad(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  const int groups=B/G; const size_t smem=((DM==1?DELTA_WORDS:0)+(B/G)*(80+PAD))*4u;
  k_group<G,N,DM,PAD><<<(outer_count+groups-1)/groups,B,smem>>>(0,outer_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}



template<int G,int N,int DM,int PAD=0>
static Row bench_group_compact(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; const int groups=B/G; const dim3 block(B),grid((unsigned)((outer_count+groups-1)/groups));
  const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*sizeof(uint32_t);
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_group_compact<G,N,DM,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_group_compact<G,N,DM,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(outer_count)*64.0*repeats); resources(r,k_group_compact<G,N,DM,PAD>,B,smem,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

template<int G,int N,int DM,int PAD=0>
static bool correctness_group_compact(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  const int groups=B/G; const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*4u;
  k_group_compact<G,N,DM,PAD><<<(outer_count+groups-1)/groups,B,smem>>>(0,outer_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}


template<bool NC,int PAD=0>
static Row bench_group_compact_g4_vec4(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; constexpr int G=4; const int groups=B/G; const dim3 block(B),grid((unsigned)((outer_count+groups-1)/groups));
  const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*sizeof(uint32_t);
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_group_compact_g4_vec4<NC,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_group_compact_g4_vec4<NC,PAD><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(outer_count)*64.0*repeats); resources(r,k_group_compact_g4_vec4<NC,PAD>,B,smem,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

template<bool NC,int PAD=0>
static bool correctness_group_compact_g4_vec4(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice)); constexpr int G=4;
  const int groups=B/G; const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*4u;
  k_group_compact_g4_vec4<NC,PAD><<<(outer_count+groups-1)/groups,B,smem>>>(0,outer_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}


template<int G,int V,int CACHE,int PAD=0,bool COMBINE_MATCH=false>
static Row bench_group_compact_vec(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; const int groups=B/G; const dim3 block(B),grid((unsigned)((outer_count+groups-1)/groups));
  const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*sizeof(uint32_t);
  cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
  k_group_compact_vec<G,V,CACHE,PAD,COMBINE_MATCH><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a)); for(int i=0;i<repeats;i++) k_group_compact_vec<G,V,CACHE,PAD,COMBINE_MATCH><<<grid,block,smem>>>(0,outer_count,target,0u,dwin); CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));
  r.ghs=elapsed_ghs(a,b,double(outer_count)*64.0*repeats); resources(r,k_group_compact_vec<G,V,CACHE,PAD,COMBINE_MATCH>,B,smem,p);
  CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b)); return r;
}

template<int G,int V,int CACHE,int PAD=0,bool COMBINE_MATCH=false>
static bool correctness_group_compact_vec(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  const int groups=B/G; const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*4u;
  k_group_compact_vec<G,V,CACHE,PAD,COMBINE_MATCH><<<(outer_count+groups-1)/groups,B,smem>>>(0,outer_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}


template<int G,int V,int CACHE,int REP,int PAD=0>
static Row bench_pair52(const char *name,uint64_t outer6_equiv,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p,int B) {
  Row r{name}; const int groups=B/G;
  // A 5-char outer covers 4096 candidates instead of 64, so divide the
  // six-char baseline outer count by 64 to benchmark approximately equal work.
  const uint64_t outer5_count=std::max<uint64_t>(1,(outer6_equiv+63)/64);
  const dim3 block(B),grid((unsigned)((outer5_count+groups-1)/groups));
  const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*sizeof(uint32_t);
  cudaEvent_t x,y; CUDA_OK(cudaEventCreate(&x));CUDA_OK(cudaEventCreate(&y));
  k_pair52<G,V,CACHE,REP,PAD><<<grid,block,smem>>>(0,outer5_count,target,0u,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(x)); for(int i=0;i<repeats;i++) k_pair52<G,V,CACHE,REP,PAD><<<grid,block,smem>>>(0,outer5_count,target,0u,dwin); CUDA_OK(cudaEventRecord(y));CUDA_OK(cudaEventSynchronize(y));
  r.ghs=elapsed_ghs(x,y,double(outer5_count)*4096.0*repeats); resources(r,k_pair52<G,V,CACHE,REP,PAD>,B,smem,p);
  CUDA_OK(cudaEventDestroy(x));CUDA_OK(cudaEventDestroy(y)); return r;
}

template<int G,int V,int CACHE,int REP,int PAD=0>
static bool correctness_pair52(uint64_t outer5_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  const int groups=B/G; const size_t smem=(B/G)*(80-FIRST_WORD+PAD)*4u;
  k_pair52<G,V,CACHE,REP,PAD><<<(outer5_count+groups-1)/groups,B,smem>>>(0,outer5_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}

static bool cpu_verify_id(uint64_t id,uint32_t expected_h0,const uint32_t hin[5]) {
  uint32_t w[16],d[5];make_words_for_id(id,w);cpu_compress(hin,w,d);return d[0]==expected_h0;
}

static bool correctness_named80(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice)); const uint64_t count=outer_count*64u;
  k_named80<<<(count+127)/128,128>>>(0,count,target,0xffffffffu,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost)); return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}

template<int G,int N,int DM>
static bool correctness_group(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5],int B) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  const int groups=B/G; const size_t smem=((DM==1?DELTA_WORDS:0)+(B/G)*80u)*4u;
  k_group<G,N,DM><<<(outer_count+groups-1)/groups,B,smem>>>(0,outer_count,target,0xffffffffu,dwin);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost));
  return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}

static bool correctness_one(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice)); const uint64_t count=outer_count*64u;
  k_one_thread<<<(count+255)/256,256>>>(0,count,target,0xffffffffu,dwin); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  uint64_t got;CUDA_OK(cudaMemcpy(&got,dwin,8,cudaMemcpyDeviceToHost)); return got!=NO_WINNER && cpu_verify_id(got,target,hin);
}

static void print_row(const Row &r) {
  printf("%-24s %9.3f GH/s  B=%3d regs=%3d local=%4d  smem=%5d+%5d  blocks/SM=%2d warps=%2d occ=%5.1f%%\n",
    r.name,r.ghs,r.block,r.regs,r.local_bytes,r.static_smem,r.dynamic_smem,r.active_blocks,r.warps,r.occupancy);
}
static void csv_row(FILE *f,const Row &r) {
  fprintf(f,"%d,%d,%d,%s,%.9f,%d,%d,%d,%d,%d,%d,%d,%.3f\n",NONCE_OFF,FIRST_WORD,INNER_WORD,r.name,r.ghs,r.block,r.regs,r.local_bytes,r.static_smem,r.dynamic_smem,r.active_blocks,r.warps,r.occupancy);
}

#ifndef STUDY_NO_MAIN

template<int KIND>
static Row r4_bench_base(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  if constexpr (KIND==0) return bench_one(outer_count,repeats,target,dwin,p);
  else return bench_named80(outer_count,repeats,target,dwin,p);
}
template<int KIND>
static bool r4_correct_base(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  if constexpr (KIND==0) return correctness_one(outer_count,target,dwin,hin);
  else return correctness_named80(outer_count,target,dwin,hin);
}

template<int G,int N,int DM,int PAD,int B>
static Row r4_bench_full(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  return bench_group_pad<G,N,DM,PAD>(name,outer_count,repeats,target,dwin,p,B);
}
template<int G,int N,int DM,int PAD,int B>
static bool r4_correct_full(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  return correctness_group_pad<G,N,DM,PAD>(outer_count,target,dwin,hin,B);
}

template<int G,int N,int DM,int PAD,int B>
static Row r4_bench_compact(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  return bench_group_compact<G,N,DM,PAD>(name,outer_count,repeats,target,dwin,p,B);
}
template<int G,int N,int DM,int PAD,int B>
static bool r4_correct_compact(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  return correctness_group_compact<G,N,DM,PAD>(outer_count,target,dwin,hin,B);
}

template<bool NC,int PAD,int B>
static Row r4_bench_vec4(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  return bench_group_compact_g4_vec4<NC,PAD>(name,outer_count,repeats,target,dwin,p,B);
}
template<bool NC,int PAD,int B>
static bool r4_correct_vec4(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  return correctness_group_compact_g4_vec4<NC,PAD>(outer_count,target,dwin,hin,B);
}


template<int G,int V,int CACHE,int PAD,int B,bool COMBINE>
static Row r5_bench_vec(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  return bench_group_compact_vec<G,V,CACHE,PAD,COMBINE>(name,outer_count,repeats,target,dwin,p,B);
}
template<int G,int V,int CACHE,int PAD,int B,bool COMBINE>
static bool r5_correct_vec(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  return correctness_group_compact_vec<G,V,CACHE,PAD,COMBINE>(outer_count,target,dwin,hin,B);
}


template<int G,int V,int CACHE,int REP,int PAD,int B>
static Row r6_bench_pair(const char *name,uint64_t outer_count,int repeats,uint32_t target,uint64_t *dwin,const cudaDeviceProp &p) {
  return bench_pair52<G,V,CACHE,REP,PAD>(name,outer_count,repeats,target,dwin,p,B);
}
template<int G,int V,int CACHE,int REP,int PAD,int B>
static bool r6_correct_pair(uint64_t outer_count,uint32_t target,uint64_t *dwin,const uint32_t hin[5]) {
  return correctness_pair52<G,V,CACHE,REP,PAD>(outer_count,target,dwin,hin,B);
}

using R4BenchFn = Row (*)(const char*,uint64_t,int,uint32_t,uint64_t*,const cudaDeviceProp&);
using R4CorrectFn = bool (*)(uint64_t,uint32_t,uint64_t*,const uint32_t*);
struct R4Variant {
  const char *name,*family,*memory;
  int g,n,dm,pad,block;
  R4BenchFn bench;
  R4CorrectFn correct;
};

static const R4Variant R4_VARIANTS[] = {
#define R4_BASE(NAME,KIND) {NAME,"baseline","-",0,0,-1,0,(KIND)==0?256:128,&r4_bench_base<KIND>,&r4_correct_base<KIND>},
#define R4_FULL(NAME,G,N,DM,PAD,B) {NAME,"full",(DM)==0?"global":((DM)==1?"shared":((DM)==2?"constant":"readonly")),G,N,DM,PAD,B,&r4_bench_full<G,N,DM,PAD,B>,&r4_correct_full<G,N,DM,PAD,B>},
#define R4_COMPACT(NAME,G,N,DM,PAD,B) {NAME,"compact",(DM)==0?"global":((DM)==2?"constant":"readonly"),G,N,DM,PAD,B,&r4_bench_compact<G,N,DM,PAD,B>,&r4_correct_compact<G,N,DM,PAD,B>},
#define R4_VEC4(NAME,NC,PAD,B) {NAME,"vec4",(NC)?"nc":"ca",4,4,(NC)?4:5,PAD,B,&r4_bench_vec4<(bool)NC,PAD,B>,&r4_correct_vec4<(bool)NC,PAD,B>},
#define R5_VEC(NAME,G,V,CACHE,PAD,B,COMBINE) {NAME,"vec" #V,(CACHE)==0?"ca":((CACHE)==1?"nc":((CACHE)==2?"cg":"const")),G,V,CACHE,PAD,B,&r5_bench_vec<G,V,CACHE,PAD,B,(bool)COMBINE>,&r5_correct_vec<G,V,CACHE,PAD,B,(bool)COMBINE>},
#define R6_PAIR(NAME,G,V,CACHE,REP,PAD,B) {NAME,(REP)==0?"pair52-sep":"pair52-table",(CACHE)==0?"ca":((CACHE)==1?"nc":"cg"),G,V,CACHE,PAD,B,&r6_bench_pair<G,V,CACHE,REP,PAD,B>,&r6_correct_pair<G,V,CACHE,REP,PAD,B>},
#include "variants.inc"
#undef R4_BASE
#undef R4_FULL
#undef R4_COMPACT
#undef R4_VEC4
#undef R5_VEC
#undef R6_PAIR
};
static constexpr size_t R4_VARIANT_COUNT=sizeof(R4_VARIANTS)/sizeof(R4_VARIANTS[0]);

static size_t r4_required_dynamic_smem(const R4Variant &v) {
  if(v.g<=0) return 0;
  const size_t groups=(size_t)v.block/(size_t)v.g;
  if(!strcmp(v.family,"full")) {
    size_t words=groups*(size_t)(80+v.pad);
    if(!strcmp(v.memory,"shared")) words += DELTA_WORDS;
    return words*sizeof(uint32_t);
  }
  // compact, old vec4, and generic vecN all store only W[FIRST_WORD..79].
  return groups*(size_t)(80-FIRST_WORD+v.pad)*sizeof(uint32_t);
}

static bool r4_name_selected(const char *name,const std::vector<std::string> &only,const std::vector<std::string> &filters,const std::vector<std::string> &exclude) {
  if(!only.empty()) {
    bool hit=false; for(const auto &x:only) if(x==name){hit=true;break;} if(!hit)return false;
  }
  if(!filters.empty()) {
    bool hit=false; for(const auto &x:filters) if(strstr(name,x.c_str())){hit=true;break;} if(!hit)return false;
  }
  for(const auto &x:exclude) if(x==name) return false;
  return true;
}

static void r4_write_csv_header(FILE *f) {
  fprintf(f,"sample,nonce_off,first_word,inner_word,variant,family,memory,g,ilp,pad,ghs,block,regs,local_bytes,static_smem,dynamic_smem,active_blocks_per_sm,warps_per_sm,occupancy_pct\n");
}
static void r4_write_csv(FILE *f,int sample,const R4Variant &v,const Row &r) {
  fprintf(f,"%d,%d,%d,%d,%s,%s,%s,%d,%d,%d,%.9f,%d,%d,%d,%d,%d,%d,%d,%.3f\n",
    sample,NONCE_OFF,FIRST_WORD,INNER_WORD,v.name,v.family,v.memory,v.g,v.n,v.pad,r.ghs,r.block,r.regs,r.local_bytes,r.static_smem,r.dynamic_smem,r.active_blocks,r.warps,r.occupancy);
}

int main(int argc,char **argv) {
  uint64_t outer_count=1ull<<18; int repeats=BENCH_REPEATS; int samples=1;
  const char *csv_path=nullptr; bool skip_correctness=false,correctness_only=false,list_only=false;
  bool shuffle=false; uint64_t shuffle_seed=0x5a17c0deULL;
  std::vector<std::string> only,filters,exclude;
  for(int i=1;i<argc;i++) {
    if(!strcmp(argv[i],"--outer") && i+1<argc) outer_count=strtoull(argv[++i],nullptr,0);
    else if(!strcmp(argv[i],"--repeats") && i+1<argc) repeats=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--samples") && i+1<argc) samples=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--csv") && i+1<argc) csv_path=argv[++i];
    else if(!strcmp(argv[i],"--only") && i+1<argc) only.emplace_back(argv[++i]);
    else if(!strcmp(argv[i],"--filter") && i+1<argc) filters.emplace_back(argv[++i]);
    else if(!strcmp(argv[i],"--exclude") && i+1<argc) exclude.emplace_back(argv[++i]);
    else if(!strcmp(argv[i],"--skip-correctness")) skip_correctness=true;
    else if(!strcmp(argv[i],"--correctness-only")) correctness_only=true;
    else if(!strcmp(argv[i],"--list")) list_only=true;
    else if(!strcmp(argv[i],"--shuffle")) { shuffle=true; if(i+1<argc && argv[i+1][0]!='-') shuffle_seed=strtoull(argv[++i],nullptr,0); }
    else { fprintf(stderr,"usage: %s [--outer N] [--repeats N] [--samples N] [--csv path] [--only exact]... [--filter substr]... [--exclude exact]... [--shuffle [seed]] [--skip-correctness|--correctness-only] [--list]\n",argv[0]); return 2; }
  }
  if(repeats<1||samples<1){fprintf(stderr,"repeats/samples must be >=1\n");return 2;}

  std::vector<const R4Variant*> selected;
  for(const auto &v:R4_VARIANTS) if(r4_name_selected(v.name,only,filters,exclude)) selected.push_back(&v);
  if(list_only) {
    printf("variants=%zu selected=%zu\n",R4_VARIANT_COUNT,selected.size());
    for(const auto *v:selected) printf("%-30s family=%-8s mem=%-8s G=%2d N=%2d pad=%d B=%3d\n",v->name,v->family,v->memory,v->g,v->n,v->pad,v->block);
    return selected.empty()?6:0;
  }
  if(selected.empty()){fprintf(stderr,"no variants matched selection\n");return 6;}

  int dev=0;CUDA_OK(cudaSetDevice(dev));cudaDeviceProp p{};CUDA_OK(cudaGetDeviceProperties(&p,dev));
  printf("GPU: %s  cc=%d.%d SMs=%d maxThreads/SM=%d shared/block=%zu shared/SM=%zu\n",p.name,p.major,p.minor,p.multiProcessorCount,p.maxThreadsPerMultiProcessor,p.sharedMemPerBlock,p.sharedMemPerMultiprocessor);
  printf("layout: PREFIX_BLOCKS=%d DATA_LEN=%d NONCE_OFF=%d NONCE_LEN=%d FIRST_WORD=%d INNER_WORD=%d total_data_bytes=%d\n",
    PREFIX_BLOCKS,DATA_LEN,NONCE_OFF,NONCE_LEN,FIRST_WORD,INNER_WORD,PREFIX_BLOCKS*64+DATA_LEN);
  printf("compact: words=%d bank_period=%d g4_auto_pad=%d variants=%zu selected=%zu\n",COMPACT_WORDS,COMPACT_BANK_PERIOD,G4_COMPACT_AUTO_PAD,R4_VARIANT_COUNT,selected.size());

  // Never launch a shape that requires architecture-specific >default dynamic
  // shared memory unless a future experiment explicitly opts it in. This keeps
  // resource failures distinct from hash correctness failures.
  {
    std::vector<const R4Variant*> runnable; runnable.reserve(selected.size());
    size_t skipped=0;
    for(const auto *v:selected) {
      const size_t need=r4_required_dynamic_smem(*v);
      if(need>(size_t)p.sharedMemPerBlock) {
        fprintf(stderr,"SKIP_RESOURCE: %s dynamic_smem=%zu default_limit=%zu\n",v->name,need,(size_t)p.sharedMemPerBlock);
        skipped++;
      } else runnable.push_back(v);
    }
    if(skipped) printf("resource preflight: runnable=%zu skipped=%zu\n",runnable.size(),skipped);
    selected.swap(runnable);
    if(selected.empty()){fprintf(stderr,"no runnable variants after resource preflight\n");return 6;}
  }

  uint32_t base16[16],hin[5],pre[5],delta[DELTA_WORDS];make_setup(base16,hin,pre,delta);
  { uint32_t a[80],b[80];cpu_expand_classic(base16,a);cpu_expand_alt(base16,b);if(memcmp(a,b,sizeof(a))){fprintf(stderr,"alt schedule identity FAIL\n");return 3;} }
  uint32_t base80_upload[80]; cpu_expand_classic(base16,base80_upload);
  CUDA_OK(cudaMemcpyToSymbol(C_BASE16,base16,sizeof(base16)));
  CUDA_OK(cudaMemcpyToSymbol(C_BASE80,base80_upload,sizeof(base80_upload)));
  CUDA_OK(cudaMemcpyToSymbol(C_HIN,hin,sizeof(hin)));
  CUDA_OK(cudaMemcpyToSymbol(C_PRE,pre,sizeof(pre)));
  CUDA_OK(cudaMemcpyToSymbol(C_CHARSET,HOST_CHARSET,64));
  CUDA_OK(cudaMemcpyToSymbol(G_DELTA,delta,sizeof(delta)));
  CUDA_OK(cudaMemcpyToSymbol(C_DELTA,delta,sizeof(delta)));
  std::vector<uint32_t> pair_a,pair_b,pair_table; make_pair_setup(pair_a,pair_b,pair_table);
  CUDA_OK(cudaMemcpyToSymbol(G_PAIR_A,pair_a.data(),pair_a.size()*sizeof(uint32_t)));
  CUDA_OK(cudaMemcpyToSymbol(G_PAIR_B,pair_b.data(),pair_b.size()*sizeof(uint32_t)));
  CUDA_OK(cudaMemcpyToSymbol(G_PAIR_TABLE,pair_table.data(),pair_table.size()*sizeof(uint32_t)));

  const uint64_t test_outer=1234;
  static constexpr unsigned test_inners[3]={0u,37u,63u};
  uint32_t correctness_targets[3];
  for(int ti=0;ti<3;ti++) {
    const uint64_t id=(test_outer<<6)|test_inners[ti]; uint32_t tw[16],td[5];
    make_words_for_id(id,tw);cpu_compress(hin,tw,td);correctness_targets[ti]=td[0];
  }
  const uint64_t test_id=(test_outer<<6)|37u; const uint32_t target=correctness_targets[1];
  uint64_t *dwin=nullptr;CUDA_OK(cudaMalloc(&dwin,8)); const uint64_t c_outer=test_outer+1;

  if(!skip_correctness) {
    size_t passed=0,resource_skipped=0,failed=0;
    std::vector<const R4Variant*> verified;
    verified.reserve(selected.size());
    for(const auto *v:selected) {
      bool vok=true; unsigned bad_inner=0;
      for(int ti=0;ti<3;ti++) {
        if(!v->correct(c_outer,correctness_targets[ti],dwin,hin)) { vok=false; bad_inner=test_inners[ti]; break; }
      }
      if(!vok) {
        fprintf(stderr,"GPU correctness FAIL_HASH: %s inner=%u\n",v->name,bad_inner);
        failed++;
        continue;
      }
      verified.push_back(v); passed++;
    }
    printf("correctness: checked=%zu pass=%zu resource_skip=%zu hash_fail=%zu probes={0,37,63} middle_H0=%08x known_middle_id=%" PRIu64 "\n",
      selected.size(),passed,resource_skipped,failed,target,test_id);
    if(failed) { CUDA_OK(cudaFree(dwin)); return 4; }
    selected.swap(verified);
    if(selected.empty()) { fprintf(stderr,"no runnable verified variants remain\n"); CUDA_OK(cudaFree(dwin)); return 6; }
  }
  if(correctness_only){CUDA_OK(cudaFree(dwin));return 0;}

  const uint32_t bench_target=0x13579bdfu; uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  FILE *csv=nullptr;if(csv_path){csv=fopen(csv_path,"w");if(!csv){perror(csv_path);return 5;}r4_write_csv_header(csv);}
  std::mt19937_64 rng(shuffle_seed);
  for(int sample=0;sample<samples;sample++) {
    std::vector<const R4Variant*> order=selected;
    if(shuffle) std::shuffle(order.begin(),order.end(),rng);
    printf("\nsample %d/%d (%" PRIu64 " outer = %.0f hashes/launch, repeats=%d%s):\n",sample+1,samples,outer_count,double(outer_count)*64.0,repeats,shuffle?", shuffled":"");
    for(const auto *v:order) {
      Row r=v->bench(v->name,outer_count,repeats,bench_target,dwin,p);
      print_row(r);
      if(csv) r4_write_csv(csv,sample,v[0],r);
    }
  }
  if(csv){fclose(csv);printf("CSV: %s\n",csv_path);} CUDA_OK(cudaFree(dwin)); return 0;
}
#endif // STUDY_NO_MAIN
