#!/usr/bin/env python3
import argparse, json
from pathlib import Path
MASK=0xffffffff

def rol(x,n):return ((x<<n)&MASK)|(x>>(32-n))
def expand(w):
 w=list(w)+[0]*64
 for t in range(16,80):w[t]=rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1)
 return w
def c32(x):return f'0x{x:08x}u'

def operand(idx,base80):
    if idx==12:return 'outer'
    if idx==13:return 'W13_BASE'
    if idx==14:return 'W14'
    if idx==15:return 'W15'
    if idx<12:return c32(base80[idx])
    return f'sched[{idx-16}]'

def gen_schedule(base80):
    out=[]
    for t in range(16,32):
      a,b,c,d=(operand(t-k,base80) for k in (3,8,14,16))
      out.append(f'    sched[{t-16}]=rol32(({a})^({b})^({c})^({d}),1);')
    for t in range(32,80):
      a,b,c,d=(operand(t-k,base80) for k in (6,16,28,32))
      out.append(f'    sched[{t-16}]=rol32(({a})^({b})^({c})^({d}),2);')
    return '\n'.join(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('-o','--output');ap.add_argument('--prefix-hex',default='13579bdf');a=ap.parse_args();d=Path(a.job);m=json.loads((d/'job.json').read_text())
 if m['mode']!='binary-tail-4+1' or m['nonce_within_block']!=48:raise SystemExit('requires binary-tail-4+1 offset-48 job')
 b16=[int(x,16) for x in m['base16']];b80=expand(b16);pre=[int(x,16) for x in m['pre']];hin=[int(x,16) for x in m['hin']]
 prefix=a.prefix_hex.lower().removeprefix('0x')
 if not prefix or len(prefix)>40 or any(c not in '0123456789abcdef' for c in prefix):raise SystemExit('--prefix-hex must contain 1..40 hex digits')
 prefix_bits=4*len(prefix);target160=int(prefix,16)<<(160-prefix_bits);target_words=[(target160>>(128-32*i))&MASK for i in range(5)]
 target=target_words[0];adj=(target-hin[0])&MASK
 # one known valid candidate for GPU correctness gate
 test_outer=0x01020304;test_inner=0x05
 # derive test adjusted target with regular SHA-1 compression in Python
 def comp(h,w):
  w=expand(w);aa,bb,cc,dd,ee=h
  for t in range(80):
   if t<20:f=(bb&cc)|((~bb)&dd);k=0x5a827999
   elif t<40:f=bb^cc^dd;k=0x6ed9eba1
   elif t<60:f=(bb&cc)|(bb&dd)|(cc&dd);k=0x8f1bbcdc
   else:f=bb^cc^dd;k=0xca62c1d6
   z=(rol(aa,5)+f+ee+k+w[t])&MASK;ee=dd;dd=cc;cc=rol(bb,30);bb=aa;aa=z
  return [(h[0]+aa)&MASK,(h[1]+bb)&MASK,(h[2]+cc)&MASK,(h[3]+dd)&MASK,(h[4]+ee)&MASK]
 tw=b16.copy();tw[12]=test_outer;tw[13]|=test_inner<<24;test_digest=comp(hin,tw);th0=test_digest[0];test_adj=(th0-hin[0])&MASK
 # Build compile-time match helpers. For <=32 bits, subtracting HIN0 turns the
 # H0 prefix into a modular interval over a80. For >32 bits H0 is exact; H1..H4
 # are reconstructed only inside that rare H0-match branch.
 if prefix_bits<32:
  span=1<<(32-prefix_bits);base=((target_words[0]&((MASK<<(32-prefix_bits))&MASK))-hin[0])&MASK
  hot=f'uint32_t(a80-{c32(base)}) < {span}u'
 else:
  hot=f'a80=={c32((target_words[0]-hin[0])&MASK)}'
 extra=[]
 expr=['',f'(s.a+{c32(hin[1])})',f'(rol32(s.b,30)+{c32(hin[2])})',f'(s.c+{c32(hin[3])})',f'(s.d+{c32(hin[4])})']
 for wi in range(1,5):
  used=prefix_bits-32*wi
  if used<=0:break
  take=min(32,used);mask=MASK if take==32 else ((MASK<<(32-take))&MASK);want=target_words[wi]&mask
  extra.append(f'if((({expr[wi]})&{c32(mask)})!={c32(want)})return false;')
 prod_match='if(!('+hot+'))return false;'+''.join(extra)+'return true;'
 test_extra=[f'if(({expr[i]})!={c32(test_digest[i])})return false;' for i in range(1,5)]
 test_match=f'if(a80!={c32(test_adj)})return false;'+''.join(test_extra)+'return true;'
 schedule=gen_schedule(b80)
 src=f'''// AUTO-GENERATED exact Git binary-tail CUDA kernel. Do not edit by hand.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
static constexpr uint32_t W13_BASE={c32(b16[13])},W14={c32(b16[14])},W15={c32(b16[15])};
static constexpr uint32_t TARGET_ADJ={c32(adj)};
static constexpr int PREFIX_BITS={prefix_bits};
static constexpr uint32_t TEST_ADJ={c32(test_adj)};
static constexpr uint32_t TEST_OUTER={c32(test_outer)}; static constexpr unsigned TEST_INNER=0x{test_inner:02x}u;
#define CUDA_OK(x) do{{cudaError_t e_=(x);if(e_!=cudaSuccess){{fprintf(stderr,"CUDA error %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}}}while(0)
__device__ __forceinline__ uint32_t rol32(uint32_t x,unsigned n){{return __funnelshift_l(x,x,n);}}
__device__ __forceinline__ uint32_t fch(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xca;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fpa(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0x96;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fmj(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xe8;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
struct S{{uint32_t a,b,c,d,e;}};
__device__ __forceinline__ void sch(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fch(s.b,s.c,s.d)+s.e+K0+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void sp1(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K1+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void smj(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fmj(s.b,s.c,s.d)+s.e+K2+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void sp3(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ uint32_t fina(const S&s,uint32_t w){{return rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;}}
constexpr uint32_t cr(uint32_t x,int n){{n&=31;return n?((x<<n)|(x>>(32-n))):x;}}
constexpr uint32_t pmask(int t){{uint32_t p[80]{{}};p[13]=1;for(int i=16;i<=t;i++)p[i]=cr(p[i-3]^p[i-8]^p[i-14]^p[i-16],1);return p[t];}}
constexpr int pc(uint32_t x){{int n=0;while(x){{n+=x&1u;x>>=1;}}return n;}} constexpr int ctz(uint32_t x){{int n=0;while(!(x&1u)){{n++;x>>=1;}}return n;}}
constexpr int cidx(int t){{int n=0;for(int q=16;q<t;q++)if(pc(pmask(q))>1)n++;return n;}}
template<int V>struct P{{uint32_t x[V];}};
template<int V>__device__ __forceinline__ P<V> rawp(unsigned j){{P<V>z{{}};#pragma unroll\n for(int i=0;i<V;i++)z.x[i]=uint32_t(j+i)<<24;return z;}}
template<int V>__device__ __forceinline__ P<V> rotp(const P<V>&p,int r){{P<V>z{{}};#pragma unroll\n for(int i=0;i<V;i++)z.x[i]=rol32(p.x[i],r);return z;}}
template<int V,int CACHE>__device__ __forceinline__ P<V> loadp(const uint32_t*t,int row,unsigned j){{P<V>z{{}};const uint32_t*p=t+row*256+j;#pragma unroll\n for(int q=0;q<V;q+=4){{uint4 a;if constexpr(CACHE==0)asm volatile("ld.global.ca.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));else asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));z.x[q]=a.x;z.x[q+1]=a.y;z.x[q+2]=a.z;z.x[q+3]=a.w;}}return z;}}
template<int T,int V,int CACHE>__device__ __forceinline__ P<V> delta(const uint32_t*t,const P<V>&raw,unsigned j,const P<V>&r7,const P<V>&r3){{constexpr uint32_t m=pmask(T);constexpr int n=pc(m);if constexpr(n==0)return P<V>{{}};else if constexpr(n==1){{constexpr int e=ctz(m);if constexpr(e==7)return r7;else if constexpr(e==3)return r3;else return rotp<V>(raw,e);}}else return loadp<V,CACHE>(t,cidx(T),j);}}
__device__ __forceinline__ bool zero_byte(uint32_t x){{return ((x-0x01010101u)&~x&0x80808080u)!=0;}}
__device__ __forceinline__ bool prod_match(uint32_t a80,const S&s){{{prod_match}}}
__device__ __forceinline__ bool test_match(uint32_t a80,const S&s){{{test_match}}}

template<int G,int V,int CACHE,bool TEST=false>
__global__ void kernel(uint64_t outer_base,uint64_t outer_count,const uint32_t*table,uint64_t*winner){{
 static_assert(256%(G*V)==0,"mapping");constexpr int STRIDE=65;extern __shared__ uint32_t sm[];int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;uint64_t gi=uint64_t(blockIdx.x)*groups+gl,ov=outer_base+gi;bool active=gi<outer_count&&ov<=0xffffffffull;uint32_t outer=(uint32_t)ov;uint32_t*sched=sm+gl*STRIDE;
 if(active&&lane==0){{
{schedule}
 }}
 __syncwarp();if(!active)return;
 S common{{{','.join(c32(x) for x in pre)}}};sch(common,outer);constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{unsigned j=chunk+unsigned(lane)*V;S st[V];#pragma unroll\n for(int i=0;i<V;i++)st[i]=common;auto raw=rawp<V>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);#pragma unroll\n for(int i=0;i<V;i++)sch(st[i],W13_BASE+raw.x[i]);#pragma unroll\n for(int i=0;i<V;i++)sch(st[i],W14);#pragma unroll\n for(int i=0;i<V;i++)sch(st[i],W15);
 #define R(T,F) do{{auto dv=delta<T,V,CACHE>(table,raw,j,r7,r3);uint32_t bw=sched[(T)-16];_Pragma("unroll") for(int i=0;i<V;i++)F(st[i],bw^dv.x[i]);}}while(0)
 R(16,sch);R(17,sch);R(18,sch);R(19,sch);R(20,sp1);R(21,sp1);R(22,sp1);R(23,sp1);R(24,sp1);R(25,sp1);R(26,sp1);R(27,sp1);R(28,sp1);R(29,sp1);R(30,sp1);R(31,sp1);R(32,sp1);R(33,sp1);R(34,sp1);R(35,sp1);R(36,sp1);R(37,sp1);R(38,sp1);R(39,sp1);R(40,smj);R(41,smj);R(42,smj);R(43,smj);R(44,smj);R(45,smj);R(46,smj);R(47,smj);R(48,smj);R(49,smj);R(50,smj);R(51,smj);R(52,smj);R(53,smj);R(54,smj);R(55,smj);R(56,smj);R(57,smj);R(58,smj);R(59,smj);R(60,sp3);R(61,sp3);R(62,sp3);R(63,sp3);R(64,sp3);R(65,sp3);R(66,sp3);R(67,sp3);R(68,sp3);R(69,sp3);R(70,sp3);R(71,sp3);R(72,sp3);R(73,sp3);R(74,sp3);R(75,sp3);R(76,sp3);R(77,sp3);R(78,sp3);#undef R
 {{auto dv=delta<79,V,CACHE>(table,raw,j,r7,r3);uint32_t bw=sched[63];#pragma unroll\n for(int i=0;i<V;i++){{uint32_t a80=fina(st[i],bw^dv.x[i]);bool hit=TEST?test_match(a80,st[i]):prod_match(a80,st[i]);if(hit){{unsigned in=j+i;if(in&& !zero_byte(outer))atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((uint64_t(outer)<<8)|in));}}}}}}
 }}
}}

static std::vector<uint32_t> read_table(const char*p){{FILE*f=fopen(p,"rb");if(!f){{perror(p);exit(2);}}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint32_t>v((size_t)n/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(2);fclose(f);return v;}}
template<int G,int V,int CACHE>static bool correct(const uint32_t*t,uint64_t*w){{uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));constexpr int B=96;int groups=B/G;size_t sm=(B/G)*65u*4u;kernel<G,V,CACHE,true><<<1,B,sm>>>(TEST_OUTER,1,t,w);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint64_t got;CUDA_OK(cudaMemcpy(&got,w,8,cudaMemcpyDeviceToHost));return got==((uint64_t(TEST_OUTER)<<8)|TEST_INNER);}}
template<int G,int V,int CACHE>static void bench(const char*name,uint64_t n,int reps,const uint32_t*t,uint64_t*w,const cudaDeviceProp&p){{constexpr int B=96;int groups=B/G;size_t sm=(B/G)*65u*4u;dim3 grid((unsigned)((n+groups-1)/groups));cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));kernel<G,V,CACHE><<<grid,B,sm>>>(0,n,t,w);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaEventRecord(a));for(int r=0;r<reps;r++)kernel<G,V,CACHE><<<grid,B,sm>>>(0,n,t,w);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));cudaFuncAttributes fa{{}};CUDA_OK(cudaFuncGetAttributes(&fa,kernel<G,V,CACHE>));printf("%-18s %.6f GH/s regs=%d local=%zu smem=%zu\\n",name,double(n)*256.0*reps/(double(ms)*1e6),fa.numRegs,fa.localSizeBytes,sm);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
int main(int argc,char**argv){{const char*path=argc>1?argv[1]:"complex_delta.bin";uint64_t n=argc>2?strtoull(argv[2],nullptr,0):(1ull<<18);int reps=argc>3?atoi(argv[3]):6;auto h=read_table(path);if(h.size()!=32u*256u){{fprintf(stderr,"bad table size\\n");return 2;}}uint32_t*d=nullptr;uint64_t*w=nullptr;CUDA_OK(cudaMalloc(&d,h.size()*4));CUDA_OK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&w,8));cudaDeviceProp p{{}};CUDA_OK(cudaGetDeviceProperties(&p,0));if(!correct<4,4,0>(d,w)||!correct<4,8,0>(d,w)||!correct<8,4,0>(d,w)){{fprintf(stderr,"GPU correctness FAIL\\n");return 3;}}printf("GPU correctness PASS target_h0={target:08x}\\n");bench<4,4,0>("lit-g4-v4-ca",n,reps,d,w,p);bench<4,8,0>("lit-g4-v8-ca",n,reps,d,w,p);bench<8,4,0>("lit-g8-v4-ca",n,reps,d,w,p);bench<2,8,0>("lit-g2-v8-ca",n,reps,d,w,p);bench<4,8,1>("lit-g4-v8-nc",n,reps,d,w,p);CUDA_OK(cudaFree(d));CUDA_OK(cudaFree(w));}}
'''
 src=src.replace(';#pragma unroll\n',';\n #pragma unroll\n').replace(';#undef R',';\n #undef R')
 out=Path(a.output) if a.output else d/'literal_binary_job.cu';out.write_text(src);print(out)
if __name__=='__main__':main()
