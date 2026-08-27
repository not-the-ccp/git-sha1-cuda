#include "sample-signed-tailn/tailn_job.cuh"
#define STUDY_NO_MAIN 1
#define NONCE_OFF TN_NONCE_OFF
#define NONCE_LEN TN_NONCE_LEN
#define DATA_LEN 55
#define PREFIX_BLOCKS 1
#include "finalblock_study.cu"

// Multi-block Git specialization for a contiguous nonce contained in one
// compression block. Prefix blocks are prehashed; the first mutable block uses
// the affine inner-character specialization; all following blocks are fixed.
// Their schedules are pre-expanded on the host and consumed four words/load.

template<int CACHE>
__device__ __forceinline__ uint4 tn_load4(const uint32_t *p) {
  return r6_load4<CACHE>(p);
}

template<int V>
__device__ __forceinline__ void tn_ch4(DState (&s)[V],uint4 w) {
  #pragma unroll
  for(int i=0;i<V;i++) step_ch(s[i],w.x);
  #pragma unroll
  for(int i=0;i<V;i++) step_ch(s[i],w.y);
  #pragma unroll
  for(int i=0;i<V;i++) step_ch(s[i],w.z);
  #pragma unroll
  for(int i=0;i<V;i++) step_ch(s[i],w.w);
}
template<int V>
__device__ __forceinline__ void tn_p14(DState (&s)[V],uint4 w) {
  #pragma unroll
  for(int i=0;i<V;i++) step_par1(s[i],w.x);
  #pragma unroll
  for(int i=0;i<V;i++) step_par1(s[i],w.y);
  #pragma unroll
  for(int i=0;i<V;i++) step_par1(s[i],w.z);
  #pragma unroll
  for(int i=0;i<V;i++) step_par1(s[i],w.w);
}
template<int V>
__device__ __forceinline__ void tn_maj4(DState (&s)[V],uint4 w) {
  #pragma unroll
  for(int i=0;i<V;i++) step_maj(s[i],w.x);
  #pragma unroll
  for(int i=0;i<V;i++) step_maj(s[i],w.y);
  #pragma unroll
  for(int i=0;i<V;i++) step_maj(s[i],w.z);
  #pragma unroll
  for(int i=0;i<V;i++) step_maj(s[i],w.w);
}
template<int V>
__device__ __forceinline__ void tn_p34(DState (&s)[V],uint4 w) {
  #pragma unroll
  for(int i=0;i<V;i++) step_par3(s[i],w.x);
  #pragma unroll
  for(int i=0;i<V;i++) step_par3(s[i],w.y);
  #pragma unroll
  for(int i=0;i<V;i++) step_par3(s[i],w.z);
  #pragma unroll
  for(int i=0;i<V;i++) step_par3(s[i],w.w);
}

template<int V,int CACHE,bool FF_SHARED>
__device__ __forceinline__ void tn_fixed_suffix_block(DState (&s)[V],const uint32_t *w80,volatile uint32_t *ff) {
  uint32_t oa[V],ob[V],oc[V],od[V],oe[V];
  if constexpr(FF_SHARED) {
    #pragma unroll
    for(int i=0;i<V;i++) { volatile uint32_t *q=ff+i*5; q[0]=s[i].a;q[1]=s[i].b;q[2]=s[i].c;q[3]=s[i].d;q[4]=s[i].e; }
  } else {
    #pragma unroll
    for(int i=0;i<V;i++){oa[i]=s[i].a;ob[i]=s[i].b;oc[i]=s[i].c;od[i]=s[i].d;oe[i]=s[i].e;}
  }
  #pragma unroll
  for(int q=0;q<5;q++) tn_ch4(s,tn_load4<CACHE>(w80+4*q));
  #pragma unroll
  for(int q=5;q<10;q++) tn_p14(s,tn_load4<CACHE>(w80+4*q));
  #pragma unroll
  for(int q=10;q<15;q++) tn_maj4(s,tn_load4<CACHE>(w80+4*q));
  #pragma unroll
  for(int q=15;q<20;q++) tn_p34(s,tn_load4<CACHE>(w80+4*q));
  if constexpr(FF_SHARED) {
    #pragma unroll
    for(int i=0;i<V;i++) { volatile uint32_t *q=ff+i*5; s[i].a+=q[0];s[i].b+=q[1];s[i].c+=q[2];s[i].d+=q[3];s[i].e+=q[4]; }
  } else {
    #pragma unroll
    for(int i=0;i<V;i++){s[i].a+=oa[i];s[i].b+=ob[i];s[i].c+=oc[i];s[i].d+=od[i];s[i].e+=oe[i];}
  }
}

template<int G,int V,int CACHE,bool FF_SHARED,int PAD=0>
__global__ void k_tailn(uint64_t outer_base,uint64_t outer_count,const uint32_t *suffix80,uint32_t target,uint32_t mask,uint64_t *winner) {
  static_assert((G&(G-1))==0 && G>=1 && G<=16,"G");
  static_assert(V==2 || V==4 || V==8,"V");
  static_assert(64%(G*V)==0,"G*V divides 64");
  constexpr int SW=80-FIRST_WORD,STRIDE=SW+PAD;
  extern __shared__ uint32_t sm[];
  const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;
  const uint64_t gi=uint64_t(blockIdx.x)*groups+gl; const bool active=gi<outer_count; const uint64_t outer=outer_base+gi;
  uint32_t *sched=sm+gl*STRIDE;
  if(active && lane==0) {
    #pragma unroll
    for(int t=FIRST_WORD;t<16;t++) sched[t-FIRST_WORD]=C_BASE16[t];
    #pragma unroll
    for(int k=0;k<OUTER_LEN;k++){const int p=NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));sched[wi-FIRST_WORD]|=uint32_t(C_CHARSET[(outer>>(6*k))&63u])<<sh;}
    #pragma unroll
    for(int t=16;t<32;t++){const uint32_t a=(t-3<FIRST_WORD)?C_BASE80[t-3]:sched[t-3-FIRST_WORD],b=(t-8<FIRST_WORD)?C_BASE80[t-8]:sched[t-8-FIRST_WORD],c=(t-14<FIRST_WORD)?C_BASE80[t-14]:sched[t-14-FIRST_WORD],d=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD];sched[t-FIRST_WORD]=rol32(a^b^c^d,1);}
    #pragma unroll
    for(int t=32;t<80;t++){const uint32_t a=(t-6<FIRST_WORD)?C_BASE80[t-6]:sched[t-6-FIRST_WORD],b=(t-16<FIRST_WORD)?C_BASE80[t-16]:sched[t-16-FIRST_WORD],c=(t-28<FIRST_WORD)?C_BASE80[t-28]:sched[t-28-FIRST_WORD],d=(t-32<FIRST_WORD)?C_BASE80[t-32]:sched[t-32-FIRST_WORD];sched[t-FIRST_WORD]=rol32(a^b^c^d,2);}
  }
  __syncwarp();if(!active)return;
  DState common{C_PRE[0],C_PRE[1],C_PRE[2],C_PRE[3],C_PRE[4]};
  #pragma unroll
  for(int t=FIRST_WORD;t<INNER_WORD;t++) step_ch(common,sched[t-FIRST_WORD]);
  constexpr unsigned CHUNK=G*V;
  const size_t sched_words=(size_t)groups*STRIDE;
  volatile uint32_t *ffbase=(volatile uint32_t*)(sm+sched_words);
  #pragma unroll 1
  for(unsigned chunk=0;chunk<64;chunk+=CHUNK) {
    const unsigned jbase=chunk+unsigned(lane)*V;DState st[V];
    #pragma unroll
    for(int i=0;i<V;i++)st[i]=common;
    auto dv=r5_load_pack<V,CACHE>(0,jbase);const uint32_t wi=sched[INNER_WORD-FIRST_WORD];
    #pragma unroll
    for(int i=0;i<V;i++)step_ch(st[i],wi+dv.x[i]);
    #pragma unroll
    for(int t=INNER_WORD+1;t<16;t++){const uint32_t bw=sched[t-FIRST_WORD];
      #pragma unroll
      for(int i=0;i<V;i++)step_ch(st[i],bw);}
    #pragma unroll
    for(int t=16;t<20;t++){const uint32_t bw=sched[t-FIRST_WORD];dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++)step_ch(st[i],bw^dv.x[i]);}
    #pragma unroll
    for(int t=20;t<40;t++){const uint32_t bw=sched[t-FIRST_WORD];dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++)step_par1(st[i],bw^dv.x[i]);}
    #pragma unroll
    for(int t=40;t<60;t++){const uint32_t bw=sched[t-FIRST_WORD];dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++)step_maj(st[i],bw^dv.x[i]);}
    #pragma unroll
    for(int t=60;t<80;t++){const uint32_t bw=sched[t-FIRST_WORD];dv=r5_load_pack<V,CACHE>(1+t-16,jbase);
      #pragma unroll
      for(int i=0;i<V;i++)step_par3(st[i],bw^dv.x[i]);}
    #pragma unroll
    for(int i=0;i<V;i++){st[i].a+=C_HIN[0];st[i].b+=C_HIN[1];st[i].c+=C_HIN[2];st[i].d+=C_HIN[3];st[i].e+=C_HIN[4];}
    volatile uint32_t *ff=ffbase+((size_t)threadIdx.x*V)*5;
    #pragma unroll 1
    for(int b=0;b<TN_SUFFIX_BLOCKS;b++) tn_fixed_suffix_block<V,CACHE,FF_SHARED>(st,suffix80+b*80,ff);
    #pragma unroll
    for(int i=0;i<V;i++)if((st[i].a&mask)==target)atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((outer<<6)|(jbase+unsigned(i))));
  }
}

#ifndef TAILN_NO_MAIN
static std::vector<uint32_t> read_u32_file(const char *path){FILE*f=fopen(path,"rb");if(!f){perror(path);exit(2);}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint32_t>v((size_t)n/4);if(fread(v.data(),4,v.size(),f)!=v.size()){perror("fread");exit(2);}fclose(f);return v;}

template<int G,int V,int CACHE,bool FF_SHARED>
static void run_variant(const char*name,uint64_t outer,int reps,const uint32_t*d_suffix,uint64_t*dwin,const cudaDeviceProp&p){
  constexpr int B=96;const int groups=B/G;const size_t sched=(B/G)*(80-FIRST_WORD)*4u;const size_t ff=FF_SHARED?B*V*5u*4u:0;const size_t smem=sched+ff;
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dwin,&none,8,cudaMemcpyHostToDevice));
  cudaEvent_t x,y;CUDA_OK(cudaEventCreate(&x));CUDA_OK(cudaEventCreate(&y));
  k_tailn<G,V,CACHE,FF_SHARED><<<(outer+groups-1)/groups,B,smem>>>(0,outer,d_suffix,0x13579bdfu,0u,dwin);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(x));for(int r=0;r<reps;r++)k_tailn<G,V,CACHE,FF_SHARED><<<(outer+groups-1)/groups,B,smem>>>(0,outer,d_suffix,0x13579bdfu,0u,dwin);CUDA_OK(cudaEventRecord(y));CUDA_OK(cudaEventSynchronize(y));
  float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,x,y));double cand=double(outer)*64.0*reps/(ms*1e-3);cudaFuncAttributes a{};CUDA_OK(cudaFuncGetAttributes(&a,k_tailn<G,V,CACHE,FF_SHARED>));
  printf("%-20s %9.3f Mcand/s  %9.3f Gcompress/s regs=%d local=%zu smem=%zu\n",name,cand/1e6,cand*(TN_SUFFIX_BLOCKS+1)/1e9,a.numRegs,a.localSizeBytes,smem);
  CUDA_OK(cudaEventDestroy(x));CUDA_OK(cudaEventDestroy(y));
}
int main(int argc,char**argv){uint64_t outer=1ull<<14;int reps=3;const char*suffix="sample-signed-tailn/suffix80.bin";if(argc>1)outer=strtoull(argv[1],nullptr,0);if(argc>2)reps=atoi(argv[2]);if(argc>3)suffix=argv[3];
  CUDA_OK(cudaSetDevice(0));cudaDeviceProp p{};CUDA_OK(cudaGetDeviceProperties(&p,0));uint32_t base80[80];cpu_expand_classic(TN_BASE16,base80);CUDA_OK(cudaMemcpyToSymbol(C_BASE16,TN_BASE16,sizeof(TN_BASE16)));CUDA_OK(cudaMemcpyToSymbol(C_BASE80,base80,sizeof(base80)));CUDA_OK(cudaMemcpyToSymbol(C_HIN,TN_HIN,sizeof(TN_HIN)));CUDA_OK(cudaMemcpyToSymbol(C_PRE,TN_PRE,sizeof(TN_PRE)));CUDA_OK(cudaMemcpyToSymbol(C_CHARSET,HOST_CHARSET,64));
  uint32_t delta[DELTA_WORDS]{};for(int j=0;j<64;j++){uint32_t d16[16]={};d16[INNER_WORD]=uint32_t((unsigned char)HOST_CHARSET[j])<<INNER_SHIFT;uint32_t d80[80];cpu_expand_classic(d16,d80);delta[j]=d16[INNER_WORD];for(int t=16;t<80;t++)delta[(1+t-16)*64+j]=d80[t];}CUDA_OK(cudaMemcpyToSymbol(G_DELTA,delta,sizeof(delta)));
  auto hs=read_u32_file(suffix);uint32_t*d_suffix=nullptr;CUDA_OK(cudaMalloc(&d_suffix,hs.size()*4));CUDA_OK(cudaMemcpy(d_suffix,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));uint64_t*dwin=nullptr;CUDA_OK(cudaMalloc(&dwin,8));
  printf("GPU=%s suffix_blocks=%d candidate_blocks=%d outer=%llu\n",p.name,TN_SUFFIX_BLOCKS,TN_SUFFIX_BLOCKS+1,(unsigned long long)outer);
  run_variant<4,2,0,false>("g4-v2-regff",outer,reps,d_suffix,dwin,p);run_variant<4,4,0,false>("g4-v4-regff",outer,reps,d_suffix,dwin,p);run_variant<4,4,0,true>("g4-v4-shff",outer,reps,d_suffix,dwin,p);run_variant<4,8,0,true>("g4-v8-shff",outer,reps,d_suffix,dwin,p);run_variant<8,4,0,true>("g8-v4-shff",outer,reps,d_suffix,dwin,p);
  CUDA_OK(cudaFree(d_suffix));CUDA_OK(cudaFree(dwin));}
#endif
