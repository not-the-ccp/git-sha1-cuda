// Fused long-tail architecture.
// Each thread computes SV independent mutable-head blocks with a scalar rolling
// schedule (no dynamic shared memory), then reuses each fixed suffix schedule
// load across those SV candidate states.  The head is only 1/(suffix+1) of the
// work for long signed commits, so removing long-lived shared-memory occupancy
// and the split pipeline's state-buffer traffic is the design goal.
#include "tailn_job.cuh"
#define STUDY_NO_MAIN 1
#define NONCE_OFF TN_NONCE_OFF
#define NONCE_LEN TN_NONCE_LEN
#define DATA_LEN 55
#define PREFIX_BLOCKS 1
#include "finalblock_study.cu"

__host__ __device__ static inline unsigned char tf_char(unsigned j){return j<10?(unsigned char)('0'+j):j<36?(unsigned char)('A'+j-10):j<62?(unsigned char)('a'+j-36):j==62?(unsigned char)'-':(unsigned char)'_';}

template<int CACHE> __device__ __forceinline__ uint4 tf_l4(const uint32_t*p){return r6_load4<CACHE>(p);}

__device__ __forceinline__ DState tf_head_inline(uint64_t id){
 const uint64_t outer=id>>6; const unsigned inner=(unsigned)(id&63u);
 uint32_t w[16]={TN_W00,TN_W01,TN_W02,TN_W03,TN_W04,TN_W05,TN_W06,TN_W07,TN_W08,TN_W09,TN_W10,TN_W11,TN_W12,TN_W13,TN_W14,TN_W15};
 #pragma unroll
 for(int k=0;k<TN_NONCE_LEN-1;k++){const int p=TN_NONCE_OFF+k,wi=p>>2,sh=8*(3-(p&3));w[wi]|=uint32_t(tf_char((outer>>(6*k))&63u))<<sh;}
 {const int p=TN_NONCE_OFF+TN_NONCE_LEN-1,wi=p>>2,sh=8*(3-(p&3));w[wi]|=uint32_t(tf_char(inner))<<sh;}
 DState s{TN_PRE0,TN_PRE1,TN_PRE2,TN_PRE3,TN_PRE4};
 #pragma unroll
 for(int t=TN_FIRST_WORD;t<16;t++) step_ch(s,w[t]);
 #pragma unroll
 for(int t=16;t<20;t++){int i=t&15;w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1);step_ch(s,w[i]);}
 #pragma unroll
 for(int t=20;t<40;t++){int i=t&15;w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1);step_par1(s,w[i]);}
 #pragma unroll
 for(int t=40;t<60;t++){int i=t&15;w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1);step_maj(s,w[i]);}
 #pragma unroll
 for(int t=60;t<80;t++){int i=t&15;w[i]=rol32(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[i],1);step_par3(s,w[i]);}
 s.a+=TN_H0;s.b+=TN_H1;s.c+=TN_H2;s.d+=TN_H3;s.e+=TN_H4; return s;
}

// Intentionally out-of-line: for very long tails the one-time call overhead is
// tiny, while separating the rolling-16 head frame may reduce the suffix
// kernel's peak register allocation.
__device__ __attribute__((noinline)) DState tf_head_call(uint64_t id){ return tf_head_inline(id); }

template<int SV,int CACHE> __device__ __forceinline__ void tf_round80(DState (&s)[SV],const uint32_t*w){
 #pragma unroll
 for(int q=0;q<5;q++){uint4 x=tf_l4<CACHE>(w+q*4);
  #pragma unroll
  for(int i=0;i<SV;i++){step_ch(s[i],x.x);step_ch(s[i],x.y);step_ch(s[i],x.z);step_ch(s[i],x.w);}}
 #pragma unroll
 for(int q=5;q<10;q++){uint4 x=tf_l4<CACHE>(w+q*4);
  #pragma unroll
  for(int i=0;i<SV;i++){step_par1(s[i],x.x);step_par1(s[i],x.y);step_par1(s[i],x.z);step_par1(s[i],x.w);}}
 #pragma unroll
 for(int q=10;q<15;q++){uint4 x=tf_l4<CACHE>(w+q*4);
  #pragma unroll
  for(int i=0;i<SV;i++){step_maj(s[i],x.x);step_maj(s[i],x.y);step_maj(s[i],x.z);step_maj(s[i],x.w);}}
 #pragma unroll
 for(int q=15;q<20;q++){uint4 x=tf_l4<CACHE>(w+q*4);
  #pragma unroll
  for(int i=0;i<SV;i++){step_par3(s[i],x.x);step_par3(s[i],x.y);step_par3(s[i],x.z);step_par3(s[i],x.w);}}
}

template<int SV,int CACHE,bool OUTLINE_HEAD=false>
__global__ void k_tailn_fused(uint64_t first_id,uint64_t count,const uint32_t*suffix80,uint32_t target_h0,uint64_t*winner){
 static_assert(SV==1||SV==2||SV==4||SV==8,"SV");
 const uint64_t g=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,base=g*SV;
 if(base>=count)return;
 DState s[SV]; bool live[SV];
 #pragma unroll
 for(int i=0;i<SV;i++){uint64_t n=base+(unsigned)i;live[i]=n<count;if(live[i])s[i]=OUTLINE_HEAD?tf_head_call(first_id+n):tf_head_inline(first_id+n);else s[i]=DState{0,0,0,0,0};}
 #pragma unroll 1
 for(int b=0;b<TN_SUFFIX_BLOCKS-1;b++){
   DState old[SV];
   #pragma unroll
   for(int i=0;i<SV;i++)old[i]=s[i];
   tf_round80<SV,CACHE>(s,suffix80+b*80);
   #pragma unroll
   for(int i=0;i<SV;i++){s[i].a+=old[i].a;s[i].b+=old[i].b;s[i].c+=old[i].c;s[i].d+=old[i].d;s[i].e+=old[i].e;}
 }
 // Last block: H0 gate needs only the pre-block A feed-forward word.
 uint32_t olda[SV];
 #pragma unroll
 for(int i=0;i<SV;i++)olda[i]=s[i].a;
 tf_round80<SV,CACHE>(s,suffix80+(TN_SUFFIX_BLOCKS-1)*80);
 #pragma unroll
 for(int i=0;i<SV;i++)if(live[i]&&s[i].a+olda[i]==target_h0)
   atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)(first_id+base+(unsigned)i));
}

#ifndef TAILN_FUSED_NO_MAIN
static std::vector<uint32_t> tf_read(const char*p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(2);}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint32_t>v((size_t)n/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(2);fclose(f);return v;}
static inline void tf_words(uint64_t id,uint32_t w[16]){const uint64_t outer=id>>6;unsigned inner=id&63u;const uint32_t b[16]={TN_W00,TN_W01,TN_W02,TN_W03,TN_W04,TN_W05,TN_W06,TN_W07,TN_W08,TN_W09,TN_W10,TN_W11,TN_W12,TN_W13,TN_W14,TN_W15};memcpy(w,b,64);for(int k=0;k<TN_NONCE_LEN-1;k++){int p=TN_NONCE_OFF+k;w[p>>2]|=uint32_t(tf_char((outer>>(6*k))&63u))<<(8*(3-(p&3)));}int p=TN_NONCE_OFF+TN_NONCE_LEN-1;w[p>>2]|=uint32_t(tf_char(inner))<<(8*(3-(p&3)));}
static uint32_t tf_h0(uint64_t id,const std::vector<uint32_t>&suf){uint32_t w[16],h[5]={TN_H0,TN_H1,TN_H2,TN_H3,TN_H4},o[5];tf_words(id,w);cpu_compress(h,w,o);memcpy(h,o,20);for(int b=0;b<TN_SUFFIX_BLOCKS;b++){// suffix80 is expanded; reconstruct first 16 words for cpu_compress
 memcpy(w,suf.data()+b*80,64);cpu_compress(h,w,o);memcpy(h,o,20);}return h[0];}

template<int SV,int CACHE,bool OUTLINE_HEAD=false> static void run_fused(const char*name,uint64_t count,int reps,const uint32_t*ds,uint64_t*dw,const std::vector<uint32_t>&hs,const cudaDeviceProp&p){
 constexpr int B=256;const uint64_t test=0x12345ull;uint32_t th=tf_h0(test,hs);uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(dw,&none,8,cudaMemcpyHostToDevice));k_tailn_fused<SV,CACHE,OUTLINE_HEAD><<<1,B>>>(test,1,ds,th,dw);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint64_t got;CUDA_OK(cudaMemcpy(&got,dw,8,cudaMemcpyDeviceToHost));if(got!=test){fprintf(stderr,"%s correctness FAIL got=%llx expected=%llx\n",name,(unsigned long long)got,(unsigned long long)test);exit(3);}none=NO_WINNER;CUDA_OK(cudaMemcpy(dw,&none,8,cudaMemcpyHostToDevice));dim3 grid((unsigned)(((count+SV-1)/SV+B-1)/B));cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));k_tailn_fused<SV,CACHE,OUTLINE_HEAD><<<grid,B>>>(0,count,ds,0x13579bdfu,dw);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaEventRecord(a));for(int r=0;r<reps;r++)k_tailn_fused<SV,CACHE,OUTLINE_HEAD><<<grid,B>>>(0,count,ds,0x13579bdfu,dw);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));cudaFuncAttributes fa{};CUDA_OK(cudaFuncGetAttributes(&fa,k_tailn_fused<SV,CACHE,OUTLINE_HEAD>));double cps=double(count)*reps/(ms*1e-3);printf("%-16s %9.3f Mcand/s %9.3f Gcompress/s regs=%d local=%zu smem=%zu\n",name,cps/1e6,cps*(TN_SUFFIX_BLOCKS+1)/1e9,fa.numRegs,fa.localSizeBytes,fa.sharedSizeBytes);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}
int main(int argc,char**argv){const char*path=argc>1?argv[1]:"suffix80.bin";uint64_t count=argc>2?strtoull(argv[2],nullptr,0):(1ull<<20);int reps=argc>3?atoi(argv[3]):3;auto hs=tf_read(path);if(hs.size()!=size_t(TN_SUFFIX_BLOCKS)*80u){fprintf(stderr,"bad suffix\n");return 2;}uint32_t*ds=nullptr;uint64_t*dw=nullptr;CUDA_OK(cudaMalloc(&ds,hs.size()*4));CUDA_OK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&dw,8));cudaDeviceProp p{};CUDA_OK(cudaGetDeviceProperties(&p,0));printf("GPU=%s fused suffix_blocks=%d candidate_blocks=%d count=%llu\n",p.name,TN_SUFFIX_BLOCKS,TN_SUFFIX_BLOCKS+1,(unsigned long long)count);run_fused<1,0>("fused-sv1-ca",count,reps,ds,dw,hs,p);run_fused<2,0>("fused-sv2-ca",count,reps,ds,dw,hs,p);run_fused<4,0>("fused-sv4-ca",count,reps,ds,dw,hs,p);run_fused<2,0,true>("fused-sv2-call",count,reps,ds,dw,hs,p);run_fused<4,0,true>("fused-sv4-call",count,reps,ds,dw,hs,p);run_fused<8,0>("fused-sv8-ca",count,reps,ds,dw,hs,p);run_fused<2,1>("fused-sv2-nc",count,reps,ds,dw,hs,p);run_fused<4,1>("fused-sv4-nc",count,reps,ds,dw,hs,p);CUDA_OK(cudaFree(ds));CUDA_OK(cudaFree(dw));}
#endif
