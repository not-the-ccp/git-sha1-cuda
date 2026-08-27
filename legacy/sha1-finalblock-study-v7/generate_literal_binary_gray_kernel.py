#!/usr/bin/env python3
"""Generate persistent Gray-code warp kernels for exact W12/W13 binary-tail jobs.

One full warp owns one raw 32-bit outer word.  The 64 expanded W12 schedule
contributions are distributed as two uint32 registers per lane.  A warp walks
outer indices in reflected Gray-code order; between adjacent outer values only
one source bit changes, so each lane updates its two words with one coalesced
uint2 load + XORs.  Each expanded round obtains its outer contribution with one
warp shuffle from the owning lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from generate_literal_binary_stream_kernel import MASK, c32, expand, poly, parse_prefix, rol


def apply_poly(x: int, mask: int) -> int:
    out = 0
    for r in range(32):
        if (mask >> r) & 1:
            out ^= rol(x, r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("-o", "--output")
    ap.add_argument("--prefix-hex", default="13579bdf")
    args = ap.parse_args()

    d = Path(args.job)
    m = json.loads((d / "job.json").read_text())
    if m.get("mode") != "binary-tail-4+1" or (m["nonce_within_block"], m["first_word"], m["inner_word"]) != (48, 12, 13):
        raise SystemExit("Gray kernel requires exact W12/W13 binary-tail job")
    b16 = [int(x, 16) for x in m["base16"]]
    hin = [int(x, 16) for x in m["hin"]]
    pre = [int(x, 16) for x in m["pre"]]
    if b16[12] != 0 or (b16[13] & 0xFF000000):
        raise SystemExit("candidate bytes overlap nonzero base bits")
    b80 = expand(b16)
    op = poly(12)
    ip = poly(13)
    complex_rounds = [t for t in range(16, 80) if ip[t].bit_count() > 1]
    if len(complex_rounds) != 32:
        raise SystemExit("unexpected W13 sparse shape")

    # For every outer source bit, store W16..W79 contribution. Row-major by bit
    # makes all 32 lanes read one contiguous uint2 when a Gray-code bit flips.
    outer_basis: list[int] = []
    for bit in range(32):
        x = 1 << bit
        outer_basis.extend(apply_poly(x, op[t]) for t in range(16, 80))
    (d / "outer_gray_delta.bin").write_bytes(b"".join(struct.pack("<I", x) for x in outer_basis))

    prod_body = parse_prefix(args.prefix_hex, hin)
    test_index = 0x01030507
    test_outer = test_index ^ (test_index >> 1)
    # Ensure chosen Gray word has no zero byte and choose legal nonzero inner.
    if 0 in test_outer.to_bytes(4, "big"):
        test_index = 0x12345678
        test_outer = test_index ^ (test_index >> 1)
    test_inner = 0x5A
    if 0 in test_outer.to_bytes(4, "big"):
        raise AssertionError("test Gray outer unexpectedly contains zero")
    obj = bytearray((d / "object_template.bin").read_bytes())
    noff = m["nonce_object_offset"]
    obj[noff:noff + 5] = test_outer.to_bytes(4, "big") + bytes([test_inner])
    digest = hashlib.sha1(obj).digest()
    td = [int.from_bytes(digest[i:i + 4], "big") for i in range(0, 20, 4)]

    def inner_mask(t: int) -> int:
        return ip[t]

    def cidx(t: int) -> int:
        return sum(1 for q in range(16, t) if ip[q].bit_count() > 1)

    def inner_delta_expr(t: int) -> str:
        mask = inner_mask(t)
        pc = mask.bit_count()
        if pc == 0:
            return "P<V>{}"
        if pc == 1:
            r = (mask & -mask).bit_length() - 1
            if r == 7:
                return "r7"
            if r == 3:
                return "r3"
            return f"rotp<V>(raw,{r})"
        return f"loadp<V,CACHE>(inner_table,{cidx(t)},j)"

    def round_lines() -> str:
        out: list[str] = []
        phase = lambda t: "sch" if t < 20 else "sp1" if t < 40 else "smj" if t < 60 else "sp3"
        for t in range(16, 79):
            k = t - 16
            owner, odd = k >> 1, k & 1
            reg = "ow1" if odd else "ow0"
            out += [
                "    {",
                f"      const uint32_t od=__shfl_sync(0xffffffffu,{reg},{owner});",
                f"      auto dv={inner_delta_expr(t)};",
                f"      const uint32_t bw={c32(b80[t])}^od;",
                "      #pragma unroll",
                f"      for(int i=0;i<V;i++) {phase(t)}(st[i],bw^dv.x[i]);",
                "    }",
            ]
        t = 79
        owner, odd = (t - 16) >> 1, (t - 16) & 1
        reg = "ow1" if odd else "ow0"
        out += [
            "    {",
            f"      const uint32_t od=__shfl_sync(0xffffffffu,{reg},{owner});",
            f"      auto dv={inner_delta_expr(t)};",
            f"      const uint32_t bw={c32(b80[t])}^od;",
            "      #pragma unroll",
            "      for(int i=0;i<V;i++) {",
            "        const uint32_t a80=fina(st[i],bw^dv.x[i]);",
            "        const unsigned in=j+(unsigned)i;",
            "        if constexpr(DIAG) {",
            "          if(outer==TEST_OUTER && in==TEST_INNER) {",
            f"            diag[0]=a80+{c32(hin[0])};",
            f"            diag[1]=st[i].a+{c32(hin[1])};",
            f"            diag[2]=rol32(st[i].b,30)+{c32(hin[2])};",
            f"            diag[3]=st[i].c+{c32(hin[3])};",
            f"            diag[4]=st[i].d+{c32(hin[4])};",
            "          }",
            "        }",
            "        if(prod_match(a80,st[i]) && in && !zero_byte(outer))",
            "          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,",
            "                    (unsigned long long)((uint64_t(outer)<<8)|in));",
            "      }",
            "    }",
        ]
        return "\n".join(out)

    rounds = round_lines()
    src = f'''// AUTO-GENERATED persistent Gray-code exact Git binary-tail study.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
static constexpr uint32_t TEST_INDEX={c32(test_index)},TEST_OUTER={c32(test_outer)};
static constexpr unsigned TEST_INNER={test_inner}u;
static constexpr uint32_t TEST_DIGEST[5]={{{','.join(c32(x) for x in td)}}};
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
template<int V>struct P{{uint32_t x[V];}};
template<int V>__device__ __forceinline__ P<V>rawp(unsigned j){{
 P<V>z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=uint32_t(j+i)<<24;
 return z;
}}
template<int V>__device__ __forceinline__ P<V>rotp(const P<V>&p,int r){{
 P<V>z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=rol32(p.x[i],r);
 return z;
}}
template<int V,int CACHE>__device__ __forceinline__ P<V>loadp(const uint32_t*t,int row,unsigned j){{
 P<V>z{{}};const uint32_t*p=t+row*256+j;
 #pragma unroll
 for(int q=0;q<V;q+=4){{uint4 a;if constexpr(CACHE==0)asm volatile("ld.global.ca.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));else asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));z.x[q]=a.x;z.x[q+1]=a.y;z.x[q+2]=a.z;z.x[q+3]=a.w;}}
 return z;
}}
__device__ __forceinline__ uint2 load_outer2(const uint32_t*p){{uint2 a;asm volatile("ld.global.ca.v2.u32 {{%0,%1}}, [%2];":"=r"(a.x),"=r"(a.y):"l"(p));return a;}}
__device__ __forceinline__ bool zero_byte(uint32_t x){{return ((x-0x01010101u)&~x&0x80808080u)!=0;}}
__device__ __forceinline__ bool prod_match(uint32_t a80,const S&s){{{prod_body}}}

template<int V,int CACHE,int OUTERS_PER_WARP,bool DIAG=false>
__global__ void kernel_gray(uint64_t index_base,uint64_t index_count,const uint32_t*inner_table,const uint32_t*outer_basis,uint64_t*winner,uint32_t*diag){{
 static_assert(V==4||V==8,"V");static_assert(256%(32*V)==0,"mapping");static_assert(OUTERS_PER_WARP>=1,"loop");
 const unsigned lane=threadIdx.x&31u;const uint64_t warp=uint64_t(blockIdx.x)*(blockDim.x>>5)+(threadIdx.x>>5);
 const uint64_t first=index_base+warp*OUTERS_PER_WARP,end=index_base+index_count;if(first>=end||first>0xffffffffull)return;
 const uint64_t stop=(first+OUTERS_PER_WARP<end)?first+OUTERS_PER_WARP:end;
 uint32_t outer=(uint32_t)first^((uint32_t)first>>1);uint32_t ow0=0,ow1=0;
 // Initialize this lane's two distributed W16..W79 words from set outer bits.
 #pragma unroll
 for(int bit=0;bit<32;bit++)if((outer>>bit)&1u){{uint2 z=load_outer2(outer_basis+bit*64+lane*2);ow0^=z.x;ow1^=z.y;}}
 for(uint64_t ix=first;ix<stop;ix++){{
   S common{{{','.join(c32(x) for x in pre)}}};sch(common,outer);constexpr unsigned CHUNK=32*V;
   #pragma unroll 1
   for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{
     const unsigned j=chunk+lane*V;S st[V];
     #pragma unroll
     for(int i=0;i<V;i++)st[i]=common;
     auto raw=rawp<V>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
     #pragma unroll
     for(int i=0;i<V;i++)sch(st[i],{c32(b16[13])}^raw.x[i]);
     #pragma unroll
     for(int i=0;i<V;i++)sch(st[i],{c32(b16[14])});
     #pragma unroll
     for(int i=0;i<V;i++)sch(st[i],{c32(b16[15])});
{rounds}
   }}
   if(ix+1<stop){{const unsigned bit=(unsigned)(__ffs((int)((uint32_t)ix+1u))-1);uint2 z=load_outer2(outer_basis+bit*64+lane*2);ow0^=z.x;ow1^=z.y;outer^=1u<<bit;}}
 }}
}}
#ifndef GRAY_NO_MAIN
static std::vector<uint32_t>read_words(const char*p){{FILE*f=fopen(p,"rb");if(!f){{perror(p);exit(2);}}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint32_t>v((size_t)n/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(2);fclose(f);return v;}}
template<int V,int LOOP,int CACHE=0>static void bench(const char*name,uint64_t n,int reps,const uint32_t*it,const uint32_t*ob,uint64_t*w){{constexpr int B=256;uint64_t warps=(n+LOOP-1)/LOOP;dim3 grid((unsigned)((warps+7)/8));uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));kernel_gray<V,CACHE,LOOP,false><<<grid,B>>>(0,n,it,ob,w,nullptr);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));for(int r=0;r<reps;r++)kernel_gray<V,CACHE,LOOP,false><<<grid,B>>>(0,n,it,ob,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));cudaFuncAttributes fa{{}};CUDA_OK(cudaFuncGetAttributes(&fa,kernel_gray<V,CACHE,LOOP,false>));printf("%-19s %.6f GH/s regs=%d local=%zu smem=%zu\\n",name,double(n)*256.0*reps/(double(ms)*1e6),fa.numRegs,fa.localSizeBytes,fa.sharedSizeBytes);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
int main(int argc,char**argv){{const char*ipath=argc>1?argv[1]:"complex_delta.bin";const char*opath=argc>2?argv[2]:"outer_gray_delta.bin";uint64_t n=argc>3?strtoull(argv[3],nullptr,0):(1ull<<20);int reps=argc>4?atoi(argv[4]):4;auto ih=read_words(ipath),oh=read_words(opath);if(ih.size()!=32u*256u||oh.size()!=32u*64u){{fprintf(stderr,"bad tables\\n");return 2;}}uint32_t*it=nullptr,*ob=nullptr,*diag=nullptr;uint64_t*w=nullptr;CUDA_OK(cudaMalloc(&it,ih.size()*4));CUDA_OK(cudaMemcpy(it,ih.data(),ih.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&ob,oh.size()*4));CUDA_OK(cudaMemcpy(ob,oh.data(),oh.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&diag,20));CUDA_OK(cudaMemset(diag,0,20));CUDA_OK(cudaMalloc(&w,8));uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));kernel_gray<8,0,1,true><<<1,32>>>(TEST_INDEX,1,it,ob,w,diag);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint32_t got[5]={{}};CUDA_OK(cudaMemcpy(got,diag,20,cudaMemcpyDeviceToHost));if(memcmp(got,TEST_DIGEST,20)){{fprintf(stderr,"GPU digest correctness FAIL\\n");for(int i=0;i<5;i++)fprintf(stderr," H%d got=%08x want=%08x\\n",i,got[i],TEST_DIGEST[i]);return 3;}}printf("GPU digest correctness PASS gray-index=%08x outer=%08x inner=%02x\\n",TEST_INDEX,TEST_OUTER,TEST_INNER);bench<8,1>("gray-v8-loop1",n,reps,it,ob,w);bench<8,4>("gray-v8-loop4",n,reps,it,ob,w);bench<8,16>("gray-v8-loop16",n,reps,it,ob,w);bench<8,64>("gray-v8-loop64",n,reps,it,ob,w);bench<8,256>("gray-v8-loop256",n,reps,it,ob,w);bench<4,16>("gray-v4-loop16",n,reps,it,ob,w);bench<4,64>("gray-v4-loop64",n,reps,it,ob,w);bench<8,64,1>("gray-v8-nc-l64",n,reps,it,ob,w);CUDA_OK(cudaFree(it));CUDA_OK(cudaFree(ob));CUDA_OK(cudaFree(diag));CUDA_OK(cudaFree(w));}}
#endif
'''
    src = src.replace(";#pragma unroll\\n", ";\\n#pragma unroll\\n")
    out = Path(args.output) if args.output else d / "literal_binary_gray.cu"
    out.write_text(src)
    analysis = {
        "outer_basis_bytes": len(outer_basis) * 4,
        "outer_words_per_lane": 2,
        "expanded_round_shuffles_per_outer_v8": 64,
        "gray_update_load_bytes_per_warp_outer": 32 * 8,
        "test_index": test_index,
        "test_outer": f"{test_outer:08x}",
        "test_inner": f"{test_inner:02x}",
        "test_digest": [f"{x:08x}" for x in td],
    }
    Path(str(out) + ".json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
