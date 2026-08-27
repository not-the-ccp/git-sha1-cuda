#!/usr/bin/env python3
"""Generate a no-shared-memory exact Git binary-tail CUDA study.

The job must have a five-byte raw nonce at final-block offsets 48..52:
  W12 = four-byte raw outer counter
  W13[31:24] = one-byte raw inner value

The generated kernels map one outer counter to a lane group and stream the
expanded schedule directly from the two affine source words.  No outer
schedule is written to shared memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MASK = 0xFFFFFFFF


def rol(x: int, n: int) -> int:
    n &= 31
    return ((x << n) & MASK) | (x >> ((32 - n) & 31)) if n else x


def expand(w16: list[int]) -> list[int]:
    w = list(w16) + [0] * 64
    for t in range(16, 80):
        w[t] = rol(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1)
    return w


def poly(source_word: int) -> list[int]:
    """Rotation-polynomial masks for one symbolic source word."""
    p = [0] * 80
    p[source_word] = 1
    for t in range(16, 80):
        p[t] = rol(p[t - 3] ^ p[t - 8] ^ p[t - 14] ^ p[t - 16], 1)
    return p


def c32(x: int) -> str:
    return f"0x{x & MASK:08x}u"


CACHE_ROTATIONS = {
    0: {},
    2: {4: "or4", 8: "or8"},
    4: {4: "or4", 8: "or8", 3: "or3", 6: "or6"},
    6: {4: "or4", 8: "or8", 3: "or3", 6: "or6", 7: "or7", 12: "or12"},
    8: {
        4: "or4", 8: "or8", 3: "or3", 6: "or6",
        7: "or7", 12: "or12", 11: "or11", 5: "or5",
    },
}


def outer_expr(mask: int, cache_level: int) -> str:
    if not mask:
        return "0u"
    names = CACHE_ROTATIONS[cache_level]
    xs: list[str] = []
    for r in range(32):
        if not ((mask >> r) & 1):
            continue
        if r == 0:
            xs.append("outer")
        elif r in names:
            xs.append(names[r])
        else:
            xs.append(f"rol32(outer,{r})")
    return "^".join(xs)


def parse_prefix(prefix_hex: str, hin: list[int]) -> str:
    p = prefix_hex.lower().removeprefix("0x")
    if not 1 <= len(p) <= 40 or any(ch not in "0123456789abcdef" for ch in p):
        raise SystemExit("--prefix-hex must be 1..40 hexadecimal digits")
    bits = len(p) * 4
    full = (p + "0" * 40)[:40]
    target_words = [int(full[i:i + 8], 16) for i in range(0, 40, 8)]

    # H0 = a80 + HIN0.  For >=32 bits, subtract HIN0 once at generation time.
    # For a short prefix, unsigned subtraction implements the possibly-wrapped
    # modular interval [target-HIN0, target-HIN0+span).
    if bits < 32:
        span = 1 << (32 - bits)
        mask = (MASK << (32 - bits)) & MASK
        want = target_words[0] & mask
        base = (want - hin[0]) & MASK
        first = f"if(uint32_t(a80-{c32(base)}) >= {span}u) return false;"
    else:
        adjusted = (target_words[0] - hin[0]) & MASK
        first = f"if(a80!={c32(adjusted)}) return false;"

    # At the start of round 79, after computing a80, the other final SHA state
    # words are b80=s.a, c80=ROL30(s.b), d80=s.c, e80=s.d.
    expressions = [
        "",
        f"(s.a+{c32(hin[1])})",
        f"(rol32(s.b,30)+{c32(hin[2])})",
        f"(s.c+{c32(hin[3])})",
        f"(s.d+{c32(hin[4])})",
    ]
    checks = [first]
    for wi in range(1, 5):
        used = bits - 32 * wi
        if used <= 0:
            break
        take = min(32, used)
        mask = MASK if take == 32 else ((MASK << (32 - take)) & MASK)
        want = target_words[wi] & mask
        checks.append(
            f"if((({expressions[wi]})&{c32(mask)})!={c32(want)}) return false;"
        )
    checks.append("return true;")
    return "".join(checks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="binary-tail job directory")
    ap.add_argument("-o", "--output")
    ap.add_argument("--prefix-hex", default="13579bdf")
    args = ap.parse_args()

    d = Path(args.job)
    meta = json.loads((d / "job.json").read_text())
    b16 = [int(x, 16) for x in meta["base16"]]
    hin = [int(x, 16) for x in meta["hin"]]
    pre = [int(x, 16) for x in meta["pre"]]

    if meta.get("mode") != "binary-tail-4+1":
        raise SystemExit("job is not binary-tail-4+1")
    if (meta["nonce_within_block"], meta["first_word"], meta["inner_word"]) != (48, 12, 13):
        raise SystemExit("stream kernel requires exact W12/W13 binary-tail layout")
    if b16[12] != 0 or (b16[13] & 0xFF000000) != 0:
        raise SystemExit("binary-tail job base words overlap the raw W12/W13 candidate field")

    b80 = expand(b16)
    outer_poly = poly(12)
    inner_poly = poly(13)
    complex_rounds = [t for t in range(16, 80) if inner_poly[t].bit_count() > 1]
    if len(complex_rounds) != 32:
        raise SystemExit(f"unexpected W13 complex-row count: {len(complex_rounds)}")

    prod_body = parse_prefix(args.prefix_hex, hin)

    # Independent digest-capture candidate.  All five bytes are nonzero, so it
    # is legal under the normal Git plumbing policy.
    test_outer = 0x01020304
    test_inner = 0x05
    test_obj = bytearray((d / "object_template.bin").read_bytes())
    noff = meta["nonce_object_offset"]
    test_obj[noff:noff + 5] = test_outer.to_bytes(4, "big") + bytes([test_inner])
    td = hashlib.sha1(test_obj).digest()
    test_words = [int.from_bytes(td[i:i + 4], "big") for i in range(0, 20, 4)]

    def round_body(cache_level: int) -> str:
        lines: list[str] = []
        phase = lambda t: "sch" if t < 20 else "sp1" if t < 40 else "smj" if t < 60 else "sp3"
        for t in range(16, 79):
            bw = f"({c32(b80[t])}^({outer_expr(outer_poly[t], cache_level)}))"
            lines += [
                "  {",
                f"    auto dv=idelta<{t},V,CACHE>(table,raw,j,r7,r3);",
                f"    const uint32_t bw={bw};",
                "    #pragma unroll",
                f"    for(int i=0;i<V;i++) {phase(t)}(st[i],bw^dv.x[i]);",
                "  }",
            ]
        t = 79
        bw = f"({c32(b80[t])}^({outer_expr(outer_poly[t], cache_level)}))"
        lines += [
            "  {",
            "    auto dv=idelta<79,V,CACHE>(table,raw,j,r7,r3);",
            f"    const uint32_t bw={bw};",
            "    #pragma unroll",
            "    for(int i=0;i<V;i++) {",
            "      const uint32_t a80=fina(st[i],bw^dv.x[i]);",
            "      const unsigned in=j+(unsigned)i;",
            "      if constexpr(DIAG) {",
            "        if(outer==TEST_OUTER && in==TEST_INNER) {",
            f"          diag[0]=a80+{c32(hin[0])};",
            f"          diag[1]=st[i].a+{c32(hin[1])};",
            f"          diag[2]=rol32(st[i].b,30)+{c32(hin[2])};",
            f"          diag[3]=st[i].c+{c32(hin[3])};",
            f"          diag[4]=st[i].d+{c32(hin[4])};",
            "        }",
            "      }",
            "      if(prod_match(a80,st[i]) && in && !zero_byte(outer))",
            "        atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,",
            "                  (unsigned long long)((uint64_t(outer)<<8)|in));",
            "    }",
            "  }",
        ]
        return "\n".join(lines)

    cache_branches = []
    for idx, cache_level in enumerate((0, 2, 4, 6, 8)):
        kw = "if" if idx == 0 else "else if" if cache_level != 8 else "else"
        cond = f" constexpr(OCACHE=={cache_level})" if kw != "else" else ""
        cache_branches.append(f"    {kw}{cond} {{\n{round_body(cache_level)}\n    }}")

    src = f'''// AUTO-GENERATED no-shared streamed exact Git binary-tail study.
// job mode: W12 raw outer + W13 high-byte raw inner; target={args.prefix_hex.lower()}
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
static constexpr uint32_t TEST_OUTER={c32(test_outer)};
static constexpr unsigned TEST_INNER={test_inner}u;
static constexpr uint32_t TEST_DIGEST[5]={{{','.join(c32(x) for x in test_words)}}};
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
constexpr uint32_t imask(int t){{uint32_t p[80]{{}};p[13]=1;for(int i=16;i<=t;i++)p[i]=cr(p[i-3]^p[i-8]^p[i-14]^p[i-16],1);return p[t];}}
constexpr int pc(uint32_t x){{int n=0;while(x){{n+=x&1u;x>>=1;}}return n;}}
constexpr int ctz(uint32_t x){{int n=0;while(!(x&1u)){{n++;x>>=1;}}return n;}}
constexpr int cidx(int t){{int n=0;for(int q=16;q<t;q++)if(pc(imask(q))>1)n++;return n;}}

template<int V> struct P{{uint32_t x[V];}};
template<int V> __device__ __forceinline__ P<V> rawp(unsigned j){{
 P<V> z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=uint32_t(j+i)<<24;
 return z;
}}
template<int V> __device__ __forceinline__ P<V> rotp(const P<V>&p,int r){{
 P<V> z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=rol32(p.x[i],r);
 return z;
}}
template<int V,int CACHE> __device__ __forceinline__ P<V> loadp(const uint32_t*t,int row,unsigned j){{
 P<V> z{{}};const uint32_t*p=t+row*256+j;
 #pragma unroll
 for(int q=0;q<V;q+=4){{uint4 a;if constexpr(CACHE==0)asm volatile("ld.global.ca.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));else asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));z.x[q]=a.x;z.x[q+1]=a.y;z.x[q+2]=a.z;z.x[q+3]=a.w;}}
 return z;
}}
template<int T,int V,int CACHE> __device__ __forceinline__ P<V> idelta(const uint32_t*t,const P<V>&raw,unsigned j,const P<V>&r7,const P<V>&r3){{constexpr uint32_t m=imask(T);constexpr int n=pc(m);if constexpr(n==0)return P<V>{{}};else if constexpr(n==1){{constexpr int e=ctz(m);if constexpr(e==7)return r7;else if constexpr(e==3)return r3;else return rotp<V>(raw,e);}}else return loadp<V,CACHE>(t,cidx(T),j);}}
__device__ __forceinline__ bool zero_byte(uint32_t x){{return ((x-0x01010101u)&~x&0x80808080u)!=0;}}
__device__ __forceinline__ bool prod_match(uint32_t a80,const S&s){{{prod_body}}}

template<int G,int V,int CACHE,int OCACHE,bool DIAG=false>
__global__ void kernel_stream(uint64_t outer_base,uint64_t outer_count,const uint32_t*table,uint64_t*winner,uint32_t*diag){{
 static_assert(G==8||G==16||G==32,"G");static_assert(V==4||V==8,"V");static_assert(256%(G*V)==0,"mapping");
 const int lane_group=threadIdx.x&(G-1),groups=blockDim.x/G;
 const uint64_t gi=uint64_t(blockIdx.x)*groups+(threadIdx.x/G),ov=outer_base+gi;
 if(gi>=outer_count||ov>0xffffffffull)return;
 const uint32_t outer=(uint32_t)ov;
 uint32_t or4=0,or8=0,or3=0,or6=0,or7=0,or12=0,or11=0,or5=0;
 if constexpr(OCACHE>=2){{or4=rol32(outer,4);or8=rol32(outer,8);}}
 if constexpr(OCACHE>=4){{or3=rol32(outer,3);or6=rol32(outer,6);}}
 if constexpr(OCACHE>=6){{or7=rol32(outer,7);or12=rol32(outer,12);}}
 if constexpr(OCACHE>=8){{or11=rol32(outer,11);or5=rol32(outer,5);}}
 S common{{{','.join(c32(x) for x in pre)}}};
 sch(common,outer); // W12
 constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{
   const unsigned j=chunk+unsigned(lane_group)*V;
   S st[V];
   #pragma unroll
   for(int i=0;i<V;i++)st[i]=common;
   auto raw=rawp<V>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[13])}^raw.x[i]);
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[14])});
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[15])});
{chr(10).join(cache_branches)}
 }}
}}

#ifndef STREAM_NO_MAIN
static std::vector<uint32_t> read_table(const char*p){{FILE*f=fopen(p,"rb");if(!f){{perror(p);exit(2);}}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint32_t>v((size_t)n/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(2);fclose(f);return v;}}
template<int G,int V,int CACHE,int OCACHE> static void bench(const char*n,uint64_t outer_count,int reps,const uint32_t*t,uint64_t*w){{
 constexpr int B=256;dim3 grid((unsigned)((outer_count+B/G-1)/(B/G)));uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));
 kernel_stream<G,V,CACHE,OCACHE,false><<<grid,B>>>(0,outer_count,t,w,nullptr);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
 CUDA_OK(cudaEventRecord(a));for(int r=0;r<reps;r++)kernel_stream<G,V,CACHE,OCACHE,false><<<grid,B>>>(0,outer_count,t,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));cudaFuncAttributes fa{{}};CUDA_OK(cudaFuncGetAttributes(&fa,kernel_stream<G,V,CACHE,OCACHE,false>));printf("%-20s %.6f GH/s regs=%d local=%zu smem=%zu\\n",n,double(outer_count)*256.0*reps/(double(ms)*1e6),fa.numRegs,fa.localSizeBytes,fa.sharedSizeBytes);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));
}}
int main(int argc,char**argv){{
 const char*path=argc>1?argv[1]:"complex_delta.bin";uint64_t n=argc>2?strtoull(argv[2],nullptr,0):(1ull<<18);int reps=argc>3?atoi(argv[3]):6;
 auto h=read_table(path);if(h.size()!=32u*256u){{fprintf(stderr,"bad table\\n");return 2;}}
 uint32_t*d=nullptr,*diag=nullptr;uint64_t*w=nullptr;CUDA_OK(cudaMalloc(&d,h.size()*4));CUDA_OK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&diag,20));CUDA_OK(cudaMemset(diag,0,20));CUDA_OK(cudaMalloc(&w,8));
 uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));kernel_stream<32,4,0,4,true><<<1,32>>>(TEST_OUTER,1,d,w,diag);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint32_t got[5]={{}};CUDA_OK(cudaMemcpy(got,diag,20,cudaMemcpyDeviceToHost));if(memcmp(got,TEST_DIGEST,20)){{fprintf(stderr,"GPU digest correctness FAIL\\n");for(int i=0;i<5;i++)fprintf(stderr," H%d got=%08x want=%08x\\n",i,got[i],TEST_DIGEST[i]);return 3;}}printf("GPU digest correctness PASS outer=%08x inner=%02x\\n",TEST_OUTER,TEST_INNER);
 bench<32,4,0,0>("stream-g32-v4-c0",n,reps,d,w);bench<32,4,0,4>("stream-g32-v4-c4",n,reps,d,w);bench<32,4,0,6>("stream-g32-v4-c6",n,reps,d,w);bench<32,4,0,8>("stream-g32-v4-c8",n,reps,d,w);bench<32,8,0,0>("stream-g32-v8-c0",n,reps,d,w);bench<32,8,0,4>("stream-g32-v8-c4",n,reps,d,w);bench<16,4,0,4>("stream-g16-v4-c4",n,reps,d,w);bench<16,8,0,4>("stream-g16-v8-c4",n,reps,d,w);bench<8,8,0,4>("stream-g8-v8-c4",n,reps,d,w);bench<32,8,1,4>("stream-g32-v8-nc",n,reps,d,w);
 CUDA_OK(cudaFree(d));CUDA_OK(cudaFree(diag));CUDA_OK(cudaFree(w));
}}
#endif
'''
    # Keep pragmas on their own line when compact template fragments above put
    # them after a declaration.
    src = src.replace(";#pragma unroll\\n", ";\\n#pragma unroll\\n")
    out = Path(args.output) if args.output else d / "literal_binary_stream.cu"
    out.write_text(src)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
