#!/usr/bin/env python3
"""Generate the standalone CUDA benchmark for late W12/W13 nonce layouts.

The generated constants describe real, padded SHA-1 messages: three fixed
64-byte prefix blocks followed by one final partial block.  Python's hashlib
provides an independent whole-message digest for each GPU correctness vector.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MASK = 0xFFFFFFFF
HERE = Path(__file__).resolve().parent


def rol(x: int, n: int) -> int:
    return ((x << n) & MASK) | (x >> (32 - n))


def expand(w16: list[int]) -> list[int]:
    w = list(w16) + [0] * 64
    for t in range(16, 80):
        w[t] = rol(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1)
    return w


def rounds(state: list[int], schedule: list[int], begin: int, end: int) -> list[int]:
    a, b, c, d, e = state
    for t in range(begin, end):
        if t < 20:
            f, k = d ^ (b & (c ^ d)), 0x5A827999
        elif t < 40:
            f, k = b ^ c ^ d, 0x6ED9EBA1
        elif t < 60:
            f, k = (b & c) | (d & (b | c)), 0x8F1BBCDC
        else:
            f, k = b ^ c ^ d, 0xCA62C1D6
        z = (rol(a, 5) + f + e + k + schedule[t]) & MASK
        e, d, c, b, a = d, c, rol(b, 30), a, z
    return [a, b, c, d, e]


def compress(state: list[int], block: bytes) -> list[int]:
    w = expand(words(block))
    work = rounds(state, w, 0, 80)
    return [(x + y) & MASK for x, y in zip(state, work)]


def words(block: bytes) -> list[int]:
    assert len(block) == 64
    return [int.from_bytes(block[i : i + 4], "big") for i in range(0, 64, 4)]


def pad_tail(prefix_bytes: int, data: bytes) -> bytes:
    assert len(data) <= 55
    b = bytearray(data)
    b.append(0x80)
    b.extend(b"\0" * (56 - len(b)))
    b.extend(((prefix_bytes + len(data)) * 8).to_bytes(8, "big"))
    assert len(b) == 64
    return bytes(b)


def c32(x: int) -> str:
    return f"0x{x & MASK:08x}u"


def arr(name: str, xs: list[int]) -> str:
    return f"static constexpr uint32_t {name}[{len(xs)}]={{" + ",".join(c32(x) for x in xs) + "};"


def symbolic_schedule(first: int, base16: list[int], word12: str, word13: str, prefix: str = "sched") -> str:
    """Leader-only compact schedule, using SHA-1's equivalent ROL2 recurrence."""

    def operand(idx: int) -> str:
        if idx == 12:
            return f"({word12})"
        if idx == 13:
            return f"({word13})"
        if idx < 16:
            return c32(base16[idx])
        return f"{prefix}[{idx - 16}]"

    out: list[str] = []
    for t in range(16, 32):
        terms = [operand(t - d) for d in (3, 8, 14, 16)]
        out.append(f"      {prefix}[{t - 16}]=rol32(" + "^".join(terms) + ",1);")
    for t in range(32, 80):
        terms = [operand(t - d) for d in (6, 16, 28, 32)]
        out.append(f"      {prefix}[{t - 16}]=rol32(" + "^".join(terms) + ",2);")
    return "\n".join(out)


def round_lines(table: str, shift: int, sched_expr: str, diag_condition: str, indent: str = "      ") -> str:
    out: list[str] = []
    phase = lambda t: "sch" if t < 20 else "sp1" if t < 40 else "smj" if t < 60 else "sp3"
    for t in range(16, 79):
        out += [
            f"{indent}{{ auto dv=delta<{t},V,CACHE,{shift}>({table},raw,j,r7,r3);",
            f"{indent}  const uint32_t bw={sched_expr.format(t=t)};",
            f"{indent}  #pragma unroll",
            f"{indent}  for(int i=0;i<V;i++){phase(t)}(st[i],bw^dv.x[i]); }}",
        ]
    out += [
        f"{indent}{{ auto dv=delta<79,V,CACHE,{shift}>({table},raw,j,r7,r3);",
        f"{indent}  const uint32_t bw={sched_expr.format(t=79)};",
        f"{indent}  #pragma unroll",
        f"{indent}  for(int i=0;i<V;i++){{",
        f"{indent}    const uint32_t a80=fina(st[i],bw^dv.x[i]);",
        f"{indent}    const unsigned inner=j+(unsigned)i;",
        f"{indent}    finish(a80,st[i],outer,middle,inner,target,winner,diag,DIAG && ({diag_condition}));",
        f"{indent}  }} }}",
    ]
    return "\n".join(out)


def make_layout(data_len: int, mutable: range) -> tuple[bytes, list[int]]:
    data = bytearray(((29 + 47 * i) & 255) for i in range(data_len))
    for i in mutable:
        data[i] = 0
    tail = pad_tail(3 * 64, bytes(data))
    return bytes(data), words(tail)


def main() -> int:
    prefix = b"".join(bytes(((11 + 73 * i + 31 * block) & 255) for i in range(64)) for block in range(3))
    iv = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    hin = list(iv)
    for off in range(0, len(prefix), 64):
        hin = compress(hin, prefix[off : off + 64])

    # 4+1 reference: offsets 48..52. Hierarchical 3+1+1: offsets 49..53.
    b_data, b16 = make_layout(54, range(48, 53))
    h_data, h16 = make_layout(54, range(49, 54))
    # W13-only epoch: offsets 52..54, padding is W13's low byte at offset 55.
    e_data, e16 = make_layout(55, range(52, 55))
    pre_b = rounds(hin, expand(b16), 0, 12)
    pre_h = rounds(hin, expand(h16), 0, 12)
    pre_e = rounds(hin, expand(e16), 0, 13)

    b_outer, b_inner = 0x01020304, 0x05
    b_test = bytearray(b_data)
    b_test[48:53] = b_outer.to_bytes(4, "big") + bytes([b_inner])
    b_digest = hashlib.sha1(prefix + bytes(b_test)).digest()

    h_outer, h_middle, h_inner = 0x010203, 0x04, 0x05
    h_test = bytearray(h_data)
    h_test[49:54] = h_outer.to_bytes(3, "big") + bytes([h_middle, h_inner])
    h_digest = hashlib.sha1(prefix + bytes(h_test)).digest()

    e_middle, e_inner = 0x0102, 0x03
    e_test = bytearray(e_data)
    e_test[52:55] = e_middle.to_bytes(2, "big") + bytes([e_inner])
    e_digest = hashlib.sha1(prefix + bytes(e_test)).digest()

    digest_words = lambda d: [int.from_bytes(d[i : i + 4], "big") for i in range(0, 20, 4)]

    baseline_schedule = symbolic_schedule(12, b16, "outer", c32(b16[13]))
    hier_schedule = symbolic_schedule(12, h16, f"({c32(h16[12])}|outer)", f"({c32(h16[13])}|(middle<<24))")
    epoch_schedule = symbolic_schedule(13, e16, c32(e16[12]), f"({c32(e16[13])}|(middle<<16))")

    baseline_rounds = round_lines(
        "table24", 24, "sched[{t}-16]",
        "outer==B_TEST_OUTER && inner==B_TEST_INNER",
    )
    hier_rounds = round_lines(
        "table16", 16, "sched[{t}-16]",
        "outer==H_TEST_OUTER && middle==H_TEST_MIDDLE && inner==H_TEST_INNER",
    )
    epoch_rounds = round_lines(
        "table8", 8, "sched[{t}-16]",
        "middle==E_TEST_MIDDLE && inner==E_TEST_INNER",
    )

    src = f'''// AUTO-GENERATED by generate_benchmark.py. Do not hand edit.
// Exact SHA-1 late-dependency benchmark: raw 4+1, 3+1+1, and W13-only epoch.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
{arr("B_TEST_DIGEST", digest_words(b_digest))}
{arr("H_TEST_DIGEST", digest_words(h_digest))}
{arr("E_TEST_DIGEST", digest_words(e_digest))}
static constexpr uint32_t B_TEST_OUTER={c32(b_outer)},B_TEST_INNER={b_inner}u;
static constexpr uint32_t H_TEST_OUTER={c32(h_outer)},H_TEST_MIDDLE={h_middle}u,H_TEST_INNER={h_inner}u;
static constexpr uint32_t E_TEST_MIDDLE={e_middle}u,E_TEST_INNER={e_inner}u;

#define CUDA_OK(x) do{{cudaError_t e_=(x);if(e_!=cudaSuccess){{fprintf(stderr,"CUDA error %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}}}while(0)
__host__ __device__ __forceinline__ uint32_t rol32(uint32_t x,unsigned n){{
#ifdef __CUDA_ARCH__
 return __funnelshift_l(x,x,n);
#else
 return (x<<n)|(x>>(32-n));
#endif
}}
__device__ __forceinline__ uint32_t fch(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xca;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fpa(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0x96;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fmj(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xe8;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
struct S{{uint32_t a,b,c,d,e;}};
__device__ __forceinline__ void sch(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fch(s.b,s.c,s.d)+s.e+K0+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void sp1(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K1+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void smj(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fmj(s.b,s.c,s.d)+s.e+K2+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void sp3(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ uint32_t fina(const S&s,uint32_t w){{return rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;}}

__host__ __device__ constexpr uint32_t cr(uint32_t x,int n){{n&=31;return n?((x<<n)|(x>>(32-n))):x;}}
__host__ __device__ constexpr uint32_t pmask(int t){{uint32_t p[80]{{}};p[13]=1;for(int i=16;i<=t;i++)p[i]=cr(p[i-3]^p[i-8]^p[i-14]^p[i-16],1);return p[t];}}
__host__ __device__ constexpr int pc(uint32_t x){{int n=0;while(x){{n+=x&1u;x>>=1;}}return n;}}
__host__ __device__ constexpr int ctz1(uint32_t x){{int n=0;while(!(x&1u)){{n++;x>>=1;}}return n;}}
__host__ __device__ constexpr int cidx(int t){{int n=0;for(int q=16;q<t;q++)if(pc(pmask(q))>1)n++;return n;}}
template<int V>struct P{{uint32_t x[V];}};
template<int V,int SHIFT>__device__ __forceinline__ P<V> rawp(unsigned j){{P<V>z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=uint32_t(j+i)<<SHIFT;return z;}}
template<int V>__device__ __forceinline__ P<V> rotp(const P<V>&p,int r){{P<V>z{{}};
 #pragma unroll
 for(int i=0;i<V;i++)z.x[i]=rol32(p.x[i],r);return z;}}
template<int V,int CACHE>__device__ __forceinline__ P<V> loadp(const uint32_t*t,int row,unsigned j){{P<V>z{{}};const uint32_t*p=t+row*256+j;
 #pragma unroll
 for(int q=0;q<V;q+=4){{uint4 a;if constexpr(CACHE==0)asm volatile("ld.global.ca.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));else asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));z.x[q]=a.x;z.x[q+1]=a.y;z.x[q+2]=a.z;z.x[q+3]=a.w;}}return z;}}
template<int T,int V,int CACHE,int SHIFT>__device__ __forceinline__ P<V> delta(const uint32_t*t,const P<V>&raw,unsigned j,const P<V>&r7,const P<V>&r3){{constexpr uint32_t m=pmask(T);constexpr int n=pc(m);if constexpr(n==0)return P<V>{{}};else if constexpr(n==1){{constexpr int e=ctz1(m);if constexpr(e==7)return r7;else if constexpr(e==3)return r3;else return rotp<V>(raw,e);}}else return loadp<V,CACHE>(t,cidx(T),j);}}

__device__ __forceinline__ void finish(uint32_t a80,const S&s,uint32_t outer,uint32_t middle,uint32_t inner,uint32_t target,uint64_t*winner,uint32_t*diag,bool do_diag){{
 if(do_diag){{diag[0]=a80+{c32(hin[0])};diag[1]=s.a+{c32(hin[1])};diag[2]=rol32(s.b,30)+{c32(hin[2])};diag[3]=s.c+{c32(hin[3])};diag[4]=s.d+{c32(hin[4])};}}
 if(a80==target)atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,(unsigned long long)((uint64_t(outer)<<16)|(uint64_t(middle)<<8)|inner));
}}
template<int G>__device__ __forceinline__ unsigned group_mask(){{if constexpr(G==32)return 0xffffffffu;else return ((1u<<G)-1u)<<((threadIdx.x&31)/G*G);}}

template<int G,int V,int CACHE,bool DIAG=false>
__global__ void baseline(uint64_t outer_base,uint64_t outer_count,const uint32_t*table24,uint32_t target,uint64_t*winner,uint32_t*diag){{
 static_assert(256%(G*V)==0,"mapping");constexpr int STRIDE=64;extern __shared__ uint32_t sm[];
 const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;const uint64_t gi=uint64_t(blockIdx.x)*groups+gl,ov=outer_base+gi;
 if(gi>=outer_count||ov>0xffffffffull)return;const uint32_t outer=(uint32_t)ov,middle=0;uint32_t*sched=sm+gl*STRIDE;
 if(lane==0){{
{baseline_schedule}
 }}
 __syncwarp(group_mask<G>());
 S common{{{','.join(c32(x) for x in pre_b)}}};sch(common,outer);constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{const unsigned j=chunk+unsigned(lane)*V;S st[V];
   #pragma unroll
   for(int i=0;i<V;i++)st[i]=common;auto raw=rawp<V,24>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[13])}+raw.x[i]);
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[14])});
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(b16[15])});
{baseline_rounds}
 }}
}}

template<int G,int V,int CACHE,bool DIAG=false>
__global__ void hierarchical5(uint64_t outer_base,uint64_t outer_count,const uint32_t*table16,uint32_t target,uint64_t*winner,uint32_t*diag){{
 static_assert(256%(G*V)==0,"mapping");constexpr int STRIDE=64;extern __shared__ uint32_t sm[];
 const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;const uint64_t gi=uint64_t(blockIdx.x)*groups+gl,ov=outer_base+gi;
 if(gi>=outer_count||ov>0xffffffull)return;const uint32_t outer=(uint32_t)ov;uint32_t*sched=sm+gl*STRIDE;
 S common{{{','.join(c32(x) for x in pre_h)}}};sch(common,{c32(h16[12])}|outer);constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(uint32_t middle=0;middle<256;middle++){{
   if(lane==0){{
{hier_schedule}
   }}
   __syncwarp(group_mask<G>());
   const uint32_t zbase=rol32(common.a,5)+fch(common.b,common.c,common.d)+common.e+K0+{c32(h16[13])}+(middle<<24);
   #pragma unroll 1
   for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{const unsigned j=chunk+unsigned(lane)*V;S st[V];auto raw=rawp<V,16>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
     #pragma unroll
     for(int i=0;i<V;i++)st[i]=S{{zbase+raw.x[i],common.a,rol32(common.b,30),common.c,common.d}};
     #pragma unroll
     for(int i=0;i<V;i++)sch(st[i],{c32(h16[14])});
     #pragma unroll
     for(int i=0;i<V;i++)sch(st[i],{c32(h16[15])});
{hier_rounds}
   }}
 }}
}}

// Control for the same byte layout without hierarchical reuse: one group owns
// one packed (24-bit W12 outer, 8-bit W13 middle) key and searches 256 inners.
template<int G,int V,int CACHE,bool DIAG=false>
__global__ void flat5(uint64_t key_base,uint64_t key_count,const uint32_t*table16,uint32_t target,uint64_t*winner,uint32_t*diag){{
 static_assert(256%(G*V)==0,"mapping");constexpr int STRIDE=64;extern __shared__ uint32_t sm[];
 const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;const uint64_t gi=uint64_t(blockIdx.x)*groups+gl,kv=key_base+gi;
 if(gi>=key_count||kv>0xffffffffull)return;const uint32_t key=(uint32_t)kv,outer=key>>8,middle=key&255u;uint32_t*sched=sm+gl*STRIDE;
 if(lane==0){{
{hier_schedule}
 }}
 __syncwarp(group_mask<G>());
 S common{{{','.join(c32(x) for x in pre_h)}}};sch(common,{c32(h16[12])}|outer);const uint32_t zbase=rol32(common.a,5)+fch(common.b,common.c,common.d)+common.e+K0+{c32(h16[13])}+(middle<<24);constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{const unsigned j=chunk+unsigned(lane)*V;S st[V];auto raw=rawp<V,16>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
   #pragma unroll
   for(int i=0;i<V;i++)st[i]=S{{zbase+raw.x[i],common.a,rol32(common.b,30),common.c,common.d}};
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(h16[14])});
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(h16[15])});
{hier_rounds}
 }}
}}

template<int G,int V,int CACHE,bool DIAG=false>
__global__ void epoch24(uint64_t middle_base,uint64_t middle_count,const uint32_t*table8,uint32_t target,uint64_t*winner,uint32_t*diag){{
 static_assert(256%(G*V)==0,"mapping");constexpr int STRIDE=64;extern __shared__ uint32_t sm[];
 const int gl=threadIdx.x/G,lane=threadIdx.x&(G-1),groups=blockDim.x/G;const uint64_t gi=uint64_t(blockIdx.x)*groups+gl,mv=middle_base+gi;
 if(gi>=middle_count||mv>0xffffull)return;const uint32_t outer=0,middle=(uint32_t)mv;uint32_t*sched=sm+gl*STRIDE;
 if(lane==0){{
{epoch_schedule}
 }}
 __syncwarp(group_mask<G>());
 const S common{{{','.join(c32(x) for x in pre_e)}}};const uint32_t zbase=rol32(common.a,5)+fch(common.b,common.c,common.d)+common.e+K0+{c32(e16[13])}+(middle<<16);constexpr unsigned CHUNK=G*V;
 #pragma unroll 1
 for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{const unsigned j=chunk+unsigned(lane)*V;S st[V];auto raw=rawp<V,8>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
   #pragma unroll
   for(int i=0;i<V;i++)st[i]=S{{zbase+raw.x[i],common.a,rol32(common.b,30),common.c,common.d}};
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(e16[14])});
   #pragma unroll
   for(int i=0;i<V;i++)sch(st[i],{c32(e16[15])});
{epoch_rounds}
 }}
}}

static uint32_t hrol(uint32_t x,unsigned n){{return (x<<n)|(x>>(32-n));}}
static std::vector<uint32_t> make_table(int shift){{
 std::vector<int> rows;for(int t=16;t<80;t++)if(pc(pmask(t))>1)rows.push_back(t);if(rows.size()!=32){{fprintf(stderr,"polynomial row invariant FAIL\\n");exit(2);}}
 std::vector<uint32_t> out;out.reserve(32*256);for(int t:rows)for(unsigned j=0;j<256;j++){{uint32_t w[80]{{}};w[13]=j<<shift;for(int q=16;q<80;q++)w[q]=hrol(w[q-3]^w[q-8]^w[q-14]^w[q-16],1);out.push_back(w[t]);}}return out;
}}
static double median(std::vector<float>&x){{std::sort(x.begin(),x.end());return x[x.size()/2];}}

template<typename K>static void resources(K kernel,int block,size_t sm,int&regs,size_t&local,int&blocks){{cudaFuncAttributes a{{}};CUDA_OK(cudaFuncGetAttributes(&a,kernel));CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,block,sm));regs=a.numRegs;local=a.localSizeBytes;}}
template<int G,int V,int CACHE>static void bench_base(const char*name,int block,uint64_t n,int samples,const uint32_t*t,uint64_t*w){{
 const int groups=block/G;const size_t sm=size_t(groups)*64*4;dim3 grid((unsigned)((n+groups-1)/groups));baseline<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaDeviceSynchronize());std::vector<float> times;
 for(int s=0;s<samples;s++){{cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));baseline<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
 int regs,blocks;size_t local;resources(baseline<G,V,CACHE>,block,sm,regs,local,blocks);printf("%-22s %9.4f GH/s  median=%8.3f ms B=%3d regs=%3d local=%zu smem=%zu blocks/SM=%d\\n",name,double(n)*256.0/(median(times)*1e6),median(times),block,regs,local,sm,blocks);
}}
template<int G,int V,int CACHE>static void bench_hier(const char*name,int block,uint64_t n,int samples,const uint32_t*t,uint64_t*w){{
 const int groups=block/G;const size_t sm=size_t(groups)*64*4;dim3 grid((unsigned)((n+groups-1)/groups));hierarchical5<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaDeviceSynchronize());std::vector<float> times;
 for(int s=0;s<samples;s++){{cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));hierarchical5<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
 int regs,blocks;size_t local;resources(hierarchical5<G,V,CACHE>,block,sm,regs,local,blocks);printf("%-22s %9.4f GH/s  median=%8.3f ms B=%3d regs=%3d local=%zu smem=%zu blocks/SM=%d\\n",name,double(n)*65536.0/(median(times)*1e6),median(times),block,regs,local,sm,blocks);
}}
template<int G,int V,int CACHE>static void bench_flat(const char*name,int block,uint64_t n,int samples,const uint32_t*t,uint64_t*w){{
 const int groups=block/G;const size_t sm=size_t(groups)*64*4;dim3 grid((unsigned)((n+groups-1)/groups));flat5<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaDeviceSynchronize());std::vector<float> times;
 for(int s=0;s<samples;s++){{cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));flat5<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
 int regs,blocks;size_t local;resources(flat5<G,V,CACHE>,block,sm,regs,local,blocks);printf("%-22s %9.4f GH/s  median=%8.3f ms B=%3d regs=%3d local=%zu smem=%zu blocks/SM=%d\\n",name,double(n)*256.0/(median(times)*1e6),median(times),block,regs,local,sm,blocks);
}}
template<int G,int V,int CACHE>static void bench_epoch(const char*name,int block,int batches,int samples,const uint32_t*t,uint64_t*w){{
 constexpr uint64_t n=65536;const int groups=block/G;const size_t sm=size_t(groups)*64*4;dim3 grid((unsigned)((n+groups-1)/groups));epoch24<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaDeviceSynchronize());std::vector<float> times;
 for(int s=0;s<samples;s++){{cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));for(int q=0;q<batches;q++)epoch24<G,V,CACHE><<<grid,block,sm>>>(0,n,t,0x31415926u,w,nullptr);CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms/batches);CUDA_OK(cudaEventDestroy(a));CUDA_OK(cudaEventDestroy(b));}}
 int regs,blocks;size_t local;resources(epoch24<G,V,CACHE>,block,sm,regs,local,blocks);printf("%-22s %9.4f GH/s  median=%8.3f ms B=%3d regs=%3d local=%zu smem=%zu blocks/SM=%d\\n",name,double(n)*256.0/(median(times)*1e6),median(times),block,regs,local,sm,blocks);
}}

template<typename K>static bool diag_check(K kernel,int block,size_t sm,uint64_t base,const uint32_t*table,uint32_t target,const uint32_t want[5],uint64_t*w,uint32_t*d){{CUDA_OK(cudaMemset(d,0,20));kernel<<<1,block,sm>>>(base,1,table,target,w,d);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint32_t got[5];CUDA_OK(cudaMemcpy(got,d,20,cudaMemcpyDeviceToHost));if(memcmp(got,want,20)){{for(int i=0;i<5;i++)fprintf(stderr," H%d got=%08x want=%08x\\n",i,got[i],want[i]);return false;}}return true;}}

int main(int argc,char**argv){{
 const uint64_t base_outer=argc>1?strtoull(argv[1],nullptr,0):(1ull<<21);const int samples=argc>2?atoi(argv[2]):7;if(base_outer<65536||base_outer%65536){{fprintf(stderr,"outer count must be a multiple of 65536\\n");return 2;}}
 auto h24=make_table(24),h16=make_table(16),h8=make_table(8);uint32_t *d24,*d16,*d8,*diag;uint64_t*winner;const size_t bytes=h24.size()*4;CUDA_OK(cudaMalloc(&d24,bytes));CUDA_OK(cudaMalloc(&d16,bytes));CUDA_OK(cudaMalloc(&d8,bytes));CUDA_OK(cudaMemcpy(d24,h24.data(),bytes,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(d16,h16.data(),bytes,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(d8,h8.data(),bytes,cudaMemcpyHostToDevice));CUDA_OK(cudaMalloc(&diag,20));CUDA_OK(cudaMalloc(&winner,8));uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(winner,&none,8,cudaMemcpyHostToDevice));
 const uint32_t bad=0x6a09e667u;
 bool ok=true;ok&=diag_check(baseline<4,4,0,true>,4,64*4,B_TEST_OUTER,d24,bad,B_TEST_DIGEST,winner,diag);ok&=diag_check(hierarchical5<4,4,0,true>,4,64*4,H_TEST_OUTER,d16,bad,H_TEST_DIGEST,winner,diag);ok&=diag_check(flat5<4,4,0,true>,4,64*4,(uint64_t(H_TEST_OUTER)<<8)|H_TEST_MIDDLE,d16,bad,H_TEST_DIGEST,winner,diag);ok&=diag_check(epoch24<4,4,0,true>,4,64*4,E_TEST_MIDDLE,d8,bad,E_TEST_DIGEST,winner,diag);if(!ok){{fprintf(stderr,"GPU digest correctness FAIL\\n");return 3;}}printf("GPU digest correctness PASS (all 5 words, all four kernels)\\n");
 cudaDeviceProp p{{}};CUDA_OK(cudaGetDeviceProperties(&p,0));printf("GPU: %s cc=%d.%d SMs=%d; samples=%d; reference candidates=%llu\\n",p.name,p.major,p.minor,p.multiProcessorCount,samples,(unsigned long long)(base_outer*256));
 bench_base<4,4,0>("raw4+1-g4-v4-B96",96,base_outer,samples,d24,winner);bench_base<4,4,0>("raw4+1-g4-v4-B128",128,base_outer,samples,d24,winner);bench_base<4,8,0>("raw4+1-g4-v8-B96",96,base_outer,samples,d24,winner);bench_base<8,4,0>("raw4+1-g8-v4-B128",128,base_outer,samples,d24,winner);
 const uint64_t hn=base_outer/256;bench_hier<4,4,0>("hier3+1+1-g4-v4-B96",96,hn,samples,d16,winner);bench_hier<4,4,0>("hier3+1+1-g4-v4-B128",128,hn,samples,d16,winner);bench_hier<4,8,0>("hier3+1+1-g4-v8-B96",96,hn,samples,d16,winner);bench_hier<8,4,0>("hier3+1+1-g8-v4-B128",128,hn,samples,d16,winner);
 bench_flat<4,4,0>("flat3+1+1-g4-v4-B128",128,base_outer,samples,d16,winner);bench_flat<4,8,0>("flat3+1+1-g4-v8-B96",96,base_outer,samples,d16,winner);bench_flat<8,4,0>("flat3+1+1-g8-v4-B128",128,base_outer,samples,d16,winner);
 const int batches=16;bench_epoch<4,4,0>("epochW13-g4-v4-B96",96,batches,samples,d8,winner);bench_epoch<4,4,0>("epochW13-g4-v4-B128",128,batches,samples,d8,winner);bench_epoch<4,8,0>("epochW13-g4-v8-B96",96,batches,samples,d8,winner);bench_epoch<8,4,0>("epochW13-g8-v4-B128",128,batches,samples,d8,winner);
 CUDA_OK(cudaFree(d24));CUDA_OK(cudaFree(d16));CUDA_OK(cudaFree(d8));CUDA_OK(cudaFree(diag));CUDA_OK(cudaFree(winner));return 0;
}}
'''
    out = HERE / "hierarchical_bench.cu"
    out.write_text(src)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
