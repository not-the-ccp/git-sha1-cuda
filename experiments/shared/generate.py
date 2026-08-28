#!/usr/bin/env python3
"""Generate one isolated CUDA W12/W13 shared-schedule benchmark variant.

The generated translation unit contains its fixture, lookup table builder,
independent CPU SHA-1 compression, exact GPU digest capture, and a median
benchmark harness.  It intentionally instantiates one production kernel so
that compiler/resource effects remain attributable to the named variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MASK = 0xFFFFFFFF
IV = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]


def rol(x: int, n: int) -> int:
    n &= 31
    return ((x << n) & MASK) | (x >> (32 - n)) if n else x


def expand(words: list[int]) -> list[int]:
    out = list(words) + [0] * 64
    for t in range(16, 80):
        out[t] = rol(out[t - 3] ^ out[t - 8] ^ out[t - 14] ^ out[t - 16], 1)
    return out


def compress(state: list[int], words: list[int]) -> list[int]:
    schedule = expand(words)
    a, b, c, d, e = state
    for t in range(80):
        if t < 20:
            f, k = (b & c) | ((~b) & d), 0x5A827999
        elif t < 40:
            f, k = b ^ c ^ d, 0x6ED9EBA1
        elif t < 60:
            f, k = (b & c) | (b & d) | (c & d), 0x8F1BBCDC
        else:
            f, k = b ^ c ^ d, 0xCA62C1D6
        z = (rol(a, 5) + f + e + k + schedule[t]) & MASK
        e, d, c, b, a = d, c, rol(b, 30), a, z
    return [(state[0] + a) & MASK, (state[1] + b) & MASK,
            (state[2] + c) & MASK, (state[3] + d) & MASK,
            (state[4] + e) & MASK]


def partial_rounds(state: list[int], schedule: list[int], end: int) -> list[int]:
    a, b, c, d, e = state
    for t in range(end):
        if t < 20:
            f, k = (b & c) | ((~b) & d), 0x5A827999
        elif t < 40:
            f, k = b ^ c ^ d, 0x6ED9EBA1
        elif t < 60:
            f, k = (b & c) | (b & d) | (c & d), 0x8F1BBCDC
        else:
            f, k = b ^ c ^ d, 0xCA62C1D6
        z = (rol(a, 5) + f + e + k + schedule[t]) & MASK
        e, d, c, b, a = d, c, rol(b, 30), a, z
    return [a, b, c, d, e]


def pad(data: bytes) -> bytes:
    result = bytearray(data)
    result.append(0x80)
    while len(result) % 64 != 56:
        result.append(0)
    result += (len(data) * 8).to_bytes(8, "big")
    return bytes(result)


def words(block: bytes) -> list[int]:
    return [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]


def make_job(payload: bytes) -> dict[str, object]:
    if not payload.endswith(b"\n"):
        payload += b"\n"
    label = b"X: "
    selected = None
    for filler in range(256):
        nonce_payload_offset = len(payload) + len(label) + filler
        candidate_payload = payload + label + b" " * filler + b"\0" * 5 + b"\n"
        obj = b"commit " + str(len(candidate_payload)).encode() + b"\0" + candidate_payload
        header_bytes = len(obj) - len(candidate_payload)
        nonce_object_offset = header_bytes + nonce_payload_offset
        padded = pad(obj)
        if (nonce_object_offset % 64 == 48 and
                nonce_object_offset // 64 == len(padded) // 64 - 1 and
                len(obj) % 64 == 54):
            selected = (candidate_payload, obj, padded, nonce_object_offset, filler)
            break
    if selected is None:
        raise RuntimeError("could not place five-byte nonce at final block offset 48")

    candidate_payload, obj, padded, nonce_object_offset, filler = selected
    blocks = [padded[i:i + 64] for i in range(0, len(padded), 64)]
    mutable_index = nonce_object_offset // 64
    hin = list(IV)
    for block in blocks[:mutable_index]:
        hin = compress(hin, words(block))
    final = bytearray(blocks[mutable_index])
    final[48:53] = b"\0" * 5
    base16 = words(final)
    base80 = expand(base16)
    pre = partial_rounds(hin, base80, 12)
    return {
        "payload": candidate_payload,
        "object": obj,
        "filler": filler,
        "nonce_object_offset": nonce_object_offset,
        "mutable_block": mutable_index,
        "hin": hin,
        "pre": pre,
        "base16": base16,
        "base80": base80,
        "placeholder_sha1": hashlib.sha1(obj).hexdigest(),
    }


def c32(value: int) -> str:
    return f"0x{value & MASK:08x}u"


def operand(index: int, base80: list[int]) -> str:
    if index == 12:
        return "outer"
    if index == 13:
        return "W13_BASE"
    if index == 14:
        return "W14"
    if index == 15:
        return "W15"
    if index < 12:
        return c32(base80[index])
    return f"sched[{index - 16}]"


def shared_schedule(base80: list[int]) -> str:
    lines: list[str] = []
    for t in range(16, 32):
        ops = [operand(t - k, base80) for k in (3, 8, 14, 16)]
        lines.append(f"      sched[{t - 16}]=rol32(({')^('.join(ops)}),1);")
    # For t>=32 this equivalent SHA-1 recurrence has fewer/shorter dependencies.
    for t in range(32, 80):
        ops = [operand(t - k, base80) for k in (6, 16, 28, 32)]
        lines.append(f"      sched[{t - 16}]=rol32(({')^('.join(ops)}),2);")
    return "\n".join(lines)


def phase(t: int) -> str:
    return "ch" if t < 20 else "pa1" if t < 40 else "mj" if t < 60 else "pa3"


def canonical_rounds() -> str:
    lines: list[str] = []
    for t in range(16, 79):
        lines.extend([
            "    {",
            f"      auto dv=delta<{t},V,CACHE>(table,raw,j,r7,r3);",
            f"      const uint32_t bw=sched[{t - 16}];",
            "      #pragma unroll",
            f"      for(int i=0;i<V;i++) round_{phase(t)}(st[i],bw^dv.x[i]);",
            "    }",
        ])
    lines.extend([
        "    {",
        "      auto dv=delta<79,V,CACHE>(table,raw,j,r7,r3);",
        "      const uint32_t bw=sched[63];",
        "      #pragma unroll",
        "      for(int i=0;i<V;i++) {",
        "        const uint32_t a80=final_a(st[i],bw^dv.x[i]);",
        "        const unsigned inner=j+(unsigned)i;",
        "        if constexpr(DIAG) if(inner==diag_inner) {",
        "          diag[0]=a80+HIN0; diag[1]=st[i].a+HIN1;",
        "          diag[2]=rol32(st[i].b,30)+HIN2;",
        "          diag[3]=st[i].c+HIN3; diag[4]=st[i].d+HIN4;",
        "        }",
        "        if(a80==TARGET_ADJ && inner && !zero_byte(outer))",
        "          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,",
        "                    (unsigned long long)((uint64_t(outer)<<8)|inner));",
        "      }",
        "    }",
    ])
    return "\n".join(lines)


def rotating_args(t: int, prefix: str = "st[i].") -> tuple[str, str, str, str, str]:
    names = ("a", "b", "c", "d", "e")
    r = t % 5
    args = names[-r:] + names[:-r] if r else names
    return tuple(prefix + name for name in args)  # type: ignore[return-value]


def rotating_rounds() -> str:
    lines: list[str] = []
    for t in range(16, 79):
        a, b, c, d, e = rotating_args(t)
        lines.extend([
            "    {",
            f"      auto dv=delta<{t},V,CACHE>(table,raw,j,r7,r3);",
            f"      const uint32_t bw=sched[{t - 16}];",
            "      #pragma unroll",
            f"      for(int i=0;i<V;i++) step_{phase(t)}({a},{b},{c},{d},{e},bw^dv.x[i]);",
            "    }",
        ])
    # Round 79 would use (b,c,d,e,a).  Computing only its new A avoids the
    # final rotate and preserves the pre-adjusted H0 hot comparison.
    lines.extend([
        "    {",
        "      auto dv=delta<79,V,CACHE>(table,raw,j,r7,r3);",
        "      const uint32_t bw=sched[63];",
        "      #pragma unroll",
        "      for(int i=0;i<V;i++) {",
        "        const uint32_t a80=final_a_rot(st[i],bw^dv.x[i]);",
        "        const unsigned inner=j+(unsigned)i;",
        "        if constexpr(DIAG) if(inner==diag_inner) {",
        "          diag[0]=a80+HIN0; diag[1]=st[i].b+HIN1;",
        "          diag[2]=rol32(st[i].c,30)+HIN2;",
        "          diag[3]=st[i].d+HIN3; diag[4]=st[i].e+HIN4;",
        "        }",
        "        if(a80==TARGET_ADJ && inner && !zero_byte(outer))",
        "          atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,",
        "                    (unsigned long long)((uint64_t(outer)<<8)|inner));",
        "      }",
        "    }",
    ])
    return "\n".join(lines)


def generate(args: argparse.Namespace, job: dict[str, object]) -> str:
    base16 = job["base16"]
    base80 = job["base80"]
    hin = job["hin"]
    pre = job["pre"]
    assert isinstance(base16, list) and isinstance(base80, list)
    assert isinstance(hin, list) and isinstance(pre, list)
    cache_id = {"ca": 0, "cg": 1, "nc": 2}[args.cache]
    form_id = 0 if args.round_form == "struct" else 1
    variant = (f"{args.round_form}-g{args.g}-v{args.v}-b{args.block}-"
               f"{args.cache}-s{args.stride}")
    schedule = shared_schedule(base80)
    round_body = canonical_rounds() if form_id == 0 else rotating_rounds()
    constants = ",".join(c32(x) for x in base16)
    # The rotating form resumes at round 12.  Physical variable names retain
    # their t=0 identities, so at that point (d,e,a,b,c) must present the
    # canonical (A12,B12,C12,D12,E12) state to the round macro.
    physical_pre = pre if form_id == 0 else [pre[2], pre[3], pre[4], pre[0], pre[1]]
    pre_values = ",".join(c32(x) for x in physical_pre)
    reference_object = bytearray(job["object"])
    reference_offset = int(job["nonce_object_offset"])
    reference_object[reference_offset:reference_offset + 5] = bytes.fromhex("0102030405")
    reference_digest = hashlib.sha1(reference_object).digest()
    reference_words = [int.from_bytes(reference_digest[i:i + 4], "big") for i in range(0, 20, 4)]

    return f'''// Generated by experiments/shared/generate.py; variant {variant}
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static constexpr int G={args.g},V={args.v},BLOCK={args.block},CACHE={cache_id};
static constexpr int STRIDE={args.stride},ROUND_FORM={form_id};
static constexpr const char *VARIANT="{variant}";
static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint32_t W13_BASE={c32(base16[13])},W14={c32(base16[14])},W15={c32(base16[15])};
static constexpr uint32_t HIN0={c32(hin[0])},HIN1={c32(hin[1])},HIN2={c32(hin[2])},HIN3={c32(hin[3])},HIN4={c32(hin[4])};
static constexpr uint32_t TARGET_H0=0x13579bdfu,TARGET_ADJ=TARGET_H0-HIN0;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
static constexpr uint32_t BASE16[16]={{{constants}}};
static constexpr uint32_t HASHLIB_REFERENCE[5]={{{','.join(c32(x) for x in reference_words)}}};
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
__device__ __forceinline__ void round_ch(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fch(s.b,s.c,s.d)+s.e+K0+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_pa1(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K1+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_mj(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fmj(s.b,s.c,s.d)+s.e+K2+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_pa3(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ uint32_t final_a(const S&s,uint32_t w){{return rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;}}

// Hashcat-style round: write the new A into the argument named e, rotate b,
// and let generated argument-name permutations carry state between rounds.
__device__ __forceinline__ void step_ch(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fch(b,c,d)+K0+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_pa1(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fpa(b,c,d)+K1+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_mj(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fmj(b,c,d)+K2+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_pa3(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fpa(b,c,d)+K3+w;b=rol32(b,30);}}
__device__ __forceinline__ uint32_t final_a_rot(const S&s,uint32_t w){{return s.a+rol32(s.b,5)+fpa(s.c,s.d,s.e)+K3+w;}}

__host__ __device__ constexpr uint32_t cr(uint32_t x,int n){{n&=31;return n?((x<<n)|(x>>(32-n))):x;}}
__host__ __device__ constexpr uint32_t pmask(int t){{uint32_t p[80]{{}};p[13]=1;for(int i=16;i<=t;i++)p[i]=cr(p[i-3]^p[i-8]^p[i-14]^p[i-16],1);return p[t];}}
__host__ __device__ constexpr int pc(uint32_t x){{int n=0;while(x){{n+=x&1u;x>>=1;}}return n;}}
__host__ __device__ constexpr int ctz1(uint32_t x){{int n=0;while(!(x&1u)){{n++;x>>=1;}}return n;}}
__host__ __device__ constexpr int cidx(int t){{int n=0;for(int q=16;q<t;q++)if(pc(pmask(q))>1)n++;return n;}}
template<int N>struct P{{uint32_t x[N];}};
template<int N>__device__ __forceinline__ P<N> rawp(unsigned j){{P<N>z{{}};#pragma unroll
  for(int i=0;i<N;i++)z.x[i]=uint32_t(j+i)<<24;return z;}}
template<int N>__device__ __forceinline__ P<N> rotp(const P<N>&p,int r){{P<N>z{{}};#pragma unroll
  for(int i=0;i<N;i++)z.x[i]=rol32(p.x[i],r);return z;}}
template<int N,int C>__device__ __forceinline__ P<N> loadp(const uint32_t*t,int row,unsigned j){{
  P<N>z{{}};const uint32_t*p=t+row*256+j;
  #pragma unroll
  for(int q=0;q<N;q+=4){{uint4 a;
    if constexpr(C==0)asm volatile("ld.global.ca.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));
    else if constexpr(C==1)asm volatile("ld.global.cg.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));
    else asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p+q));
    z.x[q]=a.x;z.x[q+1]=a.y;z.x[q+2]=a.z;z.x[q+3]=a.w;
  }}return z;
}}
template<int T,int N,int C>__device__ __forceinline__ P<N> delta(const uint32_t*t,const P<N>&raw,unsigned j,const P<N>&r7,const P<N>&r3){{
  constexpr uint32_t m=pmask(T);constexpr int n=pc(m);
  if constexpr(n==0)return P<N>{{}};
  else if constexpr(n==1){{constexpr int e=ctz1(m);if constexpr(e==7)return r7;else if constexpr(e==3)return r3;else return rotp<N>(raw,e);}}
  else return loadp<N,C>(t,cidx(T),j);
}}
__device__ __forceinline__ bool zero_byte(uint32_t x){{return ((x-0x01010101u)&~x&0x80808080u)!=0;}}

template<bool DIAG>
__global__ void search_kernel(uint64_t outer_base,uint64_t outer_count,const uint32_t*table,
                              uint64_t*winner,unsigned diag_inner,uint32_t*diag){{
  static_assert((G&(G-1))==0 && 256%(G*V)==0,"mapping");
  extern __shared__ uint32_t shared[];
  const int lane=threadIdx.x&(G-1),groups=blockDim.x/G,local_group=threadIdx.x/G;
  const uint64_t gi=uint64_t(blockIdx.x)*groups+local_group,ov=outer_base+gi;
  const bool active=gi<outer_count&&ov<=0xffffffffull;
  const uint32_t outer=(uint32_t)ov;
  uint32_t*sched=shared+local_group*STRIDE;
  if(active&&lane==0){{
{schedule}
  }}
  __syncwarp();
  if(!active)return;
  S common{{{pre_values}}};
  if constexpr(ROUND_FORM==0) round_ch(common,outer);
  else step_ch(common.d,common.e,common.a,common.b,common.c,outer); // t=12: (d,e,a,b,c)
  constexpr unsigned CHUNK=G*V;
  #pragma unroll 1
  for(unsigned chunk=0;chunk<256;chunk+=CHUNK){{
    const unsigned j=chunk+unsigned(lane)*V;
    S st[V];
    #pragma unroll
    for(int i=0;i<V;i++)st[i]=common;
    auto raw=rawp<V>(j),r7=rotp<V>(raw,7),r3=rotp<V>(raw,3);
    if constexpr(ROUND_FORM==0){{
      #pragma unroll
      for(int i=0;i<V;i++)round_ch(st[i],W13_BASE^raw.x[i]);
      #pragma unroll
      for(int i=0;i<V;i++)round_ch(st[i],W14);
      #pragma unroll
      for(int i=0;i<V;i++)round_ch(st[i],W15);
    }}else{{
      #pragma unroll
      for(int i=0;i<V;i++)step_ch(st[i].c,st[i].d,st[i].e,st[i].a,st[i].b,W13_BASE^raw.x[i]); // t=13
      #pragma unroll
      for(int i=0;i<V;i++)step_ch(st[i].b,st[i].c,st[i].d,st[i].e,st[i].a,W14); // t=14
      #pragma unroll
      for(int i=0;i<V;i++)step_ch(st[i].a,st[i].b,st[i].c,st[i].d,st[i].e,W15); // t=15
    }}
{round_body}
  }}
}}

static uint32_t host_rol(uint32_t x,unsigned n){{return (x<<n)|(x>>(32-n));}}
static void host_compress(const uint32_t in[5],const uint32_t block[16],uint32_t out[5]){{
  uint32_t w[80];memcpy(w,block,64);
  for(int t=16;t<80;t++)w[t]=host_rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1);
  uint32_t a=in[0],b=in[1],c=in[2],d=in[3],e=in[4];
  for(int t=0;t<80;t++){{uint32_t f,k;
    if(t<20){{f=(b&c)|(~b&d);k=K0;}}else if(t<40){{f=b^c^d;k=K1;}}
    else if(t<60){{f=(b&c)|(b&d)|(c&d);k=K2;}}else{{f=b^c^d;k=K3;}}
    uint32_t z=host_rol(a,5)+f+e+k+w[t];e=d;d=c;c=host_rol(b,30);b=a;a=z;
  }}out[0]=in[0]+a;out[1]=in[1]+b;out[2]=in[2]+c;out[3]=in[3]+d;out[4]=in[4]+e;
}}
static void print_digest(const uint32_t h[5]){{for(int i=0;i<5;i++)printf("%08x",h[i]);}}
static std::vector<uint32_t> make_table(){{
  std::vector<uint32_t> table;table.reserve(32u*256u);
  for(int t=16;t<80;t++)if(pc(pmask(t))>1)for(unsigned j=0;j<256;j++){{
    uint32_t w[80]{{}};w[13]=j<<24;
    for(int q=16;q<=t;q++)w[q]=host_rol(w[q-3]^w[q-8]^w[q-14]^w[q-16],1);
    table.push_back(w[t]);
  }}return table;
}}
static bool correctness(const uint32_t*table,uint64_t*winner,uint32_t*diag){{
  static constexpr uint32_t outers[]={{0x01020304u,0x89abcdefu,0xfedcba98u}};
  static constexpr unsigned inners[]={{0u,5u,127u,255u}};
  const uint32_t state[5]={{HIN0,HIN1,HIN2,HIN3,HIN4}};
  int checked=0;
  for(uint32_t outer:outers)for(unsigned inner:inners){{
    uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(winner,&none,8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(diag,0,20));
    search_kernel<true><<<1,BLOCK,(BLOCK/G)*STRIDE*4u>>>(outer,1,table,winner,inner,diag);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
    uint32_t got[5],block[16],want[5];memcpy(block,BASE16,64);block[12]=outer;
    block[13]=(block[13]&0x00ffffffu)|(inner<<24);host_compress(state,block,want);
    CUDA_OK(cudaMemcpy(got,diag,20,cudaMemcpyDeviceToHost));
    if(checked<4){{printf("DIGEST outer=%08x inner=%02x gpu=",outer,inner);print_digest(got);printf(" cpu=");print_digest(want);printf("\\n");}}
    if(memcmp(got,want,20)){{fprintf(stderr,"digest mismatch outer=%08x inner=%02x\\n",outer,inner);return false;}}
    if(outer==0x01020304u&&inner==5u&&memcmp(got,HASHLIB_REFERENCE,20)){{fprintf(stderr,"whole-object hashlib reference mismatch\\n");return false;}}
    checked++;
  }}printf("CORRECT variant=%s exact_digests=%d\\n",VARIANT,checked);return true;
}}
static double median(std::vector<double> x){{std::sort(x.begin(),x.end());return x[x.size()/2];}}
int main(int argc,char**argv){{
  const uint64_t n=argc>1?strtoull(argv[1],nullptr,0):(1ull<<20);
  const int launches=argc>2?atoi(argv[2]):4,samples=argc>3?atoi(argv[3]):9;
  if(n==0||launches<1||samples<3||!(samples&1)){{fprintf(stderr,"usage: %s [outer_count] [launches] [odd_samples>=3]\\n",argv[0]);return 2;}}
  cudaDeviceProp prop{{}};CUDA_OK(cudaGetDeviceProperties(&prop,0));
  if(BLOCK%G){{fprintf(stderr,"BLOCK must be divisible by G\\n");return 2;}}
  const size_t dynamic_smem=(BLOCK/G)*STRIDE*4u;
  if(dynamic_smem>prop.sharedMemPerBlock){{fprintf(stderr,"dynamic shared memory %zu exceeds default limit %zu\\n",dynamic_smem,prop.sharedMemPerBlock);return 2;}}
  auto host_table=make_table();if(host_table.size()!=32u*256u){{fprintf(stderr,"table rows=%zu, expected 8192\\n",host_table.size());return 2;}}
  uint32_t*table=nullptr,*diag=nullptr;uint64_t*winner=nullptr;
  CUDA_OK(cudaMalloc(&table,host_table.size()*4));CUDA_OK(cudaMemcpy(table,host_table.data(),host_table.size()*4,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMalloc(&diag,20));CUDA_OK(cudaMalloc(&winner,8));
  if(!correctness(table,winner,diag))return 3;
  const int groups=BLOCK/G;const dim3 grid((unsigned)((n+groups-1)/groups));
  uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(winner,&none,8,cudaMemcpyHostToDevice));
  for(int r=0;r<8;r++)search_kernel<false><<<grid,BLOCK,dynamic_smem>>>(0,n,table,winner,0,nullptr);
  CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t begin,end;CUDA_OK(cudaEventCreate(&begin));CUDA_OK(cudaEventCreate(&end));
  std::vector<double> rates;rates.reserve(samples);
  for(int sample=0;sample<samples;sample++){{
    CUDA_OK(cudaEventRecord(begin));
    for(int r=0;r<launches;r++)search_kernel<false><<<grid,BLOCK,dynamic_smem>>>(0,n,table,winner,0,nullptr);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaEventRecord(end));CUDA_OK(cudaEventSynchronize(end));
    float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,begin,end));
    double rate=double(n)*256.0*launches/(double(ms)*1e6);rates.push_back(rate);
    printf("SAMPLE variant=%s index=%d ms=%.3f ghs=%.6f\\n",VARIANT,sample,ms,rate);
  }}
  cudaFuncAttributes attr{{}};CUDA_OK(cudaFuncGetAttributes(&attr,search_kernel<false>));
  auto sorted=rates;std::sort(sorted.begin(),sorted.end());const double med=median(rates);
  printf("RESULT variant=%s median_ghs=%.6f min_ghs=%.6f max_ghs=%.6f regs=%d local=%zu static_smem=%zu dynamic_smem=%zu block=%d grid=%u gpu=\\\"%s\\\" cc=%d.%d\\n",
         VARIANT,med,sorted.front(),sorted.back(),attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,dynamic_smem,BLOCK,grid.x,prop.name,prop.major,prop.minor);
  CUDA_OK(cudaEventDestroy(begin));CUDA_OK(cudaEventDestroy(end));CUDA_OK(cudaFree(table));CUDA_OK(cudaFree(diag));CUDA_OK(cudaFree(winner));
  return 0;
}}
'''.replace(";#pragma unroll\n", ";\n  #pragma unroll\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=Path(__file__).with_name("fixture_commit.txt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--g", type=int, default=4, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--v", type=int, default=4, choices=(4, 8))
    parser.add_argument("--block", type=int, default=96, choices=(64, 96, 128, 192, 256))
    parser.add_argument("--cache", choices=("ca", "cg", "nc"), default="ca")
    parser.add_argument("--stride", type=int, choices=(64, 65), default=65)
    parser.add_argument("--round-form", choices=("struct", "rotating"), default="struct")
    args = parser.parse_args()
    if args.block % args.g or 256 % (args.g * args.v):
        parser.error("BLOCK must be divisible by G and G*V must divide 256")
    job = make_job(args.payload.read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate(args, job))
    if args.metadata:
        meta = {k: v for k, v in job.items() if k not in ("payload", "object", "base80")}
        meta.update(vars(args))
        for key, value in list(meta.items()):
            if isinstance(value, Path):
                meta[key] = str(value)
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(meta, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
