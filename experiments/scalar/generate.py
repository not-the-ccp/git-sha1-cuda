#!/usr/bin/env python3
"""Generate one isolated scalar/persistent final-block CUDA benchmark.

Every lane hashes one candidate at a time, or a compile-time ILP bundle of two
or four candidates.  A fixed-size grid walks the requested interval in a
grid-stride loop, making the kernel persistent without changing the exact host
candidate mapping.  Each generated translation unit instantiates only one
production kernel so ptxas resource counts remain attributable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.git_sha1_job import (  # noqa: E402
    MASK32,
    TargetPrefix,
    block_words,
    build_raw_tail_job,
    expand_schedule,
)


def c32(value: int) -> str:
    return f"0x{value & MASK32:08x}u"


def rol(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (32 - amount))) & MASK32


def partial_rounds(state: list[int], schedule: list[int], end: int) -> list[int]:
    a, b, c, d, e = state
    for t in range(end):
        if t < 20:
            f, k = d ^ (b & (c ^ d)), 0x5A827999
        elif t < 40:
            f, k = b ^ c ^ d, 0x6ED9EBA1
        elif t < 60:
            f, k = (b & c) | (d & (b | c)), 0x8F1BBCDC
        else:
            f, k = b ^ c ^ d, 0xCA62C1D6
        z = (rol(a, 5) + f + e + k + schedule[t]) & MASK32
        e, d, c, b, a = d, c, rol(b, 30), a, z
    return [a, b, c, d, e]


def phase(t: int) -> str:
    return "ch" if t < 20 else "pa1" if t < 40 else "mj" if t < 60 else "pa3"


def rotating_args(t: int, prefix: str) -> tuple[str, str, str, str, str]:
    names = ("a", "b", "c", "d", "e")
    amount = t % 5
    ordered = names[-amount:] + names[:-amount] if amount else names
    return tuple(prefix + name for name in ordered)  # type: ignore[return-value]


def polynomial_masks(source_word: int) -> list[int]:
    masks = [0] * 80
    masks[source_word] = 1
    for t in range(16, 80):
        masks[t] = rol(masks[t - 3] ^ masks[t - 8] ^ masks[t - 14] ^ masks[t - 16], 1)
    return masks


def delta_expression(variable: str, mask: int) -> str:
    terms = [variable if bit == 0 else f"rol32({variable},{bit})" for bit in range(32) if mask & (1 << bit)]
    return "^".join(terms) if terms else "0u"


def schedule_word(
    schedule_kind: str,
    lane: int,
    t: int,
    base80: list[int],
    outer_masks: list[int],
    inner_masks: list[int],
) -> tuple[list[str], str]:
    if schedule_kind == "rolling":
        if t < 16:
            return [], f"w{lane}[{t}]"
        index = t & 15
        lines = [
            f"w{lane}[{index}]=rol32(w{lane}[{(t-3)&15}]^w{lane}[{(t-8)&15}]^"
            f"w{lane}[{(t-14)&15}]^w{lane}[{index}],1);"
        ]
        return lines, f"w{lane}[{index}]"

    if schedule_kind == "rolling32":
        if t < 16:
            return [], f"w{lane}[{t}]"
        index = t & 31
        if t < 32:
            expression = (
                f"rol32(w{lane}[{(t-3)&31}]^w{lane}[{(t-8)&31}]^"
                f"w{lane}[{(t-14)&31}]^w{lane}[{(t-16)&31}],1)"
            )
        else:
            expression = (
                f"rol32(w{lane}[{(t-6)&31}]^w{lane}[{(t-16)&31}]^"
                f"w{lane}[{(t-28)&31}]^w{lane}[{(t-32)&31}],2)"
            )
        return [f"w{lane}[{index}]={expression};"], f"w{lane}[{index}]"

    outer_delta = delta_expression(f"outer{lane}", outer_masks[t])
    inner_delta = delta_expression(f"innerw{lane}", inner_masks[t])
    terms = [c32(base80[t])]
    if outer_delta != "0u":
        terms.append(f"({outer_delta})")
    if inner_delta != "0u":
        terms.append(f"({inner_delta})")
    name = f"aw{lane}_{t}"
    return [f"const uint32_t {name}={'^'.join(terms)};"], name


def generated_hash_body(
    ilp: int,
    schedule_kind: str,
    round_form: str,
    pre: list[int],
    base16: list[int],
    base80: list[int],
) -> str:
    outer_masks = polynomial_masks(12)
    inner_masks = polynomial_masks(13)
    lines: list[str] = []

    for lane in range(ilp):
        lines.extend(
            [
                f"const uint32_t outer{lane}=uint32_t(id{lane}>>8);",
                f"const uint32_t innerw{lane}=uint32_t(id{lane}&0xffu)<<24;",
            ]
        )
        if schedule_kind in ("rolling", "rolling32"):
            words = 16 if schedule_kind == "rolling" else 32
            lines.append(f"uint32_t w{lane}[{words}];")
            for t in range(12):
                lines.append(f"w{lane}[{t}]={c32(base16[t])};")
            lines.extend(
                [
                    f"w{lane}[12]=outer{lane};",
                    f"w{lane}[13]={c32(base16[13])}^innerw{lane};",
                    f"w{lane}[14]={c32(base16[14])};",
                    f"w{lane}[15]={c32(base16[15])};",
                ]
            )
        if round_form == "struct":
            lines.append(f"S s{lane}{{{','.join(c32(word) for word in pre)}}};")
        else:
            physical = [pre[2], pre[3], pre[4], pre[0], pre[1]]
            lines.append(f"S s{lane}{{{','.join(c32(word) for word in physical)}}};")

    for t in range(12, 80):
        for lane in range(ilp):
            prefix, word = schedule_word(schedule_kind, lane, t, base80, outer_masks, inner_masks)
            lines.extend(prefix)
            if round_form == "struct":
                lines.append(f"round_{phase(t)}(s{lane},{word});")
            else:
                a, b, c, d, e = rotating_args(t, f"s{lane}.")
                lines.append(f"step_{phase(t)}({a},{b},{c},{d},{e},{word});")

    for lane in range(ilp):
        lines.extend(
            [
                f"if(active{lane}){{",
                f"  if constexpr(DIAG) if(id{lane}==diag_id){{",
                f"    diag[0]=s{lane}.a+HIN0;diag[1]=s{lane}.b+HIN1;diag[2]=s{lane}.c+HIN2;",
                f"    diag[3]=s{lane}.d+HIN3;diag[4]=s{lane}.e+HIN4;",
                "  }",
                f"  if(s{lane}.a==TARGET_ADJ && log_safe(id{lane}))",
                "    atomicCAS((unsigned long long*)winner,(unsigned long long)NO_WINNER,",
                f"              (unsigned long long)id{lane});",
                "}",
            ]
        )
    return "\n    ".join(lines)


def generate(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    payload = args.payload.read_bytes()
    target = TargetPrefix.from_hex(args.target)
    if target.bits != 32:
        raise ValueError("scalar benchmark currently specializes an exact 32-bit H0 prefix")
    job = build_raw_tail_job(payload, target)
    base16 = list(job.base_words)
    base80 = list(expand_schedule(base16))
    hin = list(job.prestate)
    pre = partial_rounds(hin, base80, 12)

    test_candidates = (0, 0x0102_0304_05, 0x89AB_CDEF_7F, 0xFEDC_BA98_FF)
    references = [job.digest(candidate) for candidate in test_candidates]
    reference_words = [
        [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 20, 4)]
        for digest in references
    ]
    for candidate, digest in zip(test_candidates, references):
        if job.digest_from_prestate(candidate) != digest:
            raise AssertionError("host job oracle and fixed-prestate path disagree")

    variant = f"{args.schedule}-{args.round_form}-ilp{args.ilp}-b{args.block}"
    hash_body = generated_hash_body(args.ilp, args.schedule, args.round_form, pre, base16, base80)
    src = f'''// Generated by experiments/scalar/generate.py; variant {variant}
#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static constexpr int ILP={args.ilp},BLOCK={args.block};
static constexpr const char*VARIANT="{variant}";
static constexpr uint32_t K0=0x5a827999u,K1=0x6ed9eba1u,K2=0x8f1bbcdcu,K3=0xca62c1d6u;
static constexpr uint32_t HIN0={c32(hin[0])},HIN1={c32(hin[1])},HIN2={c32(hin[2])},HIN3={c32(hin[3])},HIN4={c32(hin[4])};
static constexpr uint32_t TARGET_H0={c32(target.words[0])},TARGET_ADJ=TARGET_H0-HIN0;
static constexpr uint64_t NO_WINNER=~uint64_t(0);
static constexpr uint64_t TEST_IDS[4]={{{','.join(f'0x{x:010x}ull' for x in test_candidates)}}};
static constexpr uint32_t TEST_DIGESTS[4][5]={{{','.join('{'+','.join(c32(x) for x in row)+'}' for row in reference_words)}}};
#define CUDA_OK(x) do{{cudaError_t e_=(x);if(e_!=cudaSuccess){{fprintf(stderr,"CUDA error %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}}}while(0)

__device__ __forceinline__ uint32_t rol32(uint32_t x,unsigned n){{return __funnelshift_l(x,x,n);}}
__device__ __forceinline__ uint32_t fch(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xca;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fpa(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0x96;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
__device__ __forceinline__ uint32_t fmj(uint32_t x,uint32_t y,uint32_t z){{uint32_t r;asm("lop3.b32 %0,%1,%2,%3,0xe8;":"=r"(r):"r"(x),"r"(y),"r"(z));return r;}}
struct S{{uint32_t a,b,c,d,e;}};
__device__ __forceinline__ void round_ch(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fch(s.b,s.c,s.d)+s.e+K0+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_pa1(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K1+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_mj(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fmj(s.b,s.c,s.d)+s.e+K2+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
__device__ __forceinline__ void round_pa3(S&s,uint32_t w){{uint32_t z=rol32(s.a,5)+fpa(s.b,s.c,s.d)+s.e+K3+w;s.e=s.d;s.d=s.c;s.c=rol32(s.b,30);s.b=s.a;s.a=z;}}
// Hashcat-style role rotation: generated argument permutations move semantic
// state between physical variables, leaving only the new-A add and B rotate.
__device__ __forceinline__ void step_ch(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fch(b,c,d)+K0+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_pa1(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fpa(b,c,d)+K1+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_mj(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fmj(b,c,d)+K2+w;b=rol32(b,30);}}
__device__ __forceinline__ void step_pa3(uint32_t a,uint32_t&b,uint32_t c,uint32_t d,uint32_t&e,uint32_t w){{e+=rol32(a,5)+fpa(b,c,d)+K3+w;b=rol32(b,30);}}
__device__ __forceinline__ bool zero_byte32(uint32_t x){{return ((x-0x01010101u)&~x&0x80808080u)!=0;}}
__device__ __forceinline__ bool log_safe(uint64_t id){{return (id&0xffu)&&!zero_byte32(uint32_t(id>>8));}}

template<bool DIAG>
__global__ __launch_bounds__(BLOCK) void scalar_persistent(uint64_t first,uint64_t count,uint64_t*winner,uint64_t diag_id,uint32_t*diag){{
  const uint64_t thread=uint64_t(blockIdx.x)*BLOCK+threadIdx.x;
  const uint64_t stride=uint64_t(gridDim.x)*BLOCK;
  const uint64_t end=first+count;
  for(uint64_t base=first+thread;base<end;base+=stride*ILP){{
    {' '.join(f'const uint64_t id{i}=base+{i}ull*stride;const bool active{i}=id{i}<end;' for i in range(args.ilp))}
    {hash_body}
  }}
}}

static void print_digest(const uint32_t*h){{for(int i=0;i<5;i++)printf("%08x",h[i]);}}
static bool correctness(uint64_t*winner,uint32_t*diag){{
  for(int test=0;test<4;test++){{
    uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(winner,&none,8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(diag,0,20));
    scalar_persistent<true><<<1,BLOCK>>>(TEST_IDS[test],1,winner,TEST_IDS[test],diag);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());uint32_t got[5];CUDA_OK(cudaMemcpy(got,diag,20,cudaMemcpyDeviceToHost));
    printf("DIGEST candidate=%010llx gpu=",(unsigned long long)TEST_IDS[test]);print_digest(got);printf(" hashlib=");print_digest(TEST_DIGESTS[test]);printf("\\n");
    if(memcmp(got,TEST_DIGESTS[test],20)){{fprintf(stderr,"digest mismatch for candidate %010llx\\n",(unsigned long long)TEST_IDS[test]);return false;}}
  }}printf("CORRECT variant=%s exact_digests=4\\n",VARIANT);return true;
}}
static double median(std::vector<double>x){{std::sort(x.begin(),x.end());return x[x.size()/2];}}
int main(int argc,char**argv){{
  const uint64_t count=argc>1?strtoull(argv[1],nullptr,0):(1ull<<27);
  const int grid_mult=argc>2?atoi(argv[2]):4,launches=argc>3?atoi(argv[3]):1,samples=argc>4?atoi(argv[4]):9;
  if(!count||grid_mult<1||launches<1||samples<3||!(samples&1)){{fprintf(stderr,"usage: %s [candidate_count] [grid_mult] [launches] [odd_samples>=3]\\n",argv[0]);return 2;}}
  cudaDeviceProp prop{{}};CUDA_OK(cudaGetDeviceProperties(&prop,0));const unsigned grid=unsigned(prop.multiProcessorCount*grid_mult);
  uint64_t*winner=nullptr;uint32_t*diag=nullptr;CUDA_OK(cudaMalloc(&winner,8));CUDA_OK(cudaMalloc(&diag,20));
  if(!correctness(winner,diag))return 3;uint64_t none=NO_WINNER;CUDA_OK(cudaMemcpy(winner,&none,8,cudaMemcpyHostToDevice));
  for(int warm=0;warm<5;warm++)scalar_persistent<false><<<grid,BLOCK>>>(0,count,winner,0,nullptr);
  CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());cudaEvent_t begin,end_event;CUDA_OK(cudaEventCreate(&begin));CUDA_OK(cudaEventCreate(&end_event));
  std::vector<double>rates;rates.reserve(samples);
  for(int sample=0;sample<samples;sample++){{CUDA_OK(cudaEventRecord(begin));for(int launch=0;launch<launches;launch++)scalar_persistent<false><<<grid,BLOCK>>>(0,count,winner,0,nullptr);CUDA_OK(cudaGetLastError());CUDA_OK(cudaEventRecord(end_event));CUDA_OK(cudaEventSynchronize(end_event));float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,begin,end_event));double rate=double(count)*launches/(double(ms)*1e6);rates.push_back(rate);printf("SAMPLE variant=%s grid_mult=%d index=%d ms=%.3f ghs=%.6f\\n",VARIANT,grid_mult,sample,ms,rate);}}
  cudaFuncAttributes attr{{}};CUDA_OK(cudaFuncGetAttributes(&attr,scalar_persistent<false>));auto sorted=rates;std::sort(sorted.begin(),sorted.end());
  printf("RESULT variant=%s median_ghs=%.6f min_ghs=%.6f max_ghs=%.6f regs=%d local=%zu static_smem=%zu block=%d grid=%u grid_mult=%d gpu=\\\"%s\\\" cc=%d.%d\\n",VARIANT,median(rates),sorted.front(),sorted.back(),attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,BLOCK,grid,grid_mult,prop.name,prop.major,prop.minor);
  CUDA_OK(cudaEventDestroy(begin));CUDA_OK(cudaEventDestroy(end_event));CUDA_OK(cudaFree(winner));CUDA_OK(cudaFree(diag));return 0;
}}
'''
    metadata: dict[str, object] = {
        "variant": variant,
        "schedule": args.schedule,
        "round_form": args.round_form,
        "ilp": args.ilp,
        "block": args.block,
        "job_manifest": job.manifest(),
        "pre_after_round_11": [f"{word:08x}" for word in pre],
        "test_candidates": [f"{candidate:010x}" for candidate in test_candidates],
        "test_digests": [digest.hex() for digest in references],
    }
    return src, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=Path(__file__).with_name("fixture_commit.txt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--target", default="13579bdf", help="exact eight-hex-digit H0 target")
    parser.add_argument("--schedule", choices=("rolling", "rolling32", "affine"), default="rolling")
    parser.add_argument("--round-form", choices=("struct", "rotating"), default="rotating")
    parser.add_argument("--ilp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--block",
        type=int,
        choices=(64, 96, 128, 192, 256, 320, 384, 512, 768, 1024),
        default=128,
    )
    args = parser.parse_args()
    source, metadata = generate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
