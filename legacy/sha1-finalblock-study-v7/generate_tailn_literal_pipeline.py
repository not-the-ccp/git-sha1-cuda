#!/usr/bin/env python3
"""Generate scalar fixed-suffix CUDA kernels with literal SHA-1 schedules.

This architecture is for long tail-N Git jobs: a small mutable head block is
followed by many job-fixed SHA-1 blocks.  Each candidate is scalar in the
suffix.  Fixed W0..W79 words are emitted as instruction constants, eliminating
runtime suffix-schedule loads.  --chunk-blocks controls the code-size versus
state-handoff tradeoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

MASK = 0xFFFFFFFF


def c32(x: int) -> str:
    return f"0x{x & MASK:08x}u"


def prefix_match_body(prefix_hex: str) -> str:
    p = prefix_hex.lower().removeprefix("0x")
    if not 1 <= len(p) <= 40 or any(ch not in "0123456789abcdef" for ch in p):
        raise SystemExit("--prefix-hex must be 1..40 hexadecimal digits")
    bits = len(p) * 4
    full = (p + "0" * 40)[:40]
    words = [int(full[i:i + 8], 16) for i in range(0, 40, 8)]
    fields = ["s.a", "s.b", "s.c", "s.d", "s.e"]
    out = []
    for wi in range(5):
        used = bits - 32 * wi
        if used <= 0:
            break
        take = min(32, used)
        mask = MASK if take == 32 else ((MASK << (32 - take)) & MASK)
        want = words[wi] & mask
        out.append(f"if((({fields[wi]})&{c32(mask)})!={c32(want)})return false;")
    out.append("return true;")
    return "".join(out)


def test_digest(jobdir: Path, meta: dict, test_id: int) -> list[int]:
    charset = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    x = test_id
    chars = []
    for _ in range(meta["nonce_len"]):
        chars.append(charset[x & 63])
        x >>= 6
    obj = bytearray((jobdir / "object_template.bin").read_bytes())
    off = meta["nonce_absolute_offset"]
    obj[off:off + meta["nonce_len"]] = bytes(chars)
    d = hashlib.sha1(obj).digest()
    return [int.from_bytes(d[i:i + 4], "big") for i in range(0, 20, 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate exact fixed-suffix CUDA pipeline with schedules embedded as literals")
    ap.add_argument("job")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--chunk-blocks", type=int, default=4)
    ap.add_argument("--prefix-hex", default="13579bdf")
    args = ap.parse_args()

    d = Path(args.job)
    meta = json.loads((d / "job.json").read_text())
    raw = (d / "suffix80.bin").read_bytes()
    vals = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    nb = meta["suffix_blocks"]
    C = args.chunk_blocks
    if C < 1 or C > nb:
        raise SystemExit(f"--chunk-blocks must be 1..{nb}")
    if len(vals) != nb * 80:
        raise SystemExit("suffix80 size mismatch")
    chunks = [list(range(i, min(nb, i + C))) for i in range(0, nb, C)]
    match_body = prefix_match_body(args.prefix_hex)

    test_id = 0x12345
    td = test_digest(d, meta, test_id)

    def block_code(b: int) -> str:
        w = vals[b * 80:(b + 1) * 80]
        out = ["  {", "   Sha1State old=s;"]
        for t, x in enumerate(w):
            f = "sha1_r0" if t < 20 else "sha1_r1" if t < 40 else "sha1_r2" if t < 60 else "sha1_r3"
            out.append(f"   {f}(s,{c32(x)});")
        out.append("   s.a+=old.a;s.b+=old.b;s.c+=old.c;s.d+=old.d;s.e+=old.e;")
        out.append("  }")
        return "\n".join(out)

    funcs = []
    for ci, blocks in enumerate(chunks):
        last = ci == len(chunks) - 1
        body = "\n".join(block_code(b) for b in blocks)
        if last:
            tail = """
  if(diag && i==0){diag[0]=s.a;diag[1]=s.b;diag[2]=s.c;diag[3]=s.d;diag[4]=s.e;}
  if(lp_match(s))atomicCAS((unsigned long long*)winner,(unsigned long long)SHA1_NO_WINNER,(unsigned long long)(first_id+i));
"""
        else:
            tail = """
  st[i]=s.a;st[n+i]=s.b;st[2*n+i]=s.c;st[3*n+i]=s.d;st[4*n+i]=s.e;
"""
        funcs.append(f'''__global__ void suffix_{ci:03d}(uint32_t*st,uint64_t n,uint64_t first_id,uint64_t*winner,uint32_t*diag){{
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 Sha1State s{{st[i],st[n+i],st[2*n+i],st[3*n+i],st[4*n+i]}};
{body}{tail}}}
''')

    launches = "\n".join(
        f"  suffix_{ci:03d}<<<grid,B>>>(st,n,first_id,winner,diag);CUDA_OK(cudaGetLastError());"
        for ci in range(len(chunks))
    )

    src = f'''// AUTO-GENERATED literal fixed-suffix CUDA pipeline; chunk={C} blocks; target={args.prefix_hex.lower()}.
#include "tailn_job.cuh"
#include "sha1_cuda_core.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static constexpr uint64_t LP_TEST_ID={test_id}ull;
static constexpr uint32_t LP_TEST_DIGEST[5]={{{','.join(c32(x) for x in td)}}};
#define CUDA_OK(x) do{{cudaError_t e_=(x);if(e_!=cudaSuccess){{fprintf(stderr,"CUDA error %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}}}while(0)
__host__ __device__ static inline unsigned char lp_char(unsigned j){{return j<10?(unsigned char)('0'+j):j<36?(unsigned char)('A'+j-10):j<62?(unsigned char)('a'+j-36):j==62?(unsigned char)'-':(unsigned char)'_';}}
__device__ __forceinline__ bool lp_match(const Sha1State&s){{{match_body}}}
__device__ __forceinline__ Sha1State lp_head(uint64_t id){{
 const uint64_t outer=id>>6;const unsigned inner=id&63u;
 uint32_t w[16]={{TN_W00,TN_W01,TN_W02,TN_W03,TN_W04,TN_W05,TN_W06,TN_W07,TN_W08,TN_W09,TN_W10,TN_W11,TN_W12,TN_W13,TN_W14,TN_W15}};
 #pragma unroll
 for(int k=0;k<TN_NONCE_LEN-1;k++){{int p=TN_NONCE_OFF+k;w[p>>2]|=uint32_t(lp_char((outer>>(6*k))&63u))<<(8*(3-(p&3)));}}
 int p=TN_NONCE_OFF+TN_NONCE_LEN-1;w[p>>2]|=uint32_t(lp_char(inner))<<(8*(3-(p&3)));
 Sha1State s{{TN_PRE0,TN_PRE1,TN_PRE2,TN_PRE3,TN_PRE4}};
 #pragma unroll
 for(int t=TN_FIRST_WORD;t<16;t++)sha1_r0(s,w[t]);
 #pragma unroll
 for(int t=16;t<20;t++){{int q=t&15;w[q]=sha1_rol(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[q],1);sha1_r0(s,w[q]);}}
 #pragma unroll
 for(int t=20;t<40;t++){{int q=t&15;w[q]=sha1_rol(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[q],1);sha1_r1(s,w[q]);}}
 #pragma unroll
 for(int t=40;t<60;t++){{int q=t&15;w[q]=sha1_rol(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[q],1);sha1_r2(s,w[q]);}}
 #pragma unroll
 for(int t=60;t<80;t++){{int q=t&15;w[q]=sha1_rol(w[(t-3)&15]^w[(t-8)&15]^w[(t-14)&15]^w[q],1);sha1_r3(s,w[q]);}}
 s.a+=TN_H0;s.b+=TN_H1;s.c+=TN_H2;s.d+=TN_H3;s.e+=TN_H4;return s;
}}
__global__ void head_kernel(uint64_t first_id,uint64_t n,uint32_t*st){{uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;Sha1State s=lp_head(first_id+i);st[i]=s.a;st[n+i]=s.b;st[2*n+i]=s.c;st[3*n+i]=s.d;st[4*n+i]=s.e;}}
{''.join(funcs)}
#ifndef LITERAL_PIPELINE_NO_MAIN
static void launch(uint64_t first_id,uint64_t n,uint32_t*st,uint64_t*winner,uint32_t*diag){{constexpr int B=256;dim3 grid((unsigned)((n+B-1)/B));head_kernel<<<grid,B>>>(first_id,n,st);CUDA_OK(cudaGetLastError());
{launches}
}}
int main(int argc,char**argv){{
 uint64_t n=argc>1?strtoull(argv[1],nullptr,0):(1ull<<18);int reps=argc>2?atoi(argv[2]):3;
 uint32_t*st=nullptr,*diag=nullptr;uint64_t*w=nullptr;CUDA_OK(cudaMalloc(&st,size_t(n)*5*4));CUDA_OK(cudaMalloc(&diag,20));CUDA_OK(cudaMemset(diag,0,20));CUDA_OK(cudaMalloc(&w,8));uint64_t none=SHA1_NO_WINNER;
 CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));launch(LP_TEST_ID,1,st,w,diag);CUDA_OK(cudaDeviceSynchronize());uint32_t gotd[5]={{}};CUDA_OK(cudaMemcpy(gotd,diag,20,cudaMemcpyDeviceToHost));if(memcmp(gotd,LP_TEST_DIGEST,20)){{fprintf(stderr,"GPU digest correctness FAIL\\n");for(int i=0;i<5;i++)fprintf(stderr," H%d got=%08x want=%08x\\n",i,gotd[i],LP_TEST_DIGEST[i]);return 3;}}printf("GPU digest correctness PASS test=%llx\\n",(unsigned long long)LP_TEST_ID);
 CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));launch(0,n,st,w,nullptr);CUDA_OK(cudaDeviceSynchronize());cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));for(int r=0;r<reps;r++){{CUDA_OK(cudaMemcpy(w,&none,8,cudaMemcpyHostToDevice));launch(0,n,st,w,nullptr);}}CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));double cps=double(n)*reps/(ms*1e-3);printf("literal-pipeline-c{C}: %.3f Mcand/s %.3f Gcompress/s chunks={len(chunks)} state_traffic~{len(chunks)*40} B/candidate source_chunks={len(chunks)}\\n",cps/1e6,cps*({nb}+1)/1e9);
 CUDA_OK(cudaFree(st));CUDA_OK(cudaFree(diag));CUDA_OK(cudaFree(w));
}}
#endif
'''
    out = Path(args.output)
    out.write_text(src)
    metadata = {
        "chunk_blocks": C,
        "suffix_blocks": nb,
        "chunks": len(chunks),
        "target_hex": args.prefix_hex.lower().removeprefix("0x"),
        "approx_state_bytes_per_candidate": len(chunks) * 40,
        "source_bytes": len(src),
        "max_literal_rounds_per_kernel": max(len(x) for x in chunks) * 80,
        "test_id": test_id,
        "test_digest": [f"{x:08x}" for x in td],
    }
    Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
