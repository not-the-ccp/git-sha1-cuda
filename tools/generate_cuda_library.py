#!/usr/bin/env python3
"""Generate the unrolled CUDA implementation used by the public C ABI."""

from __future__ import annotations

from pathlib import Path


def operand(index: int) -> str:
    if index == 12:
        return "outer"
    if index < 16:
        return f"C_JOB.base_words[{index}]"
    return f"sched[{index - 16}]"


def schedule_body() -> str:
    lines: list[str] = []
    for t in range(16, 32):
        ops = [operand(t - k) for k in (3, 8, 14, 16)]
        lines.append(f"      sched[{t - 16}] = rol32(({'^'.join(ops)}), 1);")
    for t in range(32, 80):
        ops = [operand(t - k) for k in (6, 16, 28, 32)]
        lines.append(f"      sched[{t - 16}] = rol32(({'^'.join(ops)}), 2);")
    return "\n".join(lines)


def phase(t: int) -> str:
    return "ch" if t < 20 else "pa1" if t < 40 else "mj" if t < 60 else "pa3"


def round_body() -> str:
    lines: list[str] = []
    for t in range(16, 79):
        lines += [
            "    {",
            f"      auto dv = delta<{t}, V>(table, raw, j, r7, r3);",
            f"      const uint32_t bw = sched[{t - 16}];",
            "      #pragma unroll",
            f"      for (int i = 0; i < V; ++i) round_{phase(t)}(st[i], bw ^ dv.x[i]);",
            "    }",
        ]
    lines += [
        "    {",
        "      auto dv = delta<79, V>(table, raw, j, r7, r3);",
        "      const uint32_t bw = sched[63];",
        "      #pragma unroll",
        "      for (int i = 0; i < V; ++i) {",
        "        const uint32_t a80 = final_a(st[i], bw ^ dv.x[i]);",
        "        const unsigned inner = j + unsigned(i);",
        "        if constexpr (DOMAIN == 1) { if (inner >= INNER_END) continue; }",
        "        S digest{a80 + C_JOB.prestate[0],",
        "                 st[i].a + C_JOB.prestate[1],",
        "                 rol32(st[i].b, 30) + C_JOB.prestate[2],",
        "                 st[i].c + C_JOB.prestate[3],",
        "                 st[i].d + C_JOB.prestate[4]};",
        "        if constexpr (HEADER) compress_suffix(digest, suffix_schedules, suffix_blocks);",
        "        if constexpr (DIAG) {",
        "          if (inner == diag_inner) {",
        "            diag[0] = digest.a;",
        "            diag[1] = digest.b;",
        "            diag[2] = digest.c;",
        "            diag[3] = digest.d;",
        "            diag[4] = digest.e;",
        "          }",
        "        } else if ((HEADER ? target_match_digest(digest) : target_match<MODE>(a80, st[i])) &&",
        "                   nonce_valid(outer, inner, nonce_policy)) {",
        "          atomicCAS(reinterpret_cast<unsigned long long *>(winner),",
        "                    static_cast<unsigned long long>(GSV_NO_WINNER),",
        "                    static_cast<unsigned long long>((uint64_t(outer) << 8) | inner));",
        "        }",
        "      }",
        "    }",
    ]
    return "\n".join(lines)


def masked_word(candidate: str, first_shift: int) -> str:
    terms = ["0x20202020u"]
    for byte in range(4):
        candidate_shift = first_shift - byte * 5
        word_shift = 24 - byte * 8
        terms.append(f"(uint32_t(({candidate}>>{candidate_shift})&0x1full)<<{word_shift})")
    return "|".join(terms)


def masked_round_body(ilp: int = 4) -> str:
    lines: list[str] = []
    for lane in range(ilp):
        lines.extend(
            [
                f"const uint32_t mw11_{lane}={masked_word(f'mid{lane}', 35)};",
                f"const uint32_t mw12_{lane}={masked_word(f'mid{lane}', 15)};",
                f"uint32_t mw{lane}[32];",
            ]
        )
        for t in range(11):
            lines.append(f"mw{lane}[{t}]=C_JOB.base_words[{t}];")
        lines.extend(
            [
                f"mw{lane}[11]=mw11_{lane};",
                f"mw{lane}[12]=mw12_{lane};",
                f"mw{lane}[13]=C_JOB.base_words[13];",
                f"mw{lane}[14]=C_JOB.base_words[14];",
                f"mw{lane}[15]=C_JOB.base_words[15];",
                f"S ms{lane}{{C_PRE11[0],C_PRE11[1],C_PRE11[2],C_PRE11[3],C_PRE11[4]}};",
            ]
        )
    for t in range(11, 80):
        for lane in range(ilp):
            if t >= 16:
                index = t & 31
                if t < 32:
                    expression = (
                        f"rol32(mw{lane}[{(t-3)&31}]^mw{lane}[{(t-8)&31}]^"
                        f"mw{lane}[{(t-14)&31}]^mw{lane}[{(t-16)&31}],1)"
                    )
                else:
                    expression = (
                        f"rol32(mw{lane}[{(t-6)&31}]^mw{lane}[{(t-16)&31}]^"
                        f"mw{lane}[{(t-28)&31}]^mw{lane}[{(t-32)&31}],2)"
                    )
                lines.append(f"mw{lane}[{index}]={expression};")
            lines.append(f"round_{phase(t)}(ms{lane},mw{lane}[{t & 31}]);")
    for lane in range(ilp):
        lines.extend(
            [
                f"if(mactive{lane}){{",
                f"  S digest{{ms{lane}.a+C_JOB.prestate[0],ms{lane}.b+C_JOB.prestate[1],",
                f"           ms{lane}.c+C_JOB.prestate[2],ms{lane}.d+C_JOB.prestate[3],",
                f"           ms{lane}.e+C_JOB.prestate[4]}};",
                "  compress_suffix(digest,suffix_schedules,suffix_blocks);",
                "  if constexpr(DIAG){",
                f"    if(mid{lane}==diag_id){{diag[0]=digest.a;diag[1]=digest.b;diag[2]=digest.c;",
                "      diag[3]=digest.d;diag[4]=digest.e;}",
                "  }else if(target_match_digest(digest))",
                "    atomicCAS(reinterpret_cast<unsigned long long*>(winner),",
                "              static_cast<unsigned long long>(GSV_NO_WINNER),",
                f"              static_cast<unsigned long long>(mid{lane}));",
                "}",
            ]
        )
    return "\n    ".join(lines)


SOURCE = r'''// Generated by tools/generate_cuda_library.py. Do not hand-edit.
#include "git_sha1_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

static constexpr int G = 2, V = 8, BLOCK = 256, STRIDE = 65;
static constexpr int MASKED_ILP = 4, MASKED_BLOCK = 512;
static constexpr uint32_t K0 = 0x5a827999u, K1 = 0x6ed9eba1u;
static constexpr uint32_t K2 = 0x8f1bbcdcu, K3 = 0xca62c1d6u;
static constexpr size_t DYNAMIC_SMEM = (BLOCK / G) * STRIDE * sizeof(uint32_t);

__constant__ gsv_job C_JOB;
__constant__ uint32_t C_PRE11[5];

__host__ __device__ __forceinline__ uint32_t rol32(uint32_t x, unsigned n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_l(x, x, n);
#else
  return (x << n) | (x >> (32 - n));
#endif
}

__device__ __forceinline__ uint32_t fch(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0,%1,%2,%3,0xca;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}
__device__ __forceinline__ uint32_t fpa(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}
__device__ __forceinline__ uint32_t fmj(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t r;
  asm("lop3.b32 %0,%1,%2,%3,0xe8;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
  return r;
}

struct S { uint32_t a, b, c, d, e; };
__device__ __forceinline__ void round_ch(S &s, uint32_t w) {
  uint32_t z = rol32(s.a, 5) + fch(s.b, s.c, s.d) + s.e + K0 + w;
  s.e = s.d; s.d = s.c; s.c = rol32(s.b, 30); s.b = s.a; s.a = z;
}
__device__ __forceinline__ void round_pa1(S &s, uint32_t w) {
  uint32_t z = rol32(s.a, 5) + fpa(s.b, s.c, s.d) + s.e + K1 + w;
  s.e = s.d; s.d = s.c; s.c = rol32(s.b, 30); s.b = s.a; s.a = z;
}
__device__ __forceinline__ void round_mj(S &s, uint32_t w) {
  uint32_t z = rol32(s.a, 5) + fmj(s.b, s.c, s.d) + s.e + K2 + w;
  s.e = s.d; s.d = s.c; s.c = rol32(s.b, 30); s.b = s.a; s.a = z;
}
__device__ __forceinline__ void round_pa3(S &s, uint32_t w) {
  uint32_t z = rol32(s.a, 5) + fpa(s.b, s.c, s.d) + s.e + K3 + w;
  s.e = s.d; s.d = s.c; s.c = rol32(s.b, 30); s.b = s.a; s.a = z;
}
__device__ __forceinline__ uint32_t final_a(const S &s, uint32_t w) {
  return rol32(s.a, 5) + fpa(s.b, s.c, s.d) + s.e + K3 + w;
}

__host__ __device__ constexpr uint32_t cr(uint32_t x, int n) {
  n &= 31;
  return n ? ((x << n) | (x >> (32 - n))) : x;
}
__host__ __device__ constexpr uint32_t pmask(int t) {
  uint32_t p[80]{};
  p[13] = 1;
  for (int i = 16; i <= t; ++i) p[i] = cr(p[i - 3] ^ p[i - 8] ^ p[i - 14] ^ p[i - 16], 1);
  return p[t];
}
__host__ __device__ constexpr int pc(uint32_t x) {
  int n = 0;
  while (x) { n += x & 1u; x >>= 1; }
  return n;
}
__host__ __device__ constexpr int ctz1(uint32_t x) {
  int n = 0;
  while (!(x & 1u)) { ++n; x >>= 1; }
  return n;
}
__host__ __device__ constexpr int cidx(int t) {
  int n = 0;
  for (int q = 16; q < t; ++q) if (pc(pmask(q)) > 1) ++n;
  return n;
}

template<int N> struct P { uint32_t x[N]; };
template<int N> __device__ __forceinline__ P<N> rawp(unsigned j) {
  P<N> z{};
  #pragma unroll
  for (int i = 0; i < N; ++i) z.x[i] = uint32_t(j + i) << 24;
  return z;
}
template<int N> __device__ __forceinline__ P<N> rotp(const P<N> &p, int r) {
  P<N> z{};
  #pragma unroll
  for (int i = 0; i < N; ++i) z.x[i] = rol32(p.x[i], r);
  return z;
}
template<int N> __device__ __forceinline__ P<N> loadp(const uint32_t *table, int row, unsigned j) {
  P<N> z{};
  const uint32_t *p = table + row * 256 + j;
  #pragma unroll
  for (int q = 0; q < N; q += 4) {
    uint4 a;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p + q));
    z.x[q] = a.x; z.x[q + 1] = a.y; z.x[q + 2] = a.z; z.x[q + 3] = a.w;
  }
  return z;
}
template<int T, int N> __device__ __forceinline__ P<N> delta(
    const uint32_t *table, const P<N> &raw, unsigned j, const P<N> &r7, const P<N> &r3) {
  constexpr uint32_t m = pmask(T);
  constexpr int n = pc(m);
  if constexpr (n == 0) return P<N>{};
  else if constexpr (n == 1) {
    constexpr int e = ctz1(m);
    if constexpr (e == 7) return r7;
    else if constexpr (e == 3) return r3;
    else return rotp<N>(raw, e);
  } else return loadp<N>(table, cidx(T), j);
}

__device__ __forceinline__ bool zero_byte(uint32_t x) {
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0;
}

__device__ __forceinline__ bool printable_byte(unsigned byte) {
  return byte - 0x20u <= 0x7eu - 0x20u;
}

__device__ __forceinline__ bool nonce_valid(uint32_t outer, unsigned inner,
                                            uint32_t policy) {
  if (!inner || zero_byte(outer)) return false;
  if (policy == GSV_NONCE_HEADER_SAFE)
    return inner != 0x0au && !zero_byte(outer ^ 0x0a0a0a0au);
  if (policy == GSV_NONCE_PRINTABLE_ASCII)
    return printable_byte(inner) && printable_byte(outer >> 24) &&
           printable_byte((outer >> 16) & 0xffu) && printable_byte((outer >> 8) & 0xffu) &&
           printable_byte(outer & 0xffu);
  return true;
}

template<int MODE> __device__ __forceinline__ bool target_match(uint32_t a80, const S &s) {
  if constexpr (MODE == 0) {
    return uint32_t(a80 - C_JOB.h0_gate_base) < C_JOB.h0_gate_span;
  } else {
    if (a80 != C_JOB.h0_gate_base) return false;
    if constexpr (MODE == 2) {
      const uint32_t h1 = s.a + C_JOB.prestate[1];
      const uint32_t h2 = rol32(s.b, 30) + C_JOB.prestate[2];
      const uint32_t h3 = s.c + C_JOB.prestate[3];
      const uint32_t h4 = s.d + C_JOB.prestate[4];
      return (h1 & C_JOB.target_masks[1]) == (C_JOB.target_words[1] & C_JOB.target_masks[1]) &&
             (h2 & C_JOB.target_masks[2]) == (C_JOB.target_words[2] & C_JOB.target_masks[2]) &&
             (h3 & C_JOB.target_masks[3]) == (C_JOB.target_words[3] & C_JOB.target_masks[3]) &&
             (h4 & C_JOB.target_masks[4]) == (C_JOB.target_words[4] & C_JOB.target_masks[4]);
    }
    return true;
  }
}

__device__ __forceinline__ bool target_match_digest(const S &h) {
  const uint32_t words[5] = {h.a, h.b, h.c, h.d, h.e};
  #pragma unroll
  for (int i = 0; i < 5; ++i)
    if ((words[i] & C_JOB.target_masks[i]) != (C_JOB.target_words[i] & C_JOB.target_masks[i]))
      return false;
  return true;
}

__device__ __forceinline__ void compress_suffix(S &h, const uint32_t *schedules,
                                                uint32_t block_count) {
  for (uint32_t block = 0; block < block_count; ++block) {
    const S input = h;
    S s = h;
    const uint32_t *w = schedules + size_t(block) * 80;
    #pragma unroll
    for (int t = 0; t < 20; ++t) round_ch(s, w[t]);
    #pragma unroll
    for (int t = 20; t < 40; ++t) round_pa1(s, w[t]);
    #pragma unroll
    for (int t = 40; t < 60; ++t) round_mj(s, w[t]);
    #pragma unroll
    for (int t = 60; t < 80; ++t) round_pa3(s, w[t]);
    h.a = input.a + s.a;
    h.b = input.b + s.b;
    h.c = input.c + s.c;
    h.d = input.d + s.d;
    h.e = input.e + s.e;
  }
}

template<int MODE, bool DIAG, bool HEADER, int DOMAIN>
__global__ void search_kernel(uint64_t outer_base, uint64_t outer_count, const uint32_t *table,
                              const uint32_t *suffix_schedules, uint32_t suffix_blocks,
                              uint32_t nonce_policy, uint64_t *winner,
                              unsigned diag_inner, uint32_t *diag) {
  extern __shared__ uint32_t shared[];
  const int lane = threadIdx.x & (G - 1);
  const int groups = blockDim.x / G;
  const int local_group = threadIdx.x / G;
  const uint64_t gi = uint64_t(blockIdx.x) * groups + local_group;
  const uint64_t ov = outer_base + gi;
  constexpr uint64_t OUTER_LIMIT = DOMAIN == 1 ? 95ull * 95ull * 95ull * 95ull
                                   : DOMAIN == 2 ? 1ull << 20
                                                 : 0x100000000ull;
  const bool active = gi < outer_count && ov < OUTER_LIMIT;
  uint32_t outer = uint32_t(ov);
  if constexpr (DOMAIN == 1) {
    uint32_t ordinal = uint32_t(ov);
    outer = 0;
    #pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      outer |= (ordinal % 95u + 0x20u) << (byte * 8);
      ordinal /= 95u;
    }
  } else if constexpr (DOMAIN == 2) {
    outer = 0x20202020u | ((uint32_t(ov) & 0x0000001fu) << 0) |
            ((uint32_t(ov) & 0x000003e0u) << 3) |
            ((uint32_t(ov) & 0x00007c00u) << 6) |
            ((uint32_t(ov) & 0x000f8000u) << 9);
  }
  uint32_t *sched = shared + local_group * STRIDE;
  if (active && lane == 0) {
@@SCHEDULE@@
  }
  __syncwarp();
  if (!active) return;

  S common{C_JOB.pre12[0], C_JOB.pre12[1], C_JOB.pre12[2], C_JOB.pre12[3], C_JOB.pre12[4]};
  round_ch(common, outer);
  constexpr unsigned CHUNK = G * V;
  constexpr unsigned INNER_BEGIN = DOMAIN ? 0x20u : 0u;
  constexpr unsigned INNER_END = DOMAIN == 1 ? 0x7fu : DOMAIN == 2 ? 0x40u : 0x100u;
  #pragma unroll 1
  for (unsigned chunk = INNER_BEGIN; chunk < INNER_END; chunk += CHUNK) {
    const unsigned j = chunk + unsigned(lane) * V;
    S st[V];
    #pragma unroll
    for (int i = 0; i < V; ++i) st[i] = common;
    auto raw = rawp<V>(j), r7 = rotp<V>(raw, 7), r3 = rotp<V>(raw, 3);
    #pragma unroll
    for (int i = 0; i < V; ++i) round_ch(st[i], C_JOB.base_words[13] ^ raw.x[i]);
    #pragma unroll
    for (int i = 0; i < V; ++i) round_ch(st[i], C_JOB.base_words[14]);
    #pragma unroll
    for (int i = 0; i < V; ++i) round_ch(st[i], C_JOB.base_words[15]);
@@ROUNDS@@
  }
}

template<bool DIAG>
__global__ __launch_bounds__(MASKED_BLOCK) void masked8_kernel(
    uint64_t first, uint64_t count, const uint32_t *suffix_schedules,
    uint32_t suffix_blocks, uint64_t *winner, uint64_t diag_id, uint32_t *diag) {
  const uint64_t thread = uint64_t(blockIdx.x) * MASKED_BLOCK + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * MASKED_BLOCK;
  const uint64_t end = first + count;
  for (uint64_t base = first + thread; base < end; base += stride * MASKED_ILP) {
    const uint64_t mid0=base+0ull*stride;const bool mactive0=mid0<end;
    const uint64_t mid1=base+1ull*stride;const bool mactive1=mid1<end;
    const uint64_t mid2=base+2ull*stride;const bool mactive2=mid2<end;
    const uint64_t mid3=base+3ull*stride;const bool mactive3=mid3<end;
    @@MASKED_ROUNDS@@
  }
}

struct gsv_context {
  int32_t device = 0;
  gsv_job job{};
  uint32_t *table = nullptr;
  uint32_t *suffix_schedules = nullptr;
  size_t suffix_capacity_words = 0;
  uint32_t suffix_blocks = 0;
  bool header_mode = false;
  bool masked_mode = false;
  gsv_nonce_policy nonce_policy = GSV_NONCE_NO_NUL;
  uint32_t masked_pre11[5]{};
  int multiprocessors = 0;
  uint64_t *winner = nullptr;
  uint32_t *diag = nullptr;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  std::string error;
};

static std::mutex g_cuda_mutex;
static thread_local std::string g_last_error;

static void set_error(gsv_context *ctx, const char *message) {
  if (ctx) ctx->error = message ? message : "unknown error";
  else g_last_error = message ? message : "unknown error";
}
static gsv_status cuda_error(gsv_context *ctx, cudaError_t error, const char *operation) {
  std::string text(operation);
  text += ": ";
  text += cudaGetErrorString(error);
  set_error(ctx, text.c_str());
  return GSV_CUDA_ERROR;
}

static uint32_t host_rol(uint32_t x, unsigned n) { return (x << n) | (x >> (32 - n)); }

static void expand_schedule(const uint32_t block[16], uint32_t w[80]) {
  std::memcpy(w, block, 16 * sizeof(uint32_t));
  for (int t = 16; t < 80; ++t)
    w[t] = host_rol(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
}

static void partial_rounds(const uint32_t state[5], const uint32_t block[16], int rounds,
                           uint32_t out[5]) {
  uint32_t w[80];
  expand_schedule(block, w);
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
  for (int t = 0; t < rounds; ++t) {
    uint32_t z = host_rol(a, 5) + ((b & c) | (~b & d)) + e + K0 + w[t];
    e = d; d = c; c = host_rol(b, 30); b = a; a = z;
  }
  out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = e;
}

static void partial12(const uint32_t state[5], const uint32_t block[16], uint32_t out[5]) {
  partial_rounds(state, block, 12, out);
}

static std::vector<uint32_t> make_table() {
  std::vector<uint32_t> table;
  table.reserve(32u * 256u);
  for (int t = 16; t < 80; ++t) if (pc(pmask(t)) > 1) {
    for (unsigned j = 0; j < 256; ++j) {
      uint32_t w[80]{};
      w[13] = j << 24;
      for (int q = 16; q <= t; ++q)
        w[q] = host_rol(w[q - 3] ^ w[q - 8] ^ w[q - 14] ^ w[q - 16], 1);
      table.push_back(w[t]);
    }
  }
  return table;
}

static void masks_for_bits(uint32_t bits, uint32_t masks[5]) {
  for (int i = 0; i < 5; ++i) {
    const uint32_t used = std::min<uint32_t>(32, bits);
    masks[i] = used == 32 ? 0xffffffffu : used ? (0xffffffffu << (32 - used)) : 0u;
    bits -= used;
  }
}

static gsv_status validate_impl(const gsv_job *job, bool report) {
  auto fail = [&](gsv_status status, const char *message) {
    if (report) set_error(nullptr, message);
    return status;
  };
  if (!job) return fail(GSV_INVALID_ARGUMENT, "job is null");
  if (job->abi_version != GSV_ABI_VERSION) return fail(GSV_ABI_MISMATCH, "job ABI version mismatch");
  if (job->target_bits < 1 || job->target_bits > 160)
    return fail(GSV_INVALID_ARGUMENT, "target_bits must be in 1..160");
  if (job->base_words[12] != 0 || (job->base_words[13] & 0xff000000u) != 0)
    return fail(GSV_INVALID_ARGUMENT, "candidate bytes in base_words[12..13] must be zero");
  uint32_t masks[5], pre12[5];
  masks_for_bits(job->target_bits, masks);
  partial12(job->prestate, job->base_words, pre12);
  if (std::memcmp(masks, job->target_masks, sizeof(masks)) != 0 ||
      std::memcmp(pre12, job->pre12, sizeof(pre12)) != 0)
    return fail(GSV_INVALID_ARGUMENT, "job derived fields are inconsistent; call gsv_job_init");
  const uint32_t used = std::min<uint32_t>(32, job->target_bits);
  const uint32_t span = uint32_t(1ull << (32 - used));
  const uint32_t gate = (job->target_words[0] & masks[0]) - job->prestate[0];
  if (job->h0_gate_base != gate || job->h0_gate_span != span)
    return fail(GSV_INVALID_ARGUMENT, "job H0 gate is inconsistent; call gsv_job_init");
  return GSV_OK;
}

static void cleanup_context(gsv_context *ctx) {
  if (!ctx) return;
  if (ctx->end) cudaEventDestroy(ctx->end);
  if (ctx->begin) cudaEventDestroy(ctx->begin);
  if (ctx->diag) cudaFree(ctx->diag);
  if (ctx->winner) cudaFree(ctx->winner);
  if (ctx->suffix_schedules) cudaFree(ctx->suffix_schedules);
  if (ctx->table) cudaFree(ctx->table);
  delete ctx;
}

static gsv_status upload_job(gsv_context *ctx) {
  cudaError_t e = cudaMemcpyToSymbol(C_JOB, &ctx->job, sizeof(ctx->job));
  return e == cudaSuccess ? GSV_OK : cuda_error(ctx, e, "cudaMemcpyToSymbol(job)");
}

template<int MODE, bool HEADER>
static cudaError_t launch_search(uint64_t outer_base, uint64_t outer_count,
                                 const uint32_t *table, const uint32_t *suffix_schedules,
                                 uint32_t suffix_blocks, gsv_nonce_policy nonce_policy,
                                 uint64_t *winner, int domain) {
  const uint64_t groups = BLOCK / G;
  const uint32_t blocks = uint32_t((outer_count + groups - 1) / groups);
  if (domain == 1)
    search_kernel<MODE, false, HEADER, 1><<<blocks, BLOCK, DYNAMIC_SMEM>>>(
        outer_base, outer_count, table, suffix_schedules, suffix_blocks, nonce_policy,
        winner, 0, nullptr);
  else if (domain == 2)
    search_kernel<MODE, false, HEADER, 2><<<blocks, BLOCK, DYNAMIC_SMEM>>>(
        outer_base, outer_count, table, suffix_schedules, suffix_blocks, nonce_policy,
        winner, 0, nullptr);
  else
    search_kernel<MODE, false, HEADER, 0><<<blocks, BLOCK, DYNAMIC_SMEM>>>(
        outer_base, outer_count, table, suffix_schedules, suffix_blocks, nonce_policy,
        winner, 0, nullptr);
  return cudaGetLastError();
}

extern "C" {

uint32_t gsv_abi_version(void) { return GSV_ABI_VERSION; }

int32_t gsv_device_count(void) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) { set_error(nullptr, cudaGetErrorString(e)); return -1; }
  g_last_error.clear();
  return count;
}

gsv_status gsv_job_init(gsv_job *job, const uint32_t prestate[5],
                        const uint32_t base_words[16], uint32_t target_bits,
                        const uint32_t target_words[5]) {
  if (!job || !prestate || !base_words || !target_words) {
    set_error(nullptr, "gsv_job_init received a null pointer");
    return GSV_INVALID_ARGUMENT;
  }
  if (target_bits < 1 || target_bits > 160) {
    set_error(nullptr, "target_bits must be in 1..160");
    return GSV_INVALID_ARGUMENT;
  }
  std::memset(job, 0, sizeof(*job));
  job->abi_version = GSV_ABI_VERSION;
  std::memcpy(job->prestate, prestate, 5 * sizeof(uint32_t));
  std::memcpy(job->base_words, base_words, 16 * sizeof(uint32_t));
  std::memcpy(job->target_words, target_words, 5 * sizeof(uint32_t));
  job->target_bits = target_bits;
  masks_for_bits(target_bits, job->target_masks);
  partial12(prestate, base_words, job->pre12);
  const uint32_t used = std::min<uint32_t>(32, target_bits);
  job->h0_gate_base = (target_words[0] & job->target_masks[0]) - prestate[0];
  job->h0_gate_span = uint32_t(1ull << (32 - used));
  gsv_status status = validate_impl(job, true);
  if (status == GSV_OK) g_last_error.clear();
  return status;
}

gsv_status gsv_job_validate(const gsv_job *job) {
  gsv_status status = validate_impl(job, true);
  if (status == GSV_OK) g_last_error.clear();
  return status;
}

gsv_status gsv_context_create(int32_t device, const gsv_job *job, gsv_context **out) {
  if (!out) { set_error(nullptr, "out_context is null"); return GSV_INVALID_ARGUMENT; }
  *out = nullptr;
  gsv_status valid = validate_impl(job, true);
  if (valid != GSV_OK) return valid;
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) return cuda_error(nullptr, e, "cudaGetDeviceCount");
  if (device < 0 || device >= count) {
    set_error(nullptr, "CUDA device index is out of range");
    return GSV_INVALID_ARGUMENT;
  }
  gsv_context *ctx = new (std::nothrow) gsv_context;
  if (!ctx) { set_error(nullptr, "context allocation failed"); return GSV_INTERNAL_ERROR; }
  ctx->device = device;
  ctx->job = *job;
  if ((e = cudaSetDevice(device)) != cudaSuccess) { cleanup_context(ctx); return cuda_error(nullptr, e, "cudaSetDevice"); }
  cudaDeviceProp properties{};
  if ((e = cudaGetDeviceProperties(&properties, device)) != cudaSuccess) {
    cleanup_context(ctx); return cuda_error(nullptr, e, "cudaGetDeviceProperties");
  }
  ctx->multiprocessors = properties.multiProcessorCount;
  try {
    std::vector<uint32_t> host_table = make_table();
    if (host_table.size() != 32u * 256u) { cleanup_context(ctx); set_error(nullptr, "delta table has unexpected size"); return GSV_INTERNAL_ERROR; }
    if ((e = cudaMalloc(&ctx->table, host_table.size() * sizeof(uint32_t))) != cudaSuccess ||
        (e = cudaMemcpy(ctx->table, host_table.data(), host_table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMalloc(&ctx->winner, sizeof(uint64_t))) != cudaSuccess ||
        (e = cudaMalloc(&ctx->diag, 5 * sizeof(uint32_t))) != cudaSuccess ||
        (e = cudaEventCreate(&ctx->begin)) != cudaSuccess ||
        (e = cudaEventCreate(&ctx->end)) != cudaSuccess) {
      std::string message = std::string("CUDA context initialization: ") + cudaGetErrorString(e);
      cleanup_context(ctx); set_error(nullptr, message.c_str()); return GSV_CUDA_ERROR;
    }
  } catch (const std::bad_alloc &) {
    cleanup_context(ctx); set_error(nullptr, "host allocation failed"); return GSV_INTERNAL_ERROR;
  }
  *out = ctx;
  g_last_error.clear();
  return GSV_OK;
}

void gsv_context_destroy(gsv_context *ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  cudaSetDevice(ctx->device);
  cleanup_context(ctx);
}

gsv_status gsv_context_set_job(gsv_context *ctx, const gsv_job *job) {
  if (!ctx) { set_error(nullptr, "context is null"); return GSV_INVALID_ARGUMENT; }
  gsv_status valid = validate_impl(job, false);
  if (valid != GSV_OK) { set_error(ctx, "invalid job"); return valid; }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->job = *job;
  ctx->header_mode = false;
  ctx->masked_mode = false;
  ctx->suffix_blocks = 0;
  ctx->nonce_policy = GSV_NONCE_NO_NUL;
  ctx->error.clear();
  return GSV_OK;
}

gsv_status gsv_context_set_header_job(gsv_context *ctx, const gsv_job *job,
                                      const uint32_t *suffix_words,
                                      uint32_t suffix_block_count) {
  if (!ctx) { set_error(nullptr, "context is null"); return GSV_INVALID_ARGUMENT; }
  gsv_status valid = validate_impl(job, false);
  if (valid != GSV_OK) { set_error(ctx, "invalid job"); return valid; }
  if (suffix_block_count && !suffix_words) {
    set_error(ctx, "suffix_words is null for a non-empty suffix");
    return GSV_INVALID_ARGUMENT;
  }
  if (suffix_block_count > (UINT32_MAX / 80u)) {
    set_error(ctx, "suffix block count is too large");
    return GSV_INVALID_ARGUMENT;
  }

  std::vector<uint32_t> schedules;
  try {
    schedules.resize(size_t(suffix_block_count) * 80u);
    for (uint32_t block = 0; block < suffix_block_count; ++block)
      expand_schedule(suffix_words + size_t(block) * 16u,
                      schedules.data() + size_t(block) * 80u);
  } catch (const std::bad_alloc &) {
    set_error(ctx, "host suffix-schedule allocation failed");
    return GSV_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  cudaError_t e = cudaSetDevice(ctx->device);
  if (e != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  const size_t words = schedules.size();
  if (words > ctx->suffix_capacity_words) {
    uint32_t *replacement = nullptr;
    if ((e = cudaMalloc(&replacement, words * sizeof(uint32_t))) != cudaSuccess)
      return cuda_error(ctx, e, "allocate suffix schedules");
    if (ctx->suffix_schedules) cudaFree(ctx->suffix_schedules);
    ctx->suffix_schedules = replacement;
    ctx->suffix_capacity_words = words;
  }
  if (words && (e = cudaMemcpy(ctx->suffix_schedules, schedules.data(), words * sizeof(uint32_t),
                               cudaMemcpyHostToDevice)) != cudaSuccess)
    return cuda_error(ctx, e, "upload suffix schedules");
  ctx->job = *job;
  ctx->header_mode = true;
  ctx->masked_mode = false;
  ctx->suffix_blocks = suffix_block_count;
  ctx->nonce_policy = GSV_NONCE_HEADER_SAFE;
  ctx->error.clear();
  return GSV_OK;
}

gsv_status gsv_context_set_masked_header_job(
    gsv_context *ctx, const uint32_t prestate[5], const uint32_t base_words[16],
    uint32_t target_bits, const uint32_t target_words[5],
    const uint32_t *suffix_words, uint32_t suffix_block_count) {
  if (!ctx || !prestate || !base_words || !target_words) {
    set_error(ctx, "masked-header job received a null pointer");
    return GSV_INVALID_ARGUMENT;
  }
  if (target_bits < 1 || target_bits > 160) {
    set_error(ctx, "target_bits must be in 1..160");
    return GSV_INVALID_ARGUMENT;
  }
  if (base_words[11] || base_words[12]) {
    set_error(ctx, "masked-header candidate words W11 and W12 must be zero");
    return GSV_INVALID_ARGUMENT;
  }
  if (suffix_block_count && !suffix_words) {
    set_error(ctx, "suffix_words is null for a non-empty suffix");
    return GSV_INVALID_ARGUMENT;
  }
  if (suffix_block_count > (UINT32_MAX / 80u)) {
    set_error(ctx, "suffix block count is too large");
    return GSV_INVALID_ARGUMENT;
  }

  std::vector<uint32_t> schedules;
  try {
    schedules.resize(size_t(suffix_block_count) * 80u);
    for (uint32_t block = 0; block < suffix_block_count; ++block)
      expand_schedule(suffix_words + size_t(block) * 16u,
                      schedules.data() + size_t(block) * 80u);
  } catch (const std::bad_alloc &) {
    set_error(ctx, "host suffix-schedule allocation failed");
    return GSV_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  cudaError_t e = cudaSetDevice(ctx->device);
  if (e != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  const size_t words = schedules.size();
  if (words > ctx->suffix_capacity_words) {
    uint32_t *replacement = nullptr;
    if ((e = cudaMalloc(&replacement, words * sizeof(uint32_t))) != cudaSuccess)
      return cuda_error(ctx, e, "allocate suffix schedules");
    if (ctx->suffix_schedules) cudaFree(ctx->suffix_schedules);
    ctx->suffix_schedules = replacement;
    ctx->suffix_capacity_words = words;
  }
  if (words && (e = cudaMemcpy(ctx->suffix_schedules, schedules.data(), words * sizeof(uint32_t),
                               cudaMemcpyHostToDevice)) != cudaSuccess)
    return cuda_error(ctx, e, "upload suffix schedules");

  std::memset(&ctx->job, 0, sizeof(ctx->job));
  ctx->job.abi_version = GSV_ABI_VERSION;
  std::memcpy(ctx->job.prestate, prestate, 5 * sizeof(uint32_t));
  std::memcpy(ctx->job.base_words, base_words, 16 * sizeof(uint32_t));
  std::memcpy(ctx->job.target_words, target_words, 5 * sizeof(uint32_t));
  ctx->job.target_bits = target_bits;
  masks_for_bits(target_bits, ctx->job.target_masks);
  partial_rounds(prestate, base_words, 11, ctx->masked_pre11);
  ctx->suffix_blocks = suffix_block_count;
  ctx->header_mode = false;
  ctx->masked_mode = true;
  ctx->nonce_policy = GSV_NONCE_PRINTABLE_ASCII;
  ctx->error.clear();
  return GSV_OK;
}

gsv_status gsv_context_set_nonce_policy(gsv_context *ctx, gsv_nonce_policy policy) {
  if (!ctx) { set_error(nullptr, "context is null"); return GSV_INVALID_ARGUMENT; }
  if (policy != GSV_NONCE_NO_NUL && policy != GSV_NONCE_HEADER_SAFE &&
      policy != GSV_NONCE_PRINTABLE_ASCII) {
    set_error(ctx, "unknown nonce policy");
    return GSV_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->nonce_policy = policy;
  ctx->error.clear();
  return GSV_OK;
}

static gsv_status search_impl(gsv_context *ctx, uint64_t outer_base, uint64_t outer_count,
                              gsv_search_result *result, int domain) {
  if (!ctx || !result) { set_error(ctx, "gsv_search received a null pointer"); return GSV_INVALID_ARGUMENT; }
  if (ctx->masked_mode) {
    set_error(ctx, "masked-header jobs use gsv_search_masked_header");
    return GSV_INVALID_ARGUMENT;
  }
  std::memset(result, 0, sizeof(*result));
  result->candidate = GSV_NO_WINNER;
  const uint64_t outer_limit = domain == 1 ? 95ull * 95ull * 95ull * 95ull
                               : domain == 2 ? 1ull << 20
                                             : 0x100000000ull;
  if (!outer_count || outer_base >= outer_limit || outer_count > outer_limit - outer_base) {
    set_error(ctx, "outer range is outside the selected candidate domain");
    return GSV_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->error.clear();
  cudaError_t e;
  if ((e = cudaSetDevice(ctx->device)) != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  gsv_status uploaded = upload_job(ctx);
  if (uploaded != GSV_OK) return uploaded;
  const uint64_t none = GSV_NO_WINNER;
  if ((e = cudaMemcpy(ctx->winner, &none, sizeof(none), cudaMemcpyHostToDevice)) != cudaSuccess)
    return cuda_error(ctx, e, "reset winner");
  if ((e = cudaEventRecord(ctx->begin)) != cudaSuccess) return cuda_error(ctx, e, "record begin event");
  if (ctx->header_mode) {
    if (ctx->job.target_bits < 32)
      e = launch_search<0, true>(outer_base, outer_count, ctx->table, ctx->suffix_schedules,
                                 ctx->suffix_blocks, ctx->nonce_policy, ctx->winner, domain);
    else if (ctx->job.target_bits == 32)
      e = launch_search<1, true>(outer_base, outer_count, ctx->table, ctx->suffix_schedules,
                                 ctx->suffix_blocks, ctx->nonce_policy, ctx->winner, domain);
    else
      e = launch_search<2, true>(outer_base, outer_count, ctx->table, ctx->suffix_schedules,
                                 ctx->suffix_blocks, ctx->nonce_policy, ctx->winner, domain);
  } else {
    if (ctx->job.target_bits < 32)
      e = launch_search<0, false>(outer_base, outer_count, ctx->table, nullptr, 0,
                                  ctx->nonce_policy, ctx->winner, domain);
    else if (ctx->job.target_bits == 32)
      e = launch_search<1, false>(outer_base, outer_count, ctx->table, nullptr, 0,
                                  ctx->nonce_policy, ctx->winner, domain);
    else
      e = launch_search<2, false>(outer_base, outer_count, ctx->table, nullptr, 0,
                                  ctx->nonce_policy, ctx->winner, domain);
  }
  if (e != cudaSuccess) return cuda_error(ctx, e, "launch search kernel");
  if ((e = cudaEventRecord(ctx->end)) != cudaSuccess || (e = cudaEventSynchronize(ctx->end)) != cudaSuccess)
    return cuda_error(ctx, e, "wait for search kernel");
  if ((e = cudaEventElapsedTime(&result->milliseconds, ctx->begin, ctx->end)) != cudaSuccess)
    return cuda_error(ctx, e, "measure search kernel");
  if ((e = cudaMemcpy(&result->candidate, ctx->winner, sizeof(result->candidate), cudaMemcpyDeviceToHost)) != cudaSuccess)
    return cuda_error(ctx, e, "copy winner");
  const uint64_t inner_count = domain == 1 ? 95ull : domain == 2 ? 32ull : 256ull;
  result->candidates_hashed = outer_count * inner_count;
  if (result->milliseconds > 0.0f)
    result->billions_per_second = float(double(result->candidates_hashed) / (double(result->milliseconds) * 1.0e6));
  result->found = result->candidate != GSV_NO_WINNER;
  return result->found ? GSV_FOUND : GSV_NOT_FOUND;
}

gsv_status gsv_search(gsv_context *ctx, uint64_t outer_base, uint64_t outer_count,
                      gsv_search_result *result) {
  return search_impl(ctx, outer_base, outer_count, result, 0);
}

gsv_status gsv_search_printable(gsv_context *ctx, uint64_t outer_base, uint64_t outer_count,
                                gsv_search_result *result) {
  return search_impl(ctx, outer_base, outer_count, result, 1);
}

gsv_status gsv_search_printable_mask(gsv_context *ctx, uint64_t outer_base,
                                     uint64_t outer_count, gsv_search_result *result) {
  return search_impl(ctx, outer_base, outer_count, result, 2);
}

gsv_status gsv_search_masked_header(gsv_context *ctx, uint64_t candidate_base,
                                    uint64_t candidate_count, gsv_search_result *result) {
  if (!ctx || !result) {
    set_error(ctx, "gsv_search_masked_header received a null pointer");
    return GSV_INVALID_ARGUMENT;
  }
  if (!ctx->masked_mode) {
    set_error(ctx, "context does not contain a masked-header job");
    return GSV_INVALID_ARGUMENT;
  }
  if (!candidate_count || candidate_base >= (1ull << 40) ||
      candidate_count > (1ull << 40) - candidate_base) {
    set_error(ctx, "candidate range must be a non-empty subset of 0..2^40");
    return GSV_INVALID_ARGUMENT;
  }
  std::memset(result, 0, sizeof(*result));
  result->candidate = GSV_NO_WINNER;
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->error.clear();
  cudaError_t e;
  if ((e = cudaSetDevice(ctx->device)) != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  if ((e = cudaMemcpyToSymbol(C_JOB, &ctx->job, sizeof(ctx->job))) != cudaSuccess)
    return cuda_error(ctx, e, "cudaMemcpyToSymbol(masked job)");
  if ((e = cudaMemcpyToSymbol(C_PRE11, ctx->masked_pre11, sizeof(ctx->masked_pre11))) != cudaSuccess)
    return cuda_error(ctx, e, "cudaMemcpyToSymbol(masked pre11)");
  const uint64_t none = GSV_NO_WINNER;
  if ((e = cudaMemcpy(ctx->winner, &none, sizeof(none), cudaMemcpyHostToDevice)) != cudaSuccess)
    return cuda_error(ctx, e, "reset winner");
  if ((e = cudaEventRecord(ctx->begin)) != cudaSuccess) return cuda_error(ctx, e, "record begin event");
  const uint32_t blocks = uint32_t(ctx->multiprocessors * 4);
  masked8_kernel<false><<<blocks, MASKED_BLOCK>>>(
      candidate_base, candidate_count, ctx->suffix_schedules, ctx->suffix_blocks,
      ctx->winner, 0, nullptr);
  if ((e = cudaGetLastError()) != cudaSuccess) return cuda_error(ctx, e, "launch masked-header kernel");
  if ((e = cudaEventRecord(ctx->end)) != cudaSuccess ||
      (e = cudaEventSynchronize(ctx->end)) != cudaSuccess)
    return cuda_error(ctx, e, "wait for masked-header kernel");
  if ((e = cudaEventElapsedTime(&result->milliseconds, ctx->begin, ctx->end)) != cudaSuccess)
    return cuda_error(ctx, e, "measure masked-header kernel");
  if ((e = cudaMemcpy(&result->candidate, ctx->winner, sizeof(result->candidate),
                      cudaMemcpyDeviceToHost)) != cudaSuccess)
    return cuda_error(ctx, e, "copy winner");
  result->candidates_hashed = candidate_count;
  if (result->milliseconds > 0.0f)
    result->billions_per_second =
        float(double(candidate_count) / (double(result->milliseconds) * 1.0e6));
  result->found = result->candidate != GSV_NO_WINNER;
  return result->found ? GSV_FOUND : GSV_NOT_FOUND;
}

gsv_status gsv_digest_masked_header(gsv_context *ctx, uint64_t candidate,
                                    uint32_t digest[5]) {
  if (!ctx || !digest) {
    set_error(ctx, "gsv_digest_masked_header received a null pointer");
    return GSV_INVALID_ARGUMENT;
  }
  if (!ctx->masked_mode) {
    set_error(ctx, "context does not contain a masked-header job");
    return GSV_INVALID_ARGUMENT;
  }
  if (candidate >= (1ull << 40)) {
    set_error(ctx, "candidate must fit in 40 bits");
    return GSV_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->error.clear();
  cudaError_t e;
  if ((e = cudaSetDevice(ctx->device)) != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  if ((e = cudaMemcpyToSymbol(C_JOB, &ctx->job, sizeof(ctx->job))) != cudaSuccess ||
      (e = cudaMemcpyToSymbol(C_PRE11, ctx->masked_pre11, sizeof(ctx->masked_pre11))) != cudaSuccess)
    return cuda_error(ctx, e, "upload masked-header diagnostic job");
  masked8_kernel<true><<<1, MASKED_BLOCK>>>(candidate, 1, ctx->suffix_schedules,
                                            ctx->suffix_blocks, nullptr, candidate, ctx->diag);
  if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaDeviceSynchronize()) != cudaSuccess)
    return cuda_error(ctx, e, "masked-header digest kernel");
  if ((e = cudaMemcpy(digest, ctx->diag, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
    return cuda_error(ctx, e, "copy masked-header digest");
  return GSV_OK;
}

gsv_status gsv_digest(gsv_context *ctx, uint64_t candidate, uint32_t digest[5]) {
  if (!ctx || !digest) { set_error(ctx, "gsv_digest received a null pointer"); return GSV_INVALID_ARGUMENT; }
  if (ctx->masked_mode) {
    set_error(ctx, "masked-header jobs use gsv_digest_masked_header");
    return GSV_INVALID_ARGUMENT;
  }
  if (candidate >= (1ull << 40)) { set_error(ctx, "candidate must fit in 40 bits"); return GSV_INVALID_ARGUMENT; }
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  ctx->error.clear();
  cudaError_t e;
  if ((e = cudaSetDevice(ctx->device)) != cudaSuccess) return cuda_error(ctx, e, "cudaSetDevice");
  gsv_status uploaded = upload_job(ctx);
  if (uploaded != GSV_OK) return uploaded;
  if (ctx->header_mode)
    search_kernel<0, true, true, 0><<<1, BLOCK, DYNAMIC_SMEM>>>(
        candidate >> 8, 1, ctx->table, ctx->suffix_schedules, ctx->suffix_blocks,
        ctx->nonce_policy, nullptr, unsigned(candidate & 0xffu), ctx->diag);
  else
    search_kernel<0, true, false, 0><<<1, BLOCK, DYNAMIC_SMEM>>>(
        candidate >> 8, 1, ctx->table, nullptr, 0, ctx->nonce_policy, nullptr,
        unsigned(candidate & 0xffu), ctx->diag);
  if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaDeviceSynchronize()) != cudaSuccess)
    return cuda_error(ctx, e, "digest kernel");
  if ((e = cudaMemcpy(digest, ctx->diag, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
    return cuda_error(ctx, e, "copy digest");
  return GSV_OK;
}

const char *gsv_last_error(const gsv_context *ctx) {
  return ctx ? ctx->error.c_str() : g_last_error.c_str();
}

const char *gsv_status_string(gsv_status status) {
  switch (status) {
    case GSV_OK: return "ok";
    case GSV_FOUND: return "found";
    case GSV_NOT_FOUND: return "not found";
    case GSV_INVALID_ARGUMENT: return "invalid argument";
    case GSV_CUDA_ERROR: return "CUDA error";
    case GSV_ABI_MISMATCH: return "ABI mismatch";
    case GSV_INTERNAL_ERROR: return "internal error";
    default: return "unknown status";
  }
}

}  // extern "C"
'''


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "src" / "git_sha1_cuda.cu"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        SOURCE.replace("@@SCHEDULE@@", schedule_body())
        .replace("@@ROUNDS@@", round_body())
        .replace("@@MASKED_ROUNDS@@", masked_round_body())
    )
    output.write_text(text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
