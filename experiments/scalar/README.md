# Scalar/persistent final-block experiment

This is the deliberately conventional control for the exact raw-tail job in
`tools/git_sha1_job.py`. Each CUDA lane hashes one candidate, or a small
compile-time ILP bundle, and a fixed grid walks the candidate interval with a
grid-stride loop. There is no inter-lane schedule sharing and no lookup table.

The candidate contract is unchanged:

```text
candidate 0xAABBCCDDEE -> nonce AA BB CC DD EE
W12 = candidate >> 8
W13[31:24] = (candidate & 0xff) << 24
```

The generator obtains the exact serialized Git object, final block, prestate,
target, and hashlib references through the host job API. Every executable
checks four full 160-bit GPU digests before benchmarking. The cases include a
zero candidate, `0102030405`, a high outer value, and inner byte `ff`.

## Variants

- `rolling`: a 16-word circular schedule using the original SHA-1 recurrence.
- `rolling32`: rounds 16--31 use the original recurrence; rounds 32--79 use
  `ROL2(W[t-6] ^ W[t-16] ^ W[t-28] ^ W[t-32])`. This consumes more registers
  but shortens the schedule dependency chain.
- `affine`: directly generates every candidate-dependent schedule word from
  rotation polynomials of W12 and W13. It removes the schedule ring but needs
  substantially more boolean/rotate work.
- `struct` and `rotating`: canonical state assignment versus hashcat-style
  generated role rotation. On CUDA 13.3, ptxas made these effectively equal.
- ILP 1, 2, or 4: independent candidates interleaved round by round.

## Reproducing the winner

```bash
python3 experiments/scalar/generate.py \
  --schedule rolling32 \
  --round-form rotating \
  --ilp 4 \
  --block 512 \
  --out /tmp/scalar-ilp4.cu \
  --metadata /tmp/scalar-ilp4.json

timeout 60 flock /tmp/sha1-gpu.lock scp \
  -i /home/r34/.ssh/cuda_sandbox_ed25519 \
  -P 2222 \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/r34/.ssh/cuda_sandbox_known_hosts \
  /tmp/scalar-ilp4.cu agent@192.168.178.76:/work/scalar-ilp4.cu

timeout 120 flock /tmp/sha1-gpu.lock ssh \
  -i /home/r34/.ssh/cuda_sandbox_ed25519 \
  -p 2222 \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/r34/.ssh/cuda_sandbox_known_hosts \
  agent@192.168.178.76 \
  'cd /work && nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -Xptxas=-v scalar-ilp4.cu -o scalar-ilp4 && ./scalar-ilp4 536870912 2 1 9'
```

The benchmark arguments are candidate count, blocks per SM in the launched
grid, launches per timing sample, and odd sample count. The harness performs
five untimed warm-up launches and reports the median of CUDA-event timings.

## RTX 4060 results

CUDA 13.3.1, `sm_89`, `2^29` candidates, nine samples:

| Schedule | Form | ILP | Block/grid | Median GH/s | Registers | Local/spills |
|---|---:|---:|---:|---:|---:|---:|
| affine | rotating | 1 | 128 / 192 | 7.652 | 39 | 0 |
| rolling | rotating | 1 | 128 / 288 | 9.238 | 32 | 0 |
| rolling | struct | 1 | 128 / 288 | 9.272 | 32 | 0 |
| rolling | rotating | 2 | 256 / 288 | 9.420 | 40 | 0 |
| rolling32 | rotating | 1 | 512 / 48 | 9.562 | 47 | 0 |
| rolling32 | rotating | 2 | 384 / 192 | 9.643 | 52 | 0 |
| rolling32 | rotating | 4 | 512 / 48 | **9.646** | 59 | 0 |

The best scalar result is 88.5% of the supplied 10.9 GH/s shared-schedule
baseline; equivalently, the shared design is about 1.13x faster. The ILP-4
result is only 0.03% ahead of ILP-2, so ILP-2 is the simpler near-tie while
ILP-4 is the measured scalar winner.

## SASS evidence

Production-kernel counts from `cuobjdump --dump-sass`:

| Variant | Instructions | Instructions/candidate | SHF | LOP3 | LDL/STL |
|---|---:|---:|---:|---:|---:|
| affine, ILP-1 | 574 | 574.00 | 108 | 222 | 0 / 0 |
| rolling, ILP-2 | 1076 | 538.00 | 251 | 362 | 0 / 0 |
| rolling32, ILP-2 | 1050 | 525.00 | 245 | 340 | 0 / 0 |
| rolling32, ILP-4 | 2077 | 519.25 | 491 | 678 | 0 / 0 |

The totals include persistent-loop and hit-policy instructions, so the
per-candidate figures are normalized whole-kernel counts rather than only SHA
rounds. They still explain the direction of the timing results: the 32-word
recurrence removes about 13 instructions per candidate versus the 16-word
ring, while ILP-4 amortizes a few more loop/mapping instructions. Direct affine
generation increases the instruction count and loses decisively.
