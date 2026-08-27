# Shared-schedule CUDA SHA-1 final-block study

This directory is a clean, reproducible implementation of the fast unsigned
Git-commit path. It searches a raw five-byte big-endian candidate placed at
final-block byte offsets 48 through 52:

```text
candidate uint40:  [ b0 b1 b2 b3 | b4 ]
SHA-1 block:       [      W12     | W13 high byte ]
kernel mapping:    W12 = candidate >> 8
                   W13 |= (candidate & 0xff) << 24
```

This is the same candidate contract used by the host job tooling. GPG-signed
commits are deliberately outside this experiment: preserving an existing
signature makes the long armored suffix candidate-dependent and is a separate
kernel problem.

## What is generated

`generate.py` takes a deterministic unsigned commit payload, constructs the
exact `commit <length>\0<payload>` Git object, aligns the nonce, compresses its
five fixed SHA-1 blocks, and emits one self-contained `.cu` translation unit.
The CUDA program contains:

- the fixed input state, rounds 0--11 prestate, final block, and target gate;
- a host-built 32 x 256 table for the complex W13 schedule deltas;
- one isolated production kernel plus its diagnostic instantiation;
- an independent scalar CPU SHA-1 compression reference;
- 12 exact 160-bit GPU/CPU digest captures per executable;
- an embedded Python `hashlib` whole-object digest for candidate `0102030405`;
- eight warm-up launches and an odd-sample median event-timing harness.

The production kernel computes the W12-dependent schedule once per lane group
in shared memory. Each lane then evaluates `V` W13 bytes at a time. Simple W13
schedule deltas are rotations; only the 32 multi-term deltas use the 32 KiB
read-only table. The hot H0 comparison subtracts the fixed feed-forward word at
generation time.

Two controlled SHA-1 round forms are available:

- `struct`: explicitly updates canonical `(a,b,c,d,e)` state each round.
- `rotating`: hashcat-style steps write the new A into `e`, rotate `b`, and
  permute argument names across rounds.

Both forms produce identical digests. On this compiler/GPU, the rotating form
uses the same registers but is about 3% slower at the finalist geometry.

## Build and run

Generate and compile the long-confirmation winner with ordinary CUDA flags:

```bash
python3 experiments/shared/generate.py \
  --round-form struct --g 2 --v 8 --block 256 --cache ca --stride 65 \
  --out /tmp/shared-best.cu
nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo -Xptxas=-v \
  /tmp/shared-best.cu -o /tmp/shared-best
/tmp/shared-best 1048576 8 15
```

No legacy headers, binary tables, generated job directories, or experimental
CUDA flags are required. In particular, the compile-time helpers are valid
host/device `constexpr` functions, so plain `nvcc` does not require
`--expt-relaxed-constexpr`.

`run_remote.sh` regenerates six representative variants, compiles each in its
own translation unit, runs the correctness gate and benchmark, and captures
the finalist's resource report plus gzipped SASS. Every remote compile/run is
time-bounded and serialized through `/tmp/sha1-gpu.lock`:

```bash
experiments/shared/run_remote.sh
```

The workload can be changed without editing the script:

```bash
OUTER_COUNT=4194304 LAUNCHES=4 SAMPLES=9 \
  experiments/shared/run_remote.sh
```

## RTX 4060 result

The survey ran on an NVIDIA GeForce RTX 4060 (compute capability 8.9) with
CUDA 13.3.1. Each timed launch at the confirmation setting covers
`2^20 * 256 = 268,435,456` exact final-block hashes. A sample contains eight
launches; the reported rate is the median of 15 samples after eight warm-ups.

The best standard confirmation result is:

```text
RESULT variant=struct-g2-v8-b256-ca-s65
       median_ghs=10.996204 min_ghs=10.844216 max_ghs=11.089530
       regs=96 local=0 dynamic_smem=33280 block=256
```

A second run used `2^22` outer words, so every launch covered 1,073,741,824
hashes and every timed sample covered 4.29 billion. B256 again won, at a more
conservative sustained median of **10.933543 GH/s** (10.838--11.071 GH/s).
Accordingly, the evidence supports roughly 10.9--11.0 GH/s rather than treating
the 11.10 GH/s survey peak as the result.

The controlled results that drove the choice are:

| Variant | Median GH/s | Registers | Dynamic shared |
|---|---:|---:|---:|
| struct G2/V8/B256/CA/S65 | **10.996** | 96 | 33,280 B |
| struct G2/V8/B64/CA/S65 | 10.962 | 96 | 8,320 B |
| struct G2/V8/B192/CA/S65 | 10.944 | 96 | 24,960 B |
| struct G2/V8/B128/NC/S65 | 10.901 | 96 | 16,640 B |
| struct G4/V8/B128/CA/S65 | 10.890 | 96 | 8,320 B |
| rotating G2/V8/B128/CA/S65 | 10.565 | 96 | 16,640 B |
| struct G4/V4/B96/CA/S65 | 10.482 | 56 | 6,240 B |

All production kernels in that table have zero stack, local memory, spill
stores, and spill loads. Register caps of 88 and 80 did not improve throughput;
72 registers caused spills and reduced it to 10.25 GH/s.

Full evidence is in:

- `results/rtx4060-survey.csv`: 33 independently compiled geometry/form/cache
  variants with seven-sample medians;
- `results/rtx4060-confirmation.csv`: nine finalists with 15-sample medians;
- `results/rtx4060-large-launch.csv`: the four finalist block sizes with
  billion-candidate launches;
- `results/correctness.txt`: representative exact 160-bit digest captures;
- `results/register-caps.txt`: compiler resource and cap observations;
- `results/sass-comparison.txt`: production versus rotating-form opcode counts;
- `results/best-production.resources.txt`, `best-production.sass.gz`, and the
  corresponding `rotating-comparison` files: exact cubin evidence.

The GPU is shared with its physical host, so occasional samples show external
clock/load interference. The larger confirmation median, rather than a single
peak, is the stated result. Rates count all evaluated raw candidates. Candidates
containing a zero byte are hashed but rejected as winners, leaving a usable
fraction of `(255/256)^5`, or about 98.06%.

This harness currently specializes a 32-bit H0 gate; complete 1--160-bit target
handling belongs in the higher-level search system, while the SHA-1 core and
exact digest capture already compute all five words.
