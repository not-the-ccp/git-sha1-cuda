# Hierarchical late-dependency SHA-1 experiment

## Outcome

The proposed 3+1+1 hierarchy is correct, but it is not the fastest mapping on
the RTX 4060.  Its best confirmation result was **10.3739 GH/s**, versus
**10.8778 GH/s** for the existing raw 4+1 mapping (4.6% slower).  A flat
control using exactly the same bytes as 3+1+1 reached 10.9190 GH/s, within
0.4% of the reference.  This isolates the loss to the hierarchy itself rather
than the shifted byte positions.

The W13-only 24-bit epoch control reached 10.7243 GH/s.  Saving round 12 did
not repay the extra launch/epoch machinery, and it remained 1.4% below the
4+1 baseline before charging any epoch-update cost.

These kernels are experiments, not production search kernels.  The 4+1
reference has the same candidate contract as `tools/git_sha1_job.py`; the
other layouts would require new host job contracts and are deliberately not
integrated because they lose.

## Layouts and mapping

All byte offsets are within the final padded SHA-1 block.  Candidate integers
are big-endian byte strings.

| Kernel | Mutable offsets | Decomposition | Candidates per group setup |
|---|---:|---|---:|
| `baseline` | 48..52 | W12 outer32 + W13 inner8 | 256 |
| `hierarchical5` | 49..53 | W12 outer24 + W13 middle8 + inner8 | 65,536 |
| `flat5` | 49..53 | packed (outer24,middle8) + inner8 | 256 |
| `epoch24` | 52..54 | W13 middle16 + inner8 | 256 |

For the W13-only mode, a separate 16-bit epoch would have to alter fixed input
outside the hot final-block nonce and provide an updated chaining state and
pre-round state for each bounded 24-bit pass.  One epoch is 16,777,216 hashes.

The message schedule is linear over XOR.  For the 3+1+1 case:

```text
W[t] = Wbase[t] xor Wouter[t] xor Wmiddle[t] xor Winner[t]
```

The hierarchical kernel makes one lane build the combined outer+middle
schedule in shared memory, then amortizes it over all 256 inner bytes.  Round
13 is split exactly at its modular addition.  Given the state after round 12:

```text
zbase = ROL5(a) + Ch(b,c,d) + e + K0 + W13base + (middle << 24)
a'    = zbase + (inner << 16)                 (mod 2^32)
```

The other four post-round state words are copied from the common state.  The
W13-only kernel uses `(middle << 16)` and `(inner << 8)` in the same identity.
No approximation or carry assumption is involved.

## Correctness construction

`generate_benchmark.py` constructs three exact padded messages, each with
three fixed 64-byte blocks and one final partial block.  It computes:

- the prefix chaining state;
- the state immediately before the first mutable round;
- compact shared-schedule constants;
- affine W13 complex-row tables; and
- independent whole-message `hashlib.sha1` digests.

The CUDA executable rebuilds each 32x256 complex-delta table on the CPU from
the standard SHA-1 recurrence.  Its startup gate checks all five GPU digest
words for all four kernels.  The production-shaped benchmark keeps a runtime
H0 comparison and atomic winner side effect in every candidate path.

## RTX 4060 results

Command:

```sh
experiments/hierarchical/run_remote.sh 2097152 7
```

Each long-running row hashes 536,870,912 candidates per sample.  Epoch rows
measure the median per launch over 16 launches per sample.  CUDA events are
recorded after a warm-up; values below are medians of seven samples from one
run on CUDA 13.3.1, `sm_89`.

| Variant | GH/s | Median ms | Registers | Blocks/SM |
|---|---:|---:|---:|---:|
| raw4+1 G4/V8/B96 | **10.8778** | 49.355 | 96 | 6 |
| raw4+1 G4/V4/B128 | 10.6684 | 50.323 | 64 | 8 |
| hierarchical G8/V4/B128 | **10.3739** | 51.752 | 66 | 7 |
| hierarchical G4/V4/B128 | 10.2723 | 52.264 | 66 | 7 |
| hierarchical G4/V8/B96 | 9.6057 | 55.891 | 96 | 6 |
| flat 3+1+1 G4/V8/B96 | **10.9190** | 49.168 | 96 | 6 |
| flat 3+1+1 G4/V4/B128 | 10.6103 | 50.599 | 64 | 8 |
| W13 epoch G4/V8/B96 | **10.7243** | 1.564 | 96 | 6 |

The 24-bit epoch launch is about 1.56 ms.  A full 16-bit outer epoch sweep
would therefore add roughly 65,536 launches plus 65,536 fixed-state updates.
Even a hypothetical 10 microseconds of combined overhead per epoch is about
0.66 seconds; the measured hot hashing alone is already slower than 4+1.

## Resource and SASS evidence

`cuobjdump --dump-resource-usage` reports no local memory for any measured
kernel.  For the directly comparable G4/V4/B128 forms, the hierarchy needs 66
registers rather than 64, reducing residency from eight to seven blocks/SM.
The flat control and W13 epoch return to 64 registers and eight blocks/SM.

Static G4/V4 SASS counts are close (baseline 2,088 instructions, hierarchy
2,104, flat 2,096, epoch 2,088), but that hides the important dynamic
difference: the hierarchy's one `WARPSYNC` instruction and its 64 shared
schedule stores execute inside the 256-iteration middle loop.  Baseline, flat,
and epoch execute their schedule setup and warp synchronization once per group.
The hierarchy therefore saves one SHA-1 round per candidate while repeatedly
paying schedule rebuild, synchronization, loop bookkeeping, and lower
occupancy.  On this GPU that trade is negative.

## Reproduction

`run_remote.sh` regenerates the CUDA source, uploads it to the configured CUDA
sandbox, compiles for `sm_89`, runs correctness first, and then benchmarks.  It
wraps remote compilation and execution in `/tmp/sha1-gpu.lock` and explicit
timeouts as required by `/home/r34/CUDA-AGENTS.md`.

The first argument is the reference 4+1 outer count and must be a multiple of
65,536.  The hierarchy uses `outer_count / 256`, so every timed long row covers
the same candidate count.  The second argument is the odd median sample count.

Conclusion: keep the raw 4+1 offsets-48..52 contract.  The flat offsets-49..53
mapping is performance-equivalent but would require a needless host-contract
change, while the true hierarchy and W13 epoch add complexity without
throughput benefit.
