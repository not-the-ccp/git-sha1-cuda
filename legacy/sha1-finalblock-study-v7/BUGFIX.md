# Round 5.1 repair

The Round-5 campaign stopped at `vec8-g1-ca-p0-b192` for `NONCE_OFF=0`.

That failure point is highly diagnostic: the preceding G1/V8 CA launches at B64, B96 and B128 instantiate the same device kernel; only block geometry/dynamic shared memory differs. At offset 0 the compact schedule needs 80 words per group, so G1 consumes:

- B64: 20,480 bytes
- B96: 30,720 bytes
- B128: 40,960 bytes
- B192: 61,440 bytes
- B256: 81,920 bytes

On compute capability 8.9, >48 KiB/block dynamic shared-memory launches require explicit opt-in. These huge G1 blocks are also poor experiments for this kernel because the schedule footprint would force very low block residency. V5.1 therefore removes them instead of opting them in just to collect a predictably bad point.

## Hardening

- generated generic variants are <=48 KiB dynamic shared memory even at worst-case `FIRST_WORD=0`;
- runtime independently checks `cudaDeviceProp::sharedMemPerBlock` and emits `SKIP_RESOURCE` before launch;
- immediate `cudaGetLastError()` checks distinguish launch/configuration failures from hash failures;
- hash mismatches are named `FAIL_HASH` and never benchmarked;
- correctness probes inner nonce values 0, 37 and 63 for every variant, not just one candidate;
- a bad experimental variant is quarantined per offset so the rest of a long survey still produces useful results;
- unexpected harness/runtime failures still fail loudly, but partial results are archived automatically.

## Campaign

The research campaign surveys 230 resource-sane variants over 14 representative nonce offsets, repeatedly confirms the top 24 without recompilation, then compiles only the top six across all 49 offsets. Finally it tests register caps on the top three at offsets 16, 32 and 48. This keeps the breadth where it can change the answer while avoiding repeated compilation of hundreds of losing kernels in the full-coverage stage.
