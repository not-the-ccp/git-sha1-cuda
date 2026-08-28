# git-sha1-cuda

`git-sha1-cuda` is a CUDA implementation of SHA-1 prefix search for Git commit
objects. It evaluates a 40-bit candidate space through a C ABI and a safe Rust
wrapper, with bounded launches for progress reporting and checkpointing.

The production kernel supports:

- SHA-1 target prefixes from 1 to 160 bits;
- reusable CUDA contexts and runtime job changes;
- exact five-word digest capture for candidate verification;
- shared and static native libraries;
- deterministic Git job generation with an independent CPU SHA-1 oracle.

## Candidate layout

Each candidate is a five-byte big-endian value at byte offsets 48 through 52
of the final padded SHA-1 block:

```text
candidate 0xAABBCCDDEE -> bytes AA BB CC DD EE

final block W12        = 0xAABBCCDD
final block W13[31:24] = 0xEE
```

All earlier Git object blocks are fixed and compressed once on the CPU. The
resulting SHA-1 state becomes the GPU job's `prestate`. The final block stores
zeroes in W12 and the high byte of W13 until a candidate is materialized.

The kernel computes each W12-dependent message schedule once per lane pair.
The pair evaluates all 256 W13 bytes using direct rotations and a 32 KiB table
for multi-term schedule deltas. Candidate bytes containing NUL are hashed but
excluded from winner publication.

## Performance

Measurements on an NVIDIA GeForce RTX 4060 with CUDA 13.3:

| Target width | Throughput |
|---|---:|
| 1–32 bits | 10.95 GH/s |
| 33–160 bits | 10.79 GH/s |

The production kernel uses 96 registers, 33,280 bytes of dynamic shared
memory, and no local memory or register spills. The benchmark uses
billion-candidate launches after warm-up.

Reproducible benchmark generators, resource reports, correctness captures,
and SASS summaries are available in [`experiments/shared`](experiments/shared).

## Build

Requirements:

- CUDA Toolkit 12 or newer;
- CMake 3.24 or newer;
- a C++17 compiler supported by the installed CUDA Toolkit.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build produces:

```text
build/libgit_sha1_cuda.so
build/libgit_sha1_cuda_static.a
```

CMake targets the local GPU architecture by default. Cross-builds can set it
explicitly, for example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

## C API

The public ABI is declared in
[`include/git_sha1_cuda.h`](include/git_sha1_cuda.h).

| Function | Purpose |
|---|---|
| `gsv_job_init` | Validate a final-block job and derive masks and round-12 state |
| `gsv_context_create` | Allocate device tables, events, and result buffers |
| `gsv_context_set_job` | Reuse a context with another job |
| `gsv_search` | Search `outer_count * 256` candidates from `outer_base << 8` |
| `gsv_digest` | Compute the complete SHA-1 digest for one candidate |
| `gsv_context_destroy` | Release context resources |

`gsv_search` reports the number of evaluated candidates, CUDA event time, and
throughput. Winner selection uses a device atomic and may return any matching
candidate in the requested batch.

An outer count of `2^22` evaluates 1,073,741,824 candidates in about 0.10
seconds on the measured RTX 4060. Applications can choose batch sizes around
their desired cancellation and checkpoint latency.

## Rust

The crate under [`rust/git-sha1-cuda`](rust/git-sha1-cuda) provides
dependency-free safe bindings.

```toml
[dependencies]
git-sha1-cuda = { path = "../git-sha1-cuda/rust/git-sha1-cuda" }
```

The build script searches for the native library in the repository's `build`
directory. `GSV_LIB_DIR` selects a different directory. The platform's dynamic
loader must also be able to locate `libgit_sha1_cuda.so` at runtime.

```rust,no_run
use git_sha1_cuda::{Context, Job};

# fn search(
#     prestate: [u32; 5],
#     final_block_words: [u32; 16],
#     target_words: [u32; 5],
# ) -> Result<(), Box<dyn std::error::Error>> {
let job = Job::new(&prestate, &final_block_words, 40, &target_words)?;
let mut context = Context::new(0, &job)?;

for outer_base in (0..1_u64 << 32).step_by(1 << 22) {
    let result = context.search(outer_base, 1 << 22)?;
    if let Some(candidate) = result.candidate {
        let digest = context.digest(candidate)?;
        println!("candidate={candidate:010x} digest={digest:08x?}");
        break;
    }
}
# Ok(())
# }
```

## Git job generation

[`tools/git_sha1_job.py`](tools/git_sha1_job.py) builds the aligned commit job
and verifies Git object serialization independently of the CUDA code.

```bash
python3 tools/git_sha1_job.py commit-payload.bin \
  --target 0123456789 \
  --output job
```

The output directory contains:

| File | Contents |
|---|---|
| `job.json` | Layout, target, prestate, masks, and candidate offsets |
| `payload-template.bin` | Commit payload with the five-byte placeholder |
| `object-template.bin` | Serialized Git object template |
| `final-block-template.bin` | Padded final SHA-1 block |
| `job_constants.cuh` | Generated CUDA constants for experiments |

After a successful search, the five-byte big-endian candidate replaces the
placeholder. The CPU oracle hashes the complete serialized object and checks
the requested prefix before the object is stored.

## Correctness

The test suite covers:

- Git object framing against `git hash-object`;
- independent SHA-1 compression across padding boundaries;
- exact GPU/CPU digest agreement;
- target masks and gates from 1 through 160 bits;
- candidate byte order and W12/W13 mapping;
- C ABI search and digest capture on a CUDA device.

Run the host tests directly with:

```bash
python3 -m unittest -v
```

## License

[Mozilla Public License 2.0](LICENSE)
