# git-sha1-cuda

`git-sha1-cuda` is a CUDA implementation of SHA-1 prefix search for Git commit
objects. It prepares complete commit objects in Rust or Python and evaluates a
40-bit candidate space through a C ABI, with bounded launches for progress
reporting and checkpointing.

The production kernel supports:

- SHA-1 target prefixes from 1 to 160 bits;
- reusable CUDA contexts and runtime job changes;
- exact five-word digest capture for candidate verification;
- message-trailer and custom-header nonce carriers;
- shared and static native libraries;
- deterministic Git job generation with an independent CPU SHA-1 oracle.

## Search layouts

Each candidate is a five-byte big-endian value at byte offsets 48 through 52
of a mutable SHA-1 block:

```text
candidate 0xAABBCCDDEE -> bytes AA BB CC DD EE

final block W12        = 0xAABBCCDD
final block W13[31:24] = 0xEE
```

All earlier Git object blocks are fixed and compressed once on the CPU. The
resulting SHA-1 state becomes the GPU job's `prestate`. The mutable block stores
zeroes in W12 and the high byte of W13 until a candidate is materialized.

The kernel computes each W12-dependent message schedule once per lane pair.
The pair evaluates all 256 W13 bytes using direct rotations and a 32 KiB table
for multi-term schedule deltas.

Two carriers use this layout:

| Carrier | Commit location | Ordinary Git/GitHub message view | SHA-1 work per candidate | RTX 4060 |
|---|---|---|---:|---:|
| Message trailer | Appended to the message | Visible in the full message | 1 block | 10.90 GH/s |
| Custom header | Immediately before the header/message separator | Hidden from subject and body views | 2 blocks for a typical short message | 5.4 GH/s |

The custom header remains visible in the raw commit object through commands
such as `git cat-file commit`. Git preserves it as an ordinary unknown header,
and `git fsck --strict` accepts the resulting object. Header candidates exclude
NUL and LF bytes; trailer candidates exclude NUL bytes.

Winner publication has three selectable byte policies:

| Policy | Allowed candidate bytes | Eligible 5-byte candidates |
|---|---|---:|
| `NoNul` | Every byte except NUL | 98.06% |
| `HeaderSafe` | Every byte except NUL and LF | 96.15% |
| `PrintableAscii` | ASCII `0x20` through `0x7e` | 0.704% |

The policy changes which matching candidates are returned and does not change
raw kernel throughput. Printable ASCII is useful when raw-object readability
matters; an eight-digit target averages 56 seconds with the trailer layout or
1 minute 53 seconds with a one-suffix-block header.

## Performance

Measurements on an NVIDIA GeForce RTX 4060 with CUDA 13.3:

| Target width | Throughput |
|---|---:|
| 1–32 bits | 10.95 GH/s |
| 33–160 bits | 10.79 GH/s |

The final-block kernel uses 96 registers, 33,280 bytes of dynamic shared
memory, and no local memory or register spills. The benchmark uses
billion-candidate launches after warm-up. Fixed suffix blocks add one SHA-1
compression per block:

| Fixed suffix blocks | Throughput |
|---:|---:|
| 0 | 10.7 GH/s |
| 1 | 5.4 GH/s |
| 2 | 3.6 GH/s |
| 4 | 2.14 GH/s |
| 8 | 1.18 GH/s |

### Expected search time

The following averages use 10.95 GH/s for the message trailer and 5.4 GH/s
for a custom header with one suffix block. They include each carrier's rejected
candidate bytes.

| Leading hex digits | Candidate bits | Message trailer | Custom header |
|---:|---:|---:|---:|
| 7 | 28 | 25 ms | 52 ms |
| 8 | 32 | 0.40 s | 0.83 s |
| 9 | 36 | 6.4 s | 13.2 s |
| 10 | 40 | 1 min 42 s | 3 min 32 s |
| 11 | 44 | 27 min 18 s | 56 min 28 s |
| 12 | 48 | 7 h 17 min | 15 h 03 min |

A single job contains 40 candidate bits. Searches wider than ten hexadecimal
digits require fresh templates carrying additional fixed epoch bits. The table
assumes epochs continue until a match is found. Search time follows a geometric
distribution: 50% of searches finish within 0.693 times the average and 95%
within 3.00 times the average.

`GitJob::header_epoch` encodes a fixed epoch before the candidate while keeping
the custom-header layout aligned. Applications can reuse a CUDA context across
epochs with `GitJob::configure_context_with_policy`.

Reproducible benchmark generators, resource reports, correctness captures,
and SASS summaries are available in [`experiments/shared`](experiments/shared).
Custom-header measurements and reproduction commands are in
[`experiments/header`](experiments/header).

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
| `gsv_job_init` | Validate a mutable block and derive masks and round-12 state |
| `gsv_context_create` | Allocate device tables, events, and result buffers |
| `gsv_context_set_job` | Reuse a context with another job |
| `gsv_context_set_header_job` | Configure a mutable block followed by fixed suffix blocks |
| `gsv_context_set_nonce_policy` | Select NUL-free, header-safe, or printable winners |
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
dependency-free preparation, CPU verification, and safe CUDA bindings.

```toml
[dependencies]
git-sha1-cuda = { path = "../git-sha1-cuda/rust/git-sha1-cuda" }
```

The build script searches for the native library in the repository's `build`
directory. `GSV_LIB_DIR` selects a different directory. The platform's dynamic
loader must also be able to locate `libgit_sha1_cuda.so` at runtime.

```rust,no_run
use git_sha1_cuda::{GitJob, TargetPrefix};

# fn search(commit_payload: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
let target = TargetPrefix::from_hex("01234567")?;
let prepared = GitJob::header(commit_payload, target)?;
let mut context = prepared.create_context(0)?;

for outer_base in (0..1_u64 << 32).step_by(1 << 22) {
    let result = context.search(outer_base, 1 << 22)?;
    if let Some(candidate) = result.candidate {
        let digest = prepared.verify_candidate(candidate)?;
        let finished_payload = prepared.materialize_payload(candidate)?;
        println!("candidate={candidate:010x} digest={digest:02x?}");
        std::fs::write("commit-payload.bin", finished_payload)?;
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
  --carrier header \
  --output job
```

The output directory contains:

| File | Contents |
|---|---|
| `job.json` | Layout, target, prestate, masks, and candidate offsets |
| `payload-template.bin` | Commit payload with the five-byte placeholder |
| `object-template.bin` | Serialized Git object template |
| `mutable-block-template.bin` | Padded block containing the custom-header nonce |
| `suffix-blocks.bin` | Fixed padded blocks following the mutable block |

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
