# git-sha1-cuda

This repository provides an embeddable CUDA backend for Git commit SHA-1
prefix searches. It is a library, not a replacement CLI: an existing Git
vanity tool can prepare the commit object, pass the final-block job to this
backend, checkpoint bounded searches, and materialize the returned nonce.

The optimized path places one raw five-byte candidate at byte offsets 48..52
of the final padded SHA-1 block:

```text
candidate 0xAABBCCDDEE -> bytes AA BB CC DD EE
W12 = 0xAABBCCDD
W13 high byte = 0xEE
```

Only the final compression block varies. The production kernel shares the
W12-dependent schedule between pairs of lanes and evaluates all 256 possible
W13 bytes with a 32 KiB delta table. It supports target prefixes from 1 to 160
bits and rejects candidates containing NUL before publishing a winner.

## Build

CUDA 12+ and CMake 3.24+ are sufficient:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This produces `build/libgit_sha1_cuda.so` and, by default,
`build/libgit_sha1_cuda_static.a`. Set `CMAKE_CUDA_ARCHITECTURES` explicitly
when cross-compiling; otherwise CMake targets the local GPU.

The ABI is declared in [`include/git_sha1_cuda.h`](include/git_sha1_cuda.h).
The central calls are:

- `gsv_job_init`: validate the final-block layout and derive masks/prestate;
- `gsv_context_create`: allocate the per-device table and scratch buffers;
- `gsv_search`: search `outer_count * 256` candidates from `outer_base << 8`;
- `gsv_digest`: compute all five digest words for one candidate;
- `gsv_context_set_job`: reuse allocations when the CLI changes its target.

`gsv_search` is deliberately bounded. For example, an outer count of `2^22`
is 1,073,741,824 hashes and takes about 0.10 seconds on the tested RTX 4060.
This makes cancellation, progress reporting, and durable checkpoints simple.
The returned winner is valid but is not guaranteed to be the numerically
smallest matching candidate in the range.

## Rust

[`rust/git-sha1-cuda`](rust/git-sha1-cuda) is a dependency-free safe wrapper.
An existing Cargo project can use it directly:

```toml
[dependencies]
git-sha1-cuda = { path = "/home/r34/sha1/rust/git-sha1-cuda" }
```

Build the CUDA library first. If it is not in `/home/r34/sha1/build`, set
`GSV_LIB_DIR` while building the Rust application and ensure the dynamic
loader can find the same directory at runtime.

```rust,no_run
use git_sha1_cuda::{Context, Job};

# fn run(prestate: [u32; 5], final_block_words: [u32; 16], target: [u32; 5])
#     -> Result<(), Box<dyn std::error::Error>> {
let job = Job::new(&prestate, &final_block_words, 40, &target)?;
let mut gpu = Context::new(0, &job)?;

let batch = gpu.search(0, 1 << 22)?;
if let Some(candidate) = batch.candidate {
    let digest = gpu.digest(candidate)?;
    println!("candidate={candidate:010x} digest={digest:08x?}");
}
# Ok(())
# }
```

The `prestate` is the SHA-1 state after every fixed block before the final
block. `final_block_words` uses SHA-1's big-endian word order and must have W12
and W13's high byte cleared. `target` contains the requested digest bits
left-aligned across five words.

## Preparing Git jobs

[`tools/git_sha1_job.py`](tools/git_sha1_job.py) is the reference job builder
and independent CPU oracle. Given a raw commit payload, it aligns the nonce,
writes binary templates plus a JSON manifest, and verifies Git serialization:

```bash
python3 tools/git_sha1_job.py commit-payload.bin \
  --target 0123456789 --output job
```

The existing CLI can implement the same small preparation step directly, or
read `prestate`, `base_words`, and aligned target words from `job/job.json`.
After a winner, replace the five placeholder bytes with
`candidate.to_be_bytes()[3..]`, hash the full serialized Git object once on
the CPU, then write it through the CLI's normal Git workflow.

GPG-signed commits are intentionally deferred: a signature makes the long
armored suffix candidate-dependent and does not fit this one-variable-block
kernel.

## Measured performance

On an RTX 4060 with CUDA 13.3, the reusable runtime-job ABI sustains about
**10.95 GH/s** for targets up to 32 bits and **10.79 GH/s** for longer targets
on billion-candidate launches. The hot kernel uses 96 registers with zero
local memory and zero spills. This is within roughly 2% of the
compile-specialized research kernel.

The experiments and reproducible benchmark evidence are under
[`experiments/shared`](experiments/shared).
