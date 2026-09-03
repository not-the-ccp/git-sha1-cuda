# git-sha1-cuda

`git-sha1-cuda` creates unsigned Git commits with a chosen leading SHA-1
prefix. Commit candidates are searched on an NVIDIA GPU, verified on the CPU,
written to the object database, and installed as the repository's new `HEAD`.

The project also provides the search engine as a C library and a safe Rust
crate for applications that prepare their own commit objects.

## Install

The x86_64 Linux archive on the
[releases page](https://github.com/not-the-ccp/git-sha1-cuda/releases) contains
the CLI and its native library. Extract it and copy both directories into a
prefix such as `/usr/local`:

```bash
tar -xf git-sha1-cuda-v0.1.0-x86_64-linux.tar.xz
sudo cp -r git-sha1-cuda-v0.1.0-x86_64-linux/bin \
  git-sha1-cuda-v0.1.0-x86_64-linux/lib /usr/local/
```

The release requires an x86_64 Linux system, an NVIDIA GPU, and a compatible
NVIDIA driver.

## Create a commit

Stage the desired contents and run:

```bash
git add src/main.rs
git-sha1-cuda commit --prefix 0000000 -m "Implement the parser"
```

`--prefix` accepts one to ten hexadecimal digits. Longer prefixes take
exponentially more work. Multiple `-m` options create separate message
paragraphs, and `--device N` selects a CUDA device.

The command uses:

- the tree represented by the current Git index;
- the current `HEAD` as its single parent, when present;
- author and committer identities resolved by `git var`;
- the time at which the command starts;
- the first available matching candidate from the GPU search.

The nonce is stored in a custom `x` commit header. It does not appear in the
commit subject or body, although it remains available in the raw commit object
shown by `git cat-file commit <id>`.

Version 0.1 supports SHA-1 repositories, one-parent commits, `-m` messages, and
the current index. Git commit hooks are outside this workflow. An unchanged
index is rejected after the first commit.

## Search time

Measurements on a GeForce RTX 4060 with CUDA 13.3 reached approximately
5.4 billion candidates per second for the custom-header layout used by the
CLI. Header-safe candidate filtering is included in the estimates below.

| Leading hex digits | Average time | 95% complete by |
|---:|---:|---:|
| 6 | 3.2 ms | 9.5 ms |
| 7 | 52 ms | 0.16 s |
| 8 | 0.83 s | 2.5 s |
| 9 | 13 s | 40 s |
| 10 | 3 min 32 s | 10 min 36 s |

Search time follows a geometric distribution. The median is about 69% of the
average, so individual runs vary considerably.

The message-trailer layout available through the library searches about
10.9 billion candidates per second because its mutable block is the final
SHA-1 block. Its nonce is part of the commit message. The custom-header layout
typically evaluates one additional fixed block per candidate to keep the nonce
outside the message.

## Build from source

Requirements:

- CUDA Toolkit 12 or newer;
- CMake 3.24 or newer;
- Rust 1.82 or newer;
- Git.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

GSV_LIB_DIR="$PWD/build" \
  cargo build --release --manifest-path rust/git-sha1-cuda/Cargo.toml
```

The CLI binary is written to
`rust/git-sha1-cuda/target/release/git-sha1-cuda`. The native build produces
`build/libgit_sha1_cuda.so` and `build/libgit_sha1_cuda_static.a`.

CMake targets the local GPU architecture by default. Set
`CMAKE_CUDA_ARCHITECTURES` for a distributable or cross-compiled build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

## Rust API

The dependency-free crate under
[`rust/git-sha1-cuda`](rust/git-sha1-cuda) prepares complete Git commit jobs,
verifies results with an independent SHA-1 implementation, and exposes the
CUDA context through safe Rust types.

```rust,no_run
use git_sha1_cuda::{GitJob, TargetPrefix};

# fn search(commit_payload: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
let target = TargetPrefix::from_hex("01234567")?;
let prepared = GitJob::header(commit_payload, target)?;
let mut context = prepared.create_context(0)?;

for outer_base in (0..1_u64 << 32).step_by(1 << 22) {
    let result = context.search(outer_base, 1 << 22)?;
    if let Some(candidate) = result.candidate {
        prepared.verify_candidate(candidate)?;
        let payload = prepared.materialize_payload(candidate)?;
        std::fs::write("commit-payload.bin", payload)?;
        break;
    }
}
# Ok(())
# }
```

`GSV_LIB_DIR` selects the directory containing `libgit_sha1_cuda.so` while
building the crate.

## C API

[`include/git_sha1_cuda.h`](include/git_sha1_cuda.h) declares the stable C ABI.
Its reusable contexts support target prefixes from 1 to 160 bits, bounded
search batches, full digest capture, runtime job replacement, custom-header
suffix blocks, and selectable nonce-byte policies.

An outer batch of `2^22` covers 1,073,741,824 candidates. This takes about
0.20 seconds for a typical custom-header job on the measured RTX 4060, keeping
launch and transfer overhead small while allowing regular progress updates.

The Python job generator provides an independent preparation and verification
path:

```bash
python3 tools/git_sha1_job.py commit-payload.bin \
  --target 0123456789 \
  --carrier header \
  --output job
```

Benchmark generators and captured results are stored under
[`experiments`](experiments).

## License

[Mozilla Public License 2.0](LICENSE)
