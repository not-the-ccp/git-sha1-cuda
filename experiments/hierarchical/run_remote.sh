#!/usr/bin/env bash
set -euo pipefail

outer_count="${1:-2097152}"
samples="${2:-7}"
if [[ ! "$outer_count" =~ ^[0-9]+$ || ! "$samples" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [reference_outer_count_multiple_of_65536] [median_samples]" >&2
  exit 2
fi
if (( outer_count < 65536 || outer_count % 65536 != 0 || samples < 1 || samples % 2 == 0 )); then
  echo "outer count must be a positive multiple of 65536; samples must be positive and odd" >&2
  exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
key=/home/r34/.ssh/cuda_sandbox_ed25519
known=/home/r34/.ssh/cuda_sandbox_known_hosts
host=agent@192.168.178.76
ssh_opts=(-i "$key" -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$known")
scp_opts=(-i "$key" -P 2222 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$known")

timeout 30 python3 "$here/generate_benchmark.py"
timeout 60 scp "${scp_opts[@]}" "$here/hierarchical_bench.cu" "$host:/work/hierarchical_bench.cu"
timeout 240 flock -w 220 /tmp/sha1-gpu.lock \
  ssh "${ssh_opts[@]}" "$host" \
  'cd /work && timeout 210 nvcc -O3 -std=c++17 -arch=sm_89 hierarchical_bench.cu -o hierarchical_bench'
timeout 240 flock -w 220 /tmp/sha1-gpu.lock \
  ssh "${ssh_opts[@]}" "$host" \
  "cd /work && timeout 210 ./hierarchical_bench '$outer_count' '$samples'"
