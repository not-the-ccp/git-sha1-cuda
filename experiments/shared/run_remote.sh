#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir="$here/build"
lock_file=/tmp/sha1-gpu.lock
key=/home/r34/.ssh/cuda_sandbox_ed25519
known_hosts=/home/r34/.ssh/cuda_sandbox_known_hosts
remote=agent@192.168.178.76
remote_dir=/work/shared-schedule
outer_count=${OUTER_COUNT:-1048576}
launches=${LAUNCHES:-8}
samples=${SAMPLES:-15}

ssh_opts=(-i "$key" -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$known_hosts")
scp_opts=(-i "$key" -P 2222 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$known_hosts")

mkdir -p "$build_dir"

# One translation unit per line: form G V block cache stride.  This avoids the
# compiler/resource cross-talk caused by a large all-variants executable.
variants=(
  "struct 4 4 96 ca 65"
  "struct 2 8 64 ca 65"
  "struct 2 8 128 ca 65"
  "struct 2 8 256 ca 65"
  "struct 2 8 128 nc 65"
  "rotating 2 8 128 ca 65"
)

timeout 60 flock -w 55 "$lock_file" \
  ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_dir'"

for spec in "${variants[@]}"; do
  read -r form group vector block cache stride <<<"$spec"
  name="${form}-g${group}-v${vector}-b${block}-${cache}-s${stride}"
  source_file="$build_dir/$name.cu"
  binary="$name.bin"
  timeout 30 python3 "$here/generate.py" \
    --round-form "$form" --g "$group" --v "$vector" --block "$block" \
    --cache "$cache" --stride "$stride" --out "$source_file"
  timeout 60 flock -w 55 "$lock_file" \
    scp "${scp_opts[@]}" "$source_file" "$remote:$remote_dir/$name.cu"
  timeout 240 flock -w 235 "$lock_file" \
    ssh "${ssh_opts[@]}" "$remote" \
      "cd '$remote_dir' && timeout 150 nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo -Xptxas=-v '$name.cu' -o '$binary' 2>&1 && timeout 75 './$binary' '$outer_count' '$launches' '$samples'"
done

# Capture auditable machine code and resource metadata for the production
# finalist. cuobjdump is used because nvdisasm 13.3 rejects this line-info ELF.
best=struct-g2-v8-b256-ca-s65
rotating=rotating-g2-v8-b128-ca-s65
timeout 120 flock -w 115 "$lock_file" \
  ssh "${ssh_opts[@]}" "$remote" \
    "cd '$remote_dir' && timeout 30 cuobjdump --dump-resource-usage '$best.bin' > best-production.resources.txt && timeout 45 cuobjdump --dump-sass '$best.bin' | gzip -9 > best-production.sass.gz && timeout 30 cuobjdump --dump-resource-usage '$rotating.bin' > rotating-comparison.resources.txt && timeout 45 cuobjdump --dump-sass '$rotating.bin' | gzip -9 > rotating-comparison.sass.gz"
timeout 60 flock -w 55 "$lock_file" \
  scp "${scp_opts[@]}" "$remote:$remote_dir/best-production.resources.txt" "$remote:$remote_dir/best-production.sass.gz" "$remote:$remote_dir/rotating-comparison.resources.txt" "$remote:$remote_dir/rotating-comparison.sass.gz" "$build_dir/"

printf 'SASS and resources: %s\n' "$build_dir"
