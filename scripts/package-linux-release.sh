#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-"${repository}/dist"}
if [[ -n "$(git -C "${repository}" status --porcelain --untracked-files=normal)" ]]; then
  echo "release builds require a clean Git worktree" >&2
  exit 1
fi
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' \
  "${repository}/rust/git-sha1-cuda/Cargo.toml" | head -1)
cmake_version=$(sed -n 's/^project(git_sha1_cuda VERSION \([^ ]*\).*/\1/p' \
  "${repository}/CMakeLists.txt")

if [[ -z "${version}" || "${version}" != "${cmake_version}" ]]; then
  echo "Cargo and CMake project versions must match" >&2
  exit 1
fi

archive="git-sha1-cuda-v${version}-x86_64-linux.tar.xz"
if [[ -e "${output}/${archive}" || -e "${output}/${archive}.sha256" ]]; then
  echo "release output already exists: ${output}/${archive}" >&2
  exit 1
fi

mkdir -p "${output}"
source_date_epoch=$(git -C "${repository}" log -1 --format=%ct)
timeout "${GSV_RELEASE_TIMEOUT:-3600}" docker buildx build \
  --file "${repository}/packaging/Dockerfile.x86_64-linux" \
  --build-arg "GSV_VERSION=${version}" \
  --build-arg "SOURCE_DATE_EPOCH=${source_date_epoch}" \
  --output "type=local,dest=${output}" \
  "${repository}"

(cd "${output}" && sha256sum --check "${archive}.sha256")
printf '%s\n' "${output}/${archive}" "${output}/${archive}.sha256"
