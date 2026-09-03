#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ARCHIVE" >&2
  exit 2
fi

archive=$(realpath "$1")
workspace=$(mktemp -d "${TMPDIR:-/tmp}/git-sha1-cuda-smoke.XXXXXX")
trap 'rm -rf "${workspace}"' EXIT
timeout 60 tar -xf "${archive}" -C "${workspace}"

mapfile -t binaries < <(find "${workspace}" -mindepth 3 -maxdepth 3 \
  -type f -path '*/bin/git-sha1-cuda')
if [[ ${#binaries[@]} -ne 1 ]]; then
  echo "archive must contain one git-sha1-cuda package" >&2
  exit 1
fi
binary=${binaries[0]}
package=${binary%/bin/git-sha1-cuda}

timeout 30 "${binary}" --version
timeout 30 "${binary}" devices
test -f "${package}/include/git_sha1_cuda.h"
test -f "${package}/lib/libgit_sha1_cuda.so"
test -f "${package}/lib/cmake/git_sha1_cuda/git_sha1_cudaConfig.cmake"
test -f "${package}/LICENSE"
test -f "${package}/README.md"

repository=${workspace}/repository
git -C "${workspace}" init -q "${repository}"
git -C "${repository}" config user.name "Release Test"
git -C "${repository}" config user.email "release-test@localhost"

(
  cd "${repository}"
  timeout 60 "${binary}" commit --prefix 000000 -m "Header carrier smoke test"
  timeout 60 "${binary}" commit --allow-empty --carrier trailer \
    --prefix 111111 -m "Trailer carrier smoke test"
)

test "$(git -C "${repository}" rev-parse HEAD)" = \
  "$(git -C "${repository}" rev-parse '111111^{commit}')"
test "$(git -C "${repository}" rev-parse HEAD^)" = \
  "$(git -C "${repository}" rev-parse '000000^{commit}')"
timeout 30 git -C "${repository}" fsck --strict
echo "release smoke test: OK"
