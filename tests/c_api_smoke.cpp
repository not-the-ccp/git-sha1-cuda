#include "git_sha1_cuda.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr uint32_t PRESTATE[5] = {
    0xa5ff205eu, 0xabec3215u, 0xb70842d4u, 0x7a006bdbu, 0xf23f547bu};
static constexpr uint32_t BASE_WORDS[16] = {
    0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu,
    0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu,
    0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu, 0x5f5f5f5fu,
    0x00000000u, 0x000a8000u, 0x00000000u, 0x000003b0u};
static constexpr uint32_t EXPECTED[5] = {
    0xc23c85a4u, 0xc8d6a511u, 0x58ff2395u, 0x4bc4f78fu, 0xce4b14e1u};
static constexpr uint64_t CANDIDATE = 0x0102030405ull;

static constexpr uint32_t HEADER_PRESTATE[5] = {
    0xa907a856u, 0xaa8f6d0bu, 0x852a9528u, 0x4a16b8ccu, 0x7985d9d1u};
static constexpr uint32_t HEADER_BASE_WORDS[16] = {
    0x2b303030u, 0x300a7820u, 0x20202020u, 0x20202020u,
    0x20202020u, 0x20202020u, 0x20202020u, 0x20202020u,
    0x20202020u, 0x20202020u, 0x20202020u, 0x20202020u,
    0x00000000u, 0x000a0a76u, 0x69736962u, 0x6c652073u};
static constexpr uint32_t HEADER_SUFFIX_WORDS[16] = {
    0x75626a65u, 0x63740a80u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000638u};
static constexpr uint32_t HEADER_EXPECTED[5] = {
    0xfb542602u, 0xa40a02bbu, 0x2fe6613au, 0x1ca13d74u, 0x89d83246u};
static constexpr uint64_t HEADER_CANDIDATE = 0x6162636465ull;
static constexpr uint32_t HEADER_UNSAFE_INNER_EXPECTED[5] = {
    0x3bab2187u, 0xc3d2daefu, 0x50ce4134u, 0x70c75bfeu, 0x7192e47fu};
static constexpr uint32_t HEADER_UNSAFE_OUTER_EXPECTED[5] = {
    0x76eb321eu, 0xaa7ee1ccu, 0x24189162u, 0x4c76fa34u, 0x0ad2b9c2u};
static constexpr uint32_t HEADER_CONTROL_EXPECTED[5] = {
    0xea1f27afu, 0x59f8313cu, 0x6e5aafcfu, 0x969ebc02u, 0x455985e6u};

static int fail(const char *what, gsv_status status, const gsv_context *context = nullptr) {
  std::fprintf(stderr, "%s: %s: %s\n", what, gsv_status_string(status), gsv_last_error(context));
  return 1;
}

int main(int argc, char **argv) {
  int32_t devices = gsv_device_count();
  if (devices < 1) {
    std::fprintf(stderr, "no CUDA device: %s\n", gsv_last_error(nullptr));
    return 77;
  }
  gsv_job job{};
  gsv_status status = gsv_job_init(&job, PRESTATE, BASE_WORDS, 160, EXPECTED);
  if (status != GSV_OK) return fail("job init", status);
  gsv_context *context = nullptr;
  status = gsv_context_create(0, &job, &context);
  if (status != GSV_OK) return fail("context create", status);

  uint32_t digest[5]{};
  status = gsv_digest(context, CANDIDATE, digest);
  if (status != GSV_OK) { int rc = fail("digest", status, context); gsv_context_destroy(context); return rc; }
  if (std::memcmp(digest, EXPECTED, sizeof(digest)) != 0) {
    std::fprintf(stderr, "digest mismatch\n");
    gsv_context_destroy(context);
    return 1;
  }

  gsv_search_result result{};
  status = gsv_search(context, CANDIDATE >> 8, 1, &result);
  if (status != GSV_FOUND) { int rc = fail("exact search", status, context); gsv_context_destroy(context); return rc; }
  if (!result.found || result.candidate != CANDIDATE || result.candidates_hashed != 256) {
    std::fprintf(stderr, "unexpected exact-search result\n");
    gsv_context_destroy(context);
    return 1;
  }
  static constexpr uint32_t WIDTHS[] = {1, 31, 32, 33, 64, 159, 160};
  for (uint32_t bits : WIDTHS) {
    status = gsv_job_init(&job, PRESTATE, BASE_WORDS, bits, EXPECTED);
    if (status != GSV_OK) { int rc = fail("prefix job init", status); gsv_context_destroy(context); return rc; }
    status = gsv_context_set_job(context, &job);
    if (status != GSV_OK) { int rc = fail("set prefix job", status, context); gsv_context_destroy(context); return rc; }
    status = gsv_search(context, CANDIDATE >> 8, 1, &result);
    if (status != GSV_FOUND) { int rc = fail("prefix search", status, context); gsv_context_destroy(context); return rc; }
    status = gsv_digest(context, result.candidate, digest);
    if (status != GSV_OK) { int rc = fail("prefix digest", status, context); gsv_context_destroy(context); return rc; }
    for (int word = 0; word < 5; ++word) {
      if ((digest[word] & job.target_masks[word]) != (EXPECTED[word] & job.target_masks[word])) {
        std::fprintf(stderr, "target mismatch at %u bits\n", bits);
        gsv_context_destroy(context);
        return 1;
      }
    }
  }

  status = gsv_job_init(&job, HEADER_PRESTATE, HEADER_BASE_WORDS, 160, HEADER_EXPECTED);
  if (status != GSV_OK) { int rc = fail("header job init", status); gsv_context_destroy(context); return rc; }
  status = gsv_context_set_header_job(context, &job, HEADER_SUFFIX_WORDS, 1);
  if (status != GSV_OK) { int rc = fail("set header job", status, context); gsv_context_destroy(context); return rc; }
  status = gsv_digest(context, HEADER_CANDIDATE, digest);
  if (status != GSV_OK) { int rc = fail("header digest", status, context); gsv_context_destroy(context); return rc; }
  if (std::memcmp(digest, HEADER_EXPECTED, sizeof(digest)) != 0) {
    std::fprintf(stderr, "header digest mismatch\n");
    gsv_context_destroy(context);
    return 1;
  }
  status = gsv_search(context, HEADER_CANDIDATE >> 8, 1, &result);
  if (status != GSV_FOUND || result.candidate != HEADER_CANDIDATE) {
    int rc = fail("header exact search", status, context);
    gsv_context_destroy(context);
    return rc;
  }

  struct UnsafeHeaderCase {
    uint64_t candidate;
    const uint32_t *expected;
  };
  static constexpr UnsafeHeaderCase UNSAFE_HEADER_CASES[] = {
      {0x616263640aull, HEADER_UNSAFE_INNER_EXPECTED},
      {0x610a636465ull, HEADER_UNSAFE_OUTER_EXPECTED},
  };
  for (const auto &unsafe : UNSAFE_HEADER_CASES) {
    status = gsv_job_init(&job, HEADER_PRESTATE, HEADER_BASE_WORDS, 160, unsafe.expected);
    if (status != GSV_OK) { int rc = fail("unsafe header job init", status); gsv_context_destroy(context); return rc; }
    status = gsv_context_set_header_job(context, &job, HEADER_SUFFIX_WORDS, 1);
    if (status != GSV_OK) { int rc = fail("set unsafe header job", status, context); gsv_context_destroy(context); return rc; }
    status = gsv_digest(context, unsafe.candidate, digest);
    if (status != GSV_OK || std::memcmp(digest, unsafe.expected, sizeof(digest)) != 0) {
      int rc = fail("unsafe header digest", status, context);
      gsv_context_destroy(context);
      return rc;
    }
    status = gsv_search(context, unsafe.candidate >> 8, 1, &result);
    if (status != GSV_NOT_FOUND || result.found) {
      std::fprintf(stderr, "unsafe header candidate was published\n");
      gsv_context_destroy(context);
      return 1;
    }
  }
  status = gsv_job_init(&job, HEADER_PRESTATE, HEADER_BASE_WORDS, 160, HEADER_CONTROL_EXPECTED);
  if (status != GSV_OK) { int rc = fail("control-byte job init", status); gsv_context_destroy(context); return rc; }
  status = gsv_context_set_header_job(context, &job, HEADER_SUFFIX_WORDS, 1);
  if (status != GSV_OK) { int rc = fail("set control-byte job", status, context); gsv_context_destroy(context); return rc; }
  status = gsv_search(context, 0x61626364u, 1, &result);
  if (status != GSV_FOUND || result.candidate != 0x6162636401ull) {
    int rc = fail("header-safe control-byte search", status, context);
    gsv_context_destroy(context);
    return rc;
  }
  status = gsv_context_set_nonce_policy(context, GSV_NONCE_PRINTABLE_ASCII);
  if (status != GSV_OK) { int rc = fail("set printable policy", status, context); gsv_context_destroy(context); return rc; }
  status = gsv_search(context, 0x61626364u, 1, &result);
  if (status != GSV_NOT_FOUND || result.found) {
    std::fprintf(stderr, "printable policy published a control byte\n");
    gsv_context_destroy(context);
    return 1;
  }
  status = gsv_context_set_nonce_policy(context, static_cast<gsv_nonce_policy>(99));
  if (status != GSV_INVALID_ARGUMENT) {
    std::fprintf(stderr, "invalid nonce policy was accepted\n");
    gsv_context_destroy(context);
    return 1;
  }
  for (uint32_t bits : WIDTHS) {
    status = gsv_job_init(&job, HEADER_PRESTATE, HEADER_BASE_WORDS, bits, HEADER_EXPECTED);
    if (status != GSV_OK) { int rc = fail("header prefix job init", status); gsv_context_destroy(context); return rc; }
    status = gsv_context_set_header_job(context, &job, HEADER_SUFFIX_WORDS, 1);
    if (status != GSV_OK) { int rc = fail("set header prefix job", status, context); gsv_context_destroy(context); return rc; }
    status = gsv_search(context, HEADER_CANDIDATE >> 8, 1, &result);
    if (status != GSV_FOUND) { int rc = fail("header prefix search", status, context); gsv_context_destroy(context); return rc; }
    status = gsv_digest(context, result.candidate, digest);
    if (status != GSV_OK) { int rc = fail("header prefix digest", status, context); gsv_context_destroy(context); return rc; }
    for (int word = 0; word < 5; ++word) {
      if ((digest[word] & job.target_masks[word]) !=
          (HEADER_EXPECTED[word] & job.target_masks[word])) {
        std::fprintf(stderr, "header target mismatch at %u bits\n", bits);
        gsv_context_destroy(context);
        return 1;
      }
    }
  }

  status = gsv_job_init(&job, PRESTATE, BASE_WORDS, 160, EXPECTED);
  if (status != GSV_OK) { int rc = fail("zero-suffix job init", status); gsv_context_destroy(context); return rc; }
  status = gsv_context_set_header_job(context, &job, nullptr, 0);
  if (status != GSV_OK) { int rc = fail("set zero-suffix header job", status, context); gsv_context_destroy(context); return rc; }
  status = gsv_digest(context, CANDIDATE, digest);
  if (status != GSV_OK || std::memcmp(digest, EXPECTED, sizeof(digest)) != 0) {
    int rc = fail("zero-suffix header digest", status, context);
    gsv_context_destroy(context);
    return rc;
  }
  std::printf("candidate=%010llx digest=%08x%08x%08x%08x%08x ghs=%.3f\n",
              static_cast<unsigned long long>(result.candidate), digest[0], digest[1], digest[2],
              digest[3], digest[4], result.billions_per_second);
  if (argc > 1) {
    const uint64_t outer_count = std::strtoull(argv[1], nullptr, 0);
    const int repetitions = argc > 2 ? std::atoi(argv[2]) : 1;
    const uint32_t benchmark_bits = argc > 3 ? uint32_t(std::strtoul(argv[3], nullptr, 0)) : 32;
    const bool benchmark_header = argc > 4 && std::strcmp(argv[4], "header") == 0;
    const uint32_t benchmark_suffix_blocks =
        benchmark_header && argc > 5 ? uint32_t(std::strtoul(argv[5], nullptr, 0)) : 1u;
    const uint32_t *benchmark_prestate = benchmark_header ? HEADER_PRESTATE : PRESTATE;
    const uint32_t *benchmark_words = benchmark_header ? HEADER_BASE_WORDS : BASE_WORDS;
    const uint32_t *benchmark_expected = benchmark_header ? HEADER_EXPECTED : EXPECTED;
    status = gsv_job_init(&job, benchmark_prestate, benchmark_words, benchmark_bits, benchmark_expected);
    if (status != GSV_OK) { int rc = fail("benchmark job init", status); gsv_context_destroy(context); return rc; }
    std::vector<uint32_t> benchmark_suffix_words(size_t(benchmark_suffix_blocks) * 16u);
    for (uint32_t block = 0; block < benchmark_suffix_blocks; ++block)
      std::memcpy(benchmark_suffix_words.data() + size_t(block) * 16u,
                  HEADER_SUFFIX_WORDS, sizeof(HEADER_SUFFIX_WORDS));
    status = benchmark_header
        ? gsv_context_set_header_job(context, &job, benchmark_suffix_words.data(),
                                     benchmark_suffix_blocks)
        : gsv_context_set_job(context, &job);
    if (status != GSV_OK) { int rc = fail("set benchmark job", status, context); gsv_context_destroy(context); return rc; }
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      status = gsv_search(context, 0, outer_count, &result);
      if (status != GSV_FOUND && status != GSV_NOT_FOUND) {
        int rc = fail("benchmark search", status, context);
        gsv_context_destroy(context);
        return rc;
      }
      std::printf("benchmark sample=%d outer=%llu candidates=%llu ms=%.3f ghs=%.6f\n",
                  repetition, static_cast<unsigned long long>(outer_count),
                  static_cast<unsigned long long>(result.candidates_hashed),
                  result.milliseconds, result.billions_per_second);
    }
  }
  gsv_context_destroy(context);
  return 0;
}
