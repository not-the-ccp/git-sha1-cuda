#include "git_sha1_cuda.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

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
  std::printf("candidate=%010llx digest=%08x%08x%08x%08x%08x ghs=%.3f\n",
              static_cast<unsigned long long>(result.candidate), digest[0], digest[1], digest[2],
              digest[3], digest[4], result.billions_per_second);
  if (argc > 1) {
    const uint64_t outer_count = std::strtoull(argv[1], nullptr, 0);
    const int repetitions = argc > 2 ? std::atoi(argv[2]) : 1;
    const uint32_t benchmark_bits = argc > 3 ? uint32_t(std::strtoul(argv[3], nullptr, 0)) : 32;
    status = gsv_job_init(&job, PRESTATE, BASE_WORDS, benchmark_bits, EXPECTED);
    if (status != GSV_OK) { int rc = fail("benchmark job init", status); gsv_context_destroy(context); return rc; }
    status = gsv_context_set_job(context, &job);
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
