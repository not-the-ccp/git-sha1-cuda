#ifndef GIT_SHA1_CUDA_H
#define GIT_SHA1_CUDA_H

#include <stdint.h>

#if defined(_WIN32) && defined(GSV_BUILD_SHARED)
#define GSV_API __declspec(dllexport)
#elif defined(_WIN32)
#define GSV_API __declspec(dllimport)
#elif defined(__GNUC__)
#define GSV_API __attribute__((visibility("default")))
#else
#define GSV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GSV_ABI_VERSION 1u
#define GSV_NO_WINNER UINT64_MAX

typedef struct gsv_context gsv_context;

typedef enum gsv_status {
  GSV_OK = 0,
  GSV_FOUND = 1,
  GSV_NOT_FOUND = 2,
  GSV_INVALID_ARGUMENT = -1,
  GSV_CUDA_ERROR = -2,
  GSV_ABI_MISMATCH = -3,
  GSV_INTERNAL_ERROR = -4
} gsv_status;

typedef enum gsv_nonce_policy {
  GSV_NONCE_NO_NUL = 0,
  GSV_NONCE_HEADER_SAFE = 1,
  GSV_NONCE_PRINTABLE_ASCII = 2
} gsv_nonce_policy;

/*
 * Runtime job for the optimized mutable-block layout:
 *
 *   final SHA-1 block byte 48..51 = candidate bits 39..8 (W12)
 *   final SHA-1 block byte 52     = candidate bits 7..0 (W13 high byte)
 *
 * base_words[12] and the high byte of base_words[13] must be zero. The
 * remaining words include fixed Git bytes. For final-block jobs they also
 * contain SHA-1 padding. target_words are the requested digest prefix
 * left-aligned in five words.
 */
typedef struct gsv_job {
  uint32_t abi_version;
  uint32_t prestate[5];
  uint32_t pre12[5];
  uint32_t base_words[16];
  uint32_t target_words[5];
  uint32_t target_masks[5];
  uint32_t target_bits;
  uint32_t h0_gate_base;
  uint32_t h0_gate_span;
} gsv_job;

typedef struct gsv_search_result {
  uint32_t found;
  uint32_t reserved;
  uint64_t candidate;
  uint64_t candidates_hashed;
  float milliseconds;
  float billions_per_second;
} gsv_search_result;

/* Library and device discovery. */
GSV_API uint32_t gsv_abi_version(void);
GSV_API int32_t gsv_device_count(void);

/*
 * Fill derived fields (pre12, masks and H0 gate) and validate the layout.
 * target_words must already contain the requested prefix in its most
 * significant bits; unused low bits are ignored according to target_bits.
 */
GSV_API gsv_status gsv_job_init(gsv_job *job,
                                const uint32_t prestate[5],
                                const uint32_t base_words[16],
                                uint32_t target_bits,
                                const uint32_t target_words[5]);
GSV_API gsv_status gsv_job_validate(const gsv_job *job);

/* Context owns device buffers and is bound to one CUDA device. */
GSV_API gsv_status gsv_context_create(int32_t device,
                                      const gsv_job *job,
                                      gsv_context **out_context);
GSV_API void gsv_context_destroy(gsv_context *context);
GSV_API gsv_status gsv_context_set_job(gsv_context *context,
                                       const gsv_job *job);

/*
 * Configure a custom-header job. The mutable block uses the same W12/W13
 * candidate layout as gsv_job. suffix_words contains suffix_block_count
 * complete, fixed SHA-1 blocks as big-endian words. A zero-block suffix is
 * valid. The default winner policy rejects candidate bytes containing NUL or
 * LF and can be changed with gsv_context_set_nonce_policy.
 */
GSV_API gsv_status gsv_context_set_header_job(gsv_context *context,
                                              const gsv_job *job,
                                              const uint32_t *suffix_words,
                                              uint32_t suffix_block_count);

/* Select which candidate byte strings may be published as winners. */
GSV_API gsv_status gsv_context_set_nonce_policy(gsv_context *context,
                                                gsv_nonce_policy policy);

/*
 * Search outer_count W12 values beginning at outer_base. Each outer value
 * evaluates all 256 W13 high-byte values. Keep calls bounded (for example,
 * 2^22 outer values is about 0.1 seconds on an RTX 4060) for easy checkpointing.
 */
GSV_API gsv_status gsv_search(gsv_context *context,
                              uint64_t outer_base,
                              uint64_t outer_count,
                              gsv_search_result *result);

/* Capture all five digest words for one candidate; intended for oracles/tests. */
GSV_API gsv_status gsv_digest(gsv_context *context,
                              uint64_t candidate,
                              uint32_t digest_words[5]);

GSV_API const char *gsv_last_error(const gsv_context *context);
GSV_API const char *gsv_status_string(gsv_status status);

#ifdef __cplusplus
}
#endif

#endif
