#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t M, N, K;
  int32_t dtype;
  int32_t sparsity_type;
  float   sparsity_ratio;
  int32_t layout;
} sf_matmul_desc;

typedef struct {
  int32_t sm_arch;
  int32_t num_sms;
  uint64_t shared_mem_kb;
} sf_gpu_info;

int32_t sf_select_kernel(const char* cache_path,
                         const sf_matmul_desc* desc,
                         const sf_gpu_info* gpu);

int32_t sf_cache_size(const char* cache_path);

void sf_cache_clear(const char* cache_path);

#ifdef __cplusplus
}
#endif
