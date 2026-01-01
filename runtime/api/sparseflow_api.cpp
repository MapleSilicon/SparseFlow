#include "sparseflow_api.h"
#include "../dispatch/kernel_dispatcher.h"
#include "../cache/kernel_cache.h"
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>

static std::mutex g_mu;
static std::unordered_map<std::string, std::unique_ptr<sparseflow::KernelDispatcher>> g_dispatchers;

static sparseflow::KernelDispatcher* get_dispatcher(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& ptr = g_dispatchers[path];
  if (!ptr) {
    ptr = std::make_unique<sparseflow::KernelDispatcher>(path);
  }
  return ptr.get();
}

int32_t sf_select_kernel(const char* cache_path,
                         const sf_matmul_desc* d,
                         const sf_gpu_info* g) {
  using namespace sparseflow;

  MatmulDesc desc;
  desc.M = d->M;
  desc.N = d->N;
  desc.K = d->K;
  desc.dtype = static_cast<DType>(d->dtype);
  desc.sparsity_type = static_cast<SparsityType>(d->sparsity_type);
  desc.sparsity_ratio = d->sparsity_ratio;
  desc.layout = static_cast<Layout>(d->layout);

  GPUInfo gpu;
  gpu.sm_arch = g->sm_arch;
  gpu.num_sms = g->num_sms;
  gpu.shared_mem_kb = static_cast<size_t>(g->shared_mem_kb);

  std::string path = cache_path ? cache_path : "kernel_cache.db";
  auto* dispatcher = get_dispatcher(path);

  auto kid = dispatcher->select_kernel(desc, gpu, nullptr, nullptr, nullptr, 0);
  
  return static_cast<int32_t>(kid);
}

int32_t sf_cache_size(const char* cache_path) {
  using namespace sparseflow;
  
  std::string path = cache_path ? cache_path : "kernel_cache.db";
  KernelCache cache(path);
  
  return static_cast<int32_t>(cache.size());
}

void sf_cache_clear(const char* cache_path) {
  using namespace sparseflow;
  
  std::string path = cache_path ? cache_path : "kernel_cache.db";
  KernelCache cache(path);
  cache.clear();
}
