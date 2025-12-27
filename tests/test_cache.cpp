#include "../runtime/cache/kernel_cache.h"
#include <iostream>
#include <cassert>

using namespace sparseflow;

void test_bucketing() {
  std::cout << "Testing bucketing logic..." << std::endl;
  
  assert(bucket_dimension(32) == 32);
  assert(bucket_dimension(64) == 64);
  assert(bucket_dimension(127) == 128);
  assert(bucket_dimension(128) == 128);
  assert(bucket_dimension(500) == 512);
  assert(bucket_dimension(2048) == 1024);
  
  assert(bucket_sparsity(0.1f) == 0);
  assert(bucket_sparsity(0.3f) == 1);
  assert(bucket_sparsity(0.5f) == 2);
  assert(bucket_sparsity(0.9f) == 3);
  
  std::cout << "  ✓ Bucketing tests passed" << std::endl;
}

void test_cache_operations() {
  std::cout << "Testing cache operations..." << std::endl;
  
  KernelCache cache("test_cache.db");
  cache.clear();
  
  KernelSelectionKey key;
  key.sm_arch = 86;
  key.dtype = DType::F16;
  key.M_bucket = 128;
  key.N_bucket = 128;
  key.K_bucket = 128;
  key.sparsity_type = SparsityType::NM_2_4;
  key.sparsity_bucket = 2;
  key.layout = Layout::ROW_MAJOR;
  
  auto result = cache.lookup(key);
  assert(!result.has_value());
  
  KernelSelectionValue value;
  value.kernel_id = KernelID::SPARSE_NM_32;
  value.throughput_gflops = 215.3f;
  value.confidence = 1.0f;
  value.samples = 20;
  
  cache.insert(key, value);
  
  result = cache.lookup(key);
  assert(result.has_value());
  assert(result->kernel_id == KernelID::SPARSE_NM_32);
  assert(std::abs(result->throughput_gflops - 215.3f) < 0.1f);
  
  assert(cache.size() == 1);
  
  std::cout << "  ✓ Cache operations passed" << std::endl;
}

int main() {
  std::cout << "Running SparseFlow cache tests..." << std::endl;
  
  test_bucketing();
  test_cache_operations();
  
  std::cout << "\n✓ All tests passed!" << std::endl;
  return 0;
}
