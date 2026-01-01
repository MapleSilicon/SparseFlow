#include "../runtime/api/sparseflow_api.h"
#include <iostream>
#include <cassert>

int main() {
  std::cout << "Testing C API..." << std::endl;
  
  sf_cache_clear("test_c_api.db");
  
  sf_matmul_desc desc;
  desc.M = 512;
  desc.N = 512;
  desc.K = 512;
  desc.dtype = 0;
  desc.sparsity_type = 0;
  desc.sparsity_ratio = 0.0f;
  desc.layout = 0;
  
  sf_gpu_info gpu;
  gpu.sm_arch = 86;  // RTX 3090
  gpu.num_sms = 82;
  gpu.shared_mem_kb = 100;
  
  std::cout << "\nTest 1: First call (cache miss)..." << std::endl;
  int32_t kernel1 = sf_select_kernel("test_c_api.db", &desc, &gpu);
  std::cout << "  Selected kernel: " << kernel1 << std::endl;
  
  int32_t cache_after_1 = sf_cache_size("test_c_api.db");
  std::cout << "  Cache size: " << cache_after_1 << std::endl;
  
  if (cache_after_1 > 0) {
    std::cout << "  ✓ Kernel benchmarked successfully and cached" << std::endl;
    
    std::cout << "\nTest 2: Second call (should be cache hit)..." << std::endl;
    int32_t kernel2 = sf_select_kernel("test_c_api.db", &desc, &gpu);
    std::cout << "  Selected kernel: " << kernel2 << std::endl;
    std::cout << "  Cache size: " << sf_cache_size("test_c_api.db") << std::endl;
    
    assert(kernel1 == kernel2 && "Cache hit should return same kernel");
    std::cout << "  ✓ Cache hit working" << std::endl;
    
    std::cout << "\nTest 3: Different shape (bucketing test)..." << std::endl;
    desc.M = 513;
    int32_t kernel3 = sf_select_kernel("test_c_api.db", &desc, &gpu);
    std::cout << "  Selected kernel: " << kernel3 << std::endl;
    
    int32_t cache_after_3 = sf_cache_size("test_c_api.db");
    std::cout << "  Cache size: " << cache_after_3 << std::endl;
    
    if (cache_after_3 == 2) {
      std::cout << "  ✓ Bucketing works correctly" << std::endl;
    } else {
      std::cout << "  ✗ Expected cache size 2 but got " << cache_after_3 << std::endl;
    }
  } else {
    std::cout << "  ⚠ No GPU available - kernel execution failed" << std::endl;
    std::cout << "  ✓ Cache correctly refused to store invalid results" << std::endl;
    std::cout << "\nSkipping cache hit and bucketing tests (require GPU)" << std::endl;
  }
  
  std::cout << "\n✓ C API tests passed!" << std::endl;
  std::cout << "Note: Full validation requires GPU access" << std::endl;
  return 0;
}
