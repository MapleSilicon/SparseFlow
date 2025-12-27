#pragma once
#include <cstdint>
#include <functional>

namespace sparseflow {

enum class DType : uint8_t {
  F16 = 0,
  BF16 = 1,
  F32 = 2
};

enum class SparsityType : uint8_t {
  DENSE = 0,
  NM_2_4 = 1
};

enum class Layout : uint8_t {
  ROW_MAJOR = 0,
  COL_MAJOR = 1
};

struct KernelSelectionKey {
  uint16_t sm_arch;
  DType dtype;
  uint16_t M_bucket;
  uint16_t N_bucket;
  uint16_t K_bucket;
  SparsityType sparsity_type;
  uint8_t sparsity_bucket;
  Layout layout;
  
  bool operator==(const KernelSelectionKey& other) const {
    return sm_arch == other.sm_arch &&
           dtype == other.dtype &&
           M_bucket == other.M_bucket &&
           N_bucket == other.N_bucket &&
           K_bucket == other.K_bucket &&
           sparsity_type == other.sparsity_type &&
           sparsity_bucket == other.sparsity_bucket &&
           layout == other.layout;
  }
};

inline uint16_t bucket_dimension(int dim) {
  if (dim <= 0) return 0;
  
  uint32_t x = (dim <= 32) ? 32u : (uint32_t)dim;
  
  // Round up to next power of two
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  
  if (x > 65535u) x = 65535u;
  return (uint16_t)x;
}

inline uint8_t bucket_sparsity(float sparsity_ratio) {
  if (sparsity_ratio < 0.25f) return 0;
  if (sparsity_ratio < 0.50f) return 1;
  if (sparsity_ratio < 0.75f) return 2;
  return 3;
}

} // namespace sparseflow

namespace std {
  template<>
  struct hash<sparseflow::KernelSelectionKey> {
    size_t operator()(const sparseflow::KernelSelectionKey& k) const {
      size_t h = 0;
      h ^= std::hash<uint16_t>{}(k.sm_arch) << 0;
      h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.dtype)) << 1;
      h ^= std::hash<uint16_t>{}(k.M_bucket) << 2;
      h ^= std::hash<uint16_t>{}(k.N_bucket) << 3;
      h ^= std::hash<uint16_t>{}(k.K_bucket) << 4;
      h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.sparsity_type)) << 5;
      h ^= std::hash<uint8_t>{}(k.sparsity_bucket) << 6;
      h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.layout)) << 7;
      return h;
    }
  };
}
