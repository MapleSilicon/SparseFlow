#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <memory>

namespace sparseflow {

struct CompressedBufferKey {
  uint64_t content_hash;  // Hash of actual matrix content
  uint16_t M, N, K;
  uint8_t dtype;
  uint8_t sparsity_type;
  
  bool operator==(const CompressedBufferKey& other) const {
    return content_hash == other.content_hash &&
           M == other.M && N == other.N && K == other.K &&
           dtype == other.dtype && sparsity_type == other.sparsity_type;
  }
};

struct CompressedBufferValue {
  void* d_compressed;
  void* d_compressed_buffer;
  size_t compressed_size;
  size_t compressed_buffer_size;
  uint64_t timestamp;
  
  ~CompressedBufferValue();
};

class CompressedCache {
public:
  CompressedCache();
  ~CompressedCache();
  
  bool contains(const CompressedBufferKey& key) const;
  bool has(const CompressedBufferKey& key) const;
  CompressedBufferValue* get(const CompressedBufferKey& key);
  void put(const CompressedBufferKey& key, CompressedBufferValue&& value);
  void clear();
  size_t size_bytes() const;
  size_t count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

CompressedCache& get_compressed_cache();

// Helper: compute content hash from device memory
uint64_t compute_matrix_hash(const void* d_data, size_t bytes);

}

namespace std {
  template<>
  struct hash<sparseflow::CompressedBufferKey> {
    size_t operator()(const sparseflow::CompressedBufferKey& k) const {
      size_t h = 0;
      h ^= std::hash<uint64_t>{}(k.content_hash);
      h ^= std::hash<uint16_t>{}(k.M) << 1;
      h ^= std::hash<uint16_t>{}(k.N) << 2;
      h ^= std::hash<uint16_t>{}(k.K) << 3;
      h ^= std::hash<uint8_t>{}(k.dtype) << 4;
      h ^= std::hash<uint8_t>{}(k.sparsity_type) << 5;
      return h;
    }
  };
}
