#include "compressed_cache.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <iostream>

namespace sparseflow {

uint64_t compute_matrix_hash(const void* d_data, size_t bytes) {
  size_t sample_size = (bytes < 1024) ? bytes : 1024;
  
  unsigned char* h_sample = new unsigned char[sample_size];
  
  cudaError_t err = cudaMemcpy(h_sample, d_data, sample_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[Hash] cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
    delete[] h_sample;
    return 0;
  }
  
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < sample_size; i++) {
    hash ^= h_sample[i];
    hash *= 1099511628211ULL;
  }
  
  delete[] h_sample;
  return hash;
}

CompressedBufferValue::~CompressedBufferValue() {
  // Don't free - plan owns these now
  d_compressed = nullptr;
  d_compressed_buffer = nullptr;
}

struct CompressedCache::Impl {
  std::unordered_map<CompressedBufferKey, CompressedBufferValue> cache;
  mutable std::mutex mutex;
  size_t total_bytes = 0;
  
  static constexpr size_t MAX_CACHE_BYTES = 1024 * 1024 * 1024;
};

CompressedCache::CompressedCache() : impl_(std::make_unique<Impl>()) {}

CompressedCache::~CompressedCache() = default;

bool CompressedCache::contains(const CompressedBufferKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->cache.find(key) != impl_->cache.end();
}

bool CompressedCache::has(const CompressedBufferKey& key) const {
  return contains(key);
}

CompressedBufferValue* CompressedCache::get(const CompressedBufferKey& key) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto it = impl_->cache.find(key);
  if (it == impl_->cache.end()) {
    return nullptr;
  }
  
  it->second.timestamp = std::time(nullptr);
  return &it->second;
}

void CompressedCache::put(const CompressedBufferKey& key, CompressedBufferValue&& value) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  
  size_t entry_size = value.compressed_size + value.compressed_buffer_size;
  
  if (impl_->total_bytes + entry_size > Impl::MAX_CACHE_BYTES) {
    return;
  }
  
  value.timestamp = std::time(nullptr);
  impl_->total_bytes += entry_size;
  impl_->cache.emplace(key, std::move(value));
}

void CompressedCache::clear() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->cache.clear();
  impl_->total_bytes = 0;
}

size_t CompressedCache::size_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->total_bytes;
}

size_t CompressedCache::count() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->cache.size();
}

CompressedCache& get_compressed_cache() {
  static CompressedCache cache;
  return cache;
}

}
