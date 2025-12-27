#include "kernel_cache.h"
#include <sqlite3.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <functional>

namespace sparseflow {

struct KernelCache::Impl {
  sqlite3* db_;
};

KernelCache::KernelCache(const std::string& db_path) {
  impl_ = new Impl();
  
  int rc = sqlite3_open(db_path.c_str(), &impl_->db_);
  if (rc != SQLITE_OK) {
    throw std::runtime_error("Failed to open cache database");
  }
  
  char* err_msg = nullptr;
  sqlite3_exec(impl_->db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &err_msg);
  if (err_msg) {
    sqlite3_free(err_msg);
  }
  
  const char* create_table = R"(
    CREATE TABLE IF NOT EXISTS kernel_selections (
      M_bucket INTEGER NOT NULL,
      N_bucket INTEGER NOT NULL,
      K_bucket INTEGER NOT NULL,
      dtype INTEGER NOT NULL,
      sparsity_type INTEGER NOT NULL,
      sm_arch INTEGER NOT NULL,
      kernel_id INTEGER NOT NULL,
      throughput_gflops REAL NOT NULL,
      PRIMARY KEY (M_bucket, N_bucket, K_bucket, dtype, sparsity_type, sm_arch)
    )
  )";
  
  sqlite3_exec(impl_->db_, create_table, nullptr, nullptr, &err_msg);
  if (err_msg) {
    sqlite3_free(err_msg);
  }
}

KernelCache::~KernelCache() {
  if (impl_) {
    if (impl_->db_) {
      sqlite3_close(impl_->db_);
    }
    delete impl_;
  }
}

bool KernelSelectionKey::operator==(const KernelSelectionKey& other) const {
  return M_bucket == other.M_bucket &&
         N_bucket == other.N_bucket &&
         K_bucket == other.K_bucket &&
         dtype == other.dtype &&
         sparsity_type == other.sparsity_type &&
         sm_arch == other.sm_arch;
}

bool KernelCache::lookup(const KernelSelectionKey& key, KernelSelectionValue* out_value) const {
  const char* query = R"(
    SELECT kernel_id, throughput_gflops FROM kernel_selections
    WHERE M_bucket = ? AND N_bucket = ? AND K_bucket = ?
      AND dtype = ? AND sparsity_type = ? AND sm_arch = ?
  )";
  
  sqlite3_stmt* stmt;
  if (sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr) != SQLITE_OK) {
    return false;
  }
  
  sqlite3_bind_int(stmt, 1, key.M_bucket);
  sqlite3_bind_int(stmt, 2, key.N_bucket);
  sqlite3_bind_int(stmt, 3, key.K_bucket);
  sqlite3_bind_int(stmt, 4, key.dtype);
  sqlite3_bind_int(stmt, 5, key.sparsity_type);
  sqlite3_bind_int(stmt, 6, key.sm_arch);
  
  bool found = false;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    out_value->kernel_id = static_cast<KernelID>(sqlite3_column_int(stmt, 0));
    out_value->throughput_gflops = static_cast<float>(sqlite3_column_double(stmt, 1));
    found = true;
  }
  
  sqlite3_finalize(stmt);
  return found;
}

void KernelCache::insert(const KernelSelectionKey& key, const KernelSelectionValue& value) {
  const char* query = R"(
    INSERT OR REPLACE INTO kernel_selections
    (M_bucket, N_bucket, K_bucket, dtype, sparsity_type, sm_arch, kernel_id, throughput_gflops)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  )";
  
  sqlite3_stmt* stmt;
  if (sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr) != SQLITE_OK) {
    return;
  }
  
  sqlite3_bind_int(stmt, 1, key.M_bucket);
  sqlite3_bind_int(stmt, 2, key.N_bucket);
  sqlite3_bind_int(stmt, 3, key.K_bucket);
  sqlite3_bind_int(stmt, 4, key.dtype);
  sqlite3_bind_int(stmt, 5, key.sparsity_type);
  sqlite3_bind_int(stmt, 6, key.sm_arch);
  sqlite3_bind_int(stmt, 7, static_cast<int>(value.kernel_id));
  sqlite3_bind_double(stmt, 8, value.throughput_gflops);
  
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

void KernelCache::clear() {
  const char* query = "DELETE FROM kernel_selections";
  char* err_msg = nullptr;
  sqlite3_exec(impl_->db_, query, nullptr, nullptr, &err_msg);
  if (err_msg) {
    sqlite3_free(err_msg);
  }
}

int32_t KernelCache::size() const {
  const char* query = "SELECT COUNT(*) FROM kernel_selections";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr);
  
  int count = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    count = sqlite3_column_int(stmt, 0);
  }
  
  sqlite3_finalize(stmt);
  return count;
}

} // namespace sparseflow
