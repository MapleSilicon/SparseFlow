#include "kernel_cache.h"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <cstring>

namespace sparseflow {

bool KernelSelectionKey::operator==(const KernelSelectionKey& other) const {
  return M_bucket == other.M_bucket &&
         N_bucket == other.N_bucket &&
         K_bucket == other.K_bucket &&
         dtype == other.dtype &&
         sparsity_type == other.sparsity_type &&
         sm_arch == other.sm_arch;
}

struct KernelCache::Impl {
  sqlite3* db_;
  std::string db_path_;
};

KernelCache::KernelCache(const std::string& db_path) : impl_(new Impl()) {
  impl_->db_path_ = db_path;
  
  int rc = sqlite3_open(db_path.c_str(), &impl_->db_);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to open cache DB: " << sqlite3_errmsg(impl_->db_) << "\n";
    impl_->db_ = nullptr;
    return;
  }

  const char* schema = R"(
    CREATE TABLE IF NOT EXISTS kernel_selections (
      M_bucket INTEGER,
      N_bucket INTEGER,
      K_bucket INTEGER,
      dtype INTEGER,
      sparsity_type INTEGER,
      sm_arch INTEGER,
      kernel_id INTEGER,
      throughput_gflops REAL,
      sparse_failed INTEGER DEFAULT 0,
      failure_count INTEGER DEFAULT 0,
      PRIMARY KEY (M_bucket, N_bucket, K_bucket, dtype, sparsity_type, sm_arch)
    )
  )";
  
  char* err_msg = nullptr;
  rc = sqlite3_exec(impl_->db_, schema, nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to create schema: " << err_msg << "\n";
    sqlite3_free(err_msg);
  }
}

KernelCache::~KernelCache() {
  if (impl_->db_) {
    sqlite3_close(impl_->db_);
  }
  delete impl_;
}

bool KernelCache::lookup(const KernelSelectionKey& key, KernelSelectionValue* out_value) const {
  if (!impl_->db_) return false;

  const char* query = R"(
    SELECT kernel_id, throughput_gflops, sparse_failed, failure_count
    FROM kernel_selections
    WHERE M_bucket=? AND N_bucket=? AND K_bucket=? AND dtype=? AND sparsity_type=? AND sm_arch=?
  )";

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) return false;

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
    out_value->sparse_failed = sqlite3_column_int(stmt, 2) != 0;
    out_value->failure_count = sqlite3_column_int(stmt, 3);
    found = true;
  }

  sqlite3_finalize(stmt);
  return found;
}

void KernelCache::insert(const KernelSelectionKey& key, const KernelSelectionValue& value) {
  if (!impl_->db_) return;

  const char* query = R"(
    INSERT OR REPLACE INTO kernel_selections
    (M_bucket, N_bucket, K_bucket, dtype, sparsity_type, sm_arch, kernel_id, throughput_gflops, sparse_failed, failure_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )";

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) return;

  sqlite3_bind_int(stmt, 1, key.M_bucket);
  sqlite3_bind_int(stmt, 2, key.N_bucket);
  sqlite3_bind_int(stmt, 3, key.K_bucket);
  sqlite3_bind_int(stmt, 4, key.dtype);
  sqlite3_bind_int(stmt, 5, key.sparsity_type);
  sqlite3_bind_int(stmt, 6, key.sm_arch);
  sqlite3_bind_int(stmt, 7, static_cast<int>(value.kernel_id));
  sqlite3_bind_double(stmt, 8, value.throughput_gflops);
  sqlite3_bind_int(stmt, 9, value.sparse_failed ? 1 : 0);
  sqlite3_bind_int(stmt, 10, value.failure_count);

  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

void KernelCache::record_failure(const KernelSelectionKey& key) {
  if (!impl_->db_) return;

  const char* query = R"(
    UPDATE kernel_selections
    SET failure_count = failure_count + 1, sparse_failed = 1
    WHERE M_bucket=? AND N_bucket=? AND K_bucket=? AND dtype=? AND sparsity_type=? AND sm_arch=?
  )";

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(impl_->db_, query, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) return;

  sqlite3_bind_int(stmt, 1, key.M_bucket);
  sqlite3_bind_int(stmt, 2, key.N_bucket);
  sqlite3_bind_int(stmt, 3, key.K_bucket);
  sqlite3_bind_int(stmt, 4, key.dtype);
  sqlite3_bind_int(stmt, 5, key.sparsity_type);
  sqlite3_bind_int(stmt, 6, key.sm_arch);

  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

void KernelCache::clear() {
  if (!impl_->db_) return;
  sqlite3_exec(impl_->db_, "DELETE FROM kernel_selections", nullptr, nullptr, nullptr);
}

int32_t KernelCache::size() const {
  if (!impl_->db_) return 0;
  
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
