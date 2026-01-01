#include "kernel_cache.h"
#include "kernel_key.h"
#include "../benchmark/micro_bench.h"
#include <sqlite3.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>

namespace sparseflow {

struct KernelCache::Impl {
  sqlite3* db_ = nullptr;
  std::string db_path_;
};

static void log_err(sqlite3* db, const char* where) {
  std::cerr << "[KernelCache] " << where << ": " << sqlite3_errmsg(db) << "\n";
}

static bool exec_sql(sqlite3* db, const std::string& sql) {
  char* err = nullptr;
  int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
  if (rc != SQLITE_OK) {
    std::cerr << "[KernelCache] exec failed: " << (err ? err : "unknown") << "\n";
    if (err) sqlite3_free(err);
    return false;
  }
  return true;
}

static bool column_exists(sqlite3* db, const char* table, const char* col) {
  std::string sql = "PRAGMA table_info(" + std::string(table) + ");";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return false;

  bool found = false;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const unsigned char* name = sqlite3_column_text(stmt, 1);
    if (name && std::string(reinterpret_cast<const char*>(name)) == col) {
      found = true;
      break;
    }
  }
  sqlite3_finalize(stmt);
  return found;
}

static bool ensure_schema(sqlite3* db) {
  const char* create_sql =
    "CREATE TABLE IF NOT EXISTS kernel_cache ("
    "  key_hash TEXT PRIMARY KEY,"
    "  best_kernel INTEGER NOT NULL,"
    "  best_gflops REAL NOT NULL,"
    "  sparse_failed INTEGER NOT NULL DEFAULT 0,"
    "  failure_count INTEGER NOT NULL DEFAULT 0,"
    "  last_failure_code INTEGER NOT NULL DEFAULT 0,"
    "  last_failure_ts INTEGER NOT NULL DEFAULT 0"
    ");";
  
  if (!exec_sql(db, create_sql)) return false;

  // Auto-migrate existing tables
  struct Col { const char* name; const char* alter; };
  const std::vector<Col> cols = {
    {"sparse_failed", "ALTER TABLE kernel_cache ADD COLUMN sparse_failed INTEGER NOT NULL DEFAULT 0;"},
    {"failure_count", "ALTER TABLE kernel_cache ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0;"},
    {"last_failure_code", "ALTER TABLE kernel_cache ADD COLUMN last_failure_code INTEGER NOT NULL DEFAULT 0;"},
    {"last_failure_ts", "ALTER TABLE kernel_cache ADD COLUMN last_failure_ts INTEGER NOT NULL DEFAULT 0;"},
  };

  for (const auto& c : cols) {
    if (!column_exists(db, "kernel_cache", c.name)) {
      if (!exec_sql(db, c.alter)) return false;
    }
  }

  return true;
}

KernelCache::KernelCache(const std::string& db_path) : impl_(new Impl()) {
  impl_->db_path_ = db_path;

  if (sqlite3_open(db_path.c_str(), &impl_->db_) != SQLITE_OK) {
    std::cerr << "[KernelCache] Failed to open: " << db_path << "\n";
    return;
  }

  exec_sql(impl_->db_, "PRAGMA journal_mode=WAL;");
  exec_sql(impl_->db_, "PRAGMA synchronous=NORMAL;");

  if (!ensure_schema(impl_->db_)) {
    std::cerr << "[KernelCache] Schema setup failed\n";
  }
}

KernelCache::~KernelCache() {
  if (impl_->db_) sqlite3_close(impl_->db_);
  delete impl_;
}

bool KernelCache::get(const KernelKey& key, KernelCacheEntry* out) const {
  if (!impl_->db_ || !out) return false;
  
  const std::string kh = key.to_string();
  const char* sql =
    "SELECT best_kernel, best_gflops, sparse_failed, failure_count, "
    "last_failure_code, last_failure_ts FROM kernel_cache WHERE key_hash=?1;";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(impl_->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    log_err(impl_->db_, "prepare(get)");
    return false;
  }

  sqlite3_bind_text(stmt, 1, kh.c_str(), -1, SQLITE_TRANSIENT);

  if (sqlite3_step(stmt) == SQLITE_ROW) {
    out->valid = true;
    out->best_kernel = static_cast<KernelID>(sqlite3_column_int(stmt, 0));
    out->best_gflops = sqlite3_column_double(stmt, 1);
    out->sparse_failed = sqlite3_column_int(stmt, 2) != 0;
    out->failure_count = sqlite3_column_int(stmt, 3);
    out->last_failure_code = sqlite3_column_int(stmt, 4);
    out->last_failure_ts = sqlite3_column_int64(stmt, 5);
    sqlite3_finalize(stmt);
    return true;
  }

  sqlite3_finalize(stmt);
  out->valid = false;
  return false;
}

bool KernelCache::upsert_best(const KernelKey& key, KernelID best_kernel, double best_gflops) {
  if (!impl_->db_) return false;
  
  const std::string kh = key.to_string();
  const char* sql =
    "INSERT INTO kernel_cache(key_hash, best_kernel, best_gflops, sparse_failed, "
    "failure_count, last_failure_code, last_failure_ts) "
    "VALUES(?1, ?2, ?3, 0, 0, 0, 0) "
    "ON CONFLICT(key_hash) DO UPDATE SET "
    "  best_kernel=excluded.best_kernel, "
    "  best_gflops=excluded.best_gflops, "
    "  sparse_failed=0, "
    "  failure_count=0, "
    "  last_failure_code=0, "
    "  last_failure_ts=0;";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(impl_->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    log_err(impl_->db_, "prepare(upsert)");
    return false;
  }

  sqlite3_bind_text(stmt, 1, kh.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, static_cast<int>(best_kernel));
  sqlite3_bind_double(stmt, 3, best_gflops);

  bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
  sqlite3_finalize(stmt);
  return ok;
}

bool KernelCache::mark_sparse_failed(const KernelKey& key, int failure_code) {
  if (!impl_->db_) return false;
  
  const std::string kh = key.to_string();
  const int64_t now = static_cast<int64_t>(std::time(nullptr));

  const char* sql =
    "INSERT INTO kernel_cache(key_hash, best_kernel, best_gflops, sparse_failed, "
    "failure_count, last_failure_code, last_failure_ts) "
    "VALUES(?1, 0, 0.0, 1, 1, ?2, ?3) "
    "ON CONFLICT(key_hash) DO UPDATE SET "
    "  sparse_failed=1, "
    "  failure_count=failure_count+1, "
    "  last_failure_code=excluded.last_failure_code, "
    "  last_failure_ts=excluded.last_failure_ts;";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(impl_->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    log_err(impl_->db_, "prepare(mark_failed)");
    return false;
  }

  sqlite3_bind_text(stmt, 1, kh.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, failure_code);
  sqlite3_bind_int64(stmt, 3, now);

  bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
  sqlite3_finalize(stmt);
  return ok;
}

bool KernelCache::clear_sparse_failed(const KernelKey& key) {
  if (!impl_->db_) return false;
  
  const std::string kh = key.to_string();
  const char* sql =
    "UPDATE kernel_cache SET sparse_failed=0, failure_count=0, "
    "last_failure_code=0, last_failure_ts=0 WHERE key_hash=?1;";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(impl_->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    log_err(impl_->db_, "prepare(clear)");
    return false;
  }

  sqlite3_bind_text(stmt, 1, kh.c_str(), -1, SQLITE_TRANSIENT);

  bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
  sqlite3_finalize(stmt);
  return ok;
}

bool KernelCache::is_sparse_blacklisted(const KernelKey& key) const {
  KernelCacheEntry e;
  if (!get(key, &e)) return false;
  return e.valid && e.sparse_failed;
}

// Legacy compatibility methods
bool KernelSelectionKey::operator==(const KernelSelectionKey& other) const {
  return M_bucket == other.M_bucket && N_bucket == other.N_bucket &&
         K_bucket == other.K_bucket && dtype == other.dtype &&
         sparsity_type == other.sparsity_type && sm_arch == other.sm_arch;
}

bool KernelCache::lookup(const KernelSelectionKey& key, KernelSelectionValue* out_value) const {
  // Convert to new format
  KernelKey k;
  k.M = key.M_bucket;
  k.N = key.N_bucket;
  k.K = key.K_bucket;
  k.dtype = key.dtype;
  k.sparsity_type = key.sparsity_type;
  k.sm_arch = key.sm_arch;
  
  KernelCacheEntry e;
  if (get(k, &e) && e.valid) {
    out_value->kernel_id = e.best_kernel;
    out_value->throughput_gflops = static_cast<float>(e.best_gflops);
    out_value->sparse_failed = e.sparse_failed;
    out_value->failure_count = e.failure_count;
    return true;
  }
  return false;
}

void KernelCache::insert(const KernelSelectionKey& key, const KernelSelectionValue& value) {
  KernelKey k;
  k.M = key.M_bucket;
  k.N = key.N_bucket;
  k.K = key.K_bucket;
  k.dtype = key.dtype;
  k.sparsity_type = key.sparsity_type;
  k.sm_arch = key.sm_arch;
  
  upsert_best(k, value.kernel_id, value.throughput_gflops);
}

void KernelCache::record_failure(const KernelSelectionKey& key) {
  KernelKey k;
  k.M = key.M_bucket;
  k.N = key.N_bucket;
  k.K = key.K_bucket;
  k.dtype = key.dtype;
  k.sparsity_type = key.sparsity_type;
  k.sm_arch = key.sm_arch;
  
  mark_sparse_failed(k, -1);
}

void KernelCache::clear() {
  if (impl_->db_) {
    exec_sql(impl_->db_, "DELETE FROM kernel_cache");
  }
}

int32_t KernelCache::size() const {
  if (!impl_->db_) return 0;
  
  const char* sql = "SELECT COUNT(*) FROM kernel_cache";
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(impl_->db_, sql, -1, &stmt, nullptr);
  int count = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    count = sqlite3_column_int(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return count;
}

} // namespace sparseflow
