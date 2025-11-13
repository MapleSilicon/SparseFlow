#pragma once
#include <cstdint>
struct TileCfg { int bm=64,bn=64,bk=64,banks=8,pe_rows=8,pe_cols=8; };
struct Profile { uint64_t nnz_loads=0, nz_compute=0, zero_skipped=0; double time_ms=0, energy_mj=0; };
inline Profile estimate_spmm(uint64_t nnz, uint64_t dense_ops, const TileCfg& t){
  Profile p; p.nnz_loads=nnz; p.nz_compute=nnz; p.zero_skipped = dense_ops>nnz? dense_ops-nnz:0;
  const double mem_cost=1e-6, pe_cost=5e-7; p.time_ms = mem_cost*p.nnz_loads + pe_cost*p.nz_compute;
  p.energy_mj = 0.001 * p.time_ms; return p;
}
