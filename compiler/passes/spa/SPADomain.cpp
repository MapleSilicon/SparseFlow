#include "SPADomain.h"
#include <algorithm>

namespace sparseflow {

MatrixSparsity makeDenseRows(int rows) {
  MatrixSparsity s;
  if (rows < 0)
    rows = 0;
  s.rowMask.assign(static_cast<size_t>(rows), 1);
  return s;
}

MatrixSparsity intersectRows(const MatrixSparsity &a,
                             const MatrixSparsity &b) {
  MatrixSparsity out;
  size_t n = std::min(a.rowMask.size(), b.rowMask.size());
  out.rowMask.resize(n);
  for (size_t i = 0; i < n; ++i)
    out.rowMask[i] = a.rowMask[i] & b.rowMask[i];
  return out;
}

MatrixSparsity unionRows(const MatrixSparsity &a,
                         const MatrixSparsity &b) {
  MatrixSparsity out;
  size_t n = std::min(a.rowMask.size(), b.rowMask.size());
  out.rowMask.resize(n);
  for (size_t i = 0; i < n; ++i)
    out.rowMask[i] = a.rowMask[i] | b.rowMask[i];
  return out;
}

} // namespace sparseflow
