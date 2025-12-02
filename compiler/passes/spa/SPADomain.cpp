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

MatrixSparsity makeDenseRowsCols(int rows, int cols) {
  MatrixSparsity s;
  if (rows < 0) rows = 0;
  if (cols < 0) cols = 0;
  s.rowMask.assign(static_cast<size_t>(rows), 1);
  s.colMask.assign(static_cast<size_t>(cols), 1);
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

MatrixSparsity transposeSparsity(const MatrixSparsity &m) {
  MatrixSparsity out;
  // Swap rows and columns
  out.rowMask = m.colMask;
  out.colMask = m.rowMask;
  return out;
}

} // namespace sparseflow
