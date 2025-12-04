#include "SPADomain.h"
#include <algorithm>

namespace sparseflow {

static inline int clampDim(int dim) {
  return dim < 0 ? 0 : dim;
}

MatrixSparsity makeDense(int rows, int cols) {
  rows = clampDim(rows);
  cols = clampDim(cols);
  MatrixSparsity s;
  s.rowMask.assign(static_cast<size_t>(rows), 1);
  s.colMask.assign(static_cast<size_t>(cols), 1);
  return s;
}

MatrixSparsity makeZero(int rows, int cols) {
  rows = clampDim(rows);
  cols = clampDim(cols);
  MatrixSparsity s;
  s.rowMask.assign(static_cast<size_t>(rows), 0);
  s.colMask.assign(static_cast<size_t>(cols), 0);
  return s;
}

MatrixSparsity intersect2D(const MatrixSparsity &a, const MatrixSparsity &b) {
  MatrixSparsity out;
  size_t rows = std::min(a.rowMask.size(), b.rowMask.size());
  size_t cols = std::min(a.colMask.size(), b.colMask.size());
  out.rowMask.resize(rows);
  out.colMask.resize(cols);
  for (size_t i = 0; i < rows; ++i) {
    out.rowMask[i] = a.rowMask[i] & b.rowMask[i];
  }
  for (size_t j = 0; j < cols; ++j) {
    out.colMask[j] = a.colMask[j] & b.colMask[j];
  }
  return out;
}

MatrixSparsity union2D(const MatrixSparsity &a, const MatrixSparsity &b) {
  MatrixSparsity out;
  size_t rows = std::min(a.rowMask.size(), b.rowMask.size());
  size_t cols = std::min(a.colMask.size(), b.colMask.size());
  out.rowMask.resize(rows);
  out.colMask.resize(cols);
  for (size_t i = 0; i < rows; ++i) {
    out.rowMask[i] = a.rowMask[i] | b.rowMask[i];
  }
  for (size_t j = 0; j < cols; ++j) {
    out.colMask[j] = a.colMask[j] | b.colMask[j];
  }
  return out;
}

MatrixSparsity transpose(const MatrixSparsity &m) {
  MatrixSparsity out;
  out.rowMask = m.colMask;
  out.colMask = m.rowMask;
  return out;
}

} // namespace sparseflow