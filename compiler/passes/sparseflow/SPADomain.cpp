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
  // Dense = no N:M pattern
  s.rowPattern = std::nullopt;
  s.colPattern = std::nullopt;
  return s;
}

MatrixSparsity makeZero(int rows, int cols) {
  rows = clampDim(rows);
  cols = clampDim(cols);
  MatrixSparsity s;
  s.rowMask.assign(static_cast<size_t>(rows), 0);
  s.colMask.assign(static_cast<size_t>(cols), 0);
  // Zero = no N:M pattern
  s.rowPattern = std::nullopt;
  s.colPattern = std::nullopt;
  return s;
}

// NEW: Create matrix with N:M pattern
MatrixSparsity makeNMPattern(int rows, int cols, int N, int M, bool rowWise) {
  rows = clampDim(rows);
  cols = clampDim(cols);
  
  MatrixSparsity s;
  s.rowMask.assign(static_cast<size_t>(rows), 1);
  s.colMask.assign(static_cast<size_t>(cols), 1);
  
  NMPattern pattern(N, M, rowWise);
  
  if (rowWise) {
    s.rowPattern = pattern;
    // For rowwise, set row mask based on pattern
    // Every M rows, N are active
    for (int i = 0; i < rows; i += M) {
      int activeInBlock = std::min(N, rows - i);
      for (int j = activeInBlock; j < M && (i + j) < rows; ++j) {
        s.rowMask[i + j] = 0;  // Inactive
      }
    }
  } else {
    s.colPattern = pattern;
    // For colwise, set col mask
    for (int j = 0; j < cols; j += M) {
      int activeInBlock = std::min(N, cols - j);
      for (int k = activeInBlock; k < M && (j + k) < cols; ++k) {
        s.colMask[j + k] = 0;  // Inactive
      }
    }
  }
  
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
  
  // Pattern intersection: conservative - only keep if both have same pattern
  if (a.rowPattern.has_value() && b.rowPattern.has_value()) {
    if (a.rowPattern.value() == b.rowPattern.value()) {
      out.rowPattern = a.rowPattern;
    }
  }
  
  if (a.colPattern.has_value() && b.colPattern.has_value()) {
    if (a.colPattern.value() == b.colPattern.value()) {
      out.colPattern = a.colPattern;
    }
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
  
  // Pattern union: lose structure (too conservative to propagate)
  out.rowPattern = std::nullopt;
  out.colPattern = std::nullopt;
  
  return out;
}

MatrixSparsity transpose(const MatrixSparsity &m) {
  MatrixSparsity out;
  out.rowMask = m.colMask;
  out.colMask = m.rowMask;
  
  // Swap pattern directions on transpose
  out.rowPattern = m.colPattern;
  out.colPattern = m.rowPattern;
  
  return out;
}

// NEW: Propagate N:M patterns through matmul
MatrixSparsity propagateNMPattern(
    const MatrixSparsity &lhs, 
    const MatrixSparsity &rhs,
    bool isMatMul) {
  
  MatrixSparsity result;
  
  if (!isMatMul) {
    // For non-matmul ops, do conservative intersection
    return intersect2D(lhs, rhs);
  }
  
  // For matmul: C = A × B
  // Result rows inherit from A's row pattern
  // Result cols inherit from B's col pattern
  
  result.rowMask = lhs.rowMask;
  result.colMask = rhs.colMask;
  
  // Propagate row pattern from LHS
  if (lhs.rowPattern.has_value()) {
    result.rowPattern = lhs.rowPattern;
  }
  
  // Propagate col pattern from RHS
  if (rhs.colPattern.has_value()) {
    result.colPattern = rhs.colPattern;
  }
  
  return result;
}

} // namespace sparseflow
