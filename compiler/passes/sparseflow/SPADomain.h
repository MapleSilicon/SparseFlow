#pragma once

#include <vector>
#include <cstdint>
#include <optional>
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace sparseflow {

// N:M Structured Sparsity Pattern
struct NMPattern {
  int N;  // Number of non-zero elements in block
  int M;  // Block size
  bool isRowWise;  // True if pattern applies to rows, false for columns
  
  NMPattern() : N(0), M(0), isRowWise(true) {}
  NMPattern(int n, int m, bool rowWise = true) : N(n), M(m), isRowWise(rowWise) {}
  
  // Check if this is a valid pattern
  bool isValid() const { return N > 0 && M > 0 && N <= M; }
  
  // Get density (0.0 to 1.0)
  float getDensity() const { return M > 0 ? (float)N / (float)M : 0.0f; }
  
  // Pattern name (e.g., "2:4", "1:4")
  std::string getName() const {
    return std::to_string(N) + ":" + std::to_string(M);
  }
  
  // Check if two patterns are compatible for operations
  bool isCompatibleWith(const NMPattern &other) const {
    // Same pattern is always compatible
    if (N == other.N && M == other.M) return true;
    // For now, conservatively require exact match
    // TODO: Add more sophisticated compatibility rules
    return false;
  }
  
  bool operator==(const NMPattern &other) const {
    return N == other.N && M == other.M && isRowWise == other.isRowWise;
  }
};

// Matrix sparsity information with optional N:M pattern
struct MatrixSparsity {
  std::vector<std::uint8_t> rowMask;
  std::vector<std::uint8_t> colMask;
  
  // N:M pattern information (optional)
  std::optional<NMPattern> rowPattern;  // Pattern for rows
  std::optional<NMPattern> colPattern;  // Pattern for columns
  
  // Check if this matrix has N:M structure
  bool hasNMPattern() const {
    return rowPattern.has_value() || colPattern.has_value();
  }
  
  // Get the dominant pattern (row pattern takes precedence)
  std::optional<NMPattern> getDominantPattern() const {
    if (rowPattern.has_value()) return rowPattern;
    return colPattern;
  }
};

// Factory functions for common patterns
MatrixSparsity makeDense(int rows, int cols);
MatrixSparsity makeZero(int rows, int cols);

// Create matrix with specific N:M pattern
MatrixSparsity makeNMPattern(int rows, int cols, int N, int M, bool rowWise = true);

// 2D mask operations
MatrixSparsity intersect2D(const MatrixSparsity &a, const MatrixSparsity &b);
MatrixSparsity union2D(const MatrixSparsity &a, const MatrixSparsity &b);
MatrixSparsity transpose(const MatrixSparsity &m);

// N:M pattern operations
MatrixSparsity propagateNMPattern(
    const MatrixSparsity &lhs, 
    const MatrixSparsity &rhs,
    bool isMatMul = true);

using SparsityMap = llvm::DenseMap<mlir::Value, MatrixSparsity>;

} // namespace sparseflow
