#pragma once

#include <vector>
#include <cstdint>

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace sparseflow {

struct MatrixSparsity {
  // 1 byte per row: 1 = row may be non-zero, 0 = provably all-zero
  std::vector<std::uint8_t> rowMask;
  // For transpose support: track column sparsity too (optional)
  std::vector<std::uint8_t> colMask;
};

MatrixSparsity makeDenseRows(int rows);
MatrixSparsity makeDenseRowsCols(int rows, int cols);
MatrixSparsity intersectRows(const MatrixSparsity &a,
                             const MatrixSparsity &b);
MatrixSparsity unionRows(const MatrixSparsity &a,
                         const MatrixSparsity &b);
MatrixSparsity transposeSparsity(const MatrixSparsity &m);

// Map from tensor SSA value to its row-wise sparsity info
using SparsityMap = llvm::DenseMap<mlir::Value, MatrixSparsity>;

} // namespace sparseflow
