#pragma once

#include <vector>
#include <cstdint>

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace sparseflow {

struct MatrixSparsity {
  // 1 byte per row: 1 = row may be non-zero, 0 = provably all-zero
  std::vector<std::uint8_t> rowMask;
};

MatrixSparsity makeDenseRows(int rows);
MatrixSparsity intersectRows(const MatrixSparsity &a,
                             const MatrixSparsity &b);
MatrixSparsity unionRows(const MatrixSparsity &a,
                         const MatrixSparsity &b);

// Map from tensor SSA value to its row-wise sparsity info
using SparsityMap = llvm::DenseMap<mlir::Value, MatrixSparsity>;

} // namespace sparseflow
