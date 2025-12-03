#pragma once
#include <vector>
#include <cstdint>
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace sparseflow {

struct MatrixSparsity {
  std::vector<std::uint8_t> rowMask;
  std::vector<std::uint8_t> colMask;
};

MatrixSparsity makeDense(int rows, int cols);
MatrixSparsity makeZero(int rows, int cols);
MatrixSparsity intersect2D(const MatrixSparsity &a, const MatrixSparsity &b);
MatrixSparsity union2D(const MatrixSparsity &a, const MatrixSparsity &b);
MatrixSparsity transpose(const MatrixSparsity &m);

using SparsityMap = llvm::DenseMap<mlir::Value, MatrixSparsity>;

} // namespace sparseflow
