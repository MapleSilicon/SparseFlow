#include "SPADomain.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

using sparseflow::MatrixSparsity;
using sparseflow::SparsityMap;
using sparseflow::intersectRows;
using sparseflow::makeDenseRows;
using sparseflow::unionRows;

struct SparsityPropagationPass
    : public PassWrapper<SparsityPropagationPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const final { return "sparseflow-spa"; }
  StringRef getDescription() const final {
    return "SparseFlow Sparsity Propagation Analysis (SPA) pass";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        handleMatmul(mm);
      } else if (auto addf = dyn_cast<arith::AddFOp>(op)) {
        handleAdd(addf);
      } else if (auto mulf = dyn_cast<arith::MulFOp>(op)) {
        handleMul(mulf);
      } else if (auto subf = dyn_cast<arith::SubFOp>(op)) {
        handleSub(subf);
      } else if (auto divf = dyn_cast<arith::DivFOp>(op)) {
        handleDiv(divf);
      }
    });
  }

private:
  SparsityMap S;

  MatrixSparsity normalizeRows(const MatrixSparsity &m, int64_t expectedRows) {
    MatrixSparsity out;
    if (expectedRows <= 0) return out;
    size_t rows = static_cast<size_t>(expectedRows);
    out.rowMask.assign(rows, 1);
    size_t n = std::min(rows, m.rowMask.size());
    for (size_t i = 0; i < n; ++i)
      out.rowMask[i] = m.rowMask[i] ? 1 : 0;
    return out;
  }

  MatrixSparsity getSparsity(Value v, int64_t expectedRows) {
    if (auto it = S.find(v); it != S.end())
      return normalizeRows(it->second, expectedRows);
    if (Operation *defOp = v.getDefiningOp()) {
      if (auto arr = defOp->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask")) {
        MatrixSparsity m;
        m.rowMask.reserve(arr.size());
        for (Attribute attr : arr) {
          if (auto b = dyn_cast<BoolAttr>(attr))
            m.rowMask.push_back(b.getValue() ? 1 : 0);
          else if (auto i = dyn_cast<IntegerAttr>(attr))
            m.rowMask.push_back(i.getValue().isZero() ? 0 : 1);
          else
            m.rowMask.push_back(1);
        }
        return normalizeRows(m, expectedRows);
      }
    }
    return makeDenseRows(static_cast<int>(expectedRows));
  }

  void attachRowMaskAttr(Operation *op, const MatrixSparsity &info) {
    MLIRContext *ctx = op->getContext();
    llvm::SmallVector<Attribute, 8> elems;
    elems.reserve(info.rowMask.size());
    for (std::uint8_t bit : info.rowMask)
      elems.push_back(BoolAttr::get(ctx, bit != 0));
    op->setAttr("sparseflow.spa_rowmask", ArrayAttr::get(ctx, elems));
  }

  void handleMatmul(linalg::MatmulOp mm) {
    Value result = mm.getResult(0);
    auto resultType = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    int64_t rows = resultType.getShape()[0];
    if (rows <= 0) return;
    Value lhs;
    auto inputs = mm.getInputs();
    if (!inputs.empty()) lhs = inputs.front();
    MatrixSparsity lhsS = lhs ? getSparsity(lhs, rows) : makeDenseRows(static_cast<int>(rows));
    MatrixSparsity outS = normalizeRows(lhsS, rows);
    S[result] = outS;
    attachRowMaskAttr(mm, outS);
  }

  void handleAdd(arith::AddFOp add) {
    Value res = add.getResult();
    auto resType = mlir::dyn_cast<RankedTensorType>(res.getType());
    if (!resType || resType.getRank() < 2) return;
    int64_t rows = resType.getShape()[0];
    if (rows <= 0) return;
    MatrixSparsity a = getSparsity(add.getLhs(), rows);
    MatrixSparsity b = getSparsity(add.getRhs(), rows);
    MatrixSparsity out = unionRows(a, b);
    S[res] = out;
    attachRowMaskAttr(add, out);
  }

  void handleMul(arith::MulFOp mul) {
    Value res = mul.getResult();
    auto resType = mlir::dyn_cast<RankedTensorType>(res.getType());
    if (!resType || resType.getRank() < 2) return;
    int64_t rows = resType.getShape()[0];
    if (rows <= 0) return;
    MatrixSparsity a = getSparsity(mul.getLhs(), rows);
    MatrixSparsity b = getSparsity(mul.getRhs(), rows);
    MatrixSparsity out = intersectRows(a, b);
    S[res] = out;
    attachRowMaskAttr(mul, out);
  }

  void handleSub(arith::SubFOp sub) {
    Value res = sub.getResult();
    auto resType = mlir::dyn_cast<RankedTensorType>(res.getType());
    if (!resType || resType.getRank() < 2) return;
    int64_t rows = resType.getShape()[0];
    if (rows <= 0) return;
    MatrixSparsity a = getSparsity(sub.getLhs(), rows);
    MatrixSparsity b = getSparsity(sub.getRhs(), rows);
    MatrixSparsity out = unionRows(a, b);
    S[res] = out;
    attachRowMaskAttr(sub, out);
  }

  void handleDiv(arith::DivFOp div) {
    Value res = div.getResult();
    auto resType = mlir::dyn_cast<RankedTensorType>(res.getType());
    if (!resType || resType.getRank() < 2) return;
    int64_t rows = resType.getShape()[0];
    if (rows <= 0) return;
    MatrixSparsity a = getSparsity(div.getLhs(), rows);
    MatrixSparsity b = getSparsity(div.getRhs(), rows);
    MatrixSparsity out = unionRows(a, b);
    S[res] = out;
    attachRowMaskAttr(div, out);
  }
};

}

std::unique_ptr<Pass> createSparsityPropagationPass() {
  return std::make_unique<SparsityPropagationPass>();
}

static PassRegistration<SparsityPropagationPass> regSparseFlowSPA;
