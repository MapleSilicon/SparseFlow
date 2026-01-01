#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "SPADomain.h"

using namespace mlir;
using namespace sparseflow;

namespace {

using sparseflow::makeDense;
using sparseflow::makeZero;
using sparseflow::makeNMPattern;
using sparseflow::intersect2D;
using sparseflow::union2D;
using sparseflow::propagateNMPattern;

struct SparsityPropagationPass
    : public PassWrapper<SparsityPropagationPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparsityPropagationPass)
  
  StringRef getArgument() const final { return "sparseflow-spa"; }
  StringRef getDescription() const final {
    return "Propagate 2D sparsity (rows + cols) through operations - v0.6";
  }
  
  SparsityMap S;
  
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        handleMatmul(mm);
      } else if (auto add = dyn_cast<arith::AddFOp>(op)) {
        handleAddLike(add);
      } else if (auto sub = dyn_cast<arith::SubFOp>(op)) {
        handleAddLike(sub);
      } else if (auto mul = dyn_cast<arith::MulFOp>(op)) {
        handleMulLike(mul);
      } else if (auto div = dyn_cast<arith::DivFOp>(op)) {
        handleMulLike(div);
      } else if (auto maxf = dyn_cast<arith::MaximumFOp>(op)) {
        handleMaximum(maxf);
      } else if (auto trans = dyn_cast<linalg::TransposeOp>(op)) {
        handleTranspose(trans);
      } else if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
        handleReduce(reduce);
      } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
        handleBroadcast(expand);
      }
    });
  }
  
  MatrixSparsity normalize(const MatrixSparsity &m, int rows, int cols) {
    if (rows < 0) rows = 0;
    if (cols < 0) cols = 0;
    
    MatrixSparsity out;
    out.rowMask.assign(rows, 1);
    out.colMask.assign(cols, 1);
    
    int r = std::min((int)m.rowMask.size(), rows);
    int c = std::min((int)m.colMask.size(), cols);
    
    for (int i = 0; i < r; i++) out.rowMask[i] = m.rowMask[i];
    for (int j = 0; j < c; j++) out.colMask[j] = m.colMask[j];
    
    return out;
  }
  
  MatrixSparsity getSparsity(Value v, int rows, int cols) {
    if (auto it = S.find(v); it != S.end()) {
      return normalize(it->second, rows, cols);
    }
    
    if (Operation *op = v.getDefiningOp()) {
      MatrixSparsity m;
      bool hasRowMask = false;
      bool hasColMask = false;
      
      if (auto arr = op->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask")) {
        m.rowMask.reserve(arr.size());
        for (auto a : arr) {
          m.rowMask.push_back(cast<BoolAttr>(a).getValue() ? 1 : 0);
        }
        hasRowMask = true;
      }
      
      if (auto arr = op->getAttrOfType<ArrayAttr>("sparseflow.spa_colmask")) {
        m.colMask.reserve(arr.size());
        for (auto a : arr) {
          m.colMask.push_back(cast<BoolAttr>(a).getValue() ? 1 : 0);
        }
        hasColMask = true;
      }
      
      if (hasRowMask || hasColMask) {
        if (!hasRowMask) m.rowMask.assign(rows, 1);
        if (!hasColMask) m.colMask.assign(cols, 1);
        return normalize(m, rows, cols);
      }
      
      if (auto nAttr = op->getAttrOfType<IntegerAttr>("sparseflow.n")) {
        if (auto mAttr = op->getAttrOfType<IntegerAttr>("sparseflow.m")) {
          int n = nAttr.getInt();
          int m_val = mAttr.getInt();
          
          // Check direction (default: rowwise)
          bool rowWise = true;
          if (auto dirAttr = op->getAttrOfType<StringAttr>("sparseflow.direction")) {
            rowWise = (dirAttr.getValue() == "row");
          }
          
          // Use new N:M pattern factory
          return makeNMPattern(rows, cols, n, m_val, rowWise);
        }
      }
      
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat()) {
            auto val = denseAttr.getSplatValue<APFloat>();
            if (val.isZero()) {
              return makeZero(rows, cols);
            }
          }
        }
      }
    }
    
    return makeDense(rows, cols);
  }
  
  void attach(Operation *op, const MatrixSparsity &info) {
    MLIRContext *ctx = op->getContext();
    
    // Attach row/col masks
    llvm::SmallVector<Attribute, 8> rowElems;
    rowElems.reserve(info.rowMask.size());
    for (auto bit : info.rowMask) {
      rowElems.push_back(BoolAttr::get(ctx, bit != 0));
    }
    op->setAttr("sparseflow.spa_rowmask", ArrayAttr::get(ctx, rowElems));
    
    llvm::SmallVector<Attribute, 8> colElems;
    colElems.reserve(info.colMask.size());
    for (auto bit : info.colMask) {
      colElems.push_back(BoolAttr::get(ctx, bit != 0));
    }
    op->setAttr("sparseflow.spa_colmask", ArrayAttr::get(ctx, colElems));
    
    // Attach N:M pattern metadata if present
    if (info.rowPattern.has_value()) {
      auto pattern = info.rowPattern.value();
      op->setAttr("sparseflow.nm_n", 
                  IntegerAttr::get(IntegerType::get(ctx, 32), pattern.N));
      op->setAttr("sparseflow.nm_m", 
                  IntegerAttr::get(IntegerType::get(ctx, 32), pattern.M));
      op->setAttr("sparseflow.nm_direction", StringAttr::get(ctx, "row"));
      op->setAttr("sparseflow.nm_pattern", StringAttr::get(ctx, pattern.getName()));
    } else if (info.colPattern.has_value()) {
      auto pattern = info.colPattern.value();
      op->setAttr("sparseflow.nm_n", 
                  IntegerAttr::get(IntegerType::get(ctx, 32), pattern.N));
      op->setAttr("sparseflow.nm_m", 
                  IntegerAttr::get(IntegerType::get(ctx, 32), pattern.M));
      op->setAttr("sparseflow.nm_direction", StringAttr::get(ctx, "col"));
      op->setAttr("sparseflow.nm_pattern", StringAttr::get(ctx, pattern.getName()));
    }
  }
  
  void handleMatmul(linalg::MatmulOp mm) {
    Value result = mm.getResult(0);
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    if (rows <= 0 || cols <= 0) return;
    
    // Check for explicit N:M annotation on the matmul itself
    if (auto nAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.n")) {
      if (auto mAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.m")) {
        int n = nAttr.getInt();
        int m_val = mAttr.getInt();
        
        bool rowWise = true;
        if (auto dirAttr = mm->getAttrOfType<StringAttr>("sparseflow.direction")) {
          rowWise = (dirAttr.getValue() == "row");
        }
        
        MatrixSparsity outS = makeNMPattern(rows, cols, n, m_val, rowWise);
        S[result] = outS;
        attach(mm, outS);
        return;
      }
    }
    
    auto inputs = mm.getInputs();
    if (inputs.size() < 2) return;
    
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) return;
    
    int64_t lhsCols = lhsType.getShape()[1];
    int64_t rhsRows = rhsType.getShape()[0];
    
    // Get sparsity with N:M pattern info
    MatrixSparsity lhsS = getSparsity(lhs, rows, lhsCols);
    MatrixSparsity rhsS = getSparsity(rhs, rhsRows, cols);
    
    // Use pattern-aware propagation (preserves N:M structure)
    MatrixSparsity outS = propagateNMPattern(lhsS, rhsS, /*isMatMul=*/true);
    
    S[result] = outS;
    attach(mm, outS);
  }
  
  void handleAddLike(Operation *op) {
    Value result = op->getResult(0);
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    
    MatrixSparsity lhsS = getSparsity(lhs, rows, cols);
    MatrixSparsity rhsS = getSparsity(rhs, rows, cols);
    
    MatrixSparsity outS = union2D(lhsS, rhsS);
    
    S[result] = outS;
    attach(op, outS);
  }
  
  void handleMulLike(Operation *op) {
    Value result = op->getResult(0);
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    
    MatrixSparsity lhsS = getSparsity(lhs, rows, cols);
    MatrixSparsity rhsS = getSparsity(rhs, rows, cols);
    
    MatrixSparsity outS = intersect2D(lhsS, rhsS);
    
    S[result] = outS;
    attach(op, outS);
  }
  
  void handleMaximum(arith::MaximumFOp maxOp) {
    Value result = maxOp.getResult();
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    
    Value lhs = maxOp.getLhs();
    Value rhs = maxOp.getRhs();
    
    bool lhsIsZero = false;
    bool rhsIsZero = false;
    
    if (auto lhsDef = lhs.getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(lhsDef)) {
        if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero()) {
            lhsIsZero = true;
          }
        }
      }
    }
    
    if (auto rhsDef = rhs.getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
        if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero()) {
            rhsIsZero = true;
          }
        }
      }
    }
    
    MatrixSparsity outS;
    
    if (lhsIsZero) {
      outS = getSparsity(rhs, rows, cols);
    } else if (rhsIsZero) {
      outS = getSparsity(lhs, rows, cols);
    } else {
      MatrixSparsity lhsS = getSparsity(lhs, rows, cols);
      MatrixSparsity rhsS = getSparsity(rhs, rows, cols);
      outS = union2D(lhsS, rhsS);
    }
    
    S[result] = outS;
    attach(maxOp, outS);
  }
  
  void handleTranspose(linalg::TransposeOp transOp) {
    Value result = transOp->getResult(0);
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    
    Value input = transOp.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) return;
    
    int64_t inRows = inputType.getShape()[0];
    int64_t inCols = inputType.getShape()[1];
    
    MatrixSparsity inS = getSparsity(input, inRows, inCols);
    MatrixSparsity outS = transpose(inS);
    
    S[result] = outS;
    attach(transOp, outS);
  }
  
  void handleReduce(linalg::ReduceOp reduceOp) {
    auto results = reduceOp.getResults();
    if (results.empty()) return;
    
    Value result = results[0];
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType) return;
    
    auto inputs = reduceOp.getInputs();
    if (inputs.empty()) return;
    
    Value input = inputs[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 2) return;
    
    int64_t inRows = inputType.getShape()[0];
    int64_t inCols = inputType.getShape()[1];
    
    MatrixSparsity inS = getSparsity(input, inRows, inCols);
    
    auto dims = reduceOp.getDimensions();
    if (dims.empty()) return;
    
    int64_t dim = dims[0];
    
    MatrixSparsity outS;
    if (dim == 0) {
      outS.rowMask.assign(1, 1);
      outS.colMask = inS.colMask;
    } else if (dim == 1) {
      outS.rowMask = inS.rowMask;
      outS.colMask.assign(1, 1);
    } else {
      return;
    }
    
    S[result] = outS;
    attach(reduceOp, outS);
  }
  
  void handleBroadcast(tensor::ExpandShapeOp expandOp) {
    Value result = expandOp.getResult();
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2) return;
    
    int64_t rows = resultType.getShape()[0];
    int64_t cols = resultType.getShape()[1];
    
    Value input = expandOp.getSrc();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) return;
    
    int64_t inRows = inputType.getRank() >= 1 ? inputType.getShape()[0] : 1;
    int64_t inCols = inputType.getRank() >= 2 ? inputType.getShape()[1] : 1;
    
    MatrixSparsity inS = getSparsity(input, inRows, inCols);
    
    MatrixSparsity outS;
    outS.rowMask.assign(rows, inS.rowMask.empty() ? 1 : inS.rowMask[0]);
    outS.colMask.assign(cols, inS.colMask.empty() ? 1 : inS.colMask[0]);
    
    S[result] = outS;
    attach(expandOp, outS);
  }
};

} // namespace

namespace {
inline PassRegistration<SparsityPropagationPass> registerPass;
}

void registerSparsityPropagationPass() {
  PassRegistration<SparsityPropagationPass>();
}