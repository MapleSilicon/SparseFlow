#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <sstream>
#include <string>

using namespace mlir;

namespace {

static bool extractNMFromType(Type type, int &N, int &M) {
  auto ranked = mlir::dyn_cast<RankedTensorType>(type);
  if (!ranked)
    return false;

  auto dict = mlir::dyn_cast_or_null<DictionaryAttr>(ranked.getEncoding());
  if (!dict)
    return false;

  auto nAttr = dict.get("n");
  auto mAttr = dict.get("m");
  if (!nAttr || !mAttr)
    return false;

  N = mlir::cast<IntegerAttr>(nAttr).getInt();
  M = mlir::cast<IntegerAttr>(mAttr).getInt();
  return true;
}

static std::string buildKernelName(int N, int M) {
  std::ostringstream ss;
  ss << "sparse_matmul_" << N << "_" << M;
  return ss.str();
}

struct SparseMatmulRewritePass
    : public PassWrapper<SparseMatmulRewritePass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override {
    return "sparseflow-rewrite-matmul";
  }

  StringRef getDescription() const override {
    return "Rewrite linalg.matmul ops with N:M-encoded LHS into SparseFlow runtime calls";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](linalg::MatmulOp matmul) {
      Location loc = matmul.getLoc();

      Value A = matmul.getInputs()[0];
      Value B = matmul.getInputs()[1];
      Value C = matmul.getOutputs()[0];

      int N = -1;
      int M = -1;
      if (!extractNMFromType(A.getType(), N, M)) {
        return;
      }

      std::string kernelName = buildKernelName(N, M);

      func::FuncOp targetFunc = module.lookupSymbol<func::FuncOp>(kernelName);
      if (!targetFunc) {
        OpBuilder modBuilder(module.getBodyRegion());

        SmallVector<Type> argTypes;
        argTypes.push_back(A.getType());
        argTypes.push_back(B.getType());
        argTypes.push_back(C.getType());
        argTypes.push_back(modBuilder.getI32Type());
        argTypes.push_back(modBuilder.getI32Type());
        argTypes.push_back(modBuilder.getI32Type());

        auto fnType = modBuilder.getFunctionType(argTypes, {});
        targetFunc = modBuilder.create<func::FuncOp>(loc, kernelName, fnType);
        targetFunc.setPrivate();
      }

      auto ATy = mlir::cast<RankedTensorType>(A.getType());
      auto BTy = mlir::cast<RankedTensorType>(B.getType());

      int64_t Mdim = ATy.getShape()[0];
      int64_t Kdim = ATy.getShape()[1];
      int64_t Ndim = BTy.getShape()[1];

      OpBuilder builder(matmul);

      Value Mval = builder.create<arith::ConstantIntOp>(loc, Mdim, 32);
      Value Kval = builder.create<arith::ConstantIntOp>(loc, Kdim, 32);
      Value Nval = builder.create<arith::ConstantIntOp>(loc, Ndim, 32);

      builder.create<func::CallOp>(
          loc, kernelName, TypeRange{},
          ValueRange{A, B, C, Mval, Kval, Nval});

      matmul.getResult(0).replaceAllUsesWith(C);
      matmul.erase();
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createSparseMatmulRewritePass() {
  return std::make_unique<SparseMatmulRewritePass>();
}
} // namespace mlir

static PassRegistration<SparseMatmulRewritePass> pass;
