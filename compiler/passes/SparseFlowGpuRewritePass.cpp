#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct SparseFlowGpuRewritePass
    : public PassWrapper<SparseFlowGpuRewritePass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseFlowGpuRewritePass)

  StringRef getArgument() const override {
    return "sparseflow-gpu-rewrite";
  }

  StringRef getDescription() const override {
    return "Lower SparseFlow to GPU with rowMask generation (v0.4 corrected)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<gpu::GPUDialect>();
    ctx->loadDialect<bufferization::BufferizationDialect>();
    ctx->loadDialect<tensor::TensorDialect>();
    ctx->loadDialect<memref::MemRefDialect>();
    ctx->loadDialect<scf::SCFDialect>();

    ensureGpuModule(module);

    SmallVector<func::CallOp> callsToErase;

    module.walk([&](func::CallOp call) {
      StringRef callee = call.getCallee();
      if (!callee.starts_with("sparse_matmul"))
        return;

      // ❌ ISSUE 4 FIX: Guard against GPU-side generation
      if (call->getParentOfType<gpu::GPUModuleOp>()) {
        call.emitError("RowMask generation must be host-side");
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      Location loc = call.getLoc();

      // Convert tensor operands to memrefs
      SmallVector<Value, 8> memrefArgs;
      memrefArgs.reserve(call.getNumOperands());

      for (Value v : call.getOperands()) {
        memrefArgs.push_back(toMemrefDroppingEncoding(builder, loc, v));
      }

      // ---- Step 1: Constants ----
      auto c0  = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto c1  = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto c31 = builder.create<arith::ConstantIntOp>(loc, 31, 32);
      auto c32 = builder.create<arith::ConstantIntOp>(loc, 32, 32);

      // ---- Step 2: Extract N:M from encoding ----
      auto A = call.getOperand(0);
      auto tensorType = llvm::cast<RankedTensorType>(A.getType());
      
      // ❌ ISSUE 2 FIX: Extract n,m from encoding
      auto encoding = tensorType.getEncoding();
      if (!encoding) {
        call.emitError("Tensor must have N:M encoding");
        signalPassFailure();
        return;
      }

      auto dictAttr = llvm::dyn_cast<DictionaryAttr>(encoding);
      if (!dictAttr) {
        call.emitError("Encoding must be a dictionary");
        signalPassFailure();
        return;
      }

      auto nAttr = dictAttr.get("n");
      auto mAttr = dictAttr.get("m");
      if (!nAttr || !mAttr) {
        call.emitError("Encoding must contain 'n' and 'm' attributes");
        signalPassFailure();
        return;
      }

      int64_t n = llvm::cast<IntegerAttr>(nAttr).getInt();
      int64_t m = llvm::cast<IntegerAttr>(mAttr).getInt();

      if (n != 2 || m != 4) {
        call.emitError("v0.4 supports only 2:4 sparsity");
        signalPassFailure();
        return;
      }

      // Compute row count
      int64_t rows = tensorType.getShape()[0];
      auto rowsVal = builder.create<arith::ConstantIntOp>(loc, rows, 32);

      // ❌ ISSUE 1 FIX: Correct ceil division
      // words = (rows + 31) / 32
      auto rowsPlus31 = builder.create<arith::AddIOp>(loc, rowsVal, c31);
      auto words = builder.create<arith::DivUIOp>(loc, rowsPlus31, c32);
      
      auto wordsIndex = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), words);

      // ---- Step 3: Allocate rowMask ----
      auto rowMaskType =
          MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());

      auto rowMaskOp =
          builder.create<memref::AllocOp>(loc, rowMaskType, ValueRange{wordsIndex});

      // ❌ ISSUE 2 FIX: Generate pattern from n,m
      // For N=2, M=4: pattern = 0b0011 = (1 << 2) - 1 = 3
      uint32_t maskValue = (1u << n) - 1;
      auto pattern =
          builder.create<arith::ConstantIntOp>(loc, maskValue, 32);

      // ---- Step 4: Fill mask ----
      builder.create<scf::ForOp>(
          loc, c0, wordsIndex, c1, ValueRange{},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
            b.create<memref::StoreOp>(loc, pattern, rowMaskOp, iv);
            b.create<scf::YieldOp>(loc);
          });

      // ❌ ISSUE 3 FIX: Mark rowMask as observable
      rowMaskOp->setAttr(
          "sparseflow.rowmask", builder.getUnitAttr());

      // ---- Step 5: Create gpu.launch ----
      auto launch = builder.create<gpu::LaunchOp>(
          loc,
          c1, c1, c1,
          c1, c1, c1);

      {
        Block &body = launch.getBody().front();
        OpBuilder bodyBuilder(&body, body.begin());
        bodyBuilder.create<gpu::TerminatorOp>(loc);
      }

      callsToErase.push_back(call);
    });

    for (auto call : callsToErase)
      call.erase();
  }

  Value toMemrefDroppingEncoding(OpBuilder &b, Location loc, Value tensorVal) {
    auto t = llvm::dyn_cast<TensorType>(tensorVal.getType());
    if (!t)
      return tensorVal;

    auto ranked = llvm::dyn_cast<RankedTensorType>(t);
    if (!ranked)
      return tensorVal;

    RankedTensorType plainTensor =
        RankedTensorType::get(ranked.getShape(), ranked.getElementType());

    Value plain = tensorVal;
    if (ranked != plainTensor) {
      plain = b.create<tensor::CastOp>(loc, plainTensor, tensorVal);
    }

    auto memrefTy = MemRefType::get(ranked.getShape(), ranked.getElementType());
    return b.create<bufferization::ToMemrefOp>(loc, memrefTy, plain);
  }

  void ensureGpuModule(ModuleOp module) {
    if (module.lookupSymbol<gpu::GPUModuleOp>("sf_gpu"))
      return;

    OpBuilder builder(module.getBodyRegion());
    auto gpuModule =
        builder.create<gpu::GPUModuleOp>(
            builder.getUnknownLoc(), "sf_gpu");

    Block *block = gpuModule.getBody();
    OpBuilder modBuilder(block->getTerminator());

    auto fnType = modBuilder.getFunctionType({}, {});
    auto kernel = modBuilder.create<gpu::GPUFuncOp>(
        modBuilder.getUnknownLoc(),
        "kernel",
        fnType,
        TypeRange{},
        TypeRange{});

    kernel->setAttr(
        gpu::GPUDialect::getKernelFuncAttrName(),
        modBuilder.getUnitAttr());

    Block &entry = kernel.getBody().front();
    OpBuilder bodyBuilder(&entry, entry.begin());
    bodyBuilder.create<gpu::ReturnOp>(modBuilder.getUnknownLoc());
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createSparseFlowGpuRewritePass() {
  return std::make_unique<SparseFlowGpuRewritePass>();
}
} // namespace mlir

void registerSparseFlowGpuRewritePass() {
  mlir::PassRegistration<SparseFlowGpuRewritePass>();
}
