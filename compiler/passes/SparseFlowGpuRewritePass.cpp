#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    return "Lower SparseFlow runtime calls to GPU (v0.3-alpha, bufferization-ready)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<gpu::GPUDialect>();
    ctx->loadDialect<bufferization::BufferizationDialect>();
    ctx->loadDialect<tensor::TensorDialect>();
    ctx->loadDialect<memref::MemRefDialect>();

    ensureGpuModule(module);

    SmallVector<func::CallOp> callsToErase;

    module.walk([&](func::CallOp call) {
      StringRef callee = call.getCallee();
      if (!callee.starts_with("sparse_matmul"))
        return;

      OpBuilder builder(call);
      Location loc = call.getLoc();

      // v0.4-prep: materialize memrefs (strips tensor encoding)
      SmallVector<Value, 8> memrefArgs;
      memrefArgs.reserve(call.getNumOperands());

      for (Value v : call.getOperands()) {
        memrefArgs.push_back(toMemrefDroppingEncoding(builder, loc, v));
      }

      // v0.3-alpha: still emit gpu.launch stub
      auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

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

  // Convert tensor -> memref safely even if tensor has encoding ({n,m}).
  // Step 1: tensor.cast to unencoded tensor
  // Step 2: bufferization.to_memref
  Value toMemrefDroppingEncoding(OpBuilder &b, Location loc, Value tensorVal) {
    auto t = llvm::dyn_cast<TensorType>(tensorVal.getType());
    if (!t)
      return tensorVal; // already memref or scalar

    auto ranked = llvm::dyn_cast<RankedTensorType>(t);
    if (!ranked)
      return tensorVal; // unranked

    // Drop encoding via tensor.cast
    RankedTensorType plainTensor =
        RankedTensorType::get(ranked.getShape(), ranked.getElementType());

    Value plain = tensorVal;
    if (ranked != plainTensor) {
      plain = b.create<tensor::CastOp>(loc, plainTensor, tensorVal);
    }

    // Now convert to memref
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
