#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {

struct SparseFlowGpuRewritePass
    : public PassWrapper<SparseFlowGpuRewritePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseFlowGpuRewritePass)

  StringRef getArgument() const override { return "sparseflow-gpu-rewrite"; }
  StringRef getDescription() const override {
    return "Lower sparseflow CPU calls to GPU (v0.3)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<gpu::GPUDialect>();

    ensureGpuModuleAndKernel(module);

    SmallVector<func::CallOp> callsToErase;

    module.walk([&](func::CallOp call) {
      auto callee = call.getCallee();
      if (!callee.starts_with("sparse_matmul"))
        return;

      OpBuilder builder(call);
      Location loc = call.getLoc();

      auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

      auto launch = builder.create<gpu::LaunchOp>(
          loc,
          c1, c1, c1,
          c1, c1, c1);

      Block &body = launch.getBody().front();
      OpBuilder bodyBuilder(&body, body.begin());
      bodyBuilder.create<gpu::TerminatorOp>(loc);

      callsToErase.push_back(call);
    });

    for (auto call : callsToErase)
      call.erase();
  }

  void ensureGpuModuleAndKernel(ModuleOp module) {
    // Check if already exists
    if (module.lookupSymbol<gpu::GPUModuleOp>("sf_gpu"))
      return;

    OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
    Location loc = builder.getUnknownLoc();
    
    auto gpuMod = builder.create<gpu::GPUModuleOp>(loc, "sf_gpu");
    Block *modBlock = gpuMod.getBody();
    
    // Check if module_end already exists (it might)
    bool hasEnd = false;
    for (auto &op : *modBlock) {
      if (isa<gpu::ModuleEndOp>(op)) {
        hasEnd = true;
        break;
      }
    }
    
    // Add kernel before any existing module_end
    OpBuilder modBuilder(modBlock, hasEnd ? std::prev(modBlock->end()) : modBlock->end());
    
    auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
    auto gpuFunc = modBuilder.create<gpu::GPUFuncOp>(
        loc, "kernel", funcType, TypeRange{}, TypeRange{});
    
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     builder.getUnitAttr());
    
    Block &entry = gpuFunc.getBody().front();
    OpBuilder funcBuilder = OpBuilder::atBlockEnd(&entry);
    funcBuilder.create<gpu::ReturnOp>(loc);
    
    // Only add module_end if it doesn't exist
    if (!hasEnd) {
      modBuilder.setInsertionPointToEnd(modBlock);
      modBuilder.create<gpu::ModuleEndOp>(loc);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSparseFlowGpuRewritePass() {
  return std::make_unique<SparseFlowGpuRewritePass>();
}

void registerSparseFlowGpuRewritePass() {
  mlir::PassRegistration<SparseFlowGpuRewritePass>();
}
