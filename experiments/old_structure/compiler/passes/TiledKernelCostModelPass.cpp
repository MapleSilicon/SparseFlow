#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isSharedMemory(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return false;
  auto memorySpace = memrefType.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == 3;
  return false;
}

struct TiledKernelCostModelPass
    : public PassWrapper<TiledKernelCostModelPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TiledKernelCostModelPass)

  StringRef getArgument() const final { 
    return "sparseflow-gpu-cost-model"; 
  }
  
  StringRef getDescription() const final {
    return "SparseFlow v0.7 tiled kernel cost analysis";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<TiledKernelCostModelPass>(*this);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp gpuModule = getOperation();

    gpuModule.walk([&](gpu::GPUFuncOp kernel) {
      if (kernel.getName() != "sparseflow_tiled_kernel_v07")
        return;

      unsigned globalLoads = 0;
      unsigned sharedLoads = 0;
      unsigned shuffles = 0;

      kernel.walk([&](memref::LoadOp load) {
        if (isSharedMemory(load.getMemref().getType()))
          sharedLoads++;
        else
          globalLoads++;
      });

      kernel.walk([&](gpu::ShuffleOp) { shuffles++; });

      double reuseFactor = (globalLoads > 0) 
          ? static_cast<double>(sharedLoads) / static_cast<double>(globalLoads)
          : 0.0;

      llvm::outs() << "--- SparseFlow v0.7 (tiled) Cost Report ---\n"
                   << "  global loads:  " << globalLoads << "\n"
                   << "  shared loads:  " << sharedLoads << "\n"
                   << "  shuffles:      " << shuffles << "\n"
                   << "  reuse factor:  " << reuseFactor << "x\n";
    });
  }
};

static PassRegistration<TiledKernelCostModelPass> tiledCostModelPass;

} // namespace
