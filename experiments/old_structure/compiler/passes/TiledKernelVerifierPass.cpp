#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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

static bool isInsideStepOneLoop(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>()) {
        if (stepOp.value() == 1)
          return true;
      }
    }
    parent = parent->getParentOp();
  }
  return false;
}

static bool isAfterFirstBarrier(Operation *op, gpu::GPUFuncOp kernel) {
  bool foundBarrier = false;
  for (Block &block : kernel.getBody()) {
    for (Operation &iter : block) {
      if (&iter == op)
        return foundBarrier;
      if (isa<gpu::BarrierOp>(&iter))
        foundBarrier = true;
    }
  }
  return false;
}

struct TiledKernelVerifierPass
    : public PassWrapper<TiledKernelVerifierPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TiledKernelVerifierPass)

  StringRef getArgument() const final { 
    return "sparseflow-gpu-verify-tiled"; 
  }
  
  StringRef getDescription() const final {
    return "Verify SparseFlow v0.7 tiled kernel invariants";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<TiledKernelVerifierPass>(*this);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp gpuModule = getOperation();
    bool failed = false;

    gpuModule.walk([&](gpu::GPUFuncOp kernel) {
      if (!kernel.isKernel())
        return;

      auto modeAttr = kernel->getAttrOfType<StringAttr>("mode");
      if (!modeAttr || modeAttr.getValue() != "tiled")
        return;

      unsigned barrierCount = 0;
      bool hasSharedAlloca = false;
      bool hasStepOneFor = false;
      bool hasGlobalLoadInCompute = false;

      kernel.walk([&](gpu::BarrierOp) { barrierCount++; });

      kernel.walk([&](memref::AllocaOp alloca) {
        if (isSharedMemory(alloca.getType()))
          hasSharedAlloca = true;
      });

      kernel.walk([&](scf::ForOp forOp) {
        if (auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>()) {
          if (stepOp.value() == 1)
            hasStepOneFor = true;
        }
      });

      kernel.walk([&](memref::LoadOp load) {
        Operation *op = load.getOperation();
        if (!isInsideStepOneLoop(op))
          return;
        if (!isAfterFirstBarrier(op, kernel))
          return;
        if (!isSharedMemory(load.getMemref().getType()))
          hasGlobalLoadInCompute = true;
      });

      if (barrierCount < 2) {
        kernel.emitError("tiled kernel invariant violated: expected >= 2 gpu.barrier");
        failed = true;
      }
      if (!hasSharedAlloca) {
        kernel.emitError("tiled kernel invariant violated: missing memref.alloca addrspace(3)");
        failed = true;
      }
      if (!hasStepOneFor) {
        kernel.emitError("tiled kernel invariant violated: missing scf.for step=1");
        failed = true;
      }
      if (hasGlobalLoadInCompute) {
        kernel.emitError("tiled kernel invariant violated: global loads in compute loop");
        failed = true;
      }
    });

    if (failed)
      signalPassFailure();
  }
};

static PassRegistration<TiledKernelVerifierPass> tiledVerifierPass;

} // namespace
