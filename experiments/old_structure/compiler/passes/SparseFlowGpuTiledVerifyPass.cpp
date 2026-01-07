// SparseFlowGpuTiledVerifyPass.cpp
// v0.7: Verify tiled kernel invariants for SparseFlow GPU kernels.

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isTiledKernel(gpu::GPUFuncOp f) {
  auto attr = f->getAttrOfType<StringAttr>("mode");
  return attr && attr.getValue() == "tiled";
}

static bool isAddrSpace3Memref(Type t) {
  auto mr = dyn_cast<MemRefType>(t);
  if (!mr)
    return false;

  if (auto ms = mr.getMemorySpace()) {
    if (auto i = dyn_cast<IntegerAttr>(ms))
      return i.getInt() == 3;
  }
  return false;
}

static bool isInsideStepOneFor(Operation *op) {
  Operation *cur = op->getParentOp();
  while (cur) {
    if (auto forOp = dyn_cast<scf::ForOp>(cur)) {
      auto step = forOp.getStep();
      if (auto cst = step.getDefiningOp<arith::ConstantIndexOp>()) {
        if (cst.value() == 1)
          return true;
      }
    }
    cur = cur->getParentOp();
  }
  return false;
}

static bool loadFromShared(memref::LoadOp load) {
  Value memref = load.getMemref();
  auto t = memref.getType();
  return isAddrSpace3Memref(t);
}

static bool isAfterFirstBarrier(Operation *op, gpu::GPUFuncOp func) {
  bool seenBarrier = false;
  for (auto &bb : func.getBody()) {
    for (auto &it : bb) {
      Operation *cur = &it;
      if (cur == op)
        return seenBarrier;
      if (isa<gpu::BarrierOp>(cur))
        seenBarrier = true;
    }
  }
  return false;
}

struct SparseFlowGpuTiledVerifyPass
    : public PassWrapper<SparseFlowGpuTiledVerifyPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseFlowGpuTiledVerifyPass)

  StringRef getArgument() const final { return "sparseflow-gpu-verify-tiled"; }
  StringRef getDescription() const final {
    return "Verify v0.7 tiled SparseFlow GPU kernel invariants";
  }

  void runOnOperation() override {
    gpu::GPUModuleOp mod = getOperation();
    bool anyFailed = false;

    mod.walk([&](gpu::GPUFuncOp kernel) {
      if (!kernel.isKernel())
        return;
      if (!isTiledKernel(kernel))
        return;

      int barrierCount = 0;
      bool hasSharedAlloca = false;
      bool hasStepOneFor = false;
      bool hasGlobalLoadInCompute = false;

      kernel.walk([&](gpu::BarrierOp) { barrierCount++; });

      kernel.walk([&](memref::AllocaOp a) {
        if (isAddrSpace3Memref(a.getType()))
          hasSharedAlloca = true;
      });

      kernel.walk([&](scf::ForOp forOp) {
        if (auto cst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>()) {
          if (cst.value() == 1)
            hasStepOneFor = true;
        }
      });

      kernel.walk([&](memref::LoadOp load) {
        Operation *op = load.getOperation();
        if (!isInsideStepOneFor(op))
          return;
        if (!isAfterFirstBarrier(op, kernel))
          return;
        if (!loadFromShared(load)) {
          hasGlobalLoadInCompute = true;
        }
      });

      auto fail = [&](StringRef msg) {
        kernel.emitError(msg);
        anyFailed = true;
      };

      if (barrierCount < 2)
        fail("tiled kernel invariant violated: expected >= 2 gpu.barrier ops");
      if (!hasSharedAlloca)
        fail("tiled kernel invariant violated: expected memref.alloca in addrspace(3)");
      if (!hasStepOneFor)
        fail("tiled kernel invariant violated: expected scf.for with step=1");
      if (hasGlobalLoadInCompute)
        fail("tiled kernel invariant violated: global loads in compute loop");
    });

    if (anyFailed)
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
void registerSparseFlowGpuTiledVerifyPass() {
  PassRegistration<SparseFlowGpuTiledVerifyPass>();
}
}
