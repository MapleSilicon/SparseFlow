#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct SparseFlowGpuVerifyPass
    : public PassWrapper<SparseFlowGpuVerifyPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseFlowGpuVerifyPass)

  Option<std::string> mode{
      *this, "mode",
      llvm::cl::desc("Verify mode: rowmask | warp | tiled"),
      llvm::cl::init("rowmask")};

  SparseFlowGpuVerifyPass() = default;
  SparseFlowGpuVerifyPass(const SparseFlowGpuVerifyPass &other) : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }

  StringRef getArgument() const override { return "sparseflow-gpu-verify"; }
  StringRef getDescription() const override {
    return "Verify SparseFlow GPU kernel invariants for a given mode";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto gpuMod = module.lookupSymbol<gpu::GPUModuleOp>("sf_gpu");
    if (!gpuMod) {
      module.emitError("Missing gpu.module @sf_gpu");
      signalPassFailure();
      return;
    }

    auto kernel = gpuMod.lookupSymbol<gpu::GPUFuncOp>("sparseflow_gpu_matmul_2_4_f32");
    if (!kernel) {
      gpuMod.emitError("Missing gpu.func @sparseflow_gpu_matmul_2_4_f32");
      signalPassFailure();
      return;
    }

    auto isKernelAttr = kernel->getAttr(gpu::GPUDialect::getKernelFuncAttrName());
    if (!isKernelAttr) {
      kernel.emitError("gpu.func is missing 'kernel' attribute");
      signalPassFailure();
      return;
    }

    bool sawDivui = false, sawShrui = false, sawAndi = false, sawRowmaskLoad = false;
    bool sawShuffle = false;
    bool sawBarrier = false, sawAllocaAS3 = false;

    kernel.walk([&](Operation *op) {
      if (isa<arith::DivUIOp>(op)) sawDivui = true;
      if (isa<arith::ShRUIOp>(op)) sawShrui = true;
      if (isa<arith::AndIOp>(op))  sawAndi = true;

      if (auto ld = dyn_cast<memref::LoadOp>(op)) {
        // rowmask is arg3 in kernel signature: (%arg3: memref<?xi32>)
        if (auto ba = dyn_cast<BlockArgument>(ld.getMemref())) {
          if (ba.getArgNumber() == 3) sawRowmaskLoad = true;
        }
      }

      if (isa<gpu::ShuffleOp>(op)) sawShuffle = true;
      if (isa<gpu::BarrierOp>(op)) sawBarrier = true;

      if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
        auto ty = dyn_cast<MemRefType>(alloca.getType());
        if (!ty) return;
        if (auto ms = dyn_cast_or_null<IntegerAttr>(ty.getMemorySpace())) {
          if (ms.getInt() == 3) sawAllocaAS3 = true;
        }
      }
    });

    auto m = StringRef(mode.getValue());

    auto fail = [&](StringRef msg) {
      kernel.emitError(msg);
      signalPassFailure();
    };

    if (m == "rowmask") {
      if (!sawDivui)       return fail("rowmask verify failed: missing arith.divui");
      if (!sawShrui)       return fail("rowmask verify failed: missing arith.shrui");
      if (!sawAndi)        return fail("rowmask verify failed: missing arith.andi");
      if (!sawRowmaskLoad) return fail("rowmask verify failed: missing memref.load from rowmask arg3");
      return;
    }

    if (m == "warp") {
      if (!sawShuffle) return fail("warp verify failed: missing gpu.shuffle");
      return;
    }

    if (m == "tiled") {
      if (!sawAllocaAS3) return fail("tiled verify failed: missing memref.alloca in addrspace(3)");
      if (!sawBarrier)   return fail("tiled verify failed: missing gpu.barrier");
      return;
    }

    fail("Unknown mode. Expected: rowmask | warp | tiled");
  }
};

// ✅ IMPORTANT: static registration happens at plugin load time.
static PassRegistration<SparseFlowGpuVerifyPass> sparseflowGpuVerifyPass;

} // namespace
