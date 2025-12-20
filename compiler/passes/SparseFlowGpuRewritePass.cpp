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
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct SparseFlowGpuRewritePass
    : public PassWrapper<SparseFlowGpuRewritePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseFlowGpuRewritePass)

  // IMPORTANT:
  // PassWrapper clones passes by copying PassT. Pass::Option is non-copyable.
  // So we provide a custom copy-ctor that re-creates the Option bound to *this
  // (via default member initializer) and then copies ONLY the option value.
  SparseFlowGpuRewritePass() = default;
  SparseFlowGpuRewritePass(const SparseFlowGpuRewritePass &other) : PassWrapper(other) {
    mode = other.mode.getValue();
  }

  // Pass option: rowmask | warp | tiled
  Option<std::string> mode{
      *this, "mode",
      llvm::cl::desc("Kernel mode: rowmask | warp | tiled"),
      llvm::cl::init("rowmask")};

  StringRef getArgument() const override { return "sparseflow-gpu-rewrite"; }
  StringRef getDescription() const override {
    return "SparseFlow GPU kernel generation (v0.5 rowmask + v0.6 modes)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    ctx->loadDialect<gpu::GPUDialect>();
    ctx->loadDialect<bufferization::BufferizationDialect>();
    ctx->loadDialect<tensor::TensorDialect>();
    ctx->loadDialect<memref::MemRefDialect>();
    ctx->loadDialect<scf::SCFDialect>();
    ctx->loadDialect<arith::ArithDialect>();
    ctx->loadDialect<func::FuncDialect>();

    ensureGpuModuleWithKernel(module);
  }

  void ensureGpuModuleWithKernel(ModuleOp module) {
    if (module.lookupSymbol<gpu::GPUModuleOp>("sf_gpu"))
      return;

    OpBuilder builder(module.getBodyRegion());
    Location loc = builder.getUnknownLoc();

    auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "sf_gpu");
    Block *body = gpuModule.getBody();
    OpBuilder modBuilder(body, body->begin());

    StringRef m = mode.getValue();
    if (m == "warp")
      buildWarpKernel(modBuilder, loc);
    else if (m == "tiled")
      buildTiledKernel(modBuilder, loc);
    else
      buildRowMaskKernel(modBuilder, loc);
  }

  // Kernel ABI: (A,B,C,rowMask,M,N,K)
  void buildKernelHeader(OpBuilder &builder, Location loc, gpu::GPUFuncOp &kernelOut) {
    FloatType f32 = builder.getF32Type();
    Type i32 = builder.getI32Type();

    auto memrefDynF32 =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto memrefDynI32 = MemRefType::get({ShapedType::kDynamic}, i32);

    SmallVector<Type> argTypes = {
        memrefDynF32, memrefDynF32, memrefDynF32, memrefDynI32, i32, i32, i32};

    auto fnType = builder.getFunctionType(argTypes, TypeRange{});

    auto kernel = builder.create<gpu::GPUFuncOp>(
        loc, "sparseflow_gpu_matmul_2_4_f32", fnType, TypeRange{}, TypeRange{});

    kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());
    kernelOut = kernel;
  }

  // ---------------- rowmask (v0.5) ----------------
  void buildRowMaskKernel(OpBuilder &builder, Location loc) {
    gpu::GPUFuncOp kernel;
    buildKernelHeader(builder, loc, kernel);

    Block &entry = kernel.getBody().front();
    OpBuilder b(&entry, entry.begin());

    FloatType f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Type index = b.getIndexType();

    auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto c0_i32 = b.create<arith::ConstantIntOp>(loc, 0, 32);
    auto c1_i32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
    auto c32_i32 = b.create<arith::ConstantIntOp>(loc, 32, 32);
    auto c0_f32 = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);

    auto tidx = b.create<gpu::ThreadIdOp>(loc, index, gpu::Dimension::x);
    auto bidx = b.create<gpu::BlockIdOp>(loc, index, gpu::Dimension::x);
    auto bdim = b.create<gpu::BlockDimOp>(loc, index, gpu::Dimension::x);

    auto base = b.create<arith::MulIOp>(loc, bidx, bdim);
    auto globalRow = b.create<arith::AddIOp>(loc, tidx, base);
    auto globalRowI32 = b.create<arith::IndexCastOp>(loc, i32, globalRow);

    Value A = entry.getArgument(0);
    Value B = entry.getArgument(1);
    Value C = entry.getArgument(2);
    Value rowMask = entry.getArgument(3);
    Value M = entry.getArgument(4);
    Value N = entry.getArgument(5);
    Value K = entry.getArgument(6);

    auto inBounds =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalRowI32, M);

    b.create<scf::IfOp>(loc, inBounds, [&](OpBuilder &thenB, Location thenLoc) {
      auto word = thenB.create<arith::DivUIOp>(thenLoc, globalRowI32, c32_i32);
      auto bitPos = thenB.create<arith::RemUIOp>(thenLoc, globalRowI32, c32_i32);
      auto wordIdx =
          thenB.create<arith::IndexCastOp>(thenLoc, index, word.getResult());

      auto maskWord =
          thenB.create<memref::LoadOp>(thenLoc, rowMask, ValueRange{wordIdx});
      auto shifted =
          thenB.create<arith::ShRUIOp>(thenLoc, maskWord.getResult(), bitPos);
      auto bit =
          thenB.create<arith::AndIOp>(thenLoc, shifted.getResult(), c1_i32);
      auto isActive = thenB.create<arith::CmpIOp>(
          thenLoc, arith::CmpIPredicate::ne, bit.getResult(), c0_i32);

      thenB.create<scf::IfOp>(thenLoc, isActive,
                              [&](OpBuilder &activeB, Location activeLoc) {
        auto NIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, N);
        auto KIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, K);

        activeB.create<scf::ForOp>(
            activeLoc, c0, NIdx, c1, ValueRange{},
            [&](OpBuilder &jB, Location jLoc, Value j, ValueRange) {
              auto accFor = jB.create<scf::ForOp>(
                  jLoc, c0, KIdx, c1, ValueRange{c0_f32.getResult()},
                  [&](OpBuilder &kB, Location kLoc, Value k, ValueRange iterArgs) {
                    auto aVal = kB.create<memref::LoadOp>(
                        kLoc, A, ValueRange{globalRow, k});
                    auto bVal =
                        kB.create<memref::LoadOp>(kLoc, B, ValueRange{k, j});
                    auto mul = kB.create<arith::MulFOp>(kLoc, aVal, bVal);
                    auto sum = kB.create<arith::AddFOp>(kLoc, iterArgs[0], mul);
                    kB.create<scf::YieldOp>(kLoc, sum.getResult());
                  });

              jB.create<memref::StoreOp>(jLoc, accFor.getResult(0), C,
                                        ValueRange{globalRow, j});
              jB.create<scf::YieldOp>(jLoc);
            });

        activeB.create<scf::YieldOp>(activeLoc);
      });

      thenB.create<scf::YieldOp>(thenLoc);
    });

    b.create<gpu::ReturnOp>(loc);
  }

  // ---------------- warp (warp-level shuffle demo) ----------------
  // This is NOT an optimized kernel yet. It exists to prove we can emit gpu.shuffle in MLIR19.
  // We broadcast the first lane\x27s maskWord to all lanes (IDX mode with srcLane=0).
  void buildWarpKernel(OpBuilder &builder, Location loc) {
    gpu::GPUFuncOp kernel;
    buildKernelHeader(builder, loc, kernel);

    Block &entry = kernel.getBody().front();
    OpBuilder b(&entry, entry.begin());

    FloatType f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Type index = b.getIndexType();

    auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto c0_i32 = b.create<arith::ConstantIntOp>(loc, 0, 32);
    auto c1_i32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
    auto c31_i32 = b.create<arith::ConstantIntOp>(loc, 31, 32);
    auto c32_i32 = b.create<arith::ConstantIntOp>(loc, 32, 32);
    auto c0_f32 = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);

    auto tidx = b.create<gpu::ThreadIdOp>(loc, index, gpu::Dimension::x);
    auto bidx = b.create<gpu::BlockIdOp>(loc, index, gpu::Dimension::x);
    auto bdim = b.create<gpu::BlockDimOp>(loc, index, gpu::Dimension::x);

    auto base = b.create<arith::MulIOp>(loc, bidx, bdim);
    auto globalRow = b.create<arith::AddIOp>(loc, tidx, base);
    auto globalRowI32 = b.create<arith::IndexCastOp>(loc, i32, globalRow);

    // lane = tidx & 31
    auto tidxI32 = b.create<arith::IndexCastOp>(loc, i32, tidx);
    auto lane = b.create<arith::AndIOp>(loc, tidxI32, c31_i32);

    Value A = entry.getArgument(0);
    Value B = entry.getArgument(1);
    Value C = entry.getArgument(2);
    Value rowMask = entry.getArgument(3);
    Value M = entry.getArgument(4);
    Value N = entry.getArgument(5);
    Value K = entry.getArgument(6);

    auto inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalRowI32, M);

    b.create<scf::IfOp>(loc, inBounds, [&](OpBuilder &thenB, Location thenLoc) {
      auto word = thenB.create<arith::DivUIOp>(thenLoc, globalRowI32, c32_i32);
      auto bitPos = thenB.create<arith::RemUIOp>(thenLoc, globalRowI32, c32_i32);
      auto wordIdx = thenB.create<arith::IndexCastOp>(thenLoc, index, word.getResult());

      auto maskWord = thenB.create<memref::LoadOp>(thenLoc, rowMask, ValueRange{wordIdx});

      // --- warp primitive proof ---
      // Broadcast lane0\x27s maskWord to all lanes.
      auto srcLane0 = c0_i32;
      auto width32  = c32_i32;
      auto shfl = thenB.create<gpu::ShuffleOp>(
          thenLoc,
          maskWord.getResult(),
          srcLane0,
          width32,
          gpu::ShuffleMode::IDX);

      Value maskWordUniform = shfl.getResult(0);

      auto shifted = thenB.create<arith::ShRUIOp>(thenLoc, maskWordUniform, bitPos);
      auto bit = thenB.create<arith::AndIOp>(thenLoc, shifted.getResult(), c1_i32);
      auto isActive = thenB.create<arith::CmpIOp>(
          thenLoc, arith::CmpIPredicate::ne, bit.getResult(), c0_i32);

      thenB.create<scf::IfOp>(thenLoc, isActive, [&](OpBuilder &activeB, Location activeLoc) {
        auto NIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, N);
        auto KIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, K);

        activeB.create<scf::ForOp>(
            activeLoc, c0, NIdx, c1, ValueRange{},
            [&](OpBuilder &jB, Location jLoc, Value j, ValueRange) {
              auto accFor = jB.create<scf::ForOp>(
                  jLoc, c0, KIdx, c1, ValueRange{c0_f32.getResult()},
                  [&](OpBuilder &kB, Location kLoc, Value k, ValueRange iterArgs) {
                    auto aVal = kB.create<memref::LoadOp>(kLoc, A, ValueRange{globalRow, k});
                    auto bVal = kB.create<memref::LoadOp>(kLoc, B, ValueRange{k, j});
                    auto mul = kB.create<arith::MulFOp>(kLoc, aVal, bVal);
                    auto sum = kB.create<arith::AddFOp>(kLoc, iterArgs[0], mul);
                    kB.create<scf::YieldOp>(kLoc, sum.getResult());
                  });
              jB.create<memref::StoreOp>(jLoc, accFor.getResult(0), C, ValueRange{globalRow, j});
              jB.create<scf::YieldOp>(jLoc);
            });
        activeB.create<scf::YieldOp>(activeLoc);
      });

      thenB.create<scf::YieldOp>(thenLoc);
    });

    b.create<gpu::ReturnOp>(loc);
  }

  // ---------------- tiled (shared memory + barriers) ----------------
  void buildTiledKernel(OpBuilder &builder, Location loc) {
    gpu::GPUFuncOp kernel;
    buildKernelHeader(builder, loc, kernel);

    Block &entry = kernel.getBody().front();
    OpBuilder b(&entry, entry.begin());

    FloatType f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Type index = b.getIndexType();

    auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto c0_i32 = b.create<arith::ConstantIntOp>(loc, 0, 32);
    auto c1_i32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
    auto c31_i32 = b.create<arith::ConstantIntOp>(loc, 31, 32);
    auto c32_i32 = b.create<arith::ConstantIntOp>(loc, 32, 32);
    auto cTile = b.create<arith::ConstantIndexOp>(loc, 32);
    auto c0_f32 = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);

    auto tidx = b.create<gpu::ThreadIdOp>(loc, index, gpu::Dimension::x);
    auto bidx = b.create<gpu::BlockIdOp>(loc, index, gpu::Dimension::x);
    auto bdim = b.create<gpu::BlockDimOp>(loc, index, gpu::Dimension::x);

    auto base = b.create<arith::MulIOp>(loc, bidx, bdim);
    auto globalRow = b.create<arith::AddIOp>(loc, tidx, base);
    auto globalRowI32 = b.create<arith::IndexCastOp>(loc, i32, globalRow);

    Value A = entry.getArgument(0);
    Value B = entry.getArgument(1);
    Value C = entry.getArgument(2);
    Value rowMask = entry.getArgument(3);
    Value M = entry.getArgument(4);
    Value N = entry.getArgument(5);
    Value K = entry.getArgument(6);

    // lane = tidx & 31
    auto tidxI32 = b.create<arith::IndexCastOp>(loc, i32, tidx);
    auto lane = b.create<arith::AndIOp>(loc, tidxI32, c31_i32);
    auto laneIdx = b.create<arith::IndexCastOp>(loc, index, lane.getResult());

    auto inBounds =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalRowI32, M);

    b.create<scf::IfOp>(loc, inBounds, [&](OpBuilder &thenB, Location thenLoc) {
      auto word = thenB.create<arith::DivUIOp>(thenLoc, globalRowI32, c32_i32);
      auto bitPos = thenB.create<arith::RemUIOp>(thenLoc, globalRowI32, c32_i32);
      auto wordIdx =
          thenB.create<arith::IndexCastOp>(thenLoc, index, word.getResult());
      auto maskWord =
          thenB.create<memref::LoadOp>(thenLoc, rowMask, ValueRange{wordIdx});
      auto shifted =
          thenB.create<arith::ShRUIOp>(thenLoc, maskWord.getResult(), bitPos);
      auto bit =
          thenB.create<arith::AndIOp>(thenLoc, shifted.getResult(), c1_i32);
      auto isActive = thenB.create<arith::CmpIOp>(
          thenLoc, arith::CmpIPredicate::ne, bit.getResult(), c0_i32);

      thenB.create<scf::IfOp>(thenLoc, isActive,
                              [&](OpBuilder &activeB, Location activeLoc) {
        auto NIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, N);
        auto KIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, K);

        // addrspace(3) shared memory tiles
        auto smem = activeB.getI32IntegerAttr(3);
        auto tileTy = MemRefType::get({32}, f32, MemRefLayoutAttrInterface{}, smem);
        auto A_tile = activeB.create<memref::AllocaOp>(activeLoc, tileTy);
        auto B_tile = activeB.create<memref::AllocaOp>(activeLoc, tileTy);

        activeB.create<scf::ForOp>(
            activeLoc, c0, NIdx, c1, ValueRange{},
            [&](OpBuilder &jB, Location jLoc, Value j, ValueRange) {
              Value accInit = c0_f32.getResult();

              auto kLoop = jB.create<scf::ForOp>(
                  jLoc, c0, KIdx, cTile, ValueRange{accInit},
                  [&](OpBuilder &tileB, Location tileLoc, Value k0,
                      ValueRange iterArgs) {
                    Value acc = iterArgs[0];

                    auto kLane = tileB.create<arith::AddIOp>(tileLoc, k0, laneIdx);
                    auto kLaneI32 =
                        tileB.create<arith::IndexCastOp>(tileLoc, i32, kLane);
                    auto inK = tileB.create<arith::CmpIOp>(
                        tileLoc, arith::CmpIPredicate::slt, kLaneI32, K);

                    auto ifInK = tileB.create<scf::IfOp>(
                        tileLoc, TypeRange{f32, f32}, inK.getResult(),
                        /*withElse=*/true);

                    // then: loads
                    {
                      OpBuilder tb = ifInK.getThenBodyBuilder();
                      auto aVal = tb.create<memref::LoadOp>(
                          tileLoc, A, ValueRange{globalRow, kLane});
                      auto bVal =
                          tb.create<memref::LoadOp>(tileLoc, B, ValueRange{kLane, j});
                      tb.create<scf::YieldOp>(tileLoc,
                                             ValueRange{aVal.getResult(),
                                                        bVal.getResult()});
                    }
                    // else: zeros
                    {
                      OpBuilder eb = ifInK.getElseBodyBuilder();
                      eb.create<scf::YieldOp>(tileLoc,
                                             ValueRange{c0_f32.getResult(),
                                                        c0_f32.getResult()});
                    }

                    Value aLaneVal = ifInK.getResult(0);
                    Value bLaneVal = ifInK.getResult(1);

                    tileB.create<memref::StoreOp>(tileLoc, aLaneVal, A_tile,
                                                 ValueRange{laneIdx});
                    tileB.create<memref::StoreOp>(tileLoc, bLaneVal, B_tile,
                                                 ValueRange{laneIdx});

                    tileB.create<gpu::BarrierOp>(tileLoc);

                    auto tLoop = tileB.create<scf::ForOp>(
                        tileLoc, c0, cTile, c1, ValueRange{acc},
                        [&](OpBuilder &tB, Location tLoc, Value t,
                            ValueRange accArgs) {
                          auto aSh =
                              tB.create<memref::LoadOp>(tLoc, A_tile, ValueRange{t});
                          auto bSh =
                              tB.create<memref::LoadOp>(tLoc, B_tile, ValueRange{t});
                          auto mul = tB.create<arith::MulFOp>(tLoc, aSh, bSh);
                          auto sum =
                              tB.create<arith::AddFOp>(tLoc, accArgs[0], mul);
                          tB.create<scf::YieldOp>(tLoc, sum.getResult());
                        });

                    tileB.create<gpu::BarrierOp>(tileLoc);
                    tileB.create<scf::YieldOp>(tileLoc, tLoop.getResult(0));
                  });

              jB.create<memref::StoreOp>(jLoc, kLoop.getResult(0), C,
                                        ValueRange{globalRow, j});
              jB.create<scf::YieldOp>(jLoc);
            });

        activeB.create<scf::YieldOp>(activeLoc);
      });

      thenB.create<scf::YieldOp>(thenLoc);
    });

    b.create<gpu::ReturnOp>(loc);
  }
};

} // namespace

// Keep the exact symbol your plugin expects (C++ mangled).
void registerSparseFlowGpuRewritePass() {
  PassRegistration<SparseFlowGpuRewritePass>();
}
