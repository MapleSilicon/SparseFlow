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
    return "SparseFlow GPU kernel (v0.6 warp accumulation)";
  }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<SparseFlowGpuRewritePass>(*this);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    ctx->loadDialect<gpu::GPUDialect>();
    ctx->loadDialect<bufferization::BufferizationDialect>();
    ctx->loadDialect<tensor::TensorDialect>();
    ctx->loadDialect<memref::MemRefDialect>();
    ctx->loadDialect<scf::SCFDialect>();

    ensureGpuModuleWithKernel(module);

    SmallVector<func::CallOp> callsToErase;

    module.walk([&](func::CallOp call) {
      StringRef callee = call.getCallee();
      if (!callee.starts_with("sparse_matmul"))
        return;

      if (call->getParentOfType<gpu::GPUModuleOp>()) {
        call.emitError("RowMask generation must be host-side");
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      Location loc = call.getLoc();

      SmallVector<Value, 8> memrefArgs;
      for (Value v : call.getOperands()) {
        memrefArgs.push_back(toMemrefDroppingEncoding(builder, loc, v));
      }

      auto A = call.getOperand(0);
      auto tensorType = llvm::cast<RankedTensorType>(A.getType());
      
      auto encoding = tensorType.getEncoding();
      if (!encoding) {
        call.emitError("Tensor must have N:M encoding");
        signalPassFailure();
        return;
      }

      auto dictAttr = llvm::dyn_cast<DictionaryAttr>(encoding);
      if (!dictAttr) {
        call.emitError("Encoding must be dictionary");
        signalPassFailure();
        return;
      }

      auto nAttr = dictAttr.get("n");
      auto mAttr = dictAttr.get("m");
      if (!nAttr || !mAttr) {
        call.emitError("Encoding must contain n and m");
        signalPassFailure();
        return;
      }

      int64_t n = llvm::cast<IntegerAttr>(nAttr).getInt();
      int64_t m = llvm::cast<IntegerAttr>(mAttr).getInt();

      if (n != 2 || m != 4) {
        call.emitError("v0.6 supports only 2:4 sparsity");
        signalPassFailure();
        return;
      }

      int64_t rows = tensorType.getShape()[0];
      
      auto c0  = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto c1  = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto c31 = builder.create<arith::ConstantIntOp>(loc, 31, 32);
      auto c32 = builder.create<arith::ConstantIntOp>(loc, 32, 32);

      auto rowsVal = builder.create<arith::ConstantIntOp>(loc, rows, 32);
      auto rowsPlus31 = builder.create<arith::AddIOp>(loc, rowsVal, c31);
      auto words = builder.create<arith::DivUIOp>(loc, rowsPlus31, c32);
      auto wordsIndex = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), words);

      auto rowMaskType = MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
      auto rowMaskOp = builder.create<memref::AllocOp>(loc, rowMaskType, ValueRange{wordsIndex});

      uint32_t maskValue = (1u << n) - 1;
      auto pattern = builder.create<arith::ConstantIntOp>(loc, maskValue, 32);

      builder.create<scf::ForOp>(
          loc, c0, wordsIndex, c1, ValueRange{},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
            b.create<memref::StoreOp>(loc, pattern, rowMaskOp, iv);
            b.create<scf::YieldOp>(loc);
          });

      rowMaskOp->setAttr("sparseflow.rowmask", builder.getUnitAttr());

      auto launchC1 = builder.create<arith::ConstantIndexOp>(loc, 1);

      auto launch = builder.create<gpu::LaunchOp>(
          loc,
          launchC1, launchC1, launchC1,
          launchC1, launchC1, launchC1);

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

  void ensureGpuModuleWithKernel(ModuleOp module) {
    if (module.lookupSymbol<gpu::GPUModuleOp>("sf_gpu"))
      return;

    OpBuilder builder(module.getBodyRegion());
    Location loc = builder.getUnknownLoc();
    
    auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "sf_gpu");
    Block *block = gpuModule.getBody();
    OpBuilder modBuilder(block->getTerminator());

    buildV06WarpKernel(modBuilder, loc);
  }

  void buildV06WarpKernel(OpBuilder &builder, Location loc) {
    auto f32 = builder.getF32Type();
    auto i32 = builder.getI32Type();
    auto index = builder.getIndexType();
    
    auto memrefDynF32 = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto memrefDynI32 = MemRefType::get({ShapedType::kDynamic}, i32);

    SmallVector<Type> argTypes = {
      memrefDynF32, memrefDynF32, memrefDynF32, memrefDynI32,
      i32, i32, i32
    };

    auto fnType = builder.getFunctionType(argTypes, TypeRange{});
    
    auto kernel = builder.create<gpu::GPUFuncOp>(
        loc, "sparseflow_gpu_matmul_2_4_f32", fnType,
        TypeRange{}, TypeRange{});

    kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                    builder.getUnitAttr());

    Block &entry = kernel.getBody().front();
    OpBuilder b(&entry, entry.begin());

    // Constants
    auto c0_i32 = b.create<arith::ConstantIntOp>(loc, 0, 32);
    auto c1_i32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
    auto c32_i32 = b.create<arith::ConstantIntOp>(loc, 32, 32);
    auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto c32_idx = b.create<arith::ConstantIndexOp>(loc, 32);
    auto c0_f32 = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);

    // Thread indexing
    auto tidx = b.create<gpu::ThreadIdOp>(loc, index, gpu::Dimension::x);
    auto bidx = b.create<gpu::BlockIdOp>(loc, index, gpu::Dimension::x);
    auto bdim = b.create<gpu::BlockDimOp>(loc, index, gpu::Dimension::x);
    
    auto bidTimesBdim = b.create<arith::MulIOp>(loc, bidx, bdim);
    auto globalRow = b.create<arith::AddIOp>(loc, tidx, bidTimesBdim);
    auto globalRowI32 = b.create<arith::IndexCastOp>(loc, i32, globalRow);
    
    // Lane ID
    auto laneId = b.create<arith::RemUIOp>(loc, globalRowI32, c32_i32);
    auto laneIdIdx = b.create<arith::IndexCastOp>(loc, index, laneId);
    
    // Get kernel args
    Value A = entry.getArgument(0);
    Value B = entry.getArgument(1);
    Value C = entry.getArgument(2);
    Value rowMask = entry.getArgument(3);
    Value M = entry.getArgument(4);
    Value N = entry.getArgument(5);
    Value K = entry.getArgument(6);
    
    auto inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalRowI32, M);
    
    b.create<scf::IfOp>(loc, inBounds, [&](OpBuilder &thenB, Location thenLoc) {
      // RowMask check
      auto word = thenB.create<arith::DivUIOp>(thenLoc, globalRowI32, c32_i32);
      auto bitPos = thenB.create<arith::RemUIOp>(thenLoc, globalRowI32, c32_i32);
      auto wordIdx = thenB.create<arith::IndexCastOp>(thenLoc, index, word);
      auto maskWord = thenB.create<memref::LoadOp>(thenLoc, rowMask, ValueRange{wordIdx});
      auto shifted = thenB.create<arith::ShRUIOp>(thenLoc, maskWord.getResult(), bitPos);
      auto bit = thenB.create<arith::AndIOp>(thenLoc, shifted, c1_i32);
      auto isActive = thenB.create<arith::CmpIOp>(thenLoc, arith::CmpIPredicate::ne, bit, c0_i32);
      
      thenB.create<scf::IfOp>(thenLoc, isActive, [&](OpBuilder &activeB, Location activeLoc) {
        auto NIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, N);
        auto KIdx = activeB.create<arith::IndexCastOp>(activeLoc, index, K);
        
        // Loop over output columns
        activeB.create<scf::ForOp>(
            activeLoc, c0, NIdx, c1, ValueRange{},
            [&](OpBuilder &jB, Location jLoc, Value j, ValueRange) {
              
              // WARP ACCUMULATION: Each lane does K/32 elements
              auto partialSum = jB.create<scf::ForOp>(
                  jLoc, laneIdIdx, KIdx, c32_idx, ValueRange{c0_f32.getResult()},
                  [&](OpBuilder &kB, Location kLoc, Value k, ValueRange iterArgs) {
                    auto aVal = kB.create<memref::LoadOp>(kLoc, A, ValueRange{globalRow, k});
                    auto bVal = kB.create<memref::LoadOp>(kLoc, B, ValueRange{k, j});
                    auto mul = kB.create<arith::MulFOp>(kLoc, aVal, bVal);
                    auto sum = kB.create<arith::AddFOp>(kLoc, iterArgs[0], mul);
                    kB.create<scf::YieldOp>(kLoc, sum.getResult());
                  });
              
              // WARP REDUCTION: 5 shuffle steps (log2(32) = 5)
              Value reduced = partialSum.getResult(0);
              
              // offset = 16, 8, 4, 2, 1
              for (int offset = 16; offset >= 1; offset /= 2) {
                auto offsetVal = jB.create<arith::ConstantIntOp>(jLoc, offset, 32);
                auto shuffled = jB.create<gpu::ShuffleOp>(
                    jLoc, reduced, offsetVal, c32_i32, gpu::ShuffleMode::XOR);
                reduced = jB.create<arith::AddFOp>(jLoc, reduced, shuffled.getShuffleResult());
              }
              
              // SINGLE WRITER: Only lane 0 writes result
              auto isLane0 = jB.create<arith::CmpIOp>(jLoc, arith::CmpIPredicate::eq, laneId, c0_i32);
              jB.create<scf::IfOp>(jLoc, isLane0, [&](OpBuilder &writeB, Location writeLoc) {
                writeB.create<memref::StoreOp>(writeLoc, reduced, C, ValueRange{globalRow, j});
                writeB.create<scf::YieldOp>(writeLoc);
              });
              
              jB.create<scf::YieldOp>(jLoc);
            });
        
        activeB.create<scf::YieldOp>(activeLoc);
      });
      
      thenB.create<scf::YieldOp>(thenLoc);
    });

    b.create<gpu::ReturnOp>(loc);
  }
};

static PassRegistration<SparseFlowGpuRewritePass> sparseflowGpuRewritePass;

} // namespace

namespace mlir {
std::unique_ptr<Pass> createSparseFlowGpuRewritePass() {
  return std::make_unique<SparseFlowGpuRewritePass>();
}
}

void registerSparseFlowGpuRewritePass() {
  // Registration handled by static PassRegistration above
}
