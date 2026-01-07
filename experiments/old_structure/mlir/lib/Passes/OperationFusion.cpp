//===- OperationFusion.cpp - Fuse compatible operations -------------------===//
//
// Automatically fuses sparse GEMM with compatible epilogue operations:
// - GEMM + ReLU -> FusedGEMM(epilogue=relu)
// - GEMM + SiLU -> FusedGEMM(epilogue=silu)
// - GEMM + Bias + ReLU -> FusedGEMM(epilogue=bias_relu)
//
//===----------------------------------------------------------------------===//

#include "SparseFlow/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace sparseflow {

namespace {

// Pattern: GEMM + ReLU -> Fused GEMM
struct FuseGemmReLU : public OpRewritePattern</* ReluOp */> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(/* ReluOp */ relu_op,
                                   PatternRewriter &rewriter) const override {
        // Check if input is from SparseGemmOp
        auto gemm_op = relu_op.getInput().getDefiningOp</* SparseGemmOp */>();
        if (!gemm_op) {
            return failure();
        }
        
        // Check if GEMM has only one use (this ReLU)
        if (!gemm_op.getResult().hasOneUse()) {
            return failure();
        }
        
        // Create fused operation
        auto fused_op = rewriter.create</* FusedSparseGemmOp */>(
            gemm_op.getLoc(),
            gemm_op.getResult().getType(),
            gemm_op.getA(),
            gemm_op.getBc(),
            /* bias */ nullptr,
            rewriter.getStringAttr("relu"),
            gemm_op.getTileM(),
            gemm_op.getTileN(),
            gemm_op.getTileK()
        );
        
        // Replace ReLU output with fused op output
        rewriter.replaceOp(relu_op, fused_op.getResult());
        
        // Erase original GEMM
        rewriter.eraseOp(gemm_op);
        
        return success();
    }
};

// Pattern: GEMM + SiLU -> Fused GEMM
struct FuseGemmSiLU : public OpRewritePattern</* SiLUOp */> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(/* SiLUOp */ silu_op,
                                   PatternRewriter &rewriter) const override {
        // Similar to FuseGemmReLU, but with "silu" epilogue
        auto gemm_op = silu_op.getInput().getDefiningOp</* SparseGemmOp */>();
        if (!gemm_op || !gemm_op.getResult().hasOneUse()) {
            return failure();
        }
        
        auto fused_op = rewriter.create</* FusedSparseGemmOp */>(
            gemm_op.getLoc(),
            gemm_op.getResult().getType(),
            gemm_op.getA(),
            gemm_op.getBc(),
            nullptr,
            rewriter.getStringAttr("silu"),
            gemm_op.getTileM(),
            gemm_op.getTileN(),
            gemm_op.getTileK()
        );
        
        rewriter.replaceOp(silu_op, fused_op.getResult());
        rewriter.eraseOp(gemm_op);
        
        return success();
    }
};

// Pattern: GEMM + Bias + ReLU -> Fused GEMM
struct FuseGemmBiasReLU : public OpRewritePattern</* ReluOp */> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(/* ReluOp */ relu_op,
                                   PatternRewriter &rewriter) const override {
        // Check pattern: relu(add(gemm, bias))
        auto add_op = relu_op.getInput().getDefiningOp</* AddOp */>();
        if (!add_op) {
            return failure();
        }
        
        // One input should be GEMM, other should be bias
        auto gemm_op = add_op.getLhs().getDefiningOp</* SparseGemmOp */>();
        Value bias = add_op.getRhs();
        
        if (!gemm_op) {
            // Try reversed order
            gemm_op = add_op.getRhs().getDefiningOp</* SparseGemmOp */>();
            bias = add_op.getLhs();
        }
        
        if (!gemm_op) {
            return failure();
        }
        
        // Create fused operation with bias
        auto fused_op = rewriter.create</* FusedSparseGemmOp */>(
            gemm_op.getLoc(),
            gemm_op.getResult().getType(),
            gemm_op.getA(),
            gemm_op.getBc(),
            bias,
            rewriter.getStringAttr("bias_relu"),
            gemm_op.getTileM(),
            gemm_op.getTileN(),
            gemm_op.getTileK()
        );
        
        rewriter.replaceOp(relu_op, fused_op.getResult());
        rewriter.eraseOp(add_op);
        rewriter.eraseOp(gemm_op);
        
        return success();
    }
};

// The pass
struct OperationFusionPass 
    : public PassWrapper<OperationFusionPass, OperationPass<ModuleOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperationFusionPass)
    
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        
        // Apply fusion patterns
        RewritePatternSet patterns(context);
        patterns.add<FuseGemmReLU>(context);
        patterns.add<FuseGemmSiLU>(context);
        patterns.add<FuseGemmBiasReLU>(context);
        
        if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                                 std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const final { 
        return "sparseflow-fuse-ops"; 
    }
    
    StringRef getDescription() const final {
        return "Fuse SparseFlow operations with compatible epilogues";
    }
};

} // anonymous namespace

std::unique_ptr<Pass> createOperationFusionPass() {
    return std::make_unique<OperationFusionPass>();
}

} // namespace sparseflow
} // namespace mlir
