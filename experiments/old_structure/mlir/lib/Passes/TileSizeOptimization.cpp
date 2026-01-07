//===- TileSizeOptimization.cpp - Optimize tile sizes --------------------===//
//
// Automatically selects optimal tile sizes based on:
// - Matrix dimensions
// - GPU architecture (SM count, register count)
// - Memory bandwidth
//
//===----------------------------------------------------------------------===//

#include "SparseFlow/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace sparseflow {

namespace {

// GPU architecture parameters
struct GPUArchInfo {
    int sm_count;           // Number of SMs
    int max_threads_per_sm; // Max threads per SM
    int registers_per_sm;   // Registers per SM
    int shared_mem_per_sm;  // Shared memory per SM (bytes)
    
    static GPUArchInfo getForArch(int compute_capability) {
        GPUArchInfo info;
        switch(compute_capability) {
            case 80: // A100
                info.sm_count = 108;
                info.max_threads_per_sm = 2048;
                info.registers_per_sm = 65536;
                info.shared_mem_per_sm = 163840;
                break;
            case 86: // RTX 3090
                info.sm_count = 82;
                info.max_threads_per_sm = 1536;
                info.registers_per_sm = 65536;
                info.shared_mem_per_sm = 102400;
                break;
            case 89: // RTX 4090
                info.sm_count = 128;
                info.max_threads_per_sm = 1536;
                info.registers_per_sm = 65536;
                info.shared_mem_per_sm = 102400;
                break;
            case 90: // H100
                info.sm_count = 132;
                info.max_threads_per_sm = 2048;
                info.registers_per_sm = 65536;
                info.shared_mem_per_sm = 227328;
                break;
            default:
                // Default to Ampere-like
                info.sm_count = 80;
                info.max_threads_per_sm = 1536;
                info.registers_per_sm = 65536;
                info.shared_mem_per_sm = 102400;
        }
        return info;
    }
};

// Tile configuration
struct TileConfig {
    int tile_m;
    int tile_n;
    int tile_k;
    
    // Compute occupancy score (higher is better)
    float getOccupancyScore(const GPUArchInfo& arch) const {
        // Shared memory required
        size_t smem_bytes = (tile_m * tile_k + tile_k * tile_n) * sizeof(float);
        
        // Check if fits in shared memory
        if (smem_bytes > arch.shared_mem_per_sm) {
            return 0.0f; // Invalid configuration
        }
        
        // Threads per block (8 warps = 256 threads)
        int threads_per_block = 256;
        
        // Blocks per SM (limited by shared memory)
        int blocks_per_sm = arch.shared_mem_per_sm / smem_bytes;
        blocks_per_sm = std::min(blocks_per_sm, 
                                 arch.max_threads_per_sm / threads_per_block);
        
        // Occupancy = active threads / max threads
        float occupancy = (blocks_per_sm * threads_per_block) / 
                         (float)arch.max_threads_per_sm;
        
        // Work per thread
        float work_per_thread = (tile_m * tile_n * tile_k) / 
                                (float)threads_per_block;
        
        // Score balances occupancy and work per thread
        return occupancy * std::log2(1 + work_per_thread);
    }
};

// Tile size optimizer
class TileSizeOptimizer {
public:
    TileSizeOptimizer(int compute_capability) 
        : arch_(GPUArchInfo::getForArch(compute_capability)) {}
    
    TileConfig selectOptimalTile(int M, int N, int K) {
        // Candidate tile sizes
        std::vector<TileConfig> candidates = {
            {64, 64, 32},
            {64, 128, 32},
            {128, 64, 32},
            {128, 128, 32},
            {128, 128, 64},
            {128, 256, 32},
            {256, 128, 32},
            {256, 256, 32},
        };
        
        // Find best tile size
        TileConfig best = candidates[0];
        float best_score = 0.0f;
        
        for (const auto& config : candidates) {
            // Skip if tiles don't divide dimensions well
            if (M % config.tile_m > config.tile_m / 2 ||
                N % config.tile_n > config.tile_n / 2) {
                continue;
            }
            
            float score = config.getOccupancyScore(arch_);
            if (score > best_score) {
                best_score = score;
                best = config;
            }
        }
        
        return best;
    }
    
private:
    GPUArchInfo arch_;
};

// Pattern to optimize tile sizes
struct OptimizeTileSizes : public OpRewritePattern</* SparseGemmOp */> {
    OptimizeTileSizes(MLIRContext *context, int compute_capability)
        : OpRewritePattern(context), 
          optimizer_(compute_capability) {}
    
    LogicalResult matchAndRewrite(/* SparseGemmOp */ op,
                                   PatternRewriter &rewriter) const override {
        // Get matrix dimensions
        auto A_type = op.getA().getType().cast<RankedTensorType>();
        int M = A_type.getShape()[0];
        int K = A_type.getShape()[1];
        
        auto Bc_type = op.getBc().getType().cast<RankedTensorType>();
        int N = Bc_type.getShape()[1];
        
        // Skip if tile sizes already set
        if (op.getTileM() && op.getTileN() && op.getTileK()) {
            return failure();
        }
        
        // Compute optimal tile sizes
        TileConfig optimal = optimizer_.selectOptimalTile(M, N, K);
        
        // Update operation with optimal tiles
        rewriter.updateRootInPlace(op, [&]() {
            op.setTileMAttr(rewriter.getI32IntegerAttr(optimal.tile_m));
            op.setTileNAttr(rewriter.getI32IntegerAttr(optimal.tile_n));
            op.setTileKAttr(rewriter.getI32IntegerAttr(optimal.tile_k));
        });
        
        return success();
    }
    
private:
    TileSizeOptimizer optimizer_;
};

// The pass
struct TileSizeOptimizationPass 
    : public PassWrapper<TileSizeOptimizationPass, OperationPass<ModuleOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileSizeOptimizationPass)
    
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        
        // TODO: Get compute capability from GPU at runtime
        int compute_capability = 86; // Default to RTX 3090
        
        // Apply tile size optimization patterns
        RewritePatternSet patterns(context);
        patterns.add<OptimizeTileSizes>(context, compute_capability);
        
        if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                                 std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const final { 
        return "sparseflow-optimize-tiles"; 
    }
    
    StringRef getDescription() const final {
        return "Optimize tile sizes for SparseFlow operations";
    }
};

} // anonymous namespace

std::unique_ptr<Pass> createTileSizeOptimizationPass() {
    return std::make_unique<TileSizeOptimizationPass>();
}

} // namespace sparseflow
} // namespace mlir
