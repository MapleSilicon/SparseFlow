#ifndef SPARSEFLOW_RUNTIME_H
#define SPARSEFLOW_RUNTIME_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace sparseflow {

// Enhanced hardware configuration from JSON metadata
struct MatmulConfig {
    std::string type;
    std::string nm_pattern;
    int32_t n;  // N in N:M
    int32_t m;  // M in N:M
    int64_t dim_m;
    int64_t dim_n; 
    int64_t dim_k;
    std::string dtype;
    int64_t tile_m;
    int64_t tile_n;
    int64_t tile_k;
};

// Performance counters
struct PerformanceStats {
    uint64_t cycles;
    uint64_t flops;
    uint64_t skipped_ops;
    uint64_t memory_bytes;
};

// Hardware register interface
class SparseFlowEngine {
public:
    SparseFlowEngine();
    ~SparseFlowEngine();
    
    // Load configuration from JSON file
    bool loadConfig(const std::string& json_file);
    
    // Program hardware registers
    bool programHardware();
    
    // Execute matmul operation
    bool executeMatmul(float* A, float* B, float* C);
    
    // Get performance statistics
    PerformanceStats getStats() const;
    
    // Software emulation (for testing without hardware)
    bool emulateMatmul(float* A, float* B, float* C);

private:
    std::vector<MatmulConfig> configs_;
    PerformanceStats stats_;
    
    // Enhanced hardware register addresses
    static constexpr uint32_t REG_N = 0x1000;
    static constexpr uint32_t REG_M = 0x1004;
    static constexpr uint32_t REG_DIM_M = 0x1008;
    static constexpr uint32_t REG_DIM_N = 0x100C;
    static constexpr uint32_t REG_DIM_K = 0x1010;
    static constexpr uint32_t REG_TILE_M = 0x1014;
    static constexpr uint32_t REG_TILE_N = 0x1018;
    static constexpr uint32_t REG_TILE_K = 0x101C;
    static constexpr uint32_t REG_DATA_TYPE = 0x1020;
    static constexpr uint32_t REG_CTRL = 0x1024;
    static constexpr uint32_t REG_STATUS = 0x1028;
    
    void writeRegister(uint32_t addr, uint32_t value);
    uint32_t readRegister(uint32_t addr);
    
    // Generate N:M mask pattern
    std::vector<bool> generateMask(int n, int m) const;
};

} // namespace sparseflow

#endif
