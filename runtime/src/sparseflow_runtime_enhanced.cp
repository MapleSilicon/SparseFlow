#include "sparseflow_runtime.h"
#include <fstream>
#include <iostream>
#include <random>
#include <cstring>
#include <sstream>
#include <regex>

namespace sparseflow {

SparseFlowEngine::SparseFlowEngine() {
    stats_ = {0, 0, 0, 0};
}

SparseFlowEngine::~SparseFlowEngine() {
}

bool SparseFlowEngine::loadConfig(const std::string& json_file) {
    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file: " << json_file << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    
    // Enhanced JSON parsing for hardware metadata
    std::regex matmul_regex(
        R"(\{"type":"matmul","nm_pattern":"(\d+):(\d+)","n":(\d+),"m":(\d+),)"
        R"("dim_m":(\d+),"dim_n":(\d+),"dim_k":(\d+),"dtype":"([^"]+)",)"
        R"("tile_m":(\d+),"tile_n":(\d+),"tile_k":(\d+))");
    
    std::smatch matches;
    
    if (std::regex_search(json_str, matches, matmul_regex)) {
        MatmulConfig config;
        config.type = "matmul";
        config.nm_pattern = matches[1].str() + ":" + matches[2].str();
        config.n = std::stoi(matches[3]);
        config.m = std::stoi(matches[4]);
        config.dim_m = std::stoll(matches[5]);
        config.dim_n = std::stoll(matches[6]);
        config.dim_k = std::stoll(matches[7]);
        config.dtype = matches[8];
        config.tile_m = std::stoll(matches[9]);
        config.tile_n = std::stoll(matches[10]);
        config.tile_k = std::stoll(matches[11]);
        
        configs_.push_back(config);
        
        std::cout << "=== Loaded Hardware Configuration ===" << std::endl;
        std::cout << "Pattern: " << config.nm_pattern << std::endl;
        std::cout << "Dimensions: " << config.dim_m << "x" << config.dim_k 
                  << " * " << config.dim_k << "x" << config.dim_n << std::endl;
        std::cout << "Data Type: " << config.dtype << std::endl;
        std::cout << "Tile Size: " << config.tile_m << "x" << config.tile_n 
                  << "x" << config.tile_k << std::endl;
        
        return true;
    }
    
    std::cerr << "No hardware configuration found in JSON" << std::endl;
    return false;
}

bool SparseFlowEngine::programHardware() {
    if (configs_.empty()) {
        std::cerr << "No configuration loaded" << std::endl;
        return false;
    }
    
    const auto& config = configs_[0];
    
    std::cout << "\n=== Programming MapleSilicon Hardware ===" << std::endl;
    std::cout << "REG_NM_PATTERN: " << config.nm_pattern << std::endl;
    std::cout << "REG_DIMENSIONS: " << config.dim_m << "x" << config.dim_n 
              << "x" << config.dim_k << std::endl;
    std::cout << "REG_DATA_TYPE: " << config.dtype << std::endl;
    std::cout << "REG_TILE_CONFIG: " << config.tile_m << "x" << config.tile_n 
              << "x" << config.tile_k << std::endl;
    
    // Simulate register writes for enhanced hardware
    writeRegister(REG_N, config.n);
    writeRegister(REG_M, config.m);
    writeRegister(REG_DIM_M, config.dim_m);
    writeRegister(REG_DIM_N, config.dim_n);
    writeRegister(REG_DIM_K, config.dim_k);
    writeRegister(REG_TILE_M, config.tile_m);
    writeRegister(REG_TILE_N, config.tile_n);
    writeRegister(REG_TILE_K, config.tile_k);
    
    // Set data type
    uint32_t dtype_encoding = 0;
    if (config.dtype == "f32") dtype_encoding = 0;
    else if (config.dtype == "f16") dtype_encoding = 1;
    else if (config.dtype == "int8") dtype_encoding = 2;
    writeRegister(REG_DATA_TYPE, dtype_encoding);
    
    return true;
}

bool SparseFlowEngine::emulateMatmul(float* A, float* B, float* C) {
    if (configs_.empty()) return false;
    
    const auto& config = configs_[0];
    const int M = config.dim_m;
    const int N = config.dim_n;
    const int K = config.dim_k;
    
    std::cout << "\n=== Executing Enhanced Sparse Matmul ===" << std::endl;
    std::cout << "Pattern: " << config.nm_pattern << std::endl;
    std::cout << "Data Type: " << config.dtype << std::endl;
    std::cout << "Tiling: " << config.tile_m << "x" << config.tile_n 
              << " tiles over " << M << "x" << N << " matrix" << std::endl;
    
    // Reset performance counters
    stats_ = {0, 0, 0, 0};
    
    // Generate N:M mask pattern
    auto mask = generateMask(config.n, config.m);
    const int block_size = config.m;
    
    // Initialize output to zero
    memset(C, 0, M * N * sizeof(float));
    
    // Enhanced tiled execution with hardware-aware scheduling
    for (int tile_i = 0; tile_i < M; tile_i += config.tile_m) {
        int tile_height = std::min(config.tile_m, M - tile_i);
        
        for (int tile_j = 0; tile_j < N; tile_j += config.tile_n) {
            int tile_width = std::min(config.tile_n, N - tile_j);
            
            // Process tile with N:M sparsity
            for (int i = tile_i; i < tile_i + tile_height; i++) {
                for (int j = tile_j; j < tile_j + tile_width; j++) {
                    float sum = 0.0f;
                    
                    for (int k_block = 0; k_block < K; k_block += block_size) {
                        for (int k_inner = 0; k_inner < block_size && (k_block + k_inner) < K; k_inner++) {
                            int k = k_block + k_inner;
                            
                            // Apply N:M sparsity
                            if (mask[k_inner]) {
                                sum += A[i * K + k] * B[k * N + j];
                                stats_.flops += 2;
                            } else {
                                stats_.skipped_ops += 2;
                            }
                            stats_.cycles++;
                        }
                    }
                    
                    C[i * N + j] = sum;
                }
            }
            
            std::cout << "Processed tile [" << tile_i/config.tile_m << "," 
                      << tile_j/config.tile_n << "]" << std::endl;
        }
    }
    
    stats_.memory_bytes = (M * K + K * N + M * N) * sizeof(float);
    
    // Calculate efficiency metrics
    double total_ops = stats_.flops + stats_.skipped_ops;
    double efficiency = (double)stats_.flops / total_ops * 100.0;
    double theoretical_max = (double)config.m / config.n;
    double achieved_speedup = total_ops / stats_.flops;
    
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "FLOPs Executed: " << stats_.flops << std::endl;
    std::cout << "Operations Skipped: " << stats_.skipped_ops << std::endl;
    std::cout << "Compute Efficiency: " << efficiency << "%" << std::endl;
    std::cout << "Theoretical Speedup: " << theoretical_max << "x" << std::endl;
    std::cout << "Achieved Speedup: " << achieved_speedup << "x" << std::endl;
    std::cout << "Memory Accessed: " << stats_.memory_bytes/1e6 << " MB" << std::endl;
    
    return true;
}
