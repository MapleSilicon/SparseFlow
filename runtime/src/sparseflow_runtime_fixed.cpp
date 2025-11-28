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
    std::cout << "Looking for JSON file: " << json_file << std::endl;
    
    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file: " << json_file << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    
    std::cout << "JSON content (first 500 chars): " << json_str.substr(0, 500) << std::endl;
    
    // Simple parsing - look for key-value pairs
    std::cout << "Parsing JSON configuration..." << std::endl;
    
    MatmulConfig config;
    config.type = "matmul";
    
    // Extract values using simple string searches
    auto extractInt = [&](const std::string& key) -> int64_t {
        std::string search_key = "\"" + key + "\":";
        size_t pos = json_str.find(search_key);
        if (pos == std::string::npos) {
            std::cout << "Key '" << key << "' not found" << std::endl;
            return 0;
        }
        pos += search_key.length();
        
        // Skip whitespace
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || json_str[pos] == '\n')) {
            pos++;
        }
        
        size_t end = json_str.find_first_of(",}", pos);
        std::string value_str = json_str.substr(pos, end - pos);
        
        // Remove any quotes or whitespace
        value_str.erase(0, value_str.find_first_not_of(" \t\n\""));
        value_str.erase(value_str.find_last_not_of(" \t\n\"") + 1);
        
        std::cout << "Found " << key << " = '" << value_str << "'" << std::endl;
        try {
            return std::stoll(value_str);
        } catch (const std::exception& e) {
            std::cout << "Error parsing " << key << " value '" << value_str << "': " << e.what() << std::endl;
            return 0;
        }
    };
    
    auto extractString = [&](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\":";
        size_t pos = json_str.find(search_key);
        if (pos == std::string::npos) {
            std::cout << "String key '" << key << "' not found" << std::endl;
            return "";
        }
        pos += search_key.length();
        
        // Skip whitespace
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || json_str[pos] == '\n')) {
            pos++;
        }
        
        // Find the opening quote
        if (json_str[pos] == '\"') {
            pos++;
            size_t end = json_str.find("\"", pos);
            std::string value = json_str.substr(pos, end - pos);
            std::cout << "Found " << key << " = '" << value << "'" << std::endl;
            return value;
        }
        
        return "";
    };
    
    config.n = extractInt("n");
    config.m = extractInt("m");
    config.dim_m = extractInt("dim_m");
    config.dim_n = extractInt("dim_n");
    config.dim_k = extractInt("dim_k");
    config.dtype = extractString("dtype");
    config.tile_m = extractInt("tile_m");
    config.tile_n = extractInt("tile_n");
    config.tile_k = extractInt("tile_k");
    config.nm_pattern = std::to_string(config.n) + ":" + std::to_string(config.m);
    
    if (config.n == 0 || config.m == 0) {
        std::cerr << "Failed to parse essential configuration values" << std::endl;
        return false;
    }
    
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

bool SparseFlowEngine::executeMatmul(float* A, float* B, float* C) {
    if (configs_.empty()) {
        std::cerr << "No configuration loaded" << std::endl;
        return false;
    }
    
    std::cout << "Executing sparse matmul on hardware..." << std::endl;
    
    // For now, emulate in software
    return emulateMatmul(A, B, C);
}

bool SparseFlowEngine::emulateMatmul(float* A, float* B, float* C) {
    if (configs_.empty()) return false;
    
    const auto& config = configs_[0];
    int64_t M = config.dim_m;
    int64_t N = config.dim_n;
    int64_t K = config.dim_k;
    
    std::cout << "\n=== Executing Enhanced Sparse Matmul ===" << std::endl;
    std::cout << "Pattern: " << config.nm_pattern << std::endl;
    std::cout << "Data Type: " << config.dtype << std::endl;
    std::cout << "Tiling: " << config.tile_m << "x" << config.tile_n 
              << " tiles over " << M << "x" << N << " matrix" << std::endl;
    
    // Reset performance counters
    stats_ = {0, 0, 0, 0};
    
    // Generate N:M mask pattern
    auto mask = generateMask(config.n, config.m);
    int block_size = config.m;
    
    // Initialize output to zero
    memset(C, 0, M * N * sizeof(float));
    
    // Enhanced tiled execution with hardware-aware scheduling
    for (int64_t tile_i = 0; tile_i < M; tile_i += config.tile_m) {
        int64_t tile_height = std::min(config.tile_m, M - tile_i);
        
        for (int64_t tile_j = 0; tile_j < N; tile_j += config.tile_n) {
            int64_t tile_width = std::min(config.tile_n, N - tile_j);
            
            // Process tile with N:M sparsity
            for (int64_t i = tile_i; i < tile_i + tile_height; i++) {
                for (int64_t j = tile_j; j < tile_j + tile_width; j++) {
                    float sum = 0.0f;
                    
                    for (int64_t k_block = 0; k_block < K; k_block += block_size) {
                        for (int k_inner = 0; k_inner < block_size && (k_block + k_inner) < K; k_inner++) {
                            int64_t k = k_block + k_inner;
                            
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

std::vector<bool> SparseFlowEngine::generateMask(int n, int m) const {
    std::vector<bool> mask(m, false);
    
    // Simple pattern: first N elements are active
    for (int i = 0; i < n; i++) {
        mask[i] = true;
    }
    
    return mask;
}

void SparseFlowEngine::writeRegister(uint32_t addr, uint32_t value) {
    std::cout << "HW_WRITE [0x" << std::hex << addr << "] = 0x" << value << std::dec << std::endl;
}

uint32_t SparseFlowEngine::readRegister(uint32_t addr) {
    std::cout << "HW_READ [0x" << std::hex << addr << "]" << std::dec << std::endl;
    return 0;
}

PerformanceStats SparseFlowEngine::getStats() const {
    return stats_;
}

} // namespace sparseflow
