#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct SparseConfig {
    int M = 64, N = 64, K = 64;
    int n = 2, m = 4;
    std::string configPath;
};

bool loadHardwareConfig(SparseConfig &cfg) {
    std::ifstream file(cfg.configPath);
    if (!file.is_open()) {
        return false;
    }
    
    json j;
    try {
        file >> j;
    } catch (const json::exception &e) {
        return false;
    }
    
    // NEW FORMAT: operations array
    if (j.contains("operations") && j["operations"].is_array() && !j["operations"].empty()) {
        const auto &op = j["operations"][0];
        if (op.value("type", "") == "matmul") {
            cfg.M = op.value("M", 64);
            cfg.N = op.value("N", 64);
            cfg.K = op.value("K", 64);
            cfg.n = op.value("n", 2);
            cfg.m = op.value("m", 4);
            std::printf("[DEBUG] ✅ Config loaded from JSON\n");
            return true;
        }
    }
    
    return false;
}

int main() {
    SparseConfig cfg;
    
    // Try multiple relative paths
    std::vector<std::string> candidates = {
        "../../compiler/build/hardware_config.json",
        "../compiler/build/hardware_config.json",
        "compiler/build/hardware_config.json"
    };
    
    bool loaded = false;
    for (const auto& path : candidates) {
        cfg.configPath = path;
        std::printf("[DEBUG] Attempting to load: %s\n", path.c_str());
        if (loadHardwareConfig(cfg)) {
            std::printf("[DEBUG] ✅ Config loaded from: %s\n", path.c_str());
            loaded = true;
            break;
        }
    }
    
    if (!loaded) {
        std::printf("[DEBUG] ⚠️  Could not load config, using defaults\n");
    }
    
    int dim_m = cfg.M;
    int dim_n = cfg.N;
    int dim_k = cfg.K;
    float density = (cfg.m > 0) ? float(cfg.n) / float(cfg.m) : 0.0f;
    
    std::printf("\n=== Programming MapleSilicon Hardware ===\n");
    std::printf("Pattern: %d:%d\n", cfg.n, cfg.m);
    std::printf("Dimensions: %dx%dx%d\n", dim_m, dim_n, dim_k);
    std::printf("Data Type: f32\n");
    std::printf("HW_WRITE [0x1000] = %d (N)\n", cfg.n);
    std::printf("HW_WRITE [0x1004] = %d (M)\n", cfg.m);
    std::printf("HW_WRITE [0x1008] = %d (dim_m)\n", dim_m);
    std::printf("HW_WRITE [0x100c] = %d (dim_n)\n", dim_n);
    std::printf("HW_WRITE [0x1010] = %d (dim_k)\n", dim_k);
    
    std::printf("\n=== Executing Sparse Matmul ===\n");
    std::printf("Matrix: %dx%d * %dx%d\n", dim_m, dim_k, dim_k, dim_n);
    
    long long totalMACs = (long long)dim_m * dim_k * dim_n;
    long long executedMACs = (long long)(totalMACs * density);
    float speedup = 1.0f / density;
    
    std::printf("Total MACs: %lld\n", totalMACs);
    std::printf("Executed MACs: %lld\n", executedMACs);
    std::printf("Compute Efficiency: %.0f%%\n", density * 100);
    std::printf("Theoretical Speedup: %.1fx\n", speedup);
    
    std::printf("\n=== SparseFlow v0.1 Pipeline Complete! ===\n");
    return 0;
}
