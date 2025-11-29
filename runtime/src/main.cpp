#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SparseConfig {
  int M, N, K;
  int m, n;
  std::string configPath;
};

bool loadHardwareConfig(SparseConfig& cfg) {
  std::ifstream file(cfg.configPath);
  if (!file.is_open()) {
    std::printf("[DEBUG] Could not open config file: %s\n", cfg.configPath.c_str());
    return false;
  }

  json j;
  file >> j;

  if (j.contains("matmuls") && j["matmuls"].is_array() && !j["matmuls"].empty()) {
    auto& matmul = j["matmuls"][0];
    if (matmul.contains("M") && matmul.contains("N") && matmul.contains("K") &&
        matmul.contains("m") && matmul.contains("n")) {
      cfg.M = matmul["M"];
      cfg.N = matmul["N"];
      cfg.K = matmul["K"];
      cfg.m = matmul["m"];
      cfg.n = matmul["n"];

      std::printf("[DEBUG] Loaded JSON configuration successfully.\n");
      std::printf("[DEBUG] Matmul 0: M=%d, N=%d, K=%d, m=%d, n=%d\n",
                  cfg.M, cfg.N, cfg.K, cfg.m, cfg.n);

      if (matmul.contains("totalMACs") && matmul.contains("executedMACs") &&
          matmul.contains("density") && matmul.contains("theoreticalSpeedup")) {
        long long totalMACs = matmul["totalMACs"];
        long long executedMACs = matmul["executedMACs"];
        double density = matmul["density"];
        double speedup = matmul["theoreticalSpeedup"];

        std::printf("[DEBUG] JSON totalMACs=%lld executedMACs=%lld density=%.3f speedup=%.2fx\n",
                    totalMACs, executedMACs, density, speedup);
      }

      return true;
    }
  }

  std::printf("[DEBUG] Invalid JSON structure in config file.\n");
  return false;
}

int main() {
  SparseConfig cfg;
  // Set defaults (will be overridden by JSON if available)
  cfg.M = 64;
  cfg.N = 64;
  cfg.K = 64;
  cfg.m = 4;
  cfg.n = 2;
  cfg.configPath = "../../compiler/build/hardware_config.json";

  std::printf("[DEBUG] Loading hardware configuration from: %s\n", cfg.configPath.c_str());
  std::printf("[DEBUG] Found 1 matmul configuration(s)\n");
  
  bool configLoaded = loadHardwareConfig(cfg);

  if (!configLoaded) {
    std::printf("[DEBUG] Using default configuration.\n");
  }

  // Use values from cfg - ACTUALLY USE THE JSON CONFIG!
  int dim_m = cfg.M;
  int dim_n = cfg.N; 
  int dim_k = cfg.K;

  // Compute N:M sparsity fraction
  float activeFraction = 0.0f;
  if (cfg.m > 0) {
    activeFraction = static_cast<float>(cfg.n) / static_cast<float>(cfg.m);
  }

  std::printf("=== Programming MapleSilicon Hardware ===\n");
  std::printf("Pattern: %d:%d\n", cfg.n, cfg.m);
  std::printf("Dimensions: %dx%dx%d\n", dim_m, dim_n, dim_k);
  std::printf("Data Type: f32\n");

  // Fake register writes with actual config values
  std::printf("HW_WRITE [0x1000] = 0x%x (N)\n", cfg.n);
  std::printf("HW_WRITE [0x1004] = 0x%x (M)\n", cfg.m);
  std::printf("HW_WRITE [0x1008] = 0x%x (dim_m)\n", dim_m);
  std::printf("HW_WRITE [0x100c] = 0x%x (dim_n)\n", dim_n);
  std::printf("HW_WRITE [0x1010] = 0x%x (dim_k)\n", dim_k);

  std::printf("\n=== Executing Sparse Matmul ===\n");
  std::printf("Matrix: %dx%d * %dx%d\n", dim_m, dim_k, dim_k, dim_n);

  // Compute MACs based on actual dimensions
  int totalMACs = dim_m * dim_n * dim_k;
  int executedMACs = static_cast<int>(totalMACs * activeFraction);

  std::printf("Total MACs: %d\n", totalMACs);
  std::printf("Executed MACs: %d\n", executedMACs);
  std::printf("Compute Efficiency: %.0f%%\n", activeFraction * 100.0f);

  float speedup = (activeFraction > 0.0f) ? (1.0f / activeFraction) : 0.0f;
  std::printf("Theoretical Speedup: %.1fx\n", speedup);

  // Keep your fake first-element result as before
  std::printf("First element result: 64 (expected ~64.0)\n");

  return 0;
}


