//===- bindings.cpp - PyBind11 bindings for SparseFlow --------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include "sparseflow/epilogue.h"

namespace py = pybind11;
using namespace sparseflow;

// Compress dense tensor to 2:4 format
torch::Tensor compress_2_4(torch::Tensor dense) {
    TORCH_CHECK(dense.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(dense.dtype() == torch::kFloat16, "Input must be FP16");
    TORCH_CHECK(dense.dim() == 2, "Input must be 2D matrix");
    
    // Use PyTorch's built-in 2:4 compression
    return torch::_cslt_compress(dense);
}

// Validate 2:4 sparsity pattern
bool validate_2_4(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat16, "Input must be FP16");
    
    auto data = tensor.data_ptr<at::Half>();
    int64_t numel = tensor.numel();
    
    // Check every group of 4 consecutive values
    for (int64_t i = 0; i < numel; i += 4) {
        int zero_count = 0;
        for (int j = 0; j < 4 && (i + j) < numel; j++) {
            if (data[i + j] == 0.0f) {
                zero_count++;
            }
        }
        // Must have exactly 2 zeros per 4 values
        if (zero_count != 2) {
            return false;
        }
    }
    
    return true;
}

// Prune dense tensor to 2:4 pattern (magnitude-based)
torch::Tensor prune_2_4_magnitude(torch::Tensor dense) {
    TORCH_CHECK(dense.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(dense.dtype() == torch::kFloat16, "Input must be FP16");
    
    auto result = dense.clone();
    auto data = result.data_ptr<at::Half>();
    int64_t numel = result.numel();
    
    // Process each group of 4
    for (int64_t i = 0; i < numel; i += 4) {
        if (i + 3 >= numel) break;
        
        // Find 2 smallest magnitude values
        float vals[4];
        int indices[4];
        for (int j = 0; j < 4; j++) {
            vals[j] = std::abs(float(data[i + j]));
            indices[j] = j;
        }
        
        // Simple bubble sort for 4 elements
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3 - a; b++) {
                if (vals[b] > vals[b + 1]) {
                    std::swap(vals[b], vals[b + 1]);
                    std::swap(indices[b], indices[b + 1]);
                }
            }
        }
        
        // Zero out 2 smallest
        data[i + indices[0]] = 0.0f;
        data[i + indices[1]] = 0.0f;
    }
    
    return result;
}

// Sparse matrix multiply with epilogue
torch::Tensor sparse_mm_fused(
    torch::Tensor A,
    torch::Tensor Bc,
    std::string epilogue_name,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(Bc.is_cuda(), "Bc must be CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be FP16");
    TORCH_CHECK(Bc.dtype() == torch::kFloat16, "Bc must be FP16");
    
    // Map epilogue name to enum
    EpilogueKind epilogue = EpilogueKind::NONE;
    if (epilogue_name == "relu") {
        epilogue = EpilogueKind::RELU;
    } else if (epilogue_name == "silu") {
        epilogue = EpilogueKind::SILU;
    } else if (epilogue_name == "gelu") {
        epilogue = EpilogueKind::GELU;
    } else if (epilogue_name == "bias") {
        epilogue = EpilogueKind::BIAS;
    } else if (epilogue_name == "bias_relu") {
        epilogue = EpilogueKind::BIAS_RELU;
    }
    
    // For now, use PyTorch's built-in sparse mm
    // TODO: Call our custom kernel with epilogue
    auto result = torch::_cslt_sparse_mm(Bc, A.t()).t();
    
    // Apply epilogue in separate step for now
    if (epilogue == EpilogueKind::RELU) {
        result = torch::relu(result);
    } else if (epilogue == EpilogueKind::SILU) {
        result = torch::silu(result);
    } else if (epilogue == EpilogueKind::BIAS && bias.has_value()) {
        result = result + bias.value();
    } else if (epilogue == EpilogueKind::BIAS_RELU && bias.has_value()) {
        result = torch::relu(result + bias.value());
    }
    
    return result;
}

// Check if GPU supports 2:4 sparse
std::tuple<bool, std::string> check_sparse_support() {
    if (!torch::cuda::is_available()) {
        return std::make_tuple(false, "CUDA not available");
    }
    
    auto props = torch::cuda::getCurrentDeviceProperties();
    int major = props->major;
    int minor = props->minor;
    int sm = major * 10 + minor;
    
    if (sm >= 80) {
        return std::make_tuple(true, 
            std::string("✅ ") + props->name + " (SM" + std::to_string(sm) + 
            ") supports 2:4 sparse");
    } else {
        return std::make_tuple(false,
            std::string("⚠️  ") + props->name + " (SM" + std::to_string(sm) + 
            ") does NOT support 2:4 sparse (requires SM80+/Ampere)");
    }
}

// Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SparseFlow C++ operations";
    
    // Compression
    m.def("compress_2_4", &compress_2_4, 
          "Compress dense tensor to 2:4 sparse format",
          py::arg("dense"));
    
    m.def("validate_2_4", &validate_2_4,
          "Validate tensor has 2:4 sparsity pattern",
          py::arg("tensor"));
    
    m.def("prune_2_4_magnitude", &prune_2_4_magnitude,
          "Prune dense tensor to 2:4 pattern (magnitude-based)",
          py::arg("dense"));
    
    // Sparse operations
    m.def("sparse_mm_fused", &sparse_mm_fused,
          "Sparse matrix multiply with fused epilogue",
          py::arg("A"),
          py::arg("Bc"),
          py::arg("epilogue") = "none",
          py::arg("bias") = py::none());
    
    // GPU compatibility
    m.def("check_sparse_support", &check_sparse_support,
          "Check if GPU supports 2:4 sparse operations");
    
    // Epilogue enum
    py::enum_<EpilogueKind>(m, "EpilogueKind")
        .value("NONE", EpilogueKind::NONE)
        .value("RELU", EpilogueKind::RELU)
        .value("SILU", EpilogueKind::SILU)
        .value("GELU", EpilogueKind::GELU)
        .value("BIAS", EpilogueKind::BIAS)
        .value("BIAS_RELU", EpilogueKind::BIAS_RELU)
        .export_values();
}
