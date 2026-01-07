// include/sparseflow/epilogue.h
// SparseFlow Epilogue Fusion ABI v1.0

#ifndef SPARSEFLOW_EPILOGUE_H
#define SPARSEFLOW_EPILOGUE_H

#include <cstdint>

namespace sparseflow {

// ABI Version (increment on breaking changes)
constexpr int ABI_VERSION = 1;

// Supported epilogue operations
enum class EpilogueKind : int32_t {
    NONE = 0,           // No epilogue
    RELU = 1,           // max(0, x)
    SILU = 2,           // x * sigmoid(x)
    GELU = 3,           // GELU approximation
    BIAS = 4,           // x + bias
    BIAS_RELU = 5,      // max(0, x + bias)
    BIAS_SILU = 6,      // (x + bias) * sigmoid(x + bias)
    CUSTOM = 99,        // User-defined
};

// Epilogue configuration
struct EpilogueConfig {
    EpilogueKind kind;
    void* params;       // Optional parameters (e.g., bias pointer)
    int32_t param_size; // Size of params in bytes
    
    EpilogueConfig() 
        : kind(EpilogueKind::NONE), params(nullptr), param_size(0) {}
    
    EpilogueConfig(EpilogueKind k, void* p = nullptr, int32_t size = 0)
        : kind(k), params(p), param_size(size) {}
};

// Get epilogue name (for debugging/logging)
inline const char* getEpilogueName(EpilogueKind kind) {
    switch(kind) {
        case EpilogueKind::NONE: return "none";
        case EpilogueKind::RELU: return "relu";
        case EpilogueKind::SILU: return "silu";
        case EpilogueKind::GELU: return "gelu";
        case EpilogueKind::BIAS: return "bias";
        case EpilogueKind::BIAS_RELU: return "bias_relu";
        case EpilogueKind::BIAS_SILU: return "bias_silu";
        case EpilogueKind::CUSTOM: return "custom";
        default: return "unknown";
    }
}

// Validate epilogue configuration
inline bool isValidEpilogue(const EpilogueConfig& config) {
    // BIAS operations require params
    if (config.kind == EpilogueKind::BIAS ||
        config.kind == EpilogueKind::BIAS_RELU ||
        config.kind == EpilogueKind::BIAS_SILU) {
        return config.params != nullptr && config.param_size > 0;
    }
    return true;
}

} // namespace sparseflow

#endif // SPARSEFLOW_EPILOGUE_H
