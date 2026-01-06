// src/codegen/epilogue_codegen.h
// Generates CUDA code for epilogues

#ifndef SPARSEFLOW_EPILOGUE_CODEGEN_H
#define SPARSEFLOW_EPILOGUE_CODEGEN_H

#include "sparseflow/epilogue.h"
#include <string>
#include <sstream>

namespace sparseflow {
namespace codegen {

class EpilogueCodeGen {
public:
    // Generate device function for epilogue
    static std::string generateDeviceFunction(EpilogueKind kind) {
        std::stringstream ss;
        
        ss << "// Epilogue: " << getEpilogueName(kind) << "\n";
        ss << "__device__ __forceinline__ half apply_epilogue(half x";
        
        // Add bias parameter if needed
        if (requiresBias(kind)) {
            ss << ", half bias";
        }
        
        ss << ") {\n";
        ss << generateFunctionBody(kind);
        ss << "}\n\n";
        
        return ss.str();
    }
    
    // Generate FP16 version
    static std::string generateFP16Function(EpilogueKind kind) {
        std::stringstream ss;
        
        ss << "// Epilogue FP16: " << getEpilogueName(kind) << "\n";
        ss << "__device__ __forceinline__ half2 apply_epilogue_fp16(half2 x";
        
        if (requiresBias(kind)) {
            ss << ", half2 bias";
        }
        
        ss << ") {\n";
        ss << generateFP16FunctionBody(kind);
        ss << "}\n\n";
        
        return ss.str();
    }
    
private:
    static bool requiresBias(EpilogueKind kind) {
        return kind == EpilogueKind::BIAS ||
               kind == EpilogueKind::BIAS_RELU ||
               kind == EpilogueKind::BIAS_SILU;
    }
    
    static std::string generateFunctionBody(EpilogueKind kind) {
        switch(kind) {
            case EpilogueKind::NONE:
                return "    return x;\n";
                
            case EpilogueKind::RELU:
                return "    return __hmax(x, __float2half(0.0f));\n";
                
            case EpilogueKind::SILU:
                return R"(    // SiLU: x * sigmoid(x)
    half sigmoid = __hdiv(__float2half(1.0f), 
                         __hadd(__float2half(1.0f), 
                               hexp(__hneg(x))));
    return __hmul(x, sigmoid);
)";
                
            case EpilogueKind::GELU:
                return R"(    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    half x3 = __hmul(__hmul(x, x), x);
    half inner = __hadd(x, __hmul(__float2half(0.044715f), x3));
    inner = __hmul(__float2half(0.7978845608f), inner); // sqrt(2/π)
    half tanh_val = tanh(__half2float(inner)); // Note: no half tanh in CUDA
    tanh_val = __float2half(tanh_val);
    half result = __hmul(__float2half(0.5f), __hmul(x, __hadd(__float2half(1.0f), tanh_val)));
    return result;
)";
                
            case EpilogueKind::BIAS:
                return "    return __hadd(x, bias);\n";
                
            case EpilogueKind::BIAS_RELU:
                return "    return __hmax(__hadd(x, bias), __float2half(0.0f));\n";
                
            case EpilogueKind::BIAS_SILU:
                return R"(    half sum = __hadd(x, bias);
    half sigmoid = __hdiv(__float2half(1.0f), 
                         __hadd(__float2half(1.0f), 
                               hexp(__hneg(sum))));
    return __hmul(sum, sigmoid);
)";
                
            default:
                return "    return x; // Unknown epilogue\n";
        }
    }
    
    static std::string generateFP16FunctionBody(EpilogueKind kind) {
        switch(kind) {
            case EpilogueKind::NONE:
                return "    return x;\n";
                
            case EpilogueKind::RELU:
                return "    return __hmax2(x, __float2half2_rn(0.0f));\n";
                
            case EpilogueKind::SILU:
                return R"(    // SiLU vectorized
    half2 one = __float2half2_rn(1.0f);
    half2 sigmoid = h2div(one, __hadd2(one, h2exp(__hneg2(x))));
    return __hmul2(x, sigmoid);
)";
                
            case EpilogueKind::BIAS:
                return "    return __hadd2(x, bias);\n";
                
            case EpilogueKind::BIAS_RELU:
                return "    return __hmax2(__hadd2(x, bias), __float2half2_rn(0.0f));\n";
                
            default:
                return generateFunctionBody(kind); // Fall back to scalar
        }
    }
};

} // namespace codegen
} // namespace sparseflow

#endif // SPARSEFLOW_EPILOGUE_CODEGEN_H
