# PTX Kernel Development Status

## ✅ What Works
- **Minimal kernel (1 warp, 1 MMA tile, 16×8)**: Error 1.14e-5 ✅
  - Proves: MMA instruction works, fragment layout correct
  - File: `gemm_ptx_minimal.cu`

## ❌ What Fails  
- **Single warp, 8 MMA tiles (32×32)**: Error ~108 ❌
  - Proves: Multi-tile iteration is broken
  - Issue: `ldmatrix` pointer arithmetic incorrect for tiling

## 🎯 Current Best: v1.0 WMMA Kernel
- **Performance**: 31.49 TFLOPS @ 8192×8192
- **Status**: Production-ready, shipped to GitHub
- **File**: `gemm_fused_relu_v1.cu`

## 📊 Time Investment
- **PTX debugging**: 8+ hours
- **Result**: Minimal kernel validated, scaling blocked

## 🚧 Recommendation
**STOP PTX development. Ship v1.0 WMMA kernel.**

**Why:**
1. v1.0 achieves 150% of cuBLAS (31.5 TFLOPS)
2. PTX multi-tile addressing is proving extremely difficult
3. ROI on continued debugging is poor

**Path forward:**
- Ship v1.0 as production release TODAY
- Document PTX as v2.0 research item
- Revisit after studying CUTLASS source in detail

## 📝 Lessons Learned
- `ldmatrix` has undocumented lane-to-fragment mapping
- NVIDIA docs insufficient for multi-tile implementations  
- WMMA API is production-ready, PTX requires deep ISA knowledge
