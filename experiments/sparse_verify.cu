#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/numeric_types.h"

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

using Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,           cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8, 8
>;

using ElementA = typename Gemm::ElementA;
using ElementB = typename Gemm::ElementB;
using ElementC = typename Gemm::ElementC;
using ElementE = typename Gemm::ElementE;

using LayoutA  = typename Gemm::LayoutA;
using LayoutB  = typename Gemm::LayoutB;
using LayoutC  = typename Gemm::LayoutC;
using LayoutE  = typename Gemm::LayoutE;

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static size_t file_size(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return size_t(f.tellg());
}

static void load_bin(const char* name, void* dst, size_t bytes) {
    FILE* f = fopen(name, "rb");
    if (!f) {
        std::cerr << "Missing file: " << name << "\n";
        std::exit(1);
    }
    fread(dst, 1, bytes, f);
    fclose(f);
}

int main() {
    std::cout << "=== CUTLASS Sparse Verify ===\n";

    size_t e_bytes = file_size("dE.bin");
    std::cout << "dE.bin size = " << e_bytes << " bytes\n";
    if (e_bytes != 2048) {
        std::cerr << "ERROR: dE.bin MUST be 2048 bytes for 128x128\n";
        return 1;
    }

    std::vector<ElementA> hA(M * K / 2);
    std::vector<ElementB> hB(K * N);
    std::vector<uint8_t>  hE(e_bytes);
    std::vector<ElementC> hD(M * N);

    load_bin("dA.bin", hA.data(), hA.size() * sizeof(ElementA));
    load_bin("dB.bin", hB.data(), hB.size() * sizeof(ElementB));
    load_bin("dE.bin", hE.data(), e_bytes);

    void *dA, *dB, *dC, *dD, *dE;
    cuda_check(cudaMalloc(&dA, hA.size() * sizeof(ElementA)));
    cuda_check(cudaMalloc(&dB, hB.size() * sizeof(ElementB)));
    cuda_check(cudaMalloc(&dC, hD.size() * sizeof(ElementC)));
    cuda_check(cudaMalloc(&dD, hD.size() * sizeof(ElementC)));
    cuda_check(cudaMalloc(&dE, e_bytes));

    cuda_check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(ElementA), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(ElementB), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dE, hE.data(), e_bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(dC, 0, hD.size() * sizeof(ElementC)));

    cutlass::TensorRef<ElementA const, LayoutA> Aref((ElementA*)dA, LayoutA::Stride(K / 2));
    cutlass::TensorRef<ElementB const, LayoutB> Bref((ElementB*)dB, LayoutB::Stride(K));
    cutlass::TensorRef<ElementC, LayoutC>       Cref((ElementC*)dC, LayoutC::Stride(N));
    cutlass::TensorRef<ElementC, LayoutC>       Dref((ElementC*)dD, LayoutC::Stride(N));
    cutlass::TensorRef<ElementE const, LayoutE> Eref((ElementE*)dE, LayoutE::Stride(M));

    typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);

    typename Gemm::Arguments args(
        cutlass::gemm::GemmCoord(M, N, K),
        Aref, Bref, Cref, Dref, Eref,
        epilogue
    );

    Gemm gemm;
    size_t ws_bytes = gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (ws_bytes) cuda_check(cudaMalloc(&workspace, ws_bytes));

    cutlass::Status st = gemm(args, workspace);
    if (st != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS failure: " << cutlassGetStatusString(st) << "\n";
        return 1;
    }

    cuda_check(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(ElementC), cudaMemcpyDeviceToHost));

    FILE* fd = fopen("D_cutlass.bin", "wb");
    fwrite(hD.data(), sizeof(ElementC), hD.size(), fd);
    fclose(fd);

    std::cout << "SUCCESS: D_cutlass.bin written\n";
    std::cout << "First D[0..4]: " << hD[0] << " " << hD[1] << " " << hD[2] << " " << hD[3] << " " << hD[4] << "\n";
    return 0;
}
