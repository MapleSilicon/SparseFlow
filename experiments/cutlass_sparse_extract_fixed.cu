#include <cutlass/cutlass.h>

#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/gemm.h>

#include <cuda_runtime.h>
#include <cstdio>

////////////////////////////////////////////////////////////////
// Element types
////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;   // Sparse A (2:4)
using ElementB = cutlass::half_t;   // Dense B
using ElementC = float;
using ElementAccumulator = float;
using ElementE = int32_t;           // Metadata is int32_t (packed 2-bit)

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

////////////////////////////////////////////////////////////////
// Tensor Core shapes (Ampere SPARSE)
////////////////////////////////////////////////////////////////

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

////////////////////////////////////////////////////////////////
// Epilogue
////////////////////////////////////////////////////////////////

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator>;

////////////////////////////////////////////////////////////////
// Sparse GEMM Kernel (CORRECT TYPE)
////////////////////////////////////////////////////////////////

using SparseGemmKernel =
    cutlass::gemm::device::GemmSparseUniversalAdapter<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp>;

////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////

int main() {

    int M = 256;
    int N = 256;
    int K = 256;

    size_t sizeA = M * K * sizeof(ElementA);
    size_t sizeB = K * N * sizeof(ElementB);
    size_t sizeC = M * N * sizeof(ElementC);
    size_t sizeE = M * K * sizeof(ElementE); // Metadata size (dummy)

    ElementA *A;
    ElementB *B;
    ElementC *C;
    ElementE *E; // Metadata pointer

    cudaMalloc(&A, sizeA);
    cudaMalloc(&B, sizeB);
    cudaMalloc(&C, sizeC);
    cudaMalloc(&E, sizeE);

    SparseGemmKernel gemm;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename SparseGemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {A, K},
        {B, K},
        {C, N},
        {C, N},
        {1.0f, 0.0f},
        {E, K} // <--- CRITICAL: Added Metadata Argument here
    };

    cutlass::Status status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        printf("❌ CUTLASS sparse init failed\n");
        return -1;
    }

    status = gemm();
    if (status != cutlass::Status::kSuccess) {
        printf("❌ CUTLASS sparse launch failed\n");
        return -1;
    }

    cudaDeviceSynchronize();
    printf("✅ CUTLASS SPARSE GEMM EXECUTED (2:4 Tensor Cores)\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(E);

    return 0;
}
