#include <iostream>
#include "cutlass/gemm/device/sparse_gemm.h"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using OperatorClass = cutlass::arch::OpClassSparseTensorOp;
using ArchTag = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator>;

using SparseGemmKernel = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB,
    ElementC, LayoutC, ElementAccumulator,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 16>;

int main() {
    SparseGemmKernel sparse_gemm;
    std::cout << "Success" << std::endl;
    return 0;
}
