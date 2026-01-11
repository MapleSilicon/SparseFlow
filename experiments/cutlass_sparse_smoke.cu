#include <cstdio>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#define CK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

using ElementA = cutlass::half_t;     // A is compressed (2:4)
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// IMPORTANT: Use a CUTLASS-supported sparse MMA tile.
// The instruction is m16n8k32.
// Choose ThreadblockK=64, WarpK=32 -> 2 iterations (required >1 and even)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64,  64,  32>;
using InstructionShape = cutlass::gemm::GemmShape<16,  8,   32>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    1, // elements per vectorized store (safe for smoke test)
    ElementAccumulator,
    ElementAccumulator
>;

using SparseGemm = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3 // stages
>;

int main() {
  int M = 512, N = 512, K = 512;

  // CUTLASS 2:4 sparse expects:
  // A stored as "compressed" (50% of elements kept)
  // E is metadata (2 bits per element group); CUTLASS uses a specific packing.
  // For a smoke test we only need the kernel to launch; correctness isn't the goal.
  size_t size_A = size_t(M) * size_t(K) / 2 * sizeof(ElementA);
  size_t size_B = size_t(K) * size_t(N) * sizeof(ElementB);
  size_t size_C = size_t(M) * size_t(N) * sizeof(ElementC);

  // Metadata size: CUTLASS commonly uses 2 bits per element in A, packed.
  // A pragmatic "big enough" allocation for launch safety:
  size_t size_E = size_t(M) * size_t(K) / 16 * sizeof(uint32_t);

  ElementA *A = nullptr;
  ElementB *B = nullptr;
  ElementC *C = nullptr;
  uint32_t *E = nullptr;

  CK(cudaMalloc((void**)&A, size_A));
  CK(cudaMalloc((void**)&B, size_B));
  CK(cudaMalloc((void**)&C, size_C));
  CK(cudaMalloc((void**)&E, size_E));

  // Zero to avoid NaNs in epilogue
  CK(cudaMemset(A, 0, size_A));
  CK(cudaMemset(B, 0, size_B));
  CK(cudaMemset(C, 0, size_C));
  CK(cudaMemset(E, 0, size_E));

  typename EpilogueOp::Params epilogue_params(1.0f, 0.0f);

  // Leading dimensions:
  // A is compressed: lda = K/2
  // B is column major: ldb = K
  // C is row major: ldc = N
  typename SparseGemm::Arguments args(
      {M, N, K},
      {A, K/2},
      {B, K},
      {C, N},
      {C, N},
      {E, K/16},
      epilogue_params
  );

  SparseGemm op;
  auto st = op.initialize(args);
  if (st != cutlass::Status::kSuccess) {
    printf("init failed: %d\n", int(st));
    return 2;
  }

  st = op();
  if (st != cutlass::Status::kSuccess) {
    printf("run failed: %d\n", int(st));
    return 3;
  }

  CK(cudaDeviceSynchronize());
  printf("✅ CUTLASS SparseGemm launched (smoke).\n");

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(E);

  return 0;
}
