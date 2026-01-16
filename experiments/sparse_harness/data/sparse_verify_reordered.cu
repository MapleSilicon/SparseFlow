// sparse_verify_reordered.cu
// CUTLASS 3.x canonical sparse GEMM verification for SM80

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

static void load_file(const char* path, void* dst, size_t bytes) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); std::exit(1); }
  fread(dst, 1, bytes, f);
  fclose(f);
}

static void save_file(const char* path, const void* src, size_t bytes) {
  FILE* f = fopen(path, "wb");
  if (!f) { perror(path); std::exit(1); }
  fwrite(src, 1, bytes, f);
  fclose(f);
}

using Gemm = cutlass::gemm::device::SparseGemm<
  cutlass::half_t, cutlass::layout::RowMajor,
  cutlass::half_t, cutlass::layout::ColumnMajor,
  float, cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,64>,
  cutlass::gemm::GemmShape<64,64,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3, 8, 8
>;

using ElementA = Gemm::ElementA;
using ElementB = Gemm::ElementB;
using ElementC = Gemm::ElementC;
using ElementE = Gemm::ElementE;

using LayoutA = Gemm::LayoutA;
using LayoutB = Gemm::LayoutB;
using LayoutC = Gemm::LayoutC;
using LayoutE = Gemm::LayoutE;

int main() {
  std::cout << "sizeof(Gemm::ElementE) = " << sizeof(ElementE) << " bytes\n";

  std::vector<ElementA> A_dense(M*K);
  load_file("A_pruned_dense.bin", A_dense.data(), A_dense.size()*sizeof(ElementA));

  std::vector<ElementA> A_reordered(M*K/2);
  std::vector<ElementE> E_reordered(M*K/8);

  using Reorder = cutlass::gemm::host::SparseGemmReorder<
    cutlass::arch::Sm80,
    ElementA,
    LayoutA
  >;

  Reorder reorder;
  reorder(
    A_reordered.data(),
    E_reordered.data(),
    A_dense.data(),
    M, K
  );

  std::vector<ElementB> B(K*N);
  load_file("dB.bin", B.data(), B.size()*sizeof(ElementB));

  void *dA, *dB, *dC, *dD, *dE;
  cudaMalloc(&dA, A_reordered.size()*sizeof(ElementA));
  cudaMalloc(&dB, B.size()*sizeof(ElementB));
  cudaMalloc(&dE, E_reordered.size()*sizeof(ElementE));
  cudaMalloc(&dC, M*N*sizeof(ElementC));
  cudaMalloc(&dD, M*N*sizeof(ElementC));

  cudaMemcpy(dA, A_reordered.data(), A_reordered.size()*sizeof(ElementA), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size()*sizeof(ElementB), cudaMemcpyHostToDevice);
  cudaMemcpy(dE, E_reordered.data(), E_reordered.size()*sizeof(ElementE), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, M*N*sizeof(ElementC));

  cutlass::TensorRef<ElementA const, LayoutA> Aref((ElementA*)dA, LayoutA::Stride(K/2));
  cutlass::TensorRef<ElementB const, LayoutB> Bref((ElementB*)dB, LayoutB::Stride(K));
  cutlass::TensorRef<ElementC const, LayoutC> Cref((ElementC*)dC, LayoutC::Stride(N));
  cutlass::TensorRef<ElementC, LayoutC>       Dref((ElementC*)dD, LayoutC::Stride(N));
  cutlass::TensorRef<ElementE const, LayoutE> Eref((ElementE*)dE, LayoutE::Stride(M));

  typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);

  typename Gemm::Arguments args(
    {M, N, K},
    Aref, Bref, Cref, Dref, Eref,
    epilogue
  );

  Gemm gemm;
  size_t ws = gemm.get_workspace_size(args);
  void* workspace = nullptr;
  if (ws) cudaMalloc(&workspace, ws);

  auto status = gemm(args, workspace);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS error\n";
    return 1;
  }

  std::vector<ElementC> D(M*N);
  cudaMemcpy(D.data(), dD, D.size()*sizeof(ElementC), cudaMemcpyDeviceToHost);

  save_file("D_cutlass.bin", D.data(), D.size()*sizeof(ElementC));

  std::cout << "SUCCESS: D_cutlass.bin written\n";
  std::cout << "First D[0..4]: "
            << D[0] << " " << D[1] << " "
            << D[2] << " " << D[3] << " " << D[4] << "\n";

  return 0;
}
