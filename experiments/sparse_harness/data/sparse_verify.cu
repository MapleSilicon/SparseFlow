#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_sparse.h"

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

static std::vector<uint8_t> load_file_bytes(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::perror(path); std::exit(1); }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz <= 0) { std::cerr << "Bad size for " << path << "\n"; std::exit(1); }
  std::vector<uint8_t> buf((size_t)sz);
  size_t got = std::fread(buf.data(), 1, buf.size(), f);
  std::fclose(f);
  if (got != buf.size()) {
    std::cerr << "Short read for " << path << " got " << got
              << " expected " << buf.size() << "\n";
    std::exit(1);
  }
  return buf;
}

static void save_bin_or_die(const char* path, const void* src, size_t bytes) {
  FILE* f = std::fopen(path, "wb");
  if (!f) { std::perror(path); std::exit(1); }
  size_t wrote = std::fwrite(src, 1, bytes, f);
  std::fclose(f);
  if (wrote != bytes) {
    std::cerr << "Short write for " << path << " wrote " << wrote
              << " expected " << bytes << "\n";
    std::exit(1);
  }
}

int main() {
  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 128;

  using Gemm = cutlass::gemm::device::SparseGemm<
      cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      float,          cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64,  64,  64>,
      cutlass::gemm::GemmShape<16,  8,  32>,
      cutlass::epilogue::thread::LinearCombination<
          float, 4, float, float,
          cutlass::epilogue::thread::ScaleType::Default,
          cutlass::FloatRoundStyle::round_to_nearest,
          float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      3,
      8, 8
  >;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using ElementE = typename Gemm::ElementE;   // likely uint16_t here
  using LayoutE  = typename Gemm::LayoutE;    // ColumnMajorInterleaved<2>

  using LayoutA  = typename Gemm::LayoutA;
  using LayoutB  = typename Gemm::LayoutB;
  using LayoutC  = typename Gemm::LayoutC;

  std::cout << "sizeof(Gemm::ElementE) = " << sizeof(ElementE) << " bytes\n";

  // Load A/B
  std::vector<ElementA> hA(M * (K/2));
  std::vector<ElementB> hB(K * N);

  {
    FILE* fa = std::fopen("dA.bin", "rb");
    if (!fa) { std::perror("dA.bin"); return 1; }
    if (std::fread(hA.data(), sizeof(ElementA), hA.size(), fa) != hA.size()) {
      std::cerr << "Short read dA.bin\n"; return 1;
    }
    std::fclose(fa);
  }
  {
    FILE* fb = std::fopen("dB.bin", "rb");
    if (!fb) { std::perror("dB.bin"); return 1; }
    if (std::fread(hB.data(), sizeof(ElementB), hB.size(), fb) != hB.size()) {
      std::cerr << "Short read dB.bin\n"; return 1;
    }
    std::fclose(fb);
  }

  // Load metadata as raw bytes (don't guess its element type/length)
  std::vector<uint8_t> hE_bytes = load_file_bytes("dE.bin");

  void *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr, *dEbytes=nullptr;
  CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(ElementA)));
  CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(ElementB)));
  CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(ElementC)));
  CHECK_CUDA(cudaMalloc(&dD, M * N * sizeof(ElementC)));
  CHECK_CUDA(cudaMalloc(&dEbytes, hE_bytes.size()));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(ElementA), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(ElementB), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dEbytes, hE_bytes.data(), hE_bytes.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(ElementC)));
  CHECK_CUDA(cudaMemset(dD, 0, M * N * sizeof(ElementC)));

  cutlass::TensorRef<ElementA const, LayoutA> tensor_a(
      (ElementA const*)dA, LayoutA::Stride(K/2));

  cutlass::TensorRef<ElementB const, LayoutB> tensor_b(
      (ElementB const*)dB, LayoutB::Stride(K));

  cutlass::TensorRef<ElementC const, LayoutC> tensor_c(
      (ElementC const*)dC, LayoutC::Stride(N));

  cutlass::TensorRef<ElementC, LayoutC> tensor_d(
      (ElementC*)dD, LayoutC::Stride(N));

  // 🔥 KEY FIX: E must be NON-CONST TensorRef<ElementE, LayoutE> to match Arguments
  ElementE* ptrE = reinterpret_cast<ElementE*>(dEbytes);
  cutlass::TensorRef<ElementE, LayoutE> tensor_e(
      ptrE, LayoutE::Stride(M));

  typename Gemm::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);

  typename Gemm::Arguments args(
      cutlass::gemm::GemmCoord(M, N, K),
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      tensor_e,
      epilogue_params,
      1
  );

  Gemm gemm_op;

  size_t workspace_bytes = gemm_op.get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes) CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

  auto st = gemm_op.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed: " << cutlassGetStatusString(st) << "\n";
    return 1;
  }

  st = gemm_op.initialize(args, workspace);
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "initialize failed: " << cutlassGetStatusString(st) << "\n";
    return 1;
  }

  st = gemm_op();
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "run failed: " << cutlassGetStatusString(st) << "\n";
    return 1;
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<ElementC> hD(M * N);
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(ElementC), cudaMemcpyDeviceToHost));
  save_bin_or_die("D_cutlass.bin", hD.data(), hD.size() * sizeof(ElementC));

  std::cout << "Success! Wrote D_cutlass.bin\n";
  std::cout << "First D[0..4]: " << hD[0] << " " << hD[1] << " " << hD[2] << " " << hD[3] << " " << hD[4] << "\n";

  if (workspace) CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaFree(dD));
  CHECK_CUDA(cudaFree(dEbytes));
  return 0;
}
