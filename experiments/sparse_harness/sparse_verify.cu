#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error\n"); exit(1);}

int main() {
  int M=128,N=128,K=128;
  int K_HALF=K/2;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;

  using Gemm = cutlass::gemm::device::SparseGemm<ElementA, cutlass::layout::RowMajor, ElementB, cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>>;

  std::vector<ElementA> hA(M*K_HALF);
  std::vector<ElementB> hB(K*N);
  std::vector<uint8_t>  hE(M*K/8);
  std::vector<float>    hD(M*N,0), hC(M*N,0);

  FILE *fa=fopen("dA.bin","rb");
  FILE *fb=fopen("dB.bin","rb");
  FILE *fe=fopen("dE.bin","rb");
  fread(hA.data(),sizeof(ElementA),hA.size(),fa);
  fread(hB.data(),sizeof(ElementB),hB.size(),fb);
  fread(hE.data(),1,hE.size(),fe);
  fclose(fa); fclose(fb); fclose(fe);

  void *dA,*dB,*dE,*dC,*dD;
  CHECK_CUDA(cudaMalloc(&dA,hA.size()*2));
  CHECK_CUDA(cudaMalloc(&dB,hB.size()*2));
  CHECK_CUDA(cudaMalloc(&dE,hE.size()));
  CHECK_CUDA(cudaMalloc(&dC,hC.size()*4));
  CHECK_CUDA(cudaMalloc(&dD,hD.size()*4));

  CHECK_CUDA(cudaMemcpy(dA,hA.data(),hA.size()*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB.data(),hB.size()*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dE,hE.data(),hE.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC,hC.data(),hC.size()*4,cudaMemcpyHostToDevice));

  typename Gemm::Arguments args(
    {M,N,K},
    {(ElementA*)dA,K_HALF},
    {(ElementB*)dB,K},
    {(float*)dC,N},
    {(float*)dD,N},
    (uint8_t*)dE,
    {1.0f,0.0f}
  );

  Gemm gemm;
  size_t ws=gemm.get_workspace_size(args);
  void *workspace=nullptr;
  if(ws>0) CHECK_CUDA(cudaMalloc(&workspace,ws));

  if(gemm(args,workspace)!=cutlass::Status::kSuccess){
    printf("CUTLASS FAILED\n"); return 1;
  }

  CHECK_CUDA(cudaMemcpy(hD.data(),dD,hD.size()*4,cudaMemcpyDeviceToHost));
  FILE *fd=fopen("D_cutlass.bin","wb");
  fwrite(hD.data(),4,hD.size(),fd);
  fclose(fd);

  printf("OK: D_cutlass.bin written\n");
  return 0;
}
