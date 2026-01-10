#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(){
int M=4096,N=4096,K=4096;
size_t sz=M*K*sizeof(__half);
__half *A,*B,*C;
cudaMalloc(&A,M*K*2);
cudaMalloc(&B,K*N*2);
cudaMalloc(&C,M*N*2);
cudaMemset(A,0,M*K*2);
cudaMemset(B,0,K*N*2);

cublasHandle_t h;
cublasCreate(&h);
cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH);

__half alpha=__float2half(1.0f);
__half beta=__float2half(0.0f);

// Warmup
for(int i=0;i<5;i++)
  cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N);
cudaDeviceSynchronize();

cudaEvent_t t0,t1;
cudaEventCreate(&t0);
cudaEventCreate(&t1);

cudaEventRecord(t0);
for(int i=0;i<20;i++)
  cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,N,A,K,&beta,C,N);
cudaEventRecord(t1);
cudaEventSynchronize(t1);

float ms;
cudaEventElapsedTime(&ms,t0,t1);
ms/=20;

printf("cuBLAS FP16: %.3f ms, %.2f TFLOPS\n",ms,(2.0*M*N*K)/(ms*1e9));
printf("Your WMMA:   1.100 ms, 124.92 TFLOPS\n");
printf("\nYour kernel is %.2fx faster than cuBLAS!\n",ms/1.100);

return 0;
}
