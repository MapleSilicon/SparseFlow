import ctypes, time, torch

wmma = ctypes.CDLL("./dense_wmma_grid.so")
cast = ctypes.CDLL("./cast_f32_to_f16.so")

wmma.launch_dense_wmma_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
cast.launch_f32_to_f16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

torch.set_grad_enabled(False)

M=N=K=4096
A = torch.randn(M,K,device="cuda",dtype=torch.float16)
B = torch.randn(K,N,device="cuda",dtype=torch.float16)
B_col = B.t().contiguous()
C_f32 = torch.empty(M,N,device="cuda",dtype=torch.float32)
C_f16 = torch.empty(M,N,device="cuda",dtype=torch.float16)

for _ in range(5):
    wmma.launch_dense_wmma_fp32(A.data_ptr(), B_col.data_ptr(), C_f32.data_ptr(), M,N,K)
    cast.launch_f32_to_f16(C_f32.data_ptr(), C_f16.data_ptr(), M*N)
torch.cuda.synchronize()

iters=20
t0=time.time()
for _ in range(iters):
    wmma.launch_dense_wmma_fp32(A.data_ptr(), B_col.data_ptr(), C_f32.data_ptr(), M,N,K)
    cast.launch_f32_to_f16(C_f32.data_ptr(), C_f16.data_ptr(), M*N)
torch.cuda.synchronize()
dt = (time.time() - t0) / iters

tflops = (2 * M * N * K) / (dt * 1e12)
print(f"End-to-end latency (wmma fp32 + cast fp16): {dt*1000:.3f} ms")
print(f"End-to-end TFLOPS (dense-eq):              {tflops:.2f}")
