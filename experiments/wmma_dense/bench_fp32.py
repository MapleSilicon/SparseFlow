import ctypes, time, torch

lib = ctypes.CDLL("./dense_wmma_grid.so")
lib.launch_dense_wmma_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
torch.set_grad_enabled(False)

M=N=K=4096
A = torch.randn(M,K,device="cuda",dtype=torch.float16)
B = torch.randn(K,N,device="cuda",dtype=torch.float16)
B_col = B.t().contiguous()
C = torch.empty(M,N,device="cuda",dtype=torch.float32)

for _ in range(5):
    lib.launch_dense_wmma_fp32(A.data_ptr(), B_col.data_ptr(), C.data_ptr(), M,N,K)
torch.cuda.synchronize()

iters=20
t0=time.time()
for _ in range(iters):
    lib.launch_dense_wmma_fp32(A.data_ptr(), B_col.data_ptr(), C.data_ptr(), M,N,K)
torch.cuda.synchronize()
dt = (time.time() - t0) / iters

tflops = (2 * M * N * K) / (dt * 1e12)
print(f"Latency: {dt*1000:.3f} ms")
print(f"TFLOPS:  {tflops:.2f}")
