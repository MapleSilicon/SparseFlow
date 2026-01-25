import ctypes, torch

wmma = ctypes.CDLL("./dense_wmma_grid.so")
wmma.launch_dense_wmma_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]

torch.set_grad_enabled(False)

M=N=K=256
A = torch.randn(M,K,device="cuda",dtype=torch.float16)
B = torch.randn(K,N,device="cuda",dtype=torch.float16)
B_col = B.t().contiguous()
C = torch.empty(M,N,device="cuda",dtype=torch.float32)

wmma.launch_dense_wmma_fp32(A.data_ptr(), B_col.data_ptr(), C.data_ptr(), M,N,K)
torch.cuda.synchronize()

ref = (A.float() @ B.float())
max_err = (C - ref).abs().max().item()
print("max abs error:", max_err)
