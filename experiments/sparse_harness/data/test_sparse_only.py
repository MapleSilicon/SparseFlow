import torch
import ctypes
import numpy as np

print("Testing SPARSE kernel ONLY...")

sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")
sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

A_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_sparse = torch.empty(M, N, device='cuda', dtype=torch.float32)

# Run sparse kernel
sparse_lib.launch_sparse_ref_v3(
    ctypes.c_void_p(A_t.data_ptr()),
    ctypes.c_void_p(E_t.data_ptr()),
    ctypes.c_void_p(B_t.data_ptr()),
    ctypes.c_void_p(C_sparse.data_ptr()),
    M, N, K
)
torch.cuda.synchronize()

C_np = C_sparse.cpu().numpy()
max_err = np.abs(C_np - D_ref).max()
print(f"Sparse only - Max error: {max_err:.2e}")
