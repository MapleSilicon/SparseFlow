import torch
import ctypes
import numpy as np

print("Testing DENSE kernel ONLY...")

dense_lib = ctypes.CDLL("../../wmma_dense/dense_wmma_grid.so")
dense_lib.launch_dense_wmma_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

A_pruned = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M, K)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

A_t = torch.from_numpy(A_pruned).cuda()
B_t = torch.from_numpy(B).cuda()
B_col = B_t.t().contiguous()
C_dense = torch.empty(M, N, device='cuda', dtype=torch.float32)

# Run dense kernel
dense_lib.launch_dense_wmma_fp32(
    A_t.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
)
torch.cuda.synchronize()

C_np = C_dense.cpu().numpy()
max_err = np.abs(C_np - D_ref).max()
print(f"Dense only - Max error: {max_err:.2e}")
