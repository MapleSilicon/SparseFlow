print("SPARSE_REF_DEBUG_BANNER: script started", flush=True)
import os; print("PWD:", os.getcwd(), flush=True)

import ctypes
import numpy as np
import torch

M=N=K=128

# Load your files (A sparse, meta, B dense)
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E     = np.fromfile("dE.bin", np.uint16)          # size M*(K/8)
B     = np.fromfile("dB.bin", np.float16).reshape(K, N)  # stored as raw; kernel assumes column-major interpretation
Dref  = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

# Move to GPU
A_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_t = torch.zeros((M,N), dtype=torch.float32, device="cuda")

# Load kernel
lib = ctypes.CDLL("./libsparse_ref.so")
lib.launch_sparse_gemm_reference.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# Run
lib.launch_sparse_gemm_reference(
    A_t.data_ptr(),
    E_t.data_ptr(),
    B_t.data_ptr(),
    C_t.data_ptr(),
    M, N, K
)
torch.cuda.synchronize()

C = C_t.cpu().numpy()

diff = np.abs(C - Dref)
print("max abs err =", diff.max())
print("mean abs err =", diff.mean())

# If this fails, it is almost certainly B layout (row/col-major mismatch).
if diff.max() < 1e-3:
    print("✅ SparseFlow reference kernel MATCHES D_pruned_ref.bin")
else:
    print("❌ MISMATCH. Likely B is row-major, not column-major.")
    # Quick alt-check: recompute diff assuming B is actually row-major by transposing view in numpy
    # (this doesn't rerun GPU, just hints)
    print("Hint: if B is row-major, kernel must use B[col + k*N] style indexing instead.")
