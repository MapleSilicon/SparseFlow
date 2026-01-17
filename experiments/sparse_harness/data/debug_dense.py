import torch
import numpy as np

print("Debugging dense kernel expectations...")

M = N = K = 128

A_pruned = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M, K)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

print(f"A_pruned: {A_pruned.shape}")
print(f"B:        {B.shape}")
print(f"D_ref:    {D_ref.shape}")

# Compute reference on CPU with PyTorch
A_t = torch.from_numpy(A_pruned).float()
B_t = torch.from_numpy(B).float()

# Try different B layouts
D_rowmajor = torch.matmul(A_t, B_t).numpy()
D_colmajor = torch.matmul(A_t, B_t.t()).numpy()

err_rowmajor = np.abs(D_rowmajor - D_ref).max()
err_colmajor = np.abs(D_colmajor - D_ref).max()

print(f"\nCPU PyTorch reference:")
print(f"  B row-major (A @ B):   max_err = {err_rowmajor:.2e}")
print(f"  B col-major (A @ B.T): max_err = {err_colmajor:.2e}")

if err_rowmajor < 1e-3:
    print("  ✅ B is stored ROW-MAJOR")
elif err_colmajor < 1e-3:
    print("  ✅ B is stored COL-MAJOR (transposed)")
else:
    print("  ❌ Neither matches!")
