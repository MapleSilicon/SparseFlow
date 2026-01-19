import ctypes
import numpy as np
import torch

print("Testing minimal tensor core kernel...")

# Load the kernel (local path)
lib = ctypes.CDLL("./libsparse_tc_minimal.so")
lib.launch_sparse_tc_minimal.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

# Load validated data
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

# Move to GPU
A_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_t = torch.empty(M, N, device='cuda', dtype=torch.float32)

# Run kernel
lib.launch_sparse_tc_minimal(
    ctypes.c_void_p(A_t.data_ptr()),
    ctypes.c_void_p(E_t.data_ptr()),
    ctypes.c_void_p(B_t.data_ptr()),
    ctypes.c_void_p(C_t.data_ptr()),
    M, N, K
)
torch.cuda.synchronize()

# Compare
C_np = C_t.cpu().numpy()
max_err = np.abs(C_np - D_ref).max()
mean_err = np.abs(C_np - D_ref).mean()

print(f"Max error:  {max_err:.2e}")
print(f"Mean error: {mean_err:.2e}")

if max_err < 1e-4:
    print("✅ PASS - Minimal kernel works!")
else:
    print("❌ FAIL - Debug needed")
    print(f"First row comparison:")
    print(f"  Computed: {C_np[0,:5]}")
    print(f"  Expected: {D_ref[0,:5]}")
