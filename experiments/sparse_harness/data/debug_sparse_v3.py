import torch
import ctypes
import numpy as np

print("Debugging sparse_ref_v3 kernel...")

sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")
sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

# Load validated data
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

print(f"Loaded data:")
print(f"  Acomp: {Acomp.shape} {Acomp.dtype}")
print(f"  E:     {E.shape} {E.dtype}")
print(f"  B:     {B.shape} {B.dtype}")
print(f"  D_ref: {D_ref.shape} {D_ref.dtype}")

# Move to GPU
A_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_t = torch.empty(M, N, device='cuda', dtype=torch.float32)

print(f"\nGPU tensors:")
print(f"  A_t:   {A_t.shape} {A_t.dtype} contiguous={A_t.is_contiguous()}")
print(f"  E_t:   {E_t.shape} {E_t.dtype} contiguous={E_t.is_contiguous()}")
print(f"  B_t:   {B_t.shape} {B_t.dtype} contiguous={B_t.is_contiguous()}")
print(f"  C_t:   {C_t.shape} {C_t.dtype} contiguous={C_t.is_contiguous()}")

# Call kernel
sparse_lib.launch_sparse_ref_v3(
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

print(f"\nResults:")
print(f"  Max error:  {max_err:.2e}")
print(f"  Mean error: {mean_err:.2e}")

if max_err < 1e-4:
    print("  ✅ PASS")
else:
    print("  ❌ FAIL")
    print(f"\nFirst few values:")
    print(f"  C_np[0,:5]:  {C_np[0,:5]}")
    print(f"  D_ref[0,:5]: {D_ref[0,:5]}")
