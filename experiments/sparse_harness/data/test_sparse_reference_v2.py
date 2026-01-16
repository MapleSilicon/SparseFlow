import os, ctypes
import numpy as np
import torch

print("SPARSE_REF_V2: started", flush=True)
print("PWD:", os.getcwd(), flush=True)

M=N=K=128

Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E     = np.fromfile("dE.bin", np.uint16).reshape(M, K//4)
B     = np.fromfile("dB.bin", np.float16).reshape(K, N)      # row-major assumption
D_ref = np.fromfile("D_pruned_ref.bin", np.float32).reshape(M, N)

A_t = torch.from_numpy(Acomp).to("cuda")
E_t = torch.from_numpy(E).to("cuda")
B_t = torch.from_numpy(B).to("cuda")
C_t = torch.empty((M,N), device="cuda", dtype=torch.float32)

lib = ctypes.CDLL("./libsparse_ref_v2.so")
lib.launch_sparse_ref_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.launch_sparse_ref_v2(ctypes.c_void_p(A_t.data_ptr()),
                         ctypes.c_void_p(E_t.data_ptr()),
                         ctypes.c_void_p(B_t.data_ptr()),
                         ctypes.c_void_p(C_t.data_ptr()),
                         M, N, K)
torch.cuda.synchronize()

C = C_t.cpu().numpy()
diff = np.abs(C - D_ref)
print("max abs err =", float(diff.max()))
print("mean abs err =", float(diff.mean()))

if diff.max() < 1e-2:
    print("✅ SPARSE REF V2 MATCHES D_pruned_ref.bin")
else:
    print("❌ MISMATCH still. Next suspects: (a) B layout assumption, (b) codebook ordering vs stored Acomp order.")
