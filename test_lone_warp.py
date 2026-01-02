import torch, ctypes

lib = ctypes.CDLL('./gemm_ptx_lone.so')
lib.launch_gemm_ptx_lone_warp.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

M, N, K = 128, 128, 128
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

lib.launch_gemm_ptx_lone_warp(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K)
torch.cuda.synchronize()

C_ref = torch.matmul(A.float(), B.float())
err = torch.max(torch.abs(C - C_ref)).item()

print(f"Lone Warp 128×128 Test: Max Error = {err:.6f}")

if err < 0.01:
    print("✅ PASSED - Addressing is correct!")
    print("   Problem: Cooperative loading with 8 warps")
else:
    print("❌ FAILED - Store logic is broken")
