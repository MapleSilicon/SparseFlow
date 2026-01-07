import torch, ctypes

lib = ctypes.CDLL('./gemm_ptx_single_warp.so')
lib.launch_gemm_ptx_single_warp.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

print("="*60)
print("Single Warp, 2×4 MMA tiles (32×32 output)")
print("="*60)

M = 512
A = torch.randn(M, M, device='cuda', dtype=torch.float16)
B = torch.randn(M, M, device='cuda', dtype=torch.float16)
C = torch.zeros(M, M, device='cuda', dtype=torch.float32)

lib.launch_gemm_ptx_single_warp(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
torch.cuda.synchronize()

C_ref = torch.matmul(A.float(), B.float())
err = torch.max(torch.abs(C.cpu() - C_ref.cpu())).item()

print(f"Max error: {err:.6f}")
if err < 0.01:
    print("✅ PASSED - Multi-tile logic works!")
    print("   Problem is in multi-warp geometry")
else:
    print("❌ FAILED - Multi-tile logic is broken")
    print(f"C range: [{C.min():.2f}, {C.max():.2f}]")
    print(f"Ref range: [{C_ref.min():.2f}, {C_ref.max():.2f}]")
