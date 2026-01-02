import torch, ctypes

lib = ctypes.CDLL('./gemm_ptx_scaled.so')
lib.launch_gemm_ptx_scaled.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

print("="*60)
print("PER-WARP ERROR ANALYSIS")
print("="*60)

M, N, K = 128, 128, 128
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

lib.launch_gemm_ptx_scaled(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K)
torch.cuda.synchronize()

C_ref = torch.matmul(A.float(), B.float())

# Check each warp's region
print("\nWarp ID | Rows      | Cols      | Max Error")
print("-" * 60)

for warp_id in range(8):
    warp_m = warp_id >> 1  # 0-3
    warp_n = warp_id & 1   # 0-1
    
    r_start = warp_m * 32
    r_end = r_start + 32
    c_start = warp_n * 64
    c_end = c_start + 64
    
    region_err = torch.max(torch.abs(
        C[r_start:r_end, c_start:c_end] - C_ref[r_start:r_end, c_start:c_end]
    )).item()
    
    status = "✅" if region_err < 0.01 else "❌"
    print(f"Warp {warp_id}  | {r_start:3d}-{r_end:3d} | {c_start:3d}-{c_end:3d} | {region_err:10.6f} {status}")

print("="*60)
