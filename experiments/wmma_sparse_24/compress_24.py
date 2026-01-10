import torch
import numpy as np

def compress_24_weights(weights):
    """
    Compresses a dense FP16 matrix into the 2:4 structured format.
    Input: weights (M, K) - Must have K as a multiple of 4.
    Returns: 
        compressed_weights (M, K//2) - FP16 values
        metadata (M, K//16) - Int32 packed indices for Ampere
    """
    M, K = weights.shape
    assert K % 4 == 0, "K dimension must be a multiple of 4 for 2:4 sparsity"

    # 1. Reshape into groups of 4
    w_reshaped = weights.view(M, K // 4, 4)
    
    # 2. Find the mask: Pick the 2 largest absolute values in every 4 elements
    _, topk_indices = torch.topk(torch.abs(w_reshaped), 2, dim=-1, sorted=True)
    
    mask = torch.zeros_like(w_reshaped, dtype=torch.bool)
    mask.scatter_(2, topk_indices, True)

    # 3. Extract the non-zero values
    compressed_weights = w_reshaped[mask].view(M, K // 2)

    # 4. Generate Metadata (The "Magic" indices for PTX)
    metadata_indices = topk_indices.cpu().numpy()
    
    # Ampere 2:4 metadata packing for m16n8k32
    packed_meta = []
    for row in range(M):
        row_meta = []
        for group in range(0, K // 4, 8):
            # Pack 8 groups of 4-bits into one uint32
            val = np.uint32(0)
            for i in range(8):
                idx0 = metadata_indices[row, group + i, 1] # lower index
                idx1 = metadata_indices[row, group + i, 0] # higher index
                group_bits = (idx1 << 2) | idx0
                val |= (group_bits << (i * 4))
            row_meta.append(val)
        packed_meta.append(row_meta)

    metadata = torch.tensor(np.array(packed_meta), dtype=torch.int32, device=weights.device)
    
    return compressed_weights, metadata

if __name__ == "__main__":
    # Test on a small tile
    M, K = 16, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    
    W_sp, Meta = compress_24_weights(A)
    
    print(f"Original shape: {A.shape}")
    print(f"Compressed shape: {W_sp.shape}")
    print(f"Metadata shape: {Meta.shape}")
    print("\nCompression successful. Ready for PTX v3.0 kernel.")
