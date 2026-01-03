import torch
import numpy as np

print("="*60)
print("NVIDIA 2:4 COMPRESSED FORMAT")
print("="*60)

# Create simple 2:4 sparse pattern
dense = torch.tensor([
    [1.0, 0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0, 4.0],
], dtype=torch.float16)

print("Dense matrix (2x4):")
print(dense)

def compress_nvidia_2_4(dense_row):
    values = []
    metadata = []
    
    for i in range(0, len(dense_row), 4):
        group = dense_row[i:i+4]
        
        nonzero_mask = (group != 0)
        nonzero_positions = torch.where(nonzero_mask)[0]
        
        for pos in nonzero_positions:
            values.append(group[pos].item())
        
        if len(nonzero_positions) == 2:
            pos0, pos1 = nonzero_positions[0].item(), nonzero_positions[1].item()
            pattern = (1 << pos0) | (1 << pos1)
            metadata.append(pattern)
    
    return values, metadata

print("\nCompressed format:")
for row_idx in range(dense.shape[0]):
    values, meta = compress_nvidia_2_4(dense[row_idx])
    print(f"Row {row_idx}:")
    print(f"  Values: {values}")
    print(f"  Metadata: {[bin(m) for m in meta]}")

print("\n" + "="*60)
print("For 16x16 tile:")
print(f"  Dense storage: 16x16 = 256 FP16 = 512 bytes")
print(f"  Sparse storage: 16x8 = 128 FP16 = 256 bytes")
print(f"  Metadata: 256/4 groups x 4 bits = 256 bits = 32 bytes")
print(f"  Total sparse: 256 + 32 = 288 bytes (vs 512 dense)")
print(f"  Bandwidth savings: {100 - 288/512*100:.1f}%")
