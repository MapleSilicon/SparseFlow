#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

// Generate 2:4 sparse pattern: keep top 2 out of every 4 elements
void generate_24_sparse_pattern(
    const half* dense, int rows, int cols,
    half* compressed, uint32_t* metadata
) {
    int compressed_cols = cols / 2;
    int meta_cols = cols / 8;
    
    for (int r = 0; r < rows; r++) {
        for (int c_block = 0; c_block < cols; c_block += 16) {
            // Process 16 columns at a time (2x 8-element groups)
            for (int group = 0; group < 2; group++) {
                int base_col = c_block + group * 8;
                
                // Each group of 8 elements = two 2:4 blocks
                for (int block = 0; block < 2; block++) {
                    int block_base = base_col + block * 4;
                    
                    // Find top 2 magnitudes in this 4-element block
                    struct {float val; int idx;} vals[4];
                    for (int i = 0; i < 4; i++) {
                        int col = block_base + i;
                        if (col < cols) {
                            vals[i].val = fabsf(__half2float(dense[r * cols + col]));
                            vals[i].idx = i;
                        } else {
                            vals[i].val = 0.0f;
                            vals[i].idx = i;
                        }
                    }
                    
                    // Sort by magnitude (descending)
                    std::sort(vals, vals + 4, [](auto a, auto b) { 
                        return a.val > b.val; 
                    });
                    
                    // Keep top 2 indices
                    int idx0 = vals[0].idx;
                    int idx1 = vals[1].idx;
                    if (idx0 > idx1) std::swap(idx0, idx1);
                    
                    // Metadata encoding (2 bits per element)
                    int meta_bits = (idx0 << 1) | (idx1 - idx0 - 1);
                    
                    // Write compressed values
                    int out_col = (block_base - c_block) / 2 + (r * compressed_cols);
                    if (block_base + idx0 < cols)
                        compressed[out_col] = dense[r * cols + block_base + idx0];
                    if (block_base + idx1 < cols)
                        compressed[out_col + 1] = dense[r * cols + block_base + idx1];
                    
                    // Pack metadata (4 blocks per uint32_t)
                    int meta_offset = (c_block / 16) * meta_cols + group * 2 + block / 2;
                    int bit_pos = (block % 2) * 16 + (r % 16) * 2;
                    metadata[meta_offset] |= (meta_bits << bit_pos);
                }
            }
        }
    }
}

extern "C" void cpu_generate_24_sparse(
    const half* dense, int rows, int cols,
    half* compressed, uint32_t* metadata
) {
    // Zero out metadata
    int meta_size = (rows / 16) * (cols / 8);
    memset(metadata, 0, meta_size * sizeof(uint32_t));
    
    generate_24_sparse_pattern(dense, rows, cols, compressed, metadata);
}
