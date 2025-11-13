transform.sequence failures(propagate) {
  // Match all functions in the module
  %funcs = transform.module.match ops{func.func}

  // Match all linalg.matmul
  %matmuls = transform.structured.match ops{linalg.matmul} in %funcs

  // --- Block-level tiling ---
  %tile_block, %loops_block =
    transform.structured.tile_using_for %matmuls
      [32, 32, 32] (pad = true)

  // Fuse producers (optional)
  transform.structured.fuse %tile_block

  // --- Warp-level tiling ---
  %tile_warp, %loops_warp =
    transform.structured.tile_using_for %tile_block
      [16, 16, 0]

  // Vectorize inner loops
  transform.structured.vectorize %tile_warp

  // Lower vector ops to GPU-compatible form
  transform.structured.lower_vector_to_gpu %tile_warp

  // Map loops to GPU hardware
  transform.gpu.map_forall_to_blocks %loops_block
  transform.gpu.map_forall_to_threads %loops_warp
}
