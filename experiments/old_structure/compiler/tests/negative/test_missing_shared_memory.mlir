// RUN: mlir-opt-19 --load-pass-plugin=%SparseFlowPasses \
// RUN:   --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
// RUN:   %s 2>&1 | FileCheck %s

// This kernel violates the shared memory invariant (no addrspace(3) alloca)
// The verifier MUST fail

gpu.module @test {
  gpu.func @sparseflow_tiled_kernel_v07(%arg0: memref<?x?xf32>) kernel attributes {mode = "tiled"} {
    // NO shared memory allocation (missing addrspace(3))
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    gpu.barrier
    
    scf.for %i = %c0 to %c32 step %c1 {
      scf.yield
    }
    
    gpu.barrier
    
    gpu.return
  }
}

// CHECK: tiled kernel invariant violated: missing memref.alloca addrspace(3)
