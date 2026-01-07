// RUN: mlir-opt-19 --load-pass-plugin=%SparseFlowPasses \
// RUN:   --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
// RUN:   %s 2>&1 | FileCheck %s

// This kernel violates the barrier invariant (has only 1 barrier, needs ≥2)
// The verifier MUST fail with appropriate error message

gpu.module @test {
  gpu.func @sparseflow_tiled_kernel_v07(%arg0: memref<?x?xf32>) kernel attributes {mode = "tiled"} {
    %alloca = memref.alloca() : memref<32x32xf32, 3>
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    // Only ONE barrier (should be ≥2)
    gpu.barrier
    
    // Step-1 compute loop exists
    scf.for %i = %c0 to %c32 step %c1 {
      %val = memref.load %alloca[%i, %i] : memref<32x32xf32, 3>
      scf.yield
    }
    
    gpu.return
  }
}

// CHECK: tiled kernel invariant violated: expected >= 2 gpu.barrier
