// RUN: mlir-opt-19 --load-pass-plugin=%SparseFlowPasses \
// RUN:   --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
// RUN:   %s 2>&1 | FileCheck %s

// This kernel violates the compute pattern (global load in step-1 loop after barrier)
// The verifier MUST fail

gpu.module @test {
  gpu.func @sparseflow_tiled_kernel_v07(%arg0: memref<?x?xf32>) kernel attributes {mode = "tiled"} {
    %alloca = memref.alloca() : memref<32x32xf32, 3>
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    gpu.barrier
    
    // VIOLATION: Loading from global memory (%arg0) in step-1 loop after barrier
    scf.for %i = %c0 to %c32 step %c1 {
      %val = memref.load %arg0[%i, %i] : memref<?x?xf32>  // Global load!
      scf.yield
    }
    
    gpu.barrier
    
    gpu.return
  }
}

// CHECK: tiled kernel invariant violated: global loads in compute loop
