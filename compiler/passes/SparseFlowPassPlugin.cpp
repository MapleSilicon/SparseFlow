//===- SparseFlowPassPlugin.cpp ------------------------------------------===//
//
// Minimal MLIR pass plugin entrypoint for SparseFlow.
//
// All individual passes (SPA, SPAExport, FlopCounter, AnnotateNm,
// SparseMatmulRewrite, etc.) are registered via static PassRegistration
// objects in their own .cpp files.
//
// This file ONLY exposes mlirGetPassPluginInfo so that mlir-opt can
// dlopen() the plugin with --load-pass-plugin and see those passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "SparseFlowPasses",
      LLVM_VERSION_STRING,
      []() {
        // NOTE:
        // We do NOT need to call any registerXYZPass() functions here.
        // Each pass file uses static PassRegistration<...>, so loading
        // this .so is enough to register all "sparseflow-*" passes.
      }};
}
