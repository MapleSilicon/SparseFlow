#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Pass/Pass.h"

// These passes already have their own registration functions
void registerSparsityPropagationPass();
void registerSPAExportPass();

// Stubs for passes that don't exist or don't have registration
void registerAnnotateNmPass() {}
void registerFlopCounterPass() {}

// Real passes we care about
void registerSparseMatmulRewritePass();
void registerSparseFlowGpuRewritePass();

namespace mlir {
void registerExportMetadataPass() {}
}

extern "C" mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "SparseFlowPasses",
    "0.7",
    []() {
      registerAnnotateNmPass();
      mlir::registerExportMetadataPass();
      registerFlopCounterPass();
      registerSparsityPropagationPass();  // Real - exists in SparsityPropagationPass.cpp
      registerSPAExportPass();            // Real - exists in SPAExportPass.cpp
      registerSparseMatmulRewritePass();  // Real - our CPU pass
      registerSparseFlowGpuRewritePass(); // Real - our NEW GPU pass!
    }
  };
}
