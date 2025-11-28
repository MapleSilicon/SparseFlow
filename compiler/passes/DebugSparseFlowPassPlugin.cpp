#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

// Forward declarations for all SparseFlow passes.
void registerAnnotateNmPass();
void registerExportMetadataPass();
void registerFlopCounterPass();

#if defined(_WIN32)
  #define MLIR_PASSP_PLUGIN_EXPORT __declspec(dllexport)
#else
  #define MLIR_PASSP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MLIR_PASSP_PLUGIN_EXPORT mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  llvm::errs() << "DEBUG: mlirGetPassPluginInfo called for SparseFlowPasses!\n";
  
  return {
    MLIR_PLUGIN_API_VERSION,
    "SparseFlowPasses",
    "0.1",
    []() {
      llvm::errs() << "DEBUG: Registering SparseFlow passes...\n";
      registerAnnotateNmPass();
      registerExportMetadataPass();
      registerFlopCounterPass();
      llvm::errs() << "DEBUG: SparseFlow passes registered!\n";
    }
  };
}
