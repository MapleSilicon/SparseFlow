#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Pass/Pass.h"

// Forward declarations for all SparseFlow passes.
void registerFlopCounter();
void registerAnnotateNmPass();
void registerExportMetadataPass();
void registerVerifySparsePatternPass();
void registerGenerateSparseMaskPass();
void registerSimpleGPULoweringPass();
void registerSparseComputationPass();

#if defined(_WIN32)
  #define MLIR_PASSP_PLUGIN_EXPORT __declspec(dllexport)
#else
  #define MLIR_PASSP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MLIR_PASSP_PLUGIN_EXPORT mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "SparseFlowPasses",
    "0.1",
    []() {
      registerFlopCounter();
      registerAnnotateNmPass();
      registerExportMetadataPass();
      registerVerifySparsePatternPass();
      registerGenerateSparseMaskPass();
      registerSimpleGPULoweringPass();
      registerSparseComputationPass();
    }
  };
}