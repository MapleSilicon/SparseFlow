#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Pass/Pass.h"

// Forward declarations
void registerAnnotateNmPass();
void registerFlopCounterPass();
void registerSparsityPropagationPass();
void registerSPAExportPass();

namespace mlir {
void registerExportMetadataPass();
}

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
    "0.7",
    []() {
      registerAnnotateNmPass();
      mlir::registerExportMetadataPass();
      registerFlopCounterPass();
      registerSparsityPropagationPass();
      registerSPAExportPass();
    }
  };
}
