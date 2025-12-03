#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Pass/Pass.h"

// Forward declarations - NO namespace (they're global)
void registerAnnotateNmPass();
void registerFlopCounterPass();
void registerSPAExportPass();           // JSON export
void registerSparsityPropagationPass();  // ⭐ SPA v0.6

// This one IS in mlir namespace
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
    "0.1",
    []() {
      registerSPAExportPass();           // JSON export
      registerAnnotateNmPass();
      mlir::registerExportMetadataPass();
      registerFlopCounterPass();
      registerSparsityPropagationPass();  // ⭐ SPA v0.6
    }
  };
}
