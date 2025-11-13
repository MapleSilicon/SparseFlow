#include "mlir/Pass/PassRegistry.h"
#include "sparseflow/Passes.h"

// Define the namespace to avoid conflict with other MLIR components
namespace sparseflow {

void registerSparseFlowPasses() {
  ::mlir::registerPass([]() {
    return sparseflow::createAnnotateNmPass();
  });
  
  ::mlir::registerPass([]() {
    return sparseflow::createNmConsumerPass();
  });

  // CRITICAL FIX: The missing registration for the ExportMetadataPass
  ::mlir::registerPass([]() {
    return sparseflow::createExportMetadataPass();
  });
}

} // namespace sparseflow

// Utility function to call the registration
void registerAllSparseFlowPasses() {
  sparseflow::registerSparseFlowPasses();
}
