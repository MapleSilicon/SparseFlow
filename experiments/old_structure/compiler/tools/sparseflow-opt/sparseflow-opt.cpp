// sparseflow-opt: MLIR optimizer for SparseFlow passes

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include "SparseFlow/SparseFlowDialect.h"
#include "SparseFlowPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  
  // Register SparseFlow passes
  mlir::sparseflow::registerSparseFlowPasses();
  
  mlir::DialectRegistry registry;
  
  // Register all core MLIR dialects
  mlir::registerAllDialects(registry);
  
  // Register SparseFlow dialect
  registry.insert<mlir::sparseflow::SparseFlowDialect>();
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SparseFlow Optimizer", registry));
}
