//===- sparseflow-runner.cpp ---------------------------------------------===//

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  MLIRContext context;
  context.loadDialect<LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);

  auto bufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = bufferOrErr.getError()) {
    llvm::errs() << "Error: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sm, &context);

  if (!module) {
    llvm::errs() << "Parse failed\n";
    return 1;
  }

  llvm::outs() << "✓ Parsed\n";

  // Minimal ExecutionEngine options - no transformer, no shared libs
  auto maybeEngine = ExecutionEngine::create(*module);
  
  if (!maybeEngine) {
    llvm::errs() << "Engine failed: " 
                 << llvm::toString(maybeEngine.takeError()) << "\n";
    return 1;
  }

  llvm::outs() << "✓ Engine created\n";
  
  auto engine = std::move(*maybeEngine);
  auto lookup = engine->lookup("test");
  
  if (!lookup) {
    llvm::errs() << "Lookup failed\n";
    return 1;
  }

  llvm::outs() << "✓ Found test at: " << (void*)*lookup << "\n";
  
  void (*fn)() = reinterpret_cast<void(*)()>(*lookup);
  fn();
  
  llvm::outs() << "✅ Done!\n";
  return 0;
}
