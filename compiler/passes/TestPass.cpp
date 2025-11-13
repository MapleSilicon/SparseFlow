#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestPass : public PassWrapper<TestPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPass)

  StringRef getArgument() const final { return "sparseflow-test"; }
  StringRef getDescription() const final { return "Test pass"; }
  
  void runOnOperation() override {
    // Do nothing
  }
};
} // namespace

void registerTestPass() {
  ::mlir::registerPass([]() { return std::make_unique<TestPass>(); });
}
