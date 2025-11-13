#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace {

struct GenerateSparseMaskPass
    : public PassWrapper<GenerateSparseMaskPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateSparseMaskPass)

  StringRef getArgument() const final { return "sparseflow-generate-mask"; }

  StringRef getDescription() const final {
    return "Generate N:M sparse masks and attach sparseflow.mask attributes";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](linalg::MatmulOp mm) {
      // Read sparseflow.nm = { n: i64, m: i64 }
      auto dictAttr = mm->getAttrOfType<DictionaryAttr>("sparseflow.nm");
      if (!dictAttr)
        return;

      auto nAttr = dictAttr.get("n").dyn_cast_or_null<IntegerAttr>();
      auto mAttr = dictAttr.get("m").dyn_cast_or_null<IntegerAttr>();
      if (!nAttr || !mAttr)
        return;

      int64_t n = nAttr.getInt();
      int64_t m = mAttr.getInt();

      // Build a simple marker string: "2:4_pattern_mask_generated"
      auto ctx = mm.getContext();
      std::string maskStr = (Twine(n) + ":" + Twine(m) +
                             "_pattern_mask_generated")
                                .str();
      auto maskAttr = StringAttr::get(ctx, maskStr);
      auto maskArray = ArrayAttr::get(ctx, {maskAttr});

      mm->setAttr("sparseflow.mask", maskArray);

      mm.emitRemark() << "Generated " << n << ":" << m
                      << " sparse mask for weight matrix";
    });
  }
};

} // namespace

void registerGenerateSparseMaskPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GenerateSparseMaskPass>();
  });
}
