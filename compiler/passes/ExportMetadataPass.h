#ifndef EXPORT_METADATA_PASS_H
#define EXPORT_METADATA_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createExportMetadataPass();
void registerExportMetadataPass();
} // namespace mlir

#endif
