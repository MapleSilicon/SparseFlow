#include "cusparselt_context.h"
#include <mutex>
#include <iostream>

namespace sparseflow {

CuSparseLtContext& get_cusparselt_context() {
  static CuSparseLtContext ctx;
  static std::mutex m;

  std::lock_guard<std::mutex> lock(m);
  if (!ctx.initialized) {
    cusparseStatus_t status = cusparseLtInit(&ctx.handle);
    if (status == CUSPARSE_STATUS_SUCCESS) {
      ctx.initialized = true;
    }
  }
  return ctx;
}

}
