#pragma once
#include <cusparseLt.h>

namespace sparseflow {

struct CuSparseLtContext {
  cusparseLtHandle_t handle;
  bool initialized = false;
};

CuSparseLtContext& get_cusparselt_context();

}
