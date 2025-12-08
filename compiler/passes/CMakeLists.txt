cmake_minimum_required(VERSION 3.20)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_library(SparseFlowPasses MODULE
  SparseFlowPassPlugin.cpp
  AnnotateNmPass.cpp
  ExportMetadataPass.cpp
  FlopCounterPass.cpp
  SPAExportPass.cpp
  sparseflow/SparsityPropagationPass.cpp
  sparseflow/SPADomain.cpp
  SparseMatmulRewritePass.cpp
)

target_link_libraries(SparseFlowPasses
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSupport
)

set_target_properties(SparseFlowPasses PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/passes"
)
