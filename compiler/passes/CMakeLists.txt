cmake_minimum_required(VERSION 3.20)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(SparseFlowPasses MODULE
  SparseFlowPassPlugin.cpp
  AnnotateNmPass.cpp
  ExportMetadataPass.cpp
  FlopCounterPass.cpp
  spa/SPADomain.cpp
  spa/SparsityPropagationPass.cpp
)

target_compile_features(SparseFlowPasses PRIVATE cxx_std_17)

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
  OUTPUT_NAME "SparseFlowPasses"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/passes"
)
