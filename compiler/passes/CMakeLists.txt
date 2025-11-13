add_library(SparseFlowPasses MODULE
  SparseFlowPassPlugin.cpp
  AnnotateNmPass.cpp
  ExportMetadataPass.cpp
  VerifySparsePatternPass.cpp
  SimpleGPULoweringPass.cpp
  GenerateSparseMaskPass.cpp
)

target_include_directories(SparseFlowPasses PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
add_definitions(${LLVM_DEFINITIONS})

target_link_libraries(SparseFlowPasses PRIVATE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRSCFDialect
)

set_target_properties(SparseFlowPasses PROPERTIES
  OUTPUT_NAME "SparseFlowPasses"
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/passes"
  BUILD_RPATH "/usr/lib/llvm-19/lib"
  INSTALL_RPATH "/usr/lib/llvm-19/lib"
)
