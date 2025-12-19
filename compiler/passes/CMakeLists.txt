# Get all pass source files
file(GLOB PASS_SOURCES "*.cpp")

# Get sparseflow subdirectory sources if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sparseflow")
    file(GLOB SPARSEFLOW_SOURCES "sparseflow/*.cpp")
    list(APPEND PASS_SOURCES ${SPARSEFLOW_SOURCES})
endif()

# Create the plugin library
add_llvm_library(SparseFlowPasses MODULE
    ${PASS_SOURCES}
    PLUGIN_TOOL mlir-opt
)

target_include_directories(SparseFlowPasses PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sparseflow
)

target_link_libraries(SparseFlowPasses PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
)

# Ensure C++17
set_target_properties(SparseFlowPasses PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
