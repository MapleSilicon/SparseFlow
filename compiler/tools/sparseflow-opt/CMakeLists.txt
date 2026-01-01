add_executable(sparseflow-opt
  sparseflow-opt.cpp
)

target_link_libraries(sparseflow-opt
  PRIVATE
  MLIRIR
  MLIRSupport
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  SparseFlowPasses
  SparseFlowDialect
)

install(TARGETS sparseflow-opt
  RUNTIME DESTINATION bin
)
