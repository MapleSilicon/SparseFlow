# WEEK 3: MLIR Frontend + PyTorch Integration Prep

## Goals
1. ✅ **SPA pass cleanup** - Rename, document, stabilize
2. ✅ **Add MLIR dialect stubs** - Define ops without implementation
3. ✅ **Add sparseflow-opt driver** - Standalone MLIR tool
4. ✅ **Define hardware contract JSON spec** - Clear interface
5. ✅ **Start PyTorch → MLIR lowering skeleton** - Foundation

## Day-by-Day Plan

### Day 1: SPA Pass Cleanup & Renaming
- Rename passes for consistency (spa → sparseflow)
- Update documentation and comments
- Create pass registration helper

### Day 2: MLIR Dialect Stubs
- Define SparseFlow dialect structure
- Add placeholder operations
- Register with MLIR context

### Day 3: sparseflow-opt Driver
- Create standalone MLIR optimizer
- Register all SparseFlow passes
- Add basic command-line interface

### Day 4: Hardware Contract JSON Spec
- Define formal JSON schema
- Document all fields and types
- Create validation script

### Day 5: PyTorch → MLIR Skeleton
- Set up torch-mlir environment
- Create minimal conversion pipeline
- Test with simple PyTorch model

## Success Criteria
✅ All passes renamed and documented  
✅ Dialect registered and testable  
✅ `sparseflow-opt` tool built  
✅ JSON spec documented and validated  
✅ PyTorch conversion skeleton working

## Ready to begin!
