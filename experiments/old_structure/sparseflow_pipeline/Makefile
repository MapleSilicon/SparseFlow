sm ?= 86
NVCC ?= nvcc
CXXFLAGS := -O3 -lineinfo
LDFLAGS  := --shared -Xcompiler -fPIC

all: libgemm_pipeline.so test_cuda_only

libgemm_pipeline.so: gemm_pipeline_fixed.cu
	$(NVCC) -arch=sm_$(sm) $(CXXFLAGS) $(LDFLAGS) $< -o $@

test_cuda_only: test_cuda_only.cu libgemm_pipeline.so
	$(NVCC) -arch=sm_$(sm) $(CXXFLAGS) test_cuda_only.cu -L. -lgemm_pipeline -Wl,-rpath,'$$ORIGIN' -o $@

clean:
	rm -f libgemm_pipeline.so test_cuda_only
