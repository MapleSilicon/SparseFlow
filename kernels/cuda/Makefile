NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++17 --compiler-options -fPIC
CUDA_PATH = /usr/local/cuda

all: libsparseflow_wmma.so test_wmma

libsparseflow_wmma.so: wmma_dense_saturating.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $

test_wmma: test_wmma_standalone.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $

clean:
	rm -f libsparseflow_wmma.so test_wmma

.PHONY: all clean
