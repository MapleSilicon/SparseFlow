NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86 -std=c++17 -Xcompiler -fPIC

all: libsparse_24.so libsparse_utils.so

libsparse_24.so: sparse_24_wmma.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $

libsparse_utils.so: sparse_utils.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $

clean:
	rm -f *.so

run: all
	python3 bench_sparse_vs_dense.py

.PHONY: all clean run
