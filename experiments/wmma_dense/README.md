# WMMA Dense Baseline (RTX 3090 / SM86)

- FP16 inputs
- FP32 accumulation
- FP32 output (WMMA store)
- Optional second kernel: FP32 -> FP16 cast

## Build
```bash
nvcc -O3 -Xcompiler -fPIC -shared -arch=sm_86 dense_wmma_grid.cu -o dense_wmma_grid.so
nvcc -O3 -Xcompiler -fPIC -shared -arch=sm_86 cast_f32_to_f16.cu -o cast_f32_to_f16.so
```

## Run
```bash
python3 bench_fp32.py
python3 bench_fp16_out.py
```

**Note:** end-to-end fp16-out is bandwidth-limited due to extra global read/write.
