# 构建好第三方库
```bash
cmake -B build -G Ninja -DCUTLASS_NVCC_ARCHS=${YOUR_GPU_ARCH} -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON 

# eg. For Ampere GPU(A100), the GPU ARCHS is 80
# 更多请查阅：https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html#building-for-multiple-architectures

```