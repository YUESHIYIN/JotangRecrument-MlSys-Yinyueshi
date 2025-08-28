#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

__global__ void kernel(const float *__restrict__ A_ptr,
                       const float *__restrict__ B_ptr,
                       float *__restrict__ result_ptr,
                       const int num_elements) {
  // TODO: Implement the Kernel Logic
  // Sum up all the elements in the input tensor
  // 计算当前线程的全局唯一ID (Index)
// 这是CUDA编程中最核心、最常见的模式
// blockIdx.x 是当前块在网格(Grid)中的ID (第几个块)
// blockDim.x 是每个块(Block)中有多少个线程
// threadIdx.x 是当前线程在块(Block)内的ID
const int index = blockIdx.x * blockDim.x + threadIdx.x;

// 边界检查：确保线程ID没有超出数组的实际范围
// 这一点非常重要，因为我们启动的线程总数可能略大于实际元素数量
// 以确保所有元素都被覆盖到
if (index < num_elements) {
  // 核心计算逻辑：执行向量加法
  // 每个线程只负责计算结果张量中的一个元素
  result_ptr[index] = A_ptr[index] + B_ptr[index];
}
}

static void launch_kernel(const void *A_ptr, const void *B_ptr, void *output_ptr,
                          const int num_elements) {
  // TODO: Implement the LaunchKernel Logic
  // 1. 定义每个线程块 (Block) 包含多少个线程
//    256 或 512 是一个常见的、对现代GPU性能较好的选择
const int threadsPerBlock = 256;

// 2. 计算需要多少个线程块 (Block) 才能覆盖所有元素
//    这是一个标准的向上取整整数除法公式
//    (num_elements + 255) / 256
//    确保即使 num_elements 不能被整除，也能启动足够的块来处理所有数据
const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

// 3. 启动 Kernel
//    使用 <<<...>>> 语法来调用 __global__ 函数
//    第一个参数是 Grid 的维度 (有多少个 Block)
//    第二个参数是 Block 的维度 (每个 Block 有多少个 Thread)
kernel<<<blocksPerGrid, threadsPerBlock>>>(A_ptr, B_ptr, output_ptr, num_elements);
  std::cout << "\n\nWarning: Need to implement the this!!\n\n" << std::endl;
  cudaDeviceSynchronize();
}

torch::Tensor test_kernel(const torch::Tensor &A, const torch::Tensor &B) {
  torch::Tensor result_tensor = torch::empty_like(A);
  const int element_count = A.numel();

  launch_kernel(A.data_ptr<float>(), B.data_ptr<float>(), result_tensor.data_ptr<float>(),
                element_count);

  return result_tensor;
}

PYBIND11_MODULE(CUDA_Test, m) {
  m.def("test_kernel", &test_kernel, "Test kernel");
}