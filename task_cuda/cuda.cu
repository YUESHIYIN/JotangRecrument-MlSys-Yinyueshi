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
}

static void launch_kernel(const void *A_ptr, const void *B_ptr, void *output_ptr,
                          const int num_elements) {
  // TODO: Implement the LaunchKernel Logic
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