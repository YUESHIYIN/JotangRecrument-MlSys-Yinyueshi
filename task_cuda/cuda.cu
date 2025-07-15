#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void kernel(const float *__restrict__ data_in,
                       float *__restrict__ data_out, const int total_elements) {
  // TODO: Implement the Kernel Logic
  // Sum up all the elements in the input tensor
}

static void launch_kernel(const void *input_ptr, void *output_ptr,
                          const int num_elements) {
  // TODO: Implement the LaunchKernel Logic
  std::cout << "\n\nWarning: Need to implement the this!!\n\n" << std::endl;
  cudaDeviceSynchronize();
}

torch::Tensor test_kernel(const torch::Tensor &input_tensor) {
  torch::Tensor result_tensor = torch::empty_like(input_tensor);
  const int element_count = input_tensor.numel();

  launch_kernel(input_tensor.data_ptr<float>(), result_tensor.data_ptr<float>(),
                element_count);

  return result_tensor;
}

PYBIND11_MODULE(CUDA_Test, m) {
  m.def("test_kernel", &test_kernel, "Test kernel");
}