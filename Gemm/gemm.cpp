#include <iostream>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"

void test_cutlass(){
	int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
}

int main() {
    test_cutlass();
    
    return 0;
}
