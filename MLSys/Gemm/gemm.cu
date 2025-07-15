#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// 基础的CUDA GEMM kernel - 每个线程计算结果矩阵的一个元素
__global__ void gemm_baseline_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 初始化矩阵
void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    }
}

// 验证结果正确性 - 简单的CPU实现作为参考
void cpu_gemm_reference(float *A, float *B, float *C_ref, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }
}

// 验证两个矩阵是否相等
bool verify_result(float *C1, float *C2, int size, float epsilon = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(C1[i] - C2[i]) > epsilon) {
            std::cout << "验证失败：位置 " << i << ", GPU=" << C1[i] 
                     << ", CPU=" << C2[i] << ", 差值=" << std::abs(C1[i] - C2[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void test_cutlass() {
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

void GEMM_Baseline(float *A, float *B, float *C, int M, int N, int K) {
    // 设备指针
    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // 分配GPU内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    // 设置线程块和网格大小
    dim3 blockSize(16, 16);  // 16x16 线程块
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    // 启动kernel
    gemm_baseline_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 将结果复制回主机
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // 计算性能指标
    double flops = 2.0 * M * N * K; // 每次乘加算2次浮点运算
    double gflops = flops / (milliseconds * 1e6); // 转换为GFLOPS
    
    std::cout << "Baseline GEMM [" << M << "x" << K << "] x [" << K << "x" << N 
              << "] = [" << M << "x" << N << "]" << std::endl;
    std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    
    // 清理GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_cutlass();
    std::cout << "\n=== GEMM Baseline 性能测试 ===\n" << std::endl;
    
    // 设置固定随机种子，确保结果可重现
    srand(42);
    
    // 测试不同的矩阵大小
    int test_sizes[] = {256, 512, 1024};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        int M = test_sizes[t];
        int N = test_sizes[t];
        int K = test_sizes[t];
        
        std::cout << "\n--- 测试矩阵大小: " << M << "x" << K << " x " 
                  << K << "x" << N << " ---" << std::endl;
        
        // 分配主机内存
        float *A = new float[M * K];
        float *B = new float[K * N];
        float *C = new float[M * N];
        float *C_ref = new float[M * N];
        
        // 初始化输入矩阵
        init_matrix(A, M * K);
        init_matrix(B, K * N);
        
        // CPU参考实现（用于验证正确性）
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm_reference(A, B, C_ref, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        
        std::cout << "CPU参考时间: " << cpu_duration.count() << " ms" << std::endl;
        
        // CUDA Baseline实现
        GEMM_Baseline(A, B, C, M, N, K);
        
        // 验证结果正确性
        bool correct = verify_result(C, C_ref, M * N);
        std::cout << "结果验证: " << (correct ? "通过" : "失败") << std::endl;
        
        if (!correct) {
            std::cout << "警告：结果不正确，请检查实现！" << std::endl;
        }
        
        // 释放内存
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    std::cout << "提示：这是baseline实现，后续优化版本应该与此baseline进行对比" << std::endl;
    std::cout << "加速比 = baseline时间 / 优化版本时间" << std::endl;
    
    return 0;
}
