# 使用 GPU 计算  A[0:127] = b[0:127] + d[0:127] 

本项目是一个 CUDA 编程练习，目标是实现向量加法运算。使用 CUDA 在 GPU 上计算两个长度为 128 的向量的元素相加，即：
```python
 A[0:127] = B[0:127] + C[0:127] 
```

## 环境要求 (请参照招新题中的环境配置一栏)

- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit (推荐版本 11.x 或更高)
- Python 3.8+
- PyTorch (支持 CUDA)
- C++ 编译器（clang++/g++）

## 构建

```bash
pip install uv 
uv venv    //此时在当前路径下会看到.venv 文件夹
source ./.venv/bin/activate
which python   // 显示路径为 .venv/bin/python
uv pip install -e .  //构建项目
```
构建成功后
```bash 
python test_cuda.py
--------------------
Successfully imported CUDA_Test module
PyTorch version: 2.7.1+cu126
CUDA version: 12.6
Using CUDA device: ....
```

## 任务要求

需要完成 `cuda.cu` 文件中的两个 TODO 部分：

### 1. 实现 CUDA Kernel
在 `kernel` 函数中实现向量加法逻辑：
```cpp
__global__ void kernel(const float *__restrict__ A_ptr,
                       const float *__restrict__ B_ptr,
                       float *__restrict__ result_ptr,
                       const int num_elements)
```

### 2. 实现 Kernel 启动函数
在 `launch_kernel` 函数中实现 CUDA kernel 的启动逻辑：
```cpp
static void launch_kernel(const void *A_ptr,
						  const void *B_ptr, 
						  void *output_ptr,
                          const int num_elements)
```

## 运行测试

```bash
python test_cuda.py
```

测试程序将：
1. 创建两个随机的 128 元素向量
2. 调用您实现的 CUDA kernel 进行计算
3. 与 PyTorch 的 CPU 计算结果进行对比
4. 输出测试结果

## 文件说明

- `cuda.cu` - CUDA 源代码文件，包含需要实现的 kernel 和启动函数
- `test_cuda.py` - 测试脚本，验证 CUDA 实现的正确性
- `setup.py` - 编译配置文件，用于构建 PyTorch C++/CUDA 扩展
- `pyproject.toml` - 项目配置文件，定义依赖和构建要求
- `README.md` - 项目说明文档

## 编译选项说明

- `-g -O0` - 包含调试信息，禁用优化（便于调试）
- `-G` - NVCC 调试模式
- 使用 `torch.utils.cpp_extension` 进行 PyTorch 集成

## 预期输出

成功实现后，测试程序应输出：
```
Successfully imported CUDA_Test module
Using CUDA device: [您的GPU名称]
Output result: [计算结果列表]
Expected result: [期望结果列表]
Test passed! Result is correct
Test completed!
```

## 这一个章节你需要学会

1. **CUDA 基本编程模型**
   - **CUDA 线程层次结构**: Grid -> Block -> Thread
   - **基本 kernel 语法**
   - **线程索引计算**: `threadIdx.x`, `blockIdx.x`, `blockDim.x`
   - **Kernel 启动配置**: `<<<blocks, threads>>>`

2. **CUDA 内存管理**
   - **设备内存分配**
   - **主机-设备数据传输**
   - **内存访问模式**
   - **共享内存使用**

3. **PyTorch C++ 扩展开发**
   - **torch::Tensor 操作**
   - **PYBIND11 绑定**:

4. **VSCode 混合调试技巧**
   - **Python 调试配置** (`.vscode/launch.json`):
   - **C++/CUDA 调试配置**