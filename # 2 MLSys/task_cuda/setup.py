from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='CUDA_Test',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='CUDA_Test',
            sources=['cuda.cu'],
            extra_compile_args={
                'cxx': ['-g', '-O0'],
                'nvcc': ['-g', "-G", '-O0']
            }
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
