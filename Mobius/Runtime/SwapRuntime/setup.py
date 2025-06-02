# setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import torch

# 自动包含Torch头文件路径
torch_include = torch.utils.cpp_extension.include_paths()
torch_libs = torch.utils.cpp_extension.library_paths()

ext = Extension(
    name='parallel',
    sources=['wrap.cpp'],
    include_dirs=torch_include,
    library_dirs=torch_libs,
    language='c++',
    extra_compile_args=[
        '-std=c++17',
        '-O3',
        '-fvisibility=hidden',          # 关键参数
        '-fvisibility-inlines-hidden',  # 关键参数
        '-D_GLIBCXX_USE_CXX11_ABI=1'
    ],
    extra_link_args=[
        '-fvisibility=hidden',          # 链接阶段也需要
        '-fvisibility-inlines-hidden'
    ],
    libraries=['torch', 'torch_cpu','torch_cuda', 'c10', 'c10_cuda','torch_python']  # 链接Torch库
)

setup(
    name='parallel',
    version='0.1',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
