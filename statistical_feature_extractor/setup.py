from setuptools import setup, Extension
import os
import platform

# 尝试导入pybind11，如果失败则提供备选方案
try:
    import pybind11
    include_dirs = [pybind11.get_include()]
except ImportError:
    # 如果无法导入pybind11模块，尝试从conda环境路径获取include目录
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        include_dirs = [os.path.join(conda_prefix, 'Lib', 'site-packages', 'pybind11', 'include')]
    else:
        # 如果都失败，则抛出错误
        raise ImportError("无法找到pybind11。请确保已安装: pip install pybind11")

# 根据操作系统设置不同的编译参数
if platform.system() == 'Windows':
    extra_compile_args = ['/O2', '/std:c++14']
else:
    extra_compile_args = ['-O3', '-fPIC']

statistical_feature_extractor_cc = Extension(
    name='statistical_feature_extractor_cc',
    sources=['statistical_feature_extractor_cc.cpp'],
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    language='c++'
)

setup(
    name='statistical_feature_extractor_cc',
    version='1.0',
    description='C++ implementation of flow feature extraction',
    ext_modules=[statistical_feature_extractor_cc],
    install_requires=['pybind11']
)