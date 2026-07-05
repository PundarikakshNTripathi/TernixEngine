from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'ternix_core',
        ['pybind11_wrapper.cpp', '../../src/mul_mat_ternary.cpp'],
        include_dirs=[pybind11.get_include(), '../../include'],
        extra_compile_args=['-O3', '-mavx2'],
        language='c++'
    ),
]

setup(
    name='ternix_core',
    version='0.1.0',
    description='TernixEngine 1.58-bit Python bindings',
    ext_modules=ext_modules,
)
