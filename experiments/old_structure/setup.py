"""
SparseFlow - High-performance 2:4 sparse inference for NVIDIA GPUs

Installation:
    pip install -e .  # Development mode
    pip install .     # Regular install
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read version
version = "3.0.0-alpha"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# C++ extension
ext_modules = [
    CUDAExtension(
        name="sparseflow.sparseflow_cpp",
        sources=[
            "python/sparseflow_cpp/bindings.cpp",
        ],
        include_dirs=[
            "include",
            "src",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_80,code=sm_80",  # Ampere
                "-gencode", "arch=compute_86,code=sm_86",  # RTX 30
                "-gencode", "arch=compute_89,code=sm_89",  # RTX 40
                "-gencode", "arch=compute_90,code=sm_90",  # Hopper
            ],
        },
    ),
]

setup(
    name="sparseflow",
    version=version,
    author="Maple Silicon Inc.",
    author_email="engineering@maplesilicon.com",
    description="High-performance 2:4 sparse inference for NVIDIA GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MapleSilicon/SparseFlow",
    project_urls={
        "Bug Tracker": "https://github.com/MapleSilicon/SparseFlow/issues",
        "Documentation": "https://github.com/MapleSilicon/SparseFlow",
        "Source Code": "https://github.com/MapleSilicon/SparseFlow",
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
)
