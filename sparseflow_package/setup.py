from setuptools import setup, find_packages
import os

setup(
    name="sparseflow",
    version="0.7.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        'demo': ['torch>=2.0.0', 'onnx>=1.14.0'],
    },
    entry_points={
        'console_scripts': [
            'sparseflow-demo=sparseflow.cli:demo_main',
            'sparseflow-analyze=sparseflow.cli:analyze_main',
            'sparseflow-benchmark=sparseflow.cli:benchmark_main',
        ],
    },
    author="MapleSilicon",
    description="SparseFlow: Static Sparsity Analysis + Runtime for 4× MLIR Matmul Speedup",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MapleSilicon/SparseFlow",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    python_requires=">=3.8",
)
