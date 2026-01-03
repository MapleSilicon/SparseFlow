from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sparseflow",
    version="2.0.0",
    author="Maple Silicon Inc.",
    author_email="engineering@maplesilicon.com",
    description="High-performance 2:4 sparse inference for NVIDIA GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MapleSilicon/SparseFlow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    keywords="gpu sparse tensor-cores cuda performance inference",
)
