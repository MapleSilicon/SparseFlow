from setuptools import setup, find_packages

setup(
    name="sparseflow",
    version="0.7.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "demo": ["torch>=2.0.0", "onnx>=1.14.0"],
    },
    entry_points={
        "console_scripts": [
            "sparseflow-demo=sparseflow.cli:demo_main",
            "sparseflow-analyze=sparseflow.cli:analyze_main",
            "sparseflow-benchmark=sparseflow.cli:benchmark_main",
        ],
    },
    author="MapleSilicon",
    author_email="",
    url="https://github.com/MapleSilicon/SparseFlow",
    description="SparseFlow: Static sparsity analysis + runtime for 3–5× matmul speedup",
    long_description="Developer preview: hooks into SparseFlow SPA + C++ runtime.",
    long_description_content_type="text/plain",
    python_requires=">=3.9",
)
