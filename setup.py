from setuptools import setup, find_packages

setup(
    name="gvns",
    version="0.1.0",
    description="General Variable Neighborhood Search implementation for VRPTW-S",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "networkx",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "gvns-benchmark=gvns.benchmark:benchmark_folder",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
