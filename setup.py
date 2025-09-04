#!/usr/bin/env python3
"""
Setup script for the UMAF (Universal Model Adapter Fusion) project.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="umaf",
    version="0.1.0",
    author="millionthodin16",
    description="Universal Model Adapter Fusion: Capability transfer across diverse language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/millionthodin16/umaf",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "umap-learn>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "umaf=umaf.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)