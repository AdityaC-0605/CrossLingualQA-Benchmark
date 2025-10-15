"""
Setup script for Cross-Lingual Question Answering System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cross-lingual-qa-system",
    version="1.0.0",
    author="Cross-Lingual QA Team",
    author_email="contact@example.com",
    description="A comprehensive cross-lingual question answering system using mBERT and mT5",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/cross-lingual-qa-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-zero-shot=experiments.run_zero_shot:main",
            "run-few-shot=experiments.run_few_shot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    keywords="nlp, question-answering, cross-lingual, multilingual, bert, t5, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/cross-lingual-qa-system/issues",
        "Source": "https://github.com/your-repo/cross-lingual-qa-system",
        "Documentation": "https://cross-lingual-qa-system.readthedocs.io/",
    },
)
