"""
Setup script for the preprocessing package.

Install in development mode with:
    pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="preprocessing",
    version="0.1.0",
    description="Rosbag preprocessing pipeline for bagseek",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.8",
    # Map the current directory as the 'preprocessing' package
    package_dir={"preprocessing": "."},
    packages=["preprocessing"] + [
        f"preprocessing.{pkg}" 
        for pkg in find_packages(exclude=["tests", "tests.*"])
    ],
    install_requires=[
        "mcap>=0.0.10",
        "mcap-ros2-support",
        "numpy",
        "pillow",
        "python-dotenv",
        "torch",
        "torchvision",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "preprocessing=preprocessing.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
