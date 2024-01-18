from setuptools import find_packages, setup

setup(
    name="onediff",
    version="0.12.1.dev",
    description="an out-of-the-box acceleration library for diffusion models",
    url="https://github.com/siliconflow/onediff",
    author="OneDiff contributors",
    license="Apache",
    author_email="caishenghang@oneflow.org",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8.0",
    install_requires=[
        "transformers>=4.27.1",
        "diffusers>=0.19.3",
        "accelerate",
        "torch",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
