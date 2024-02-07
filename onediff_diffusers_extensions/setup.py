from setuptools import find_packages, setup

setup(
    name="onediffx",
    version="0.1.0",
    description="onediff extensions for diffusers",
    url="https://github.com/siliconflow/onediff",
    author="OneDiff contributors",
    license="Apache",
    author_email="caishenghang@oneflow.org",
    packages=find_packages(),
    python_requires=">=3.7.0",
    install_requires=[
        "transformers>=4.27.1",
        "diffusers>=0.24.0,<=0.25.1",
        "accelerate",
        "torch",
        "onefx",
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
