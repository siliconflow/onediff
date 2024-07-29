import os

from setuptools import find_packages, setup


local_scheme = os.getenv("VERSION_LOCAL_SCHEME", "node-and-date")

setup(
    name="onediffx",
    description="onediff extensions for diffusers",
    url="https://github.com/siliconflow/onediff",
    author="OneDiff contributors",
    license="Apache-2.0",
    author_email="contact@siliconflow.com",
    packages=find_packages(),
    python_requires=">=3.7.0",
    install_requires=[
        "transformers>=4.27.1",
        "diffusers>=0.19.3",
        "accelerate",
        "torch",
        "onefx",
        "omegaconf",
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    use_scm_version={
        "root": "../",
        "version_file": "./onediffx/_version.py",
        "fallback_version": "0.0.0",
        "version_scheme": "guess-next-dev",
        "local_scheme": local_scheme,
    },
    setup_requires=["setuptools_scm"],
)
