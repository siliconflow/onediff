import os

from setuptools import find_packages, setup

local_scheme = os.getenv("VERSION_LOCAL_SCHEME", "node-and-date")

setup(
    name="onediff",
    description="an out-of-the-box acceleration library for diffusion models",
    url="https://github.com/siliconflow/onediff",
    author="OneDiff contributors",
    license="Apache-2.0",
    license_files=("LICENSE",),
    author_email="contact@siliconflow.com",
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    extras_require={
        # optional dependencies, required by some features
        # dev dependencies. Install them by `pip3 install 'onediff[dev]'`
        "dev": [
            "pre-commit",
        ],
    },
    use_scm_version={
        "version_file": "src/onediff/_version.py",
        "fallback_version": "0.0.0",
        "version_scheme": "guess-next-dev",
        "local_scheme": local_scheme,
    },
    setup_requires=["setuptools_scm"],
)
