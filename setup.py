from setuptools import find_packages, setup

setup(
    name="onediff",
    version="0.2.0",
    description="OneFlow backend for diffusers",
    url="https://github.com/Oneflow-Inc/oneflow",
    author="OneFlow contributors",
    license="Apache",
    author_email="caishenghang@oneflow.co",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7.0",
    install_requires=["transformers>=4.26", "diffusers"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
