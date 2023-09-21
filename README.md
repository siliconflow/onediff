[![PyPI version](https://badge.fury.io/py/onediff.svg)](https://badge.fury.io/py/onediff)
[![Docker image build](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml/badge.svg)](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml)

# OneFlow diffusers

OneFlow backend support for diffusers

## Getting started

Please refer to this [wiki](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion)

## Quick demo

```
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.15.1"
python3 -m pip uninstall accelerate -y
python3 -m pip install -U onediff
python3 -m onediff.demo
```

## More about OneFlow

OneFlow's main [repo](https://github.com/Oneflow-Inc/oneflow)

## Development

### Clone and dev install

```
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.15.1"
python3 -m pip uninstall accelerate -y
python3 -m pip install -e .
```
### Run the examples

There is a directory for [examples](/examples/). To run examples, please run
```bash
python3 -m pip install examples/requirements.txt
```

## Release

- run examples to check it works

  ```bash
  python3 examples/text_to_image.py
  python3 examples/text_to_image_dpmsolver.py
  ```

- bump version in these files:

  ```
  setup.py
  src/onediff/__init__.py
  ```

- build wheel

  ```
  rm -rf dist
  python3 setup.py bdist_wheel
  ```

- upload to pypi

  ```bash
  twine upload dist/*
  ```
