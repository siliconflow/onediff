[![PyPI version](https://badge.fury.io/py/onediff.svg)](https://badge.fury.io/py/onediff)
[![Docker image build](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml/badge.svg)](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml)

# OneFlow diffusers

OneFlow backend support for diffusers

## Getting started

Please refer to this [wiki](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion)

## Quick demo

```
python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.15.0"
python3 -m pip uninstall accelerate -y
python3 -m pip install -U onediff
python3 -m onediff.demo
```

## More about OneFlow

OneFlow's main [repo](https://github.com/Oneflow-Inc/oneflow)

## Development

### Option 1: Fresh clone and dev install

```
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff
python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.15.0"
python3 -m pip uninstall accelerate -y
python3 -m pip install -e .
```

### Option 2: Setup if you were using the the `oneflow-fork` branch before

1. uninstall transformers and diffusers

```
python3 -m pip uninstall transformers -y
python3 -m pip uninstall diffusers -y
```

2. install transformers and diffusers

```
python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.15.0"
python3 -m pip uninstall accelerate -y
```

3. delete the main first:

```
git branch -D main
git fetch
git checkout main
python3 -m pip install -e .
```

## More examples

There is a directory for [examples](/examples/)

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
