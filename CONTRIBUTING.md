# Contributing Guidelines
Guides and instructions for maintainers and contributors of this repository.

## Release

- run examples to check it works

  ```bash
  cd onediff_diffusers_extensions
  python3 examples/text_to_image.py
  ```

- bump version in these files:

  ```
  .github/workflows/pub.yml
  src/onediff/__init__.py
  onediff_diffusers_extensions/onediffx/__init__.py
  ```

- install build package
  ```bash
  python3 -m pip install build twine
  ```

- build wheel

  ```bash
  rm -rf dist
  python3 -m build
  ```

- upload to pypi

  ```bash
  twine upload dist/*
  ```
