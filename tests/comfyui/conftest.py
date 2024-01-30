import os
import yaml
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        default="conftest.yaml",
        metavar="FILE",
        help="ComfyUI related resources configuration file",
        type=str,
    )
    return parser.parse_args()


def create_symlink(source, destination):
    try:
        print(f"Creating symlink: {destination} -> {source}")
        if os.path.exists(destination):
            os.remove(destination)
        os.symlink(source, destination)
    except Exception as e:
        # warn if symlink creation fails
        print(f"Warning: {e}")


def process_datasets(data, current_path):
    if isinstance(data, dict):
        for key, value in data.items():
            process_datasets(value, current_path / key)
    elif isinstance(data, list):
        for value in data:
            process_datasets(value, current_path)
    elif data is not None:
        base_name = Path(data).name
        current_path.mkdir(parents=True, exist_ok=True)
        create_symlink(data, current_path / base_name)


def main():
    args = parse_args()
    file_path = args.config_file

    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    comfyui_root = config_data.pop("COMFYUI_ROOT", None)

    if not comfyui_root or not os.path.exists(comfyui_root):
        raise ValueError("COMFYUI_ROOT is not set or does not exist")

    process_datasets(config_data, Path(comfyui_root))


if __name__ == "__main__":
    main()
