import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ONEDIFFBOX_ROOT = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(ONEDIFFBOX_ROOT))

from _logger import logger
from _utils import (
    build_image,
    calculate_sha256,
    gen_docker_compose_yaml,
    generate_docker_file,
    load_yaml,
    setup_repo,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build OneDiff Box")
    formatted_datetime = datetime.now().strftime("%Y%m%d-%H%M")

    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        default="config/community-default.yaml",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="onediff",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=f"benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="the output directory of Dockerfile and Docker-compose file",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        default=".",
        help="the path to build context",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="quiet mode",
    )
    args = parser.parse_args()
    return args


args = parse_args()


if __name__ == "__main__":
    image_config = load_yaml(file=args.yaml)
    file_hash = calculate_sha256(args.yaml)

    docker_file = generate_docker_file(
        args.yaml, file_hash, args.output, **image_config
    )
    version = os.path.splitext(os.path.basename(args.yaml))[0]
    image_name = f"{args.image}:{args.tag}-{version}"
    if not args.quiet:
        build_cmd = (
            f"docker build -f {docker_file} -t {args.image}:{args.tag} {args.context}"
        )
        print("Ready to build image by:")
        r = input("    " + build_cmd + " [y]/n ")
        if r == "" or r == "y" or r == "Y":
            logger.info(f"building image {image_name}")
            build_image(docker_file, image_name, args.context)
        else:
            print("building cancled")
    else:
        logger.info(f"building image {image_name}")
        build_image(docker_file, image_name, args.context)

    envs = image_config.pop("envs", [])
    volumes = image_config.pop(
        "volumes",
        [
            "$BENCHMARK_MODEL_PATH:/benchmark_model:ro",
        ],
    )
    compose_file, run_command = gen_docker_compose_yaml(
        f"onediff-benchmark-{version}", image_name, envs, volumes, args.output
    )
    logger.info(f"write docker-compose file to {compose_file}")
    logger.info(
        f"run container by:\n    {run_command}\n    and see {compose_file} for more"
    )
