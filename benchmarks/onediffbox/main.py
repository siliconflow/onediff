import argparse
from datetime import datetime
import hashlib
import logging
import os

import yaml



def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def parse_repo(repo_item: dict):
    if len(repo_item.keys()) != 1:
        raise RuntimeError(f"Only one key required, but got {repo_item.keys()}")
    else:
        for key in repo_item.keys():
            repo_name = key
            repo_item = repo_item.pop(repo_name)
            break

    repo_url = repo_item.pop("repo_url")
    branch = repo_item.pop("branch")
    commit = repo_item.pop("commit")
    cmds = repo_item.pop("cmds")

    if commit != "latest":
        docker_commands = [
            f"RUN git clone -b {branch} {repo_url} {repo_name} &&"+" \\\n",
            f"cd {repo_name} &&"+" \\\n",
            f"git checkout {commit} &&" + " \\\n",
            " && \\\n".join(cmds),
        ]
    else:
        docker_commands = [
            f"RUN git clone -b {branch} {repo_url} {repo_name} &&"+" \\\n",
            f"cd {repo_name} &&"+" \\\n",
            " && \\\n".join(cmds),
        ]
    docker_commands = " ".join(docker_commands)
    docker_commands = "\n".join(["WORKDIR /app", docker_commands])
    return docker_commands


logger = logging.getLogger("onediffbox")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
        "-i","--image",
        type=str,
        default="onediff",
    )
    parser.add_argument(
        "-t","--tag",
        type=str,
        default=f"benchmark{formatted_datetime}",
    )
    parser.add_argument(
        "-o","--output",
        type=str,
        default=os.path.join(".", "output"),
        help="the output directory of Dockerfile and Docker-compose file",
    )
    parser.add_argument(
        "-c","--context",
        type=str,
        default=os.path.join(".", "context"),
        help="the path to build context",
    )
    args = parser.parse_args()
    return args


args = parse_args()


def load_yaml(*, file):
    if not os.path.exists(file):
        raise RuntimeError(f"config file not existed: {file}")

    with open(file, "r") as file:
        yaml_content = yaml.safe_load(file)

    return yaml_content

def generate_docker_file(yaml_file, file_hash, **kwargs):
    base_image = image_config.pop('base_image', None)
    context_path = image_config.pop('context_path', None)
    oneflow_pip_index = image_config.pop('oneflow_pip_index', None)
    repos = image_config.pop('repos', None)
    proxy = image_config.pop('proxy', None)
    env = image_config.pop('env', None)

    origin_file_info = f"""#==== Generated from {yaml_file} whose SHA256: {file_hash} ====
    """

    dockerfile_head = f"""
#==== Docker Base Image ====
FROM {base_image}
"""

    dockerfile_oneflow= f"""
#==== Install the OneFlow ====
RUN pip install -f {oneflow_pip_index} oneflow
"""

    repos_cmds = []
    for repo in repos:
        repo = parse_repo(repo)
        repos_cmds.append(repo)
    repos_cmds = "\n\n".join(repos_cmds)
    dockerfile_repos = f"""
#==== Download and set up the repos ====
{repos_cmds}
"""

    dockerfile_content = "\n".join([origin_file_info, dockerfile_head, dockerfile_oneflow, dockerfile_repos])

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    dockerfile_name = os.path.join(args.output, f"Dockerfile-{file_hash[0:8]}")
    logger.info(f"Write Dockerfile to {dockerfile_name}")
    with open(dockerfile_name, "w") as f:
        f.write(dockerfile_content)    
    return dockerfile_name

if __name__ == "__main__":
    image_config = load_yaml(file=args.yaml)
    file_hash = calculate_sha256(args.yaml)

    docker_file = generate_docker_file(args.yaml, file_hash, **image_config)

