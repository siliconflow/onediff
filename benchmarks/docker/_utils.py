import hashlib
import os
import subprocess

import yaml

from _logger import logger

from git import Repo


def load_yaml(*, file):
    if not os.path.exists(file):
        raise RuntimeError(f"config file not existed: {file}")

    with open(file, "r") as file:
        yaml_content = yaml.safe_load(file)

    return yaml_content


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def setup_repo(repo_item: dict):
    if len(repo_item.keys()) != 1:
        raise RuntimeError(f"Only one key required, but got {repo_item.keys()}")
    else:
        for key in repo_item.keys():
            repo_name = key
            repo_item = repo_item.pop(repo_name)
            break

    repo_url = repo_item.pop("repo_url")
    branch = repo_item.pop("branch")
    commit = repo_item.pop("commit", None)
    cmds = repo_item.pop("cmds", None)

    repo_path = os.path.join(".", repo_name)
    if not os.path.exists(repo_path):
        logger.info(f"git clone {repo_url} to {repo_name}, branch: {branch}")
        git_repo = Repo.clone_from(repo_url, repo_path, branch=branch)
    else:
        logger.info(f"git repository {repo_name} has existed, use it")
        git_repo = Repo(repo_path)
    if commit is not None:
        git_repo.git.checkout(commit)
        logger.info(f"checkout {repo_name} to {commit}")
    docker_commands = f"COPY {repo_name} /app/{repo_name}"
    extra_cmds = ""
    if cmds is not None:
        extra_cmds = ["RUN ", " && \\\n".join(cmds)]
        extra_cmds = " ".join(extra_cmds)
        extra_cmds = "\n".join([f"WORKDIR /app/{repo_name}", extra_cmds])
    docker_commands = "\n".join([docker_commands, extra_cmds])

    return docker_commands


def generate_docker_file(yaml_file, file_hash, output_dir, **kwargs):
    image_config = kwargs

    base_image = image_config.pop("base_image", None)
    context_path = image_config.pop("context_path", None)
    oneflow_pip_index = image_config.pop("oneflow_pip_index", None)
    repos = image_config.pop("repos", None)
    proxy = image_config.pop("proxy", None)
    set_pip_mirror = image_config.pop("set_pip_mirror", None)

    origin_file_info = f"""#==== Generated from {yaml_file} ====
#     yaml file SHA256: {file_hash}
"""

    dockerfile_head = f"""
#==== Docker Base Image ====
FROM {base_image}
"""

    if set_pip_mirror is not None:
        dockerfile_set_pip_mirror = f"RUN {set_pip_mirror}"
    else:
        dockerfile_set_pip_mirror = ""

    dockerfile_oneflow = f"""
#==== Install the OneFlow ====
RUN pip install -f {oneflow_pip_index} oneflow
"""

    repos_cmds = []
    for repo in repos:
        repo = setup_repo(repo)
        repos_cmds.append(repo)
    repos_cmds = "\n\n".join(repos_cmds)
    dockerfile_repos = f"""
#==== Download and set up the repos ====
{repos_cmds}
"""

    docker_post_cmds = f"""
#==== Post setting
WORKDIR /app
"""

    dockerfile_content = "\n".join(
        [
            origin_file_info,
            dockerfile_head,
            dockerfile_set_pip_mirror,
            dockerfile_oneflow,
            dockerfile_repos,
            docker_post_cmds,
        ]
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dockerfile_name = os.path.join(output_dir, f"Dockerfile-{file_hash[0:8]}")
    logger.info(f"write Dockerfile to {dockerfile_name}")
    with open(dockerfile_name, "w") as f:
        f.write(dockerfile_content)
    return dockerfile_name


def build_image(docker_file, imagename, context):
    command = ["docker", "build", "-f", docker_file, "-t", imagename, context]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ""):
            print(line, end="")  # Print each line
        process.wait()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")


def gen_docker_compose_yaml(container_name, image, envs, volumes, output_dir):
    from collections import OrderedDict

    onediff_benchmark_service = {
        "container_name": None,
        "image": None,
        # "entrypoint": "sleep infinity",
        "tty": True,
        "stdin_open": True,
        "privileged": True,
        "shm_size": "8g",
        "network_mode": "host",
        "pids_limit": 2000,
        "cap_add": ["SYS_PTRACE"],
        "security_opt": ["seccomp=unconfined"],
        "environment": ["HF_HUB_OFFLINE=1"],
        "volumes": [],
        "working_dir": "/app",
        "restart": "no",
        "command": '/bin/bash -c "cd /app/onediff/benchmarks && bash run_benchmark.sh /benchmark_model"',
    }
    onediff_benchmark_service["container_name"] = container_name
    onediff_benchmark_service["image"] = image
    onediff_benchmark_service["environment"].extend(envs)
    onediff_benchmark_service["volumes"].extend(volumes)
    docker_compose_dict = {
        "version": "3.8",
        "services": {"onediff-benchmark": onediff_benchmark_service},
    }

    yaml_string = yaml.dump(docker_compose_dict)

    docker_compose_file = os.path.join(output_dir, f"docker-compose.{image}.yaml")

    docker_compose_readme = f"""#========
# run the OneDiff benchmark container by:
#    docker compose -f {docker_compose_file} up
#========

"""
    run_command = f"docker compose -f {docker_compose_file} up"
    with open(docker_compose_file, "w") as f:
        content = [docker_compose_readme, yaml_string]
        f.write("\n".join(content))
    return docker_compose_file, run_command
