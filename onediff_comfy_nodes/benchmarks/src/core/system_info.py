import platform
from dataclasses import dataclass, field
from typing import List

import psutil
import torch


@dataclass
class GPUInfo:
    name: str
    memory_total: float


@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    architecture: str
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percentage: float
    cpu_model: str
    cuda_available: bool = field(default=False)
    gpu_count: int = field(default=0)
    gpu_info: List[GPUInfo] = field(default_factory=list)
    cuda_version: str = field(default="N/A")
    cudnn_available: bool = field(default=False)
    cudnn_version: str = field(default="N/A")
    pytorch_version: str = field(default="N/A")

    def __str__(self):
        gpu_info_str = "\n".join(
            [
                f"    GPU {i}: {gpu.name}, Memory: {gpu.memory_total:.2f} GB"
                for i, gpu in enumerate(self.gpu_info)
            ]
        )
        return (
            f"OS Name: {self.os_name}\n"
            f"OS Version: {self.os_version}\n"
            f"Architecture: {self.architecture}\n"
            f"Total Memory: {self.total_memory:.2f} GB\n"
            f"Available Memory: {self.available_memory:.2f} GB\n"
            f"Used Memory: {self.used_memory:.2f} GB\n"
            f"Memory Usage: {self.memory_percentage:.2f}%\n"
            f"CPU Model: {self.cpu_model}\n"
            f"CUDA Available: {self.cuda_available}\n"
            f"Number of GPUs: {self.gpu_count}\n"
            f"{gpu_info_str}\n"
            f"CUDA Version: {self.cuda_version}\n"
            f"cuDNN Available: {self.cudnn_available}\n"
            f"cuDNN Version: {self.cudnn_version}\n"
            f"PyTorch Version: {self.pytorch_version}\n"
        )


def system_info_to_markdown(system_info):
    gpu_info_str = "\n".join(
        [
            f"- GPU {i}: {gpu.name}, Memory: {gpu.memory_total:.2f} GB"
            for i, gpu in enumerate(system_info.gpu_info)
        ]
    )
    markdown_str = (
        f"### System Information\n\n"
        f"| Parameter | Value |\n"
        f"| --- | --- |\n"
        f"| OS Name | {system_info.os_name} |\n"
        f"| OS Version | {system_info.os_version} |\n"
        f"| Architecture | {system_info.architecture} |\n"
        f"| Total Memory | {system_info.total_memory:.2f} GB |\n"
        f"| Available Memory | {system_info.available_memory:.2f} GB |\n"
        f"| Used Memory | {system_info.used_memory:.2f} GB |\n"
        f"| Memory Usage | {system_info.memory_percentage:.2f}% |\n"
        f"| CPU Model | {system_info.cpu_model} |\n"
        f"| CUDA Available | {system_info.cuda_available} |\n"
        f"| Number of GPUs | {system_info.gpu_count} |\n"
        f"| CUDA Version | {system_info.cuda_version} |\n"
        f"| cuDNN Available | {system_info.cudnn_available} |\n"
        f"| cuDNN Version | {system_info.cudnn_version} |\n"
        f"| PyTorch Version | {system_info.pytorch_version} |\n"
        f"\n"
        f"#### GPU Information\n\n"
        f"{gpu_info_str}\n"
    )
    return markdown_str


def get_cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            from subprocess import check_output

            return (
                check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .strip()
                .decode()
            )
    except Exception as e:
        return "Unknown"
    return "Unknown"


def get_system_info() -> SystemInfo:
    # Get OS info
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()

    # Get CPU model
    cpu_model = get_cpu_model()

    # Get memory info
    virtual_mem = psutil.virtual_memory()
    total_memory = virtual_mem.total / (1024 ** 3)  # Convert bytes to GB
    available_memory = virtual_mem.available / (1024 ** 3)  # Convert bytes to GB
    used_memory = (virtual_mem.total - virtual_mem.available) / (
        1024 ** 3
    )  # Convert bytes to GB
    memory_percentage = virtual_mem.percent

    # Get current process ID
    current_process_id = psutil.Process().pid

    # Check CUDA and cuDNN availability
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_info = []
    if cuda_available:
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            device_properties = torch.cuda.get_device_properties(i)
            gpu_memory_total = device_properties.total_memory / (
                1024 ** 3
            )  # Convert bytes to GB
            gpu_info.append(GPUInfo(name=device_name, memory_total=gpu_memory_total))

    cuda_version = torch.version.cuda if cuda_available else "N/A"
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else "N/A"
    pytorch_version = torch.__version__

    # Create SystemInfo object
    system_info = SystemInfo(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        total_memory=total_memory,
        available_memory=available_memory,
        used_memory=used_memory,
        memory_percentage=memory_percentage,
        cpu_model=cpu_model,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_info=gpu_info,
        cuda_version=cuda_version,
        cudnn_available=cudnn_available,
        cudnn_version=cudnn_version,
        pytorch_version=pytorch_version,
    )

    return system_info


if __name__ == "__main__":
    system_info = get_system_info()
    print(system_info_to_markdown(system_info))
