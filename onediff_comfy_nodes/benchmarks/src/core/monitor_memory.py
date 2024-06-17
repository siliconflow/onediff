""" 
pip install nvidia-ml-py3
"""
import time

import pynvml


def get_process_memory(pid):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for info in process_info:
            if info.pid == pid:
                return info.usedGpuMemory // 1024 // 1024  # Convert to MB
    return None


def monitor_memory(pid):
    while True:
        mem_usage = get_process_memory(pid)
        if mem_usage is not None:
            print(f"Process {pid} is using {mem_usage} MB of GPU memory")
        else:
            print(f"Process {pid} is not running or using GPU memory")
        time.sleep(0.1)


if __name__ == "__main__":
    pid = 1193828  # Replace with the process ID you want to monitor
    monitor_memory(pid)
