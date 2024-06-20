import platform
from collections import namedtuple


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


def generate_markdown_table(data, output_file="metrics.md"):
    # Extracting field names from the MetricData namedtuple
    field_names = data[0]._fields if data else []

    # Constructing the markdown table header
    markdown_table = "| " + " | ".join(field_names) + " |\n"
    markdown_table += "| " + " | ".join(["------"] * len(field_names)) + " |\n"

    # Constructing the markdown table body
    for item in data:
        markdown_table += "| " + " | ".join(map(str, item)) + " |\n"

    # Writing to Markdown file
    with open(output_file, "w") as file:
        file.write(markdown_table)


if __name__ == "__main__":
    # Example data
    MetricData = namedtuple("MetricData", ["metric", "nvidia_h800"])
    data = [
        MetricData(metric="PyTorch E2E time", nvidia_h800="1.102s"),
        MetricData(metric="OneDiff E2E time", nvidia_h800="0.865s"),
        MetricData(metric="PyTorch Max Mem Used", nvidia_h800="14.468GiB"),
        MetricData(metric="OneDiff Max Mem Used", nvidia_h800="13.970GiB"),
        MetricData(metric="PyTorch Warmup with Run time", nvidia_h800="1.741s"),
        MetricData(
            metric="OneDiff Warmup with Compilation time", nvidia_h800="718.539s"
        ),
        MetricData(metric="OneDiff Warmup with Cache time", nvidia_h800="131.776s"),
    ]

    # Call the function to generate the Markdown file
    generate_markdown_table(data)
