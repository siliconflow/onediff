import argparse
import glob
import os
import pandas as pd
import re
from skimage import color
import imageio.v2 as imageio
from skimage.metrics import structural_similarity as ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and analyze data from two dataframes."
    )
    parser.add_argument(
        "--baseline-dir", type=str, default="results/exp9", help="Baseline directory"
    )
    parser.add_argument(
        "--compare-dir", type=str, default="results/exp10", help="Compare directory"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/merged_data.csv",
        help="Output file name",
    )
    return parser.parse_args()


def calculate_ssim(image_path1, image_path2):
    image1 = imageio.imread(image_path1)
    image2 = imageio.imread(image_path2)
    if image1.ndim == 3:
        image1 = color.rgb2gray(image1)
    if image2.ndim == 3:
        image2 = color.rgb2gray(image2)
    ssim_value = ssim(image1, image2, data_range=image2.max() - image2.min())

    return ssim_value


def extract_log_info(log_file_path):
    height_width_pattern = r"ComfyGraph: height=(\d+) width=(\d+)"
    e2e_time_pattern = r"E2E time: (\d+\.\d+) seconds"
    image_path_pattern = r"Saved image to: (.+)"

    data_list = []

    with open(log_file_path, "r") as file:
        log_lines = file.readlines()

    for line in log_lines:
        height_width_match = re.search(height_width_pattern, line)
        e2e_time_match = re.search(e2e_time_pattern, line)
        image_path_match = re.search(image_path_pattern, line)

        if height_width_match:
            height = int(height_width_match.group(1))
            width = int(height_width_match.group(2))
            data_list.append({"Height": height, "Width": width})
        elif e2e_time_match:
            e2e_time = float(e2e_time_match.group(1))
            data_list[-1]["E2E Time (seconds)"] = e2e_time
        elif image_path_match:
            image_path = image_path_match.group(1)
            data_list[-1]["Image Path"] = image_path

    df = pd.DataFrame(data_list)

    return df


def compare_e2e_times(log_file_path1, log_file_path2):
    df1 = extract_log_info(log_file_path1)
    df2 = extract_log_info(log_file_path2)

    if len(df1) != len(df2):
        raise ValueError("The number of rows in the two log files does not match.")

    merged_df = pd.DataFrame(
        {
            "Height": df1["Height"],
            "Width": df1["Width"],
            "Image1": df1["Image Path"],
            "Image2": df2["Image Path"],
            "E2E_Time1_sec": df1["E2E Time (seconds)"],
            "E2E_Time2_sec": df2["E2E Time (seconds)"],
        }
    )
    ssim_values = [
        calculate_ssim(row["Image1"], row["Image2"]) for _, row in merged_df.iterrows()
    ]
    del merged_df["Image1"]
    del merged_df["Image2"]
    merged_df["SSIM"] = ssim_values
    merged_df["Time_Diff"] = merged_df["E2E_Time1_sec"] - merged_df["E2E_Time2_sec"]
    merged_df["Speedup"] = (
        merged_df["E2E_Time1_sec"] - merged_df["E2E_Time2_sec"]
    ) / merged_df["E2E_Time1_sec"]

    return merged_df


def find_unique_log_file(folder_path):
    log_files = glob.glob(os.path.join(folder_path, "*.log"))

    if len(log_files) != 1:
        print("Warning: Found", len(log_files), "log files in the folder.")
        return None
    return log_files[0]


if __name__ == "__main__":
    args = parse_args()
    log_file_path1 = find_unique_log_file(args.baseline_dir)
    log_file_path2 = find_unique_log_file(args.compare_dir)
    comparison_df = compare_e2e_times(log_file_path1, log_file_path2)
    comparison_df.to_csv(args.output_file, index=False)
