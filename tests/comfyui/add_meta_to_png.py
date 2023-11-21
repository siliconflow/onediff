import argparse
import os

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add ComfyUI API Workflow to a png file."
    )
    parser.add_argument(
        "--api",
        type=str,
        help="The json file containing the workflow for API calling.",
        required=True,
    )
    parser.add_argument(
        "--png",
        type=str,
        help="The png file containing the UI workflow meta data.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="The output png filename.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="api",
        help="The key name of the metadata.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    origin_img = Image.open(args.png)
    info_dict = origin_img.info
    metadata = PngInfo()
    if info_dict is not None:
        for k, v in info_dict.items():
            metadata.add_text(k, str(v))

    with open(args.api, "r") as f:
        c = f.read()
        metadata.add_text(args.key, c)

    origin_img.close()

    api_img = Image.open(args.png)

    if args.output == "":
        pngname, _ = os.path.splitext(os.path.basename(args.png))
        apiname, _ = os.path.splitext(os.path.basename(args.api))
        output = f"processed-{apiname}-{pngname}.png"
    else:
        output = args.output
    api_img.save(output, pnginfo=metadata)
    api_img.close()
