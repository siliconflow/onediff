import json
import os

# Prompt Styler, a custom node for ComfyUI


def read_json_file(file_path):
    """
    Reads a JSON file's content and returns it.
    Ensures content matches the expected format.
    """
    if not os.access(file_path, os.R_OK):
        print(f"Warning: No read permissions for file {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = json.load(file)
            # Check if the content matches the expected format.
            if not all(
                [
                    "name" in item and "prompt" in item and "negative_prompt" in item
                    for item in content
                ]
            ):
                print(f"Warning: Invalid content in file {file_path}")
                return None
            return content
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None


def read_sdxl_styles(json_data):
    """
    Returns style names from the provided JSON data.
    """
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return []

    return [
        item["name"] for item in json_data if isinstance(item, dict) and "name" in item
    ]


def get_all_json_files(directory):
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".json") and os.path.isfile(os.path.join(directory, file))
    ]


def load_styles_from_directory(directory):
    """
    Loads styles from all JSON files in the directory.
    Renames duplicate style names by appending a suffix.
    """
    json_files = get_all_json_files(directory)
    combined_data = []
    seen = set()

    for json_file in json_files:
        json_data = read_json_file(json_file)
        if json_data:
            for item in json_data:
                original_style = item["name"]
                style = original_style
                suffix = 1
                while style in seen:
                    style = f"{original_style}_{suffix}"
                    suffix += 1
                item["name"] = style
                seen.add(style)
                combined_data.append(item)

    unique_style_names = [
        item["name"]
        for item in combined_data
        if isinstance(item, dict) and "name" in item
    ]

    return combined_data, unique_style_names


def find_template_by_name(json_data, template_name):
    """
    Returns a template from the JSON data by name or None if not found.
    """
    for template in json_data:
        if template["name"] == template_name:
            return template
    return None


class CLIPTextEncodePromptStyle:
    @classmethod
    def INPUT_TYPES(s):
        style_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "style_template"
        )
        s.json_data, styles = load_styles_from_directory(style_dir)
        return {
            "required": {
                "text_positive": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
                "text_negative": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
                "style": ((styles),),
                "clip": ("CLIP",),
            },
            "optional": {
                "log_prompt": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
                "style_positive": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
                "style_negative": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "positive",
        "negative",
    )

    FUNCTION = "encode"

    CATEGORY = "OneDiff/Conditioning"

    def encode(
        self,
        clip,
        text_positive,
        text_negative,
        style,
        log_prompt,
        style_positive,
        style_negative,
    ):
        template = find_template_by_name(self.json_data, style)

        if style_positive:
            positive_prompt = template["prompt"].replace("{prompt}", text_positive)
        else:
            positive_prompt = text_positive

        positive_tokens = clip.tokenize(positive_prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(
            positive_tokens, return_pooled=True
        )

        if style_negative:
            negative_prompt_template = template.get("negative_prompt", "")
            negative_prompt = (
                f"{negative_prompt_template}, {text_negative}"
                if negative_prompt_template and text_negative
                else text_negative or negative_prompt_template
            )
        else:
            negative_prompt = text_negative
        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(
            negative_tokens, return_pooled=True
        )

        if log_prompt:
            print(f"{positive_prompt=}\n{negative_prompt=}")
        return (
            [[positive_cond, {"pooled_output": positive_pooled}]],
            [[negative_cond, {"pooled_output": negative_pooled}]],
        )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodePromptStyle": CLIPTextEncodePromptStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodePromptStyle": "CLIP Text Encode(PromptStyle)"
}
