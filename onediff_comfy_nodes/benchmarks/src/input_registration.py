import json

from core import create_constructor_registry, logger

# Create register and get functions
register_constructor, get_input_constructor = create_constructor_registry()


def generate_texts(min_length=50, max_length=302):
    # 50 world
    base_text = "a female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She"

    # Additional words pool
    additional_words = [
        "gracefully",
        "beautifully",
        "elegant",
        "radiant",
        "mysteriously",
        "vibrant",
        "softly",
        "gently",
        "luminescent",
        "sparkling",
        "delicately",
        "glowing",
        "brightly",
        "shimmering",
        "enchanting",
        "gloriously",
        "magnificent",
        "majestic",
        "fantastically",
        "dazzlingly",
    ]
    for i in range(min_length, max_length):
        idx = i % len(additional_words)
        base_text = base_text + " " + additional_words[idx]
        yield base_text


@register_constructor("resources/example_workflow_api.json")
def _(workflow_api_file_path):
    with open(workflow_api_file_path, "r") as fp:
        prompt = json.load(fp)
    prompt["6"]["inputs"]["text"] = "masterpiece best quality man"
    yield prompt
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            logger.info(f"Graph: {height=} {width=}")
            prompt["5"]["inputs"]["height"] = height
            prompt["5"]["inputs"]["width"] = width
            yield prompt


@register_constructor("resources/sd3_workflow_api.json")
def _(workflow_api_file_path):
    with open(workflow_api_file_path, "r") as fp:
        prompt = json.load(fp)

    prompt["135"]["inputs"]["height"] = 1024
    prompt["135"]["inputs"]["width"] = 1024
    generated_texts = list(generate_texts(max_length=101))
    generated_texts.reverse()

    for text in generate_texts():
        for height in [1024]:
            for width in [1024]:
                logger.info(f"Graph: {height=} {width=}")
                prompt["6"]["inputs"]["text"] = text
                prompt["135"]["inputs"]["height"] = height
                prompt["135"]["inputs"]["width"] = width
                yield prompt
