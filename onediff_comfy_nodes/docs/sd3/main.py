import json
from urllib import request

workflow_api_path = "./workflow_api.json"


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:9999/prompt", data=data
    )  # comfyui start port
    request.urlopen(req)


with open(workflow_api_path, "r") as fp:
    prompt = json.load(fp)


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


generated_texts = list(generate_texts(max_length=101))
generated_texts.reverse()

cout = 0
dimensions = [
    (1024, 1024),
    (1024, 768),
    (1024, 576),
    (1024, 512),
    (512, 1024),
    (768, 512),
    (512, 512),
]

for width, height in dimensions:
    # Set the width and height in the prompt
    prompt["135"]["inputs"]["width"] = width
    prompt["135"]["inputs"]["height"] = height

    # Loop through each generated text and send the prompt to the server
    for text in generated_texts:
        prompt["6"]["inputs"]["text"] = text
        queue_prompt(prompt)
        print(f"{cout=}")
        cout += 1
        break
