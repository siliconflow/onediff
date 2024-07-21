# Compile and save to oneflow graph example: python examples/text_to_image_sdxl_mp_load.py --save
# Compile and load to new device example: python examples/text_to_image_sdxl_mp_load.py --load

import argparse
import os

import torch
import oneflow as flow  # usort: skip

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
parser.add_argument("--file", type=str, required=False, default="unet_compiled")
args = parser.parse_args()


def run_sd(cmd_args, device):
    from diffusers import DiffusionPipeline
    from onediff.infer_compiler import oneflow_compile

    # Normal SDXL pipeline init.
    seed = torch.Generator(device).manual_seed(cmd_args.seed)
    output_type = "pil"
    # SDXL base: StableDiffusionXLPipeline
    base = DiffusionPipeline.from_pretrained(
        cmd_args.base,
        torch_dtype=torch.float16,
        variant=cmd_args.variant,
        use_safetensors=True,
    )
    base.to(device)

    # Compile unet with oneflow
    print("unet is compiled to oneflow.")
    base.unet = oneflow_compile(base.unet)

    # Load compiled unet with oneflow
    if cmd_args.load:
        print("loading graphs...")
        base.unet.load_graph("base_" + cmd_args.file, device)

    # Normal SDXL run
    # sizes = [1024, 896, 768]
    sizes = [1024]
    for h in sizes:
        for w in sizes:
            for i in range(1):
                image = base(
                    prompt=cmd_args.prompt,
                    height=h,
                    width=w,
                    generator=seed,
                    num_inference_steps=cmd_args.n_steps,
                    output_type=output_type,
                ).images
                image[0].save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

    # Save compiled unet with oneflow
    if cmd_args.save:
        print("saving graphs...")
        base.unet.save_graph("base_" + cmd_args.file)


if __name__ == "__main__":
    if args.save:
        run_sd(args, "cuda:0")

    if args.load:
        import torch.multiprocessing as mp

        # multi device/process run
        devices = ("cuda:0", "cuda:1")
        procs = []
        for device in devices:
            p = mp.get_context("spawn").Process(target=run_sd, args=(args, device))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            print(p)
