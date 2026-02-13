import os
from datetime import datetime

import torch
from PIL import Image

from diffsynth.core.loader.model import replace_attention_with_sla
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


def generate(prompt, prompt_neg, image_path, inference_mode="wan"):
    # choose DiT backbone
    if inference_mode in ["wan", "sla"]:
        dit_path = [
            "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
        ]
    elif inference_mode == "sla_sft":
        os.environ["INFERENCE_MODE"] = "sla_sft"
        dit_path = ["models/train/Wan2.2-TI2V-5B_sla_sft/epoch-0.safetensors"]
    elif inference_mode == "sla_sft_debug":
        os.environ["INFERENCE_MODE"] = "sla_sft"
        dit_path = ["models/train/Wan2.2-TI2V-5B_sla_sft_deubg/epoch-0.safetensors"]

    # create model
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=dit_path),
            ModelConfig(path="/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    # test inference effect of wan_sla without tuning
    if inference_mode == "sla":
        pipe.dit = replace_attention_with_sla(model=pipe.dit, sla_topk=0.15)

    # model inference
    input_image = None if not image_path else Image.open(image_path).resize((1248, 704))
    seed = torch.randint(low=0, high=1001, size=(1,)).item()
    video = pipe(
        prompt=prompt,
        negative_prompt=prompt_neg,
        seed=seed,
        tiled=True,
        height=1280,
        width=704,
        input_image=input_image,
        num_frames=121,
    )

    # save video
    os.makedirs(name=f"outputs/{inference_mode}", exist_ok=True)
    save_video(
        video,
        save_path=f"outputs/{inference_mode}/Wan2.2-TI2V-5B-{inference_mode}-{datetime.now().strftime('%m%d%H%M%S')}-seed={seed}.mp4",
        fps=24,
        quality=5,
    )


if __name__ == "__main__":
    # prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
    prompt = "主体面部表情始终保持不变。人物以挂件形式轻轻晃动，两只手先自然下垂，然后慢慢抬到胸前，手心向上。掌心周围出现闪耀的圣诞魔法光效，粒子聚集，变出一个精致的礼物盒（红色礼盒配金色蝴蝶结）。魔法光散去后，人物用双手捧着礼物盒，轻轻收在胸前，眼睛看向镜头。"
    prompt_neg = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # image_path = "data/examples/wan/cat_fightning.jpg"
    image_path = "data/train_wan/first_frame.jpg"

    # inference
    generate(
        prompt=prompt,
        prompt_neg=prompt_neg,
        image_path=image_path,
        inference_mode="sla_sft",  # 原始模型：wan，仅替换注意力：sla，微调：sla_sft，测试微调：sla_sft_debug
    )
