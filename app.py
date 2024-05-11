from __future__ import annotations
import os
import random
import uuid
import gradio as gr
import spaces
import numpy as np
import uuid
from diffusers import PixArtAlphaPipeline, LCMScheduler
import torch
from typing import Tuple
from datetime import datetime


DESCRIPTION = """ # Instant Image
        ### Super fast text to Image Generator.
        ### <span style='color: red;'>You may change the steps from 4 to 8, if you didn't get satisfied results.
        ### First Image processing takes time then images generate faster.
        ### Must Try -> Instant Video https://huggingface.co/spaces/KingNish/Instant-Video
        """
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4192"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
PORT = int(os.getenv("DEMO_PORT", "15432"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Realistic",
        "prompt": "Photorealistic {prompt} . Ulta-realistic, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, disfigured",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]


styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
NUM_IMAGES_PER_PROMPT = 1

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

if torch.cuda.is_available():

    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if os.getenv('CONSISTENCY_DECODER', False):
        print("Using DALL-E 3 Consistency Decoder")
        pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print("Loaded on Device!")
        
    # speed-up T5
    pipe.text_encoder.to_bettertransformer()

    if USE_TORCH_COMPILE:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")






def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name



def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=30)
def generate(
        prompt: str,
        negative_prompt: str = "",
        style: str = DEFAULT_STYLE_NAME,
        use_negative_prompt: bool = False,
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        inference_steps: int = 4,
        randomize_seed: bool = False,
        use_resolution_binning: bool = True,
        progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)
    
    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=0,
        num_inference_steps=inference_steps,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        use_resolution_binning=use_resolution_binning,
        output_type="pil",
    ).images

    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed


examples = [
    "A Monkey with a happy face in the Sahara desert.",
    "Eiffel Tower was Made up of ICE.",
    "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.",
    "A close-up photo of a woman. She wore a blue coat with a gray dress underneath and has blue eyes.",
    "A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
]

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row(equal_height=False):
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Gallery(label="Result", columns=NUM_IMAGES_PER_PROMPT,  show_label=False)
                    
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row():
                use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False, visible=True)
                negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )
              
#                num_imgs = gr.Slider(
 #                   label="Num Images",
  #                  minimum=1,
   #                 maximum=8,
    #                step=1,
     #               value=1,
      #          )
            style_selection = gr.Radio(
                show_label=True,
                container=True,
                interactive=True,
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
                label="Image Style",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
                )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row(visible=True):
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
        with gr.Row():
            inference_steps = gr.Slider(
                label="Steps",
                minimum=4,
                maximum=20,
                step=1,
                value=4,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )




















    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            # num_imgs,
            seed,
            width,
            height,
            inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
    # demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=11900, debug=True)














