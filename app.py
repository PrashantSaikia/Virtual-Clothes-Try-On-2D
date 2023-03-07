import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from share_btn import community_icon_html, loading_icon_html, share_js


processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")


def process_image(image, prompt):
    inputs = processor(text=prompt, images=image, padding="max_length", return_tensors="pt")
  
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits
  
    filename = "mask.png"
    preds = torch.sigmoid(preds)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    plt.imsave(filename, preds)
    return Image.open("mask.png").convert("RGB")


def read_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def predict(input_image, text_query, reference, scale, seed, step):
    width, height = input_image.size
    if width < height:
        factor = width / 512.0
        width = 512
        height = int((height / factor) / 8.0) * 8

    else:
        factor = height / 512.0
        height = 512
        width = int((width / factor) / 8.0) * 8

    init_image = input_image.convert("RGB").resize((width, height))
    mask = process_image(input_image, text_query).resize((width, height))
    #mask = dict["mask"].convert("RGB").resize((width, height))

    generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None
    output = pipe(
        image=init_image,
        mask_image=mask,
        example_image=reference,
        generator=generator,
        guidance_scale=scale,
        num_inference_steps=step,
    ).images[0]

    return output, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
'''

example = {}
ref_dir = 'examples/reference'
image_dir = 'examples/image'
ref_list = [os.path.join(ref_dir, file) for file in os.listdir(ref_dir)]
ref_list.sort()
image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
image_list.sort()


image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    # gr.HTML(read_content("header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(source="upload", elem_id="image_upload", type="pil", label="Source Image")
                    text = gr.Textbox(lines=1, placeholder="Clothing item you want to replace...")
                    reference = gr.Image(source="upload", elem_id="image_upload", type="pil", label="Reference Image")

                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img").style(height=400)
                    guidance = gr.Slider(label="Guidance scale", value=5, maximum=15,interactive=True)
                    steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1,interactive=True)

                    seed = gr.Slider(0, 10000, label='Seed (0 = random)', value=0, step=1)

                    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Paint!").style(
                            margin=False,
                            rounded=(False, True, True, False),
                            full_width=True,
                        )
                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html, visible=True)
                        loading_icon = gr.HTML(loading_icon_html, visible=True)
                        share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)

            with gr.Row():
                with gr.Column():
                    gr.Examples(
                        image_list, 
                        inputs=[image], 
                        label="Examples - Source Image",
                        examples_per_page=12
                    )
                with gr.Column():
                    gr.Examples(
                        ref_list, 
                        inputs=[reference], 
                        label="Examples - Reference Image",
                        examples_per_page=12
                    )
            
            btn.click(
                fn=predict, 
                inputs=[image, text, reference, guidance, seed, steps], 
                outputs=[image_out, community_icon, loading_icon, share_button]
            )
            share_button.click(None, [], [], _js=share_js)


image_blocks.launch(share=True)
