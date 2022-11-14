import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64


model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"

# If you are running this code locally, you need to either do a 'huggingface-cli login` or paste your User Access Token from here https://huggingface.co/settings/tokens into the use_auth_token field below.
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, use_auth_token=True, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to(device)
torch.backends.cudnn.benchmark = True

is_gpu_busy = False


def infer(prompt, samples, steps, scale, seed):
    global is_gpu_busy
    # samples = 4
    # steps = 50
    # scale = 7.5
    # seed = 42

    generator = torch.Generator(device=device).manual_seed(seed)
    # print("Is GPU busy? ", is_gpu_busy)
    images = []
    if not is_gpu_busy:
        is_gpu_busy = True
        images_list = pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        ).images
        is_gpu_busy = False
        for i, image in enumerate(images_list):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            image_b64_str = f"data:image/jpeg;base64,{image_b64}"
            images.append(image_b64_str)

    return images


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
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
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
        "A high tech solarpunk utopia in the Amazon rainforest",
        4,
        45,
        7.5,
        1024,
    ],
    [
        "A pikachu fine dining with a view to the Eiffel Tower",
        4,
        45,
        7,
        1024,
    ],
    [
        "A mecha robot in a favela in expressionist style",
        4,
        45,
        7,
        1024,
    ],
    [
        "an insect robot preparing a delicious meal",
        4,
        45,
        7,
        1024,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        4,
        45,
        7,
        1024,
    ],
]


with block:
    gr.HTML(
        f"""
        <h1 style="font-weight: 900; margin-bottom: 7px;">
        Stable Diffusion Demo: {model_id}
        </h1>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(
                mobile_collapse=False, equal_height=True
            ):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Row(elem_id="advanced-options"):
            samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(
            examples=examples,
            fn=infer,
            inputs=text,
            outputs=[gallery],
            cache_examples=False,
        )
        ex.dataset.headers = [""]

        text.submit(
            infer,
            inputs=[text, samples, steps, scale, seed],
            outputs=[gallery],
            postprocess=False,
        )
        btn.click(
            infer,
            inputs=[text, samples, steps, scale, seed],
            outputs=[gallery],
            postprocess=False,
        )

        gr.HTML(
            """
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """
        )

block.queue(concurrency_count=40, max_size=20).launch(max_threads=150)
