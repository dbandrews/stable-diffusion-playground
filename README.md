---
title: Stable Diffusion Playground
emoji: 🏃
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 3.4b2
app_file: app.py
pinned: false
license: mit
---

A very rough demo app for playing with different diffusion models.

Fully based off here:
https://huggingface.co/spaces/stabilityai/stable-diffusion/blob/main/app.py

With changes for running locally.

## Setup

Setup conda env, install with pip - struggle through getting CUDA and CuDNN installed correctly.

Then you should be good to go! If you have 12+ GB of VRAM on your GPU...

```console
conda create -n stable_diffusion python=3.9 -y
conda activate stable_diffusion
pip install -r requirements.txt
```

Run with CUDA 11.5, CuDNN 8.6.0.*

Install instructions here for CuDNN on Ubuntu within WSL:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install


Run the app with:

`python app.py`
