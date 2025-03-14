# cs236 Project : Stable Diffusion & Beyond With Pokemon

This repo contains code for 2023 fall CS236 final project: Stable Diffusion & Beyond. In this project, I explore finetuning techniques to improve performance of stable
diffusion model to generate better and more prompt aligned pokemon images.

The repo is organized as follows:

1. lora: contain files for LoRA finetuning of Stable Diffusion model:
   - cs236_clipt_score.py: script to compute CLIP-T score for images generated from LoRA finetuned stable diffusion model.
   - cs236_inference_script.py: script to run lora finetuned stable diffusion model. Also contains the prompts.
   - train_text_to_image_lora.py: training script for lora finetuning of Stable Diffusion model adapted from HuggingFace. I added code to run lora finetuning with self-atten\tion in text encoder in addition to cross-attention in unet.

2. textual_inversion: contain files for Text Inversion:
   - cs236_clip_score.py: script to compute CLIP-I and CLIP-T score for images generated from Text Inversion.
   - cs236_inference_script.py: script to run finetuned textual inversion model. Also contains the prompts.
   - textual_inversion.py: training script for text inversion. Adapted from HuggingFace.

3. dreambooth: contain files for dreambooth:
   - cs236_clip_score.py: script to compute CLIP-I and CLIP-T score for images generated from Dreambooth.
   - cs236_inference_script.py: script to run finetuned dreambooth model. Also contains the prompts.
   - train_dreambooth.py: training script for Dreambooth. Adapted from HuggingFace.
  

More details about project:
https://youtu.be/fBmY6qJKfTE
