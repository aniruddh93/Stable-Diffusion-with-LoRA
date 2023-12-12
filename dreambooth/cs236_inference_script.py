import torch
from diffusers import DiffusionPipeline
import os

model_path = "/home/aniruddh_ramrakhyani/diffusers/examples/dreambooth/pikachu/checkpoint-700"
spl_token = "sks"
pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompts_filename = {
    '_on_white_mug.png': (spl_token + ' on a white mug'),
    '_on_white_sand_beach.png': (spl_token + ' on white sand beach'),
    '_front_of_mountain_lake.png': (spl_token + ' bowing down in front of a mountain and lake'),
    '_rain.png': (spl_token + ' in rain on a green grass field'),
    '_dancing.png': (spl_token + ' in a dancing pose'),
    '_attacking.png': (spl_token + ' in attacking pose'),
    '_sunset.png': (spl_token + ' looking at the sunset'),
    '_only_mountain_lake.png': (spl_token + ' in front of a mountain and lake'),
    '_eiffel.png': (spl_token + ' in front of eiffel tower'),
    '_lunch_box.png': (spl_token + ' on a lunch box'),
    '_oil_painting.png': ('an oil painting of ' + spl_token),
    '_photo_beach.png': ('a photo of ' + spl_token + ' on the beach'),
    '_two_boat.png': ('a photo of two ' + spl_token + ' on a boat'),
    '_theme_lunchbox.png': ('a ' + spl_token + ' themed lunchbox'),
    '_hawaii.png': (spl_token + ' on a hawaii beach'),
    '_top_view.png': ('top view of ' + spl_token),
    '_bottom_view.png': ('bottom view of ' + spl_token),
    '_back_view.png': ('back view of ' + spl_token),
    '_sleeping.png': ('a sleeping ' + spl_token),
    '_sad.png': ('a sad ' + spl_token),
    '_joyous.png': ('a joyous ' + spl_token),
    '_crying.png': ('a crying ' + spl_token),
    '_screaming.png': ('a screaming ' + spl_token),
    '_police.png': (spl_token + 'in a police outfit'),
    '_chef.png': (spl_token + 'in a chef outfit'),
    '_superman.png': (spl_token + 'in a superman outfit'),
    '_nurse.png': (spl_token + 'in a nurse outfit'),
    '_fireman.png': (spl_token + 'in a firman outfit'),
    '_doctor.png': (spl_token + 'in a doctor outfit'),
    '_swimmer.png': (spl_token + 'in a swimmer outfit'),
    '_painter.png': (spl_token + 'in a painter outfit'),
    '_plumber.png': (spl_token + 'as a plumber'),
    '_blue.png': ('a blue colored ' + spl_token),
    '_red.png': ('a red colored ' + spl_token),
    '_purple.png': ('a purple colored ' + spl_token),
    '_brown.png': ('a brown colored ' + spl_token),
    '_silver.png': ('a silver colored ' + spl_token),
    '_gold.png': ('a gold colored ' + spl_token),
    '_pink.png': ('a pink colored ' + spl_token),
    '_magenta.png': ('a magenta colored ' + spl_token),
    '_crimson.png': ('a crimson colored ' + spl_token),
    '_orange.png': ('an orange colored ' + spl_token),
    '_black.png': ('a black colored ' + spl_token + 'in white background'),
    '_turqoise.png': ('an turqoise colored ' + spl_token),
    '_white.png': ('a white colored ' + spl_token + 'in black background'),
    '_black_white_sketch.png': ('a black and white sketch of ' + spl_token),
    '_serene_pond.png': ('painting of ' + spl_token + ' next to a serene pond'),
    '_stamp.png': ('picture of ' + spl_token + ' as a stamp'),
    '_mask.png': ('a ' + spl_token + ' shaped mask')
}


for filename, prompt in prompts_filename.items():
    # prompt = spl_token + prompt
    filename = spl_token + filename
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(filename)

output_path = '/mnt/disks/my_disk1/my_data/cs236_project/dreambooth/' + tinv_obj + '/'
cmd = 'mv *.png ' + output_path
os.system(cmd)
