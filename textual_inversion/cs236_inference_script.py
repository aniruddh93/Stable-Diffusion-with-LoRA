import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

model_base = "runwayml/stable-diffusion-v1-5"

# pikachu 500
# charizard 500
# meganium 2500 or 3000

tinv_obj = "pikachu"
tinv_token = "<txinv-pikachu>"

tinv_model_path = "/home/aniruddh_ramrakhyani/diffusers/examples/textual_inversion/textual_inversion_" + tinv_obj + "/learned_embeds-steps-500.safetensors"


pipeline = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion(tinv_model_path)

prompts_filename = {
    '_on_white_mug.png': (tinv_token + ' on a white mug'),
    '_on_white_sand_beach.png': (tinv_token + ' on white sand beach'),
    '_front_of_mountain_lake.png': (tinv_token + ' bowing down in front of a mountain and lake'),
    '_rain.png': (tinv_token + ' in rain on a green grass field'),
    '_dancing.png': (tinv_token + ' in a dancing pose'),
    '_attacking.png': (tinv_token + ' in attacking pose'),
    '_sunset.png': (tinv_token + ' looking at the sunset'),
    '_only_mountain_lake.png': (tinv_token + ' in front of a mountain and lake'),
    '_eiffel.png': (tinv_token + ' in front of eiffel tower'),
    '_lunch_box.png': (tinv_token + ' on a lunch box'),
    '_oil_painting.png': ('an oil painting of ' + tinv_token),
    '_photo_beach.png': ('a photo of ' + tinv_token + ' on the beach'),
    '_two_boat.png': ('a photo of two ' + tinv_token + ' on a boat'),
    '_theme_lunchbox.png': ('a ' + tinv_token + ' themed lunchbox'),
    '_hawaii.png': (tinv_token + ' on a hawaii beach'),
    '_top_view.png': ('top view of ' + tinv_token),
    '_bottom_view.png': ('bottom view of ' + tinv_token),
    '_back_view.png': ('back view of ' + tinv_token),
    '_sleeping.png': ('a sleeping ' + tinv_token),
    '_sad.png': ('a sad ' + tinv_token),
    '_joyous.png': ('a joyous ' + tinv_token),
    '_crying.png': ('a crying ' + tinv_token),
    '_screaming.png': ('a screaming ' + tinv_token),
    '_police.png': (tinv_token + 'in a police outfit'),
    '_chef.png': (tinv_token + 'in a chef outfit'),
    '_superman.png': (tinv_token + 'in a superman outfit'),
    '_nurse.png': (tinv_token + 'in a nurse outfit'),
    '_blue.png': ('a blue colored ' + tinv_token),
    '_red.png': ('a red colored ' + tinv_token),
    '_purple.png': ('a purple colored ' + tinv_token),
    '_brown.png': ('a brown colored ' + tinv_token),
    '_silver.png': ('a silver colored ' + tinv_token),
    '_gold.png': ('a gold colored ' + tinv_token),
    '_pink.png': ('a pink colored ' + tinv_token),
    '_magenta.png': ('a magenta colored ' + tinv_token),
    '_black_white_sketch.png': ('a black and white sketch of ' + tinv_token),
    '_serene_pond.png': ('painting of ' + tinv_token + ' next to a serene pond'),
    '_stamp.png': ('picture of ' + tinv_token + ' as a stamp'),
    '_mask.png': ('a ' + tinv_token + ' shaped mask')
}


for filename, prompt in prompts_filename.items():
    # prompt = tinv_token + prompt
    filename = tinv_obj + filename
    image = pipeline(prompt).images[0]
    image.save(filename)

output_path = '/mnt/disks/my_disk1/my_data/cs236_project/textual_inversion/' + tinv_obj + '/'
cmd = 'mv *.png ' + output_path
os.system(cmd)
