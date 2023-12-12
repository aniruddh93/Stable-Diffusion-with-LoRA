import os
import torch

from PIL import Image
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

input_base_path = '/mnt/disks/my_disk1/my_data/cs236_project/all_lora_rank8/'
lora_rank = 8
lora_type = 'all'
lora_scales = [1.0]

prompts_filename = {
    'pokemon_with_blue_eyes.png': 'pokemon with blue eyes',
    'dragon_pokemon_red_coat.png': 'A dragon Pokemon wearing red coat',
    'attacking_pikachu.png': 'an attacking pikachu',
    'pikachu_holding_two_green_balls.png': 'pikachu holding two green balls',
    'pink_and_blue_bird_pokemon.png': 'pink and blue bird pokemon',
    'red_pokemon_with_blue_eyes.png': 'Red pokemon with blue eyes',
    'pikachu_bowing_dragon.png': 'pikachu bowing down to a dragon pokemon in front of a mountain and lake',
    'dragon_sword_mountain.png': 'dragon pokemon polishing his sword on top of snow covered mountain',
    'half_bird_half_dragon.png': 'half bird half dragon pokemon flapping his wings in front of a lake',
    'rain_dancing.png': 'pikachu dancing with a red and black dragon in rain on a green grass field',
    'bird_sunset.png': 'yellow and black bird pokemon sitting on a tree and looking at the sunset',
    'dragon_pokemon_eiffel_tower.png': 'dragon pokemon in front of eiffel tower',
    'pikachu_eiffel_tower.png': 'pikachu in front of eiffel tower',
    'dragon_space.png': 'dragon pokemon in space spewing fire on earth',
    'dragon_attacking_dragon.png': 'dragon pokemon attacking another dragon pokemon with fire',
    'meditation.png': 'pokemon meditating under a banyan tree',
    'red_apple.png': 'pokemon and dragon eating red apple in park',
    'pikachu_boat.png': 'pikachu in a boat in a ocean with whales',
    'pikachu_dragon_sheild.png': 'pikachu sheilding from fire attack from dragon using a wooden plank',
    'green_flower.png': 'green pokemon with a red flower around the neck and big blue eyes',
    'pikachu_sunglasses.png': 'pikachu wearing sunglasses on a beach',
    'attacking_building.png': 'blue pokemon attacking a tall orange building',
}

image_text_logits = []

for scale in lora_scales:
    for filename, prompt in prompts_filename.items():
        full_filename = input_base_path + 'lora_rank' + str(lora_rank) + '_' + lora_type + '_scale_' + str(scale) + '_' + filename
        image = Image.open(full_filename)
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logit = outputs.logits_per_image
        logit_value = logit[0][0].item()
        print('logit_value: ', logit_value)
        image_text_logits.append(logit_value)

    logit_mean = torch.tensor(image_text_logits, dtype=torch.float32).mean().item()
    file_base_name = input_base_path + 'lora_rank' + str(lora_rank) + '_' + lora_type + '_scale_' + str(scale)
    print('mean for ', file_base_name, ': ', logit_mean)

