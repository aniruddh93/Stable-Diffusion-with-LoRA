import os
import torch

from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# meganium_1.png
# charizard_2.webp
# pikachu_5.jpeg

tinv_obj = "charizard"
tinv_token = "pokemon"
tinv_raw_image = "charizard_2.webp"
raw_image_path = '/mnt/disks/my_disk1/my_data/cs236_project/raw_images/' + tinv_obj + '/' + tinv_raw_image

raw_image = Image.open(raw_image_path)
raw_image_inputs = processor(images=raw_image, return_tensors="pt")
raw_image_features = model.get_image_features(**raw_image_inputs)
raw_image_features_normalized = raw_image_features / torch.linalg.vector_norm(raw_image_features, ord=2, dim=-1, keepdim=True)
raw_image_features_normalized = torch.squeeze(raw_image_features_normalized)

input_base_path = '/mnt/disks/my_disk1/my_data/cs236_project/textual_inversion/' + tinv_obj + '/'


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

image_image_logits = []
image_text_logits = []

for filename, prompt in prompts_filename.items():
    image_path = input_base_path + tinv_obj + filename
    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**image_inputs)
    image_features_normalized = image_features / torch.linalg.vector_norm(image_features, ord=2, dim=-1, keepdim=True)
    image_features_normalized = torch.squeeze(image_features_normalized)

    logits = torch.dot(raw_image_features_normalized, image_features_normalized) * 100
    logit_value = logits.item()
    print('image_image_logit_value: ', logit_value)
    image_image_logits.append(logit_value)

    # calculate image-text-logits
    text_inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)
    text_features_normalized = text_features / torch.linalg.vector_norm(text_features, ord=2, dim\
=-1, keepdim=True)
    text_features_normalized = torch.squeeze(text_features_normalized)
    txt_img_logits = torch.dot(text_features_normalized, image_features_normalized) * 100
    txt_img_logit_value = txt_img_logits.item()
    print('txt_img logit value: ', txt_img_logit_value)
    image_text_logits.append(txt_img_logit_value)

image_image_logit_mean = torch.tensor(image_image_logits, dtype=torch.float32).mean().item()
print('image-image mean logit for ', tinv_obj, ': ', image_image_logit_mean)

image_txt_logit_mean = torch.tensor(image_text_logits, dtype=torch.float32).mean().item()
print('image-text mean logit for ', tinv_obj, ': ', image_txt_logit_mean)
