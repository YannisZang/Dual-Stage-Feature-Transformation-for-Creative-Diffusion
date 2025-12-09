import sys
sys.path.append('src')
from score_util_pub import *
from inference import *
import json

model = Generator("sdxl")

num_images = 100

nouns = ["car"]
# nouns = ["chair", "car", "bicycle", "dog", "cat", "lamp", "table", "sofa", "guitar", "flower"]

for noun in nouns:
    prompt = f"a {noun}"
    if not os.path.exists(f"./dataset/{noun}/test"):
        os.makedirs(f"./dataset/{noun}/test")
    for i in range(num_images):
        output_path = os.path.join(f"./dataset/{noun}/test", f"{noun}_{i:03d}.png")
        if not os.path.exists(output_path):
            img_test = model.orig(prompt, seed=i)
            img_test.save(output_path)

model_methods = ["original", "c3", "upblock_transform", "saliency_gating", "both"]
for noun in nouns:
    for method in model_methods:
        output_dir = os.path.join(f"./dataset/{noun}", method)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

model = Generator("sdxl-light-1")

# file_path = f'./results/sdxl-light-1/house/amp_factors_80.json'
amplification_factor = [1.0]*7
file_path = f'./results/amp_factors_80.json'
with open(file_path, 'r') as file:
    data = json.load(file)
amplification_factor[0] = 1+(data[0][0]-1)*0.2
amplification_factor[1] = 1+(data[0][1]-1)*0.2
amplification_factor[2] = 1+(data[0][2]-1)*0.1
amplification_factor[3] = 1+(data[0][3]-1)*0.1

for noun in nouns:
    for method in model_methods:
        output_dir = os.path.join(f"./dataset/{noun}", method)
        test_path = os.path.join(f"./dataset/{noun}/test")
        for img_name in os.listdir(test_path):
            output_path = os.path.join(output_dir, img_name)
            
            # Check if file already exists
            if os.path.exists(output_path):
                print(f'Skipping {img_name} - already exists.')
                continue
            
            img_path = os.path.join(test_path, img_name)
            prompt = f"a creative {img_name.split('_')[0]}"
            seed = int(img_name.split('_')[1].split('.')[0])
            print(f'Generating {img_name} using {method} method.')
            if method == "original":
                img_out = model.orig(prompt=prompt, seed=seed)
            elif method == "c3":
                img_out = model.c3(prompt=prompt, seed=seed, replace_mask=amplification_factor, cutoff=[10.0,5.0,5.0,5.0,1.0,1.0,1.0])
            elif method == "upblock_transform":
                img_out = model.dual_stage(prompt=prompt, seed=seed, replace_mask=amplification_factor, cutoff=[10.0,5.0,5.0,5.0,1.0,1.0,1.0], filter_factor=0.8, saliency_fft=False)
            elif method == "saliency_gating":
                img_out = model.dual_stage(prompt=prompt, seed=seed, replace_mask=amplification_factor, cutoff=[10.0,5.0,5.0,5.0,1.0,1.0,1.0], apply_filter=False, filter_factor=0.8, saliency_fft=True)
            elif method == "both":
                img_out = dual = model.dual_stage(prompt=prompt, seed=seed, replace_mask=amplification_factor,cutoff=[10.0,5.0,5.0,5.0,1.0,1.0,1.0], filter_factor=0.8)
            img_out.save(os.path.join(output_dir, img_name))
            print(f'Saved {img_name}.')