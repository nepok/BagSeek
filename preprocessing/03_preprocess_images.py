import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import open_clip
import re
import concurrent.futures

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
IMAGES_DIR = os.path.join(BASE_DIR, "extracted_images")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_images")

model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]

# Create output directory if it doesn't exist
Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

preprocess_dict = {}

def collect_preprocess_data(model_name, pretrained_name):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name, cache_dir="/mnt/data/openclip_cache")
    preprocess_str = str(preprocess)
    model_id = f"{model_name} ({pretrained_name})"

    if preprocess_str not in preprocess_dict:
        preprocess_dict[preprocess_str] = []
    if model_id not in preprocess_dict[preprocess_str]:
        preprocess_dict[preprocess_str].append(model_id)

def get_preprocess_id(preprocess_str: str) -> str:
    summary_parts = []

    # Match Resize(size=224, ...)
    resize_match = re.search(r"Resize\(size=(\d+)", preprocess_str)
    if resize_match:
        summary_parts.append(f"resize{resize_match.group(1)}")

    # Match CenterCrop(size=(224, 224))
    crop_match = re.search(r"CenterCrop\(size=\(?(\d+)", preprocess_str)
    if crop_match:
        summary_parts.append(f"crop{crop_match.group(1)}")

    # Match ToTensor() (as a function call or object)
    if "ToTensor" in preprocess_str:
        summary_parts.append("tensor")

    # Match Normalize(...), regardless of float content
    if "Normalize" in preprocess_str:
        summary_parts.append("norm")

    # Summary name
    summary = "_".join(summary_parts)
  
    return f"{summary}"

def preprocess_images(input_dir, output_dir, model_name, pretrained_name):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name, cache_dir="/mnt/data/openclip_cache")
    print(f"Using model: {model} with preprocess: {preprocess}")

    def process_single_image(file):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            input_file_dir = os.path.join(root, file)
            relative_dir = os.path.relpath(root, input_dir)
            output_file_dir = os.path.join(output_dir, relative_dir)

            # Change the extension to .pt
            base_name = os.path.splitext(file)[0]
            output_file_path = os.path.join(output_file_dir, f"{base_name}.pt")

            if os.path.exists(output_file_path):
                return

            Path(output_file_dir).mkdir(parents=True, exist_ok=True)

            try:
                image = Image.open(input_file_dir).convert("RGB")
                image_tensor = preprocess(image)
                torch.save(image_tensor, output_file_path)
            except Exception as e:
                print(f"Error processing {input_file_dir}: {e}")

    for root, _, files in os.walk(input_dir):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_single_image, files), total=len(files), desc=f"Processing images for {root[(len(IMAGES_DIR) + 1):]}"))

def main():
    for model_name, pretrained_name in model_configs:
        collect_preprocess_data(model_name, pretrained_name)

    # Nur einen gemeinsamen Preprocessing-Hash verwenden
    if len(preprocess_dict) == 1:
        preprocess_str, models = next(iter(preprocess_dict.items()))
        shared_model_name, shared_pretrained_name = models[0].split(" (")
        shared_pretrained_name = shared_pretrained_name.rstrip(")")

        print(f"All models share the same preprocess. Proceeding with: {shared_model_name} ({shared_pretrained_name})")

        preprocess_id = get_preprocess_id(preprocess_str)
        print(f"Preprocess ID: {preprocess_id}")
        for image_folder in tqdm(os.listdir(IMAGES_DIR), desc=f"Processing image folders"):
            image_folder_path = os.path.join(IMAGES_DIR, image_folder)
            if os.path.isdir(image_folder_path):
                output_folder_path = os.path.join(PREPROCESSED_DIR, f"{preprocess_id}", image_folder)
                if not os.path.exists(output_folder_path):
                    preprocess_images(image_folder_path, output_folder_path, shared_model_name, shared_pretrained_name)
                else:
                    print(f"Skipping already processed folder: {image_folder} for preprocess {preprocess_id}")
    else:
        print("Warning: Not all model_configs share the same preprocessing function. Please check manually.")


if __name__ == "__main__":
    main()