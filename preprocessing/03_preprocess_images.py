import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from tqdm import tqdm

# Define constants for paths
BASE_DIR = "/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src"
IMAGES_DIR = os.path.join(BASE_DIR, "extracted_images")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_images")

# Create output directory if it doesn't exist
Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

# Define the preprocessing steps
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

def preprocess_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing images for {root[(len(IMAGES_DIR) + 1):]}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                input_file_dir = os.path.join(root, file)
                relative_dir = os.path.relpath(root, input_dir)
                output_file_dir = os.path.join(output_dir, relative_dir)
                
                # Change the extension to .pt
                base_name = os.path.splitext(file)[0]
                output_file_path = os.path.join(output_file_dir, f"{base_name}.pt")

                # Skip processing if the output file already exists
                if os.path.exists(output_file_path):
                    continue

                # Create output subdirectory if it doesn't exist
                Path(output_file_dir).mkdir(parents=True, exist_ok=True)

                try:
                    image = Image.open(input_file_dir).convert("RGB")
                    image_tensor = preprocess(image)
                    torch.save(image_tensor, output_file_path)
                except Exception as e:
                    print(f"Error processing {input_file_dir}: {e}")

def main():
    """Main function to iterate over all image folders and process them."""
    for image_folder in tqdm(os.listdir(IMAGES_DIR), desc="Processing image folders"):
        image_folder_path = os.path.join(IMAGES_DIR, image_folder)
        if os.path.isdir(image_folder_path):
            output_folder_path = os.path.join(PREPROCESSED_DIR, image_folder)
            if not os.path.exists(output_folder_path):
                print(f"Processing image folder: {image_folder}")
                preprocess_images(image_folder_path, output_folder_path)
            else:
                print(f"Skipping already processed folder: {image_folder}")

if __name__ == "__main__":
    main()