import os
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_images")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

models = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14", "geolocal/StreetCLIP"]

# Create output directory if it doesn't exist
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Load Hugging Face CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_embeddings(input_dir, output_dir, model, processor):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Generating embeddings for {root[(len(PREPROCESSED_DIR) + 1):]} with model {model}"):
            if file.lower().endswith('.pt'):
                input_file_path = os.path.join(root, file)
                relative_dir = os.path.relpath(root, input_dir)
                output_file_dir = os.path.join(output_dir, relative_dir)
                
                # Change the extension to _embedding.pt
                base_name = os.path.splitext(file)[0]
                output_file_path = os.path.join(output_file_dir, f"{base_name}_embedding.pt")

                # Skip processing if the output file already exists
                if os.path.exists(output_file_path):
                    continue

                # Create output subdirectory if it doesn't exist
                Path(output_file_dir).mkdir(parents=True, exist_ok=True)

                try:
                    image_tensor = torch.load(input_file_path, weights_only=True).unsqueeze(0).to(device)  # Add batch dimension
                    image = to_pil_image(image_tensor.squeeze(0).cpu())
                    inputs = processor(images=image, return_tensors="pt").to(device)

                    with torch.no_grad():
                        image_embedding = model.get_image_features(**inputs)
                        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)  # Normalize

                    torch.save(image_embedding.cpu(), output_file_path)
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

def main():
    for model_id in models:  # Use a different variable name to avoid shadowing
        EMBEDDINGS_MODEL_DIR = os.path.join(EMBEDDINGS_DIR, model_id.replace("/", "_"))  # Replace "/" with "_" for directory naming
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)  # Pass the model identifier (string) here
        print(f"Loaded model: {model_id}")
    
        """Main function to iterate over all preprocessed image folders and generate embeddings."""
        for preprocessed_folder in tqdm(os.listdir(PREPROCESSED_DIR), desc="Processing preprocessed folders"):
            preprocessed_folder_path = os.path.join(PREPROCESSED_DIR, preprocessed_folder)
            if os.path.isdir(preprocessed_folder_path):
                output_folder_path = os.path.join(EMBEDDINGS_MODEL_DIR, preprocessed_folder)
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                generate_embeddings(preprocessed_folder_path, output_folder_path, model, processor)

if __name__ == "__main__":
    main()