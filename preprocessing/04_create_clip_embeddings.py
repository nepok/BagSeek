import os
from pathlib import Path
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import open_clip

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_images")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
"""
model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k'),
    ('RN50-quickgelu', 'openai'),
    ('RN50-quickgelu', 'yfcc15m'),
    ('RN50-quickgelu', 'cc12m'),
    ('RN101-quickgelu', 'openai'),
    ('RN101-quickgelu', 'yfcc15m'),
    ('ViT-B-32-quickgelu', 'laion400m_e31'),
    ('ViT-B-32-quickgelu', 'laion400m_e32'),
    ('ViT-B-32', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-32', 'commonpool_m_clip_s128m_b4k'),
    ('ViT-B-32', 'commonpool_m_laion_s128m_b4k'),
    ('ViT-B-32', 'commonpool_m_image_s128m_b4k'),
    ('ViT-B-32', 'commonpool_m_text_s128m_b4k'),
    ('ViT-B-32', 'commonpool_m_basic_s128m_b4k'),
    ('ViT-B-32', 'commonpool_m_s128m_b4k'),
    ('ViT-B-32-quickgelu', 'metaclip_400m'),
    ('ViT-B-32-quickgelu', 'metaclip_fullcc'),
    ('ViT-B-16', 'laion400m_e31'),
    ('ViT-B-16', 'laion400m_e32'),
    ('ViT-B-16', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-16', 'datacomp_l_s1b_b8k'),
    ('ViT-B-16', 'commonpool_l_clip_s1b_b8k'),
    ('ViT-B-16', 'commonpool_l_laion_s1b_b8k'),
    ('ViT-B-16', 'commonpool_l_image_s1b_b8k'),
    ('ViT-B-16', 'commonpool_l_text_s1b_b8k'),
    ('ViT-B-16', 'commonpool_l_basic_s1b_b8k'),
    ('ViT-B-16-quickgelu', 'dfn2b'),
    ('ViT-B-16-quickgelu', 'metaclip_400m'),
    ('ViT-B-16-quickgelu', 'metaclip_fullcc'),
    ('ViT-L-14', 'laion400m_e31'),
    ('ViT-L-14', 'laion400m_e32'),
    ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
    ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'),
    ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'),
    ('ViT-L-14', 'commonpool_xl_s13b_b90k'),
    ('ViT-L-14-quickgelu', 'metaclip_400m'),
    ('ViT-L-14-quickgelu', 'metaclip_fullcc'),
    ('ViT-H-14-quickgelu', 'metaclip_fullcc'),
    ('ViT-H-14', 'metaclip_altogether'),
    ('ViT-g-14', 'laion2b_s12b_b42k'),
    ('ViT-g-14', 'laion2b_s34b_b88k'),
    ('ViT-bigG-14-quickgelu', 'metaclip_fullcc'),
]"""

model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

# Create output directory if it doesn't exist
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Load Hugging Face CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_embeddings(input_dir, output_dir, model, preprocess):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Generating embeddings for {root[(len(PREPROCESSED_DIR) + 1):]}"):
            if file.lower().endswith('.pt'):
                input_file_path = os.path.join(root, file)
                relative_dir = os.path.relpath(root, input_dir)
                output_file_dir = os.path.join(output_dir, relative_dir)

                base_name = os.path.splitext(file)[0]
                output_file_path = os.path.join(output_file_dir, f"{base_name}_embedding.pt")

                if os.path.exists(output_file_path):
                    continue

                Path(output_file_dir).mkdir(parents=True, exist_ok=True)

                try:
                    image_tensor = torch.load(input_file_path, weights_only=True).to(device)
                    image = to_pil_image(image_tensor)
                    image_input = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_embedding = model.encode_image(image_input)
                        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                    torch.save(image_embedding.cpu(), output_file_path)
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

def main():
    for model_name, pretrained in model_configs:
        print(f"Loading model: {model_name} ({pretrained})")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model = model.to(device)
            model.eval()

            embeddings_model_dir = os.path.join(EMBEDDINGS_DIR, f"{model_name.replace('/', '_')}_{pretrained}")
            Path(embeddings_model_dir).mkdir(parents=True, exist_ok=True)

            for preprocessed_folder in tqdm(os.listdir(PREPROCESSED_DIR), desc="Processing preprocessed folders"):
                preprocessed_folder_path = os.path.join(PREPROCESSED_DIR, preprocessed_folder)
                if os.path.isdir(preprocessed_folder_path):
                    output_folder_path = os.path.join(embeddings_model_dir, preprocessed_folder)
                    generate_embeddings(preprocessed_folder_path, output_folder_path, model, preprocess)

        except Exception as e:
            print(f"Failed to load model {model_name} ({pretrained}): {e}")

if __name__ == "__main__":
    main()