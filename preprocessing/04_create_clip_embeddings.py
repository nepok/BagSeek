import os
from pathlib import Path
import torch
import torch.multiprocessing as mp
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import open_clip
import gc
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define constants for paths
BASE_DIR = os.getenv("BASE_DIR")
PREPROCESSED_DIR = os.getenv("PREPROCESSED_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")

model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]


# Create output directory if it doesn't exist
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Load Hugging Face CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate embeddings for all .pt image tensors in input_dir, saving normalized embeddings to output_dir
# Loops over all files in input_dir recursively, skipping if embedding already exists
def generate_embeddings(input_dir, output_dir, model, model_name, preprocess, device):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Generating embeddings for {root[(len(PREPROCESSED_DIR) + 1):]} with Model {model_name}"):
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

# Worker function for per-GPU parallel embedding generation
# Loads model on specific GPU and processes all preprocessed images to generate embeddings
def worker(model_name, pretrained, device_id):
    print(f"Loading model: {model_name} ({pretrained}) on cuda:{device_id}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir="/mnt/data/openclip_cache")
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        embeddings_model_dir = os.path.join(EMBEDDINGS_DIR, f"{model_name.replace('/', '_')}__{pretrained}")
        Path(embeddings_model_dir).mkdir(parents=True, exist_ok=True)

        for preprocess_id in tqdm(os.listdir(PREPROCESSED_DIR), desc=f"[{model_name}] Processing preprocess folders"):
            preprocess_path = os.path.join(PREPROCESSED_DIR, preprocess_id)
            if not os.path.isdir(preprocess_path):
                continue

            for rosbag_folder in os.listdir(preprocess_path):
                rosbag_folder_path = os.path.join(preprocess_path, rosbag_folder)
                if os.path.isdir(rosbag_folder_path):
                    output_folder_path = os.path.join(embeddings_model_dir, rosbag_folder)
                    generate_embeddings(rosbag_folder_path, output_folder_path, model, model_name, preprocess, device)

        del model, preprocess
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Failed to load model {model_name} ({pretrained}): {e}")

# Main function assigns models across available GPUs using multiprocessing for parallel embedding generation
def main():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    processes = []
    for i, (model_name, pretrained) in enumerate(model_configs):
        device_id = i % num_gpus if num_gpus > 0 else "cpu"
        p = mp.Process(target=worker, args=(model_name, pretrained, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()