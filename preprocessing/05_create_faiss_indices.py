import os
from pathlib import Path
import torch
import faiss
import numpy as np
from tqdm import tqdm

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
INDICES_DIR = os.path.join(BASE_DIR, "faiss_indices")

# Create output directory if it doesn't exist
Path(INDICES_DIR).mkdir(parents=True, exist_ok=True)

# Load embeddings from .pt files, gather and convert them to numpy arrays
def load_embeddings(input_dir):
    embeddings = []
    paths = []
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Loading embeddings for {root[(len(EMBEDDINGS_DIR) + 1):]}"):
            if file.lower().endswith('_embedding.pt'):
                input_file_path = os.path.join(root, file)
                try:
                    # Load the embedding
                    embedding = torch.load(input_file_path, weights_only=True)
                    embeddings.append(embedding.cpu().numpy())  # Convert to NumPy array
                    paths.append(input_file_path)
                except Exception as e:
                    print(f"Error loading {input_file_path}: {e}")
    return embeddings, paths

# Create a FAISS index from embeddings, choosing index type based on number of embeddings
def create_faiss_index(embeddings):
    # num_embeddings is the number of vectors, dim is their dimensionality
    num_embeddings, dim = embeddings.shape
    if num_embeddings <= 50_000:
        print(f"Creating IndexFlatIP for {num_embeddings} embeddings")
        index = faiss.IndexFlatIP(dim)
    else:
        print(f"Creating IndexIVFFlat for {num_embeddings} embeddings")
        nlist = int(8 * (num_embeddings ** 0.5))  # 8 * sqrt(n) as a heuristic
        print(f"Using nlist = {nlist}")
        quantizer = faiss.IndexFlatIP(dim)  # Coarse quantizer
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    index.add(embeddings)
    return index

# Manage paths, load embeddings, create and save FAISS index for a given folder
def process_embedding_folder(model, folder_name):
    embedding_folder_path = os.path.join(EMBEDDINGS_DIR, model, folder_name)
    index_output_dir = os.path.join(INDICES_DIR, model, f"{folder_name}")
    
    print(embedding_folder_path)
    print(index_output_dir)
    # Skip processing if the index output file already exists
    if os.path.exists(os.path.join(index_output_dir, "faiss_index.index")):
        print(f"Skipping already processed folder: {folder_name}")
        return

    Path(index_output_dir).mkdir(parents=True, exist_ok=True)

    embeddings, paths = load_embeddings(embedding_folder_path)
    if embeddings:
        embeddings = np.vstack(embeddings).astype('float32')
        index = create_faiss_index(embeddings)
        faiss.write_index(index, os.path.join(index_output_dir, "faiss_index.index"))
        print(f"FAISS index saved at {index_output_dir}/faiss_index.index")

        np.save(os.path.join(index_output_dir, "embedding_paths.npy"), paths)
        print(f"Paths saved to {index_output_dir}/embedding_paths.npy")
    else:
        print(f"No embeddings found in {embedding_folder_path}")

# Loop over all model folders and their rosbags to process embeddings
def main():
    # List all model-specific embedding folders
    for model_folder in tqdm(os.listdir(EMBEDDINGS_DIR), desc="Processing model folders"):
        model_folder_path = os.path.join(EMBEDDINGS_DIR, model_folder)
        if os.path.isdir(model_folder_path):
            for rosbag in os.listdir(model_folder_path):
                process_embedding_folder(model_folder, rosbag)

if __name__ == "__main__":
    main()