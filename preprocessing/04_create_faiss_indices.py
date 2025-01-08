import os
from pathlib import Path
import torch
import faiss
import numpy as np
from tqdm import tqdm

# Define constants for paths
BASE_DIR = "/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
INDICES_DIR = os.path.join(BASE_DIR, "faiss_indices")

# Create output directory if it doesn't exist
Path(INDICES_DIR).mkdir(parents=True, exist_ok=True)

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

def create_faiss_index(embeddings):
    """Create a FAISS index from the given embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def process_embedding_folder(folder_name):
    """Process a single embedding folder and create a FAISS index."""
    embedding_folder_path = os.path.join(EMBEDDINGS_DIR, folder_name)
    index_output_dir = os.path.join(INDICES_DIR, folder_name)
    
    # Skip processing if the index output directory already exists
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

def main():
    """Main function to iterate over all embedding folders and process them."""
    for folder_name in tqdm(os.listdir(EMBEDDINGS_DIR), desc="Processing embedding folders"):
        folder_path = os.path.join(EMBEDDINGS_DIR, folder_name)
        if os.path.isdir(folder_path):
            process_embedding_folder(folder_name)

if __name__ == "__main__":
    main()