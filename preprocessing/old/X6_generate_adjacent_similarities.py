import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Root embedding folder containing model folders
EMBEDDINGS_PER_TOPIC_DIR = os.getenv("EMBEDDINGS_PER_TOPIC_DIR")

# Output folder for saving similarity plots
ADJACENT_SIMILARITIES_DIR = os.getenv("ADJACENT_SIMILARITIES_DIR")
OUTPUT_DIR = ADJACENT_SIMILARITIES_DIR

def load_embeddings(embeddings_path):
    """
    Load and return a list of numpy arrays (embeddings) from all .pt files in the rosbag_path.
    """
    embedding_files = sorted([
        os.path.join(embeddings_path, f)
        for f in os.listdir(embeddings_path)
        if f.endswith(".pt")
    ])
    embeddings = []
    for f in embedding_files:
        try:
            emb = torch.load(f, map_location="cpu")
            if isinstance(emb, torch.Tensor):
                embeddings.append(emb.numpy().squeeze())
            else:
                print(f"⚠️ Skipping {f} – not a torch.Tensor (type: {type(emb)})")
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")
    return embeddings

def compute_adjacent_similarities(embeddings):
    """
    Normalize embeddings and compute cosine similarities between adjacent embeddings.
    Returns a numpy array of similarities.
    """
    if len(embeddings) < 2:
        return None
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    similarities = [
        np.dot(embeddings[i], embeddings[i+1])
        for i in range(len(embeddings) - 1)
    ]
    return np.array(similarities)

def plot_and_save(similarities, model_name, rosbag_name, topic_name):
    """
    Plot similarities and save the image to OUTPUT_DIR/model_name/
    """
    if similarities is None or len(similarities) == 0:
        return

    # Set target aspect ratio and high resolution
    target_aspect_ratio = 112 / 9
    target_width_px = 1000
    target_height_px = int(target_width_px / target_aspect_ratio)

    dpi = 200
    height_inches = target_height_px / dpi
    width_inches = target_width_px / dpi

    if len(similarities) < target_width_px:
        # interpolate to stretch
        resized = np.interp(
            np.linspace(0, len(similarities) - 1, target_width_px),
            np.arange(len(similarities)),
            1 - similarities  # invert here
        )
    else:
        window_size = max(5, len(similarities) // 200)        
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(similarities, kernel, mode='valid')
        resized = 1 - np.interp(
            np.linspace(0, len(smoothed) - 1, target_width_px),
            np.arange(len(smoothed)),
            smoothed
        )

    plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    plt.imshow(resized[np.newaxis, :], aspect='auto', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # plt.colorbar(label='Cosine Similarity')
    # plt.title(f'Adjacent Cosine Similarities\n{rosbag_name}')
    # plt.yticks([])
    # plt.xlabel('Embedding Pair Index')

    SAVE_DIR = os.path.join(OUTPUT_DIR, model_name, rosbag_name, topic_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_filename = f"{topic_name}.png"
    output_path = os.path.join(SAVE_DIR, output_filename)
    plt.savefig(output_path)
    similarity_filename = f"{rosbag_name}.npy"
    similarity_path = os.path.join(SAVE_DIR, similarity_filename)
    np.save(similarity_path, similarities)
    plt.close()

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in os.listdir(EMBEDDINGS_PER_TOPIC_DIR):
        model_path = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model_name)
        if not os.path.isdir(model_path):
            continue

        for rosbag_name in os.listdir(model_path):
            rosbag_path = os.path.join(model_path, rosbag_name)
            if not os.path.isdir(rosbag_path):
                continue

            for topic_name in os.listdir(rosbag_path):
                topic_path = os.path.join(rosbag_path, topic_name)
                if not os.path.isdir(topic_path):
                    continue

                embeddings = load_embeddings(topic_path)
                similarities = compute_adjacent_similarities(embeddings)
                print(f"Processing: {model_name}/{rosbag_name}/{topic_name}")
                plot_and_save(similarities, model_name, rosbag_name, topic_name)

if __name__ == "__main__":
    main()
