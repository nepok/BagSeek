import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Root embedding folder containing model folders with shards
EMBEDDINGS = Path(os.getenv("EMBEDDINGS"))

# Output folder for saving similarity plots
ADJACENT_SIMILARITIES = os.getenv("ADJACENT_SIMILARITIES")

def load_embeddings_from_shards_for_topic(manifest_path: Path, shards_dir: Path, topic: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings for a specific topic from shards using the manifest, sorted by timestamp_ns.
    
    Args:
        manifest_path: Path to manifest.parquet file
        shards_dir: Path to shards directory
        topic: Topic name to filter embeddings
    
    Returns:
        Tuple of (embeddings array, manifest DataFrame filtered by topic)
        Embeddings are sorted by timestamp_ns to maintain temporal order
    """
    if not manifest_path.exists():
        print(f"⚠️ Manifest not found: {manifest_path}")
        return np.array([]), pd.DataFrame()
    
    # Read manifest
    manifest = pd.read_parquet(manifest_path)
    
    if manifest.empty:
        return np.array([]), manifest
    
    # Filter by topic
    topic_df = manifest[manifest["topic"] == topic].copy()
    
    if topic_df.empty:
        return np.array([]), topic_df
    
    # Sort by timestamp_ns to maintain temporal order
    topic_df = topic_df.sort_values("timestamp_ns").reset_index(drop=True)
    
    # Pre-load all shards that we'll need (group by shard_id for efficiency)
    shard_cache = {}
    needed_shards = topic_df["shard_id"].unique()
    
    for shard_id in needed_shards:
        shard_path = shards_dir / shard_id
        if not shard_path.exists():
            print(f"⚠️ Shard not found: {shard_path}")
            continue
        
        try:
            shard_arr = np.load(shard_path, mmap_mode="r")
            if shard_arr.dtype != np.float32:
                shard_arr = shard_arr.astype(np.float32, copy=False)
            shard_cache[shard_id] = shard_arr
        except Exception as e:
            print(f"❌ Error loading shard {shard_path}: {e}")
            continue
    
    # Extract embeddings in manifest order (preserving temporal order)
    embeddings_list = []
    for idx, row in topic_df.iterrows():
        shard_id = row["shard_id"]
        row_in_shard = int(row["row_in_shard"])
        
        if shard_id not in shard_cache:
            continue
        
        shard_arr = shard_cache[shard_id]
        
        # Extract single embedding at row_in_shard
        if 0 <= row_in_shard < len(shard_arr):
            embeddings_list.append(shard_arr[row_in_shard].copy())
        else:
            print(f"⚠️ Invalid row_in_shard {row_in_shard} for shard {shard_id} (length={len(shard_arr)})")
    
    if not embeddings_list:
        return np.array([]), topic_df
    
    # Stack embeddings (already in correct order from manifest)
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    
    return embeddings, topic_df

def compute_adjacent_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings and compute cosine similarities between adjacent embeddings.
    Returns a numpy array of similarities.
    """
    if len(embeddings) < 2:
        return None
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms
    
    # Compute dot products between adjacent embeddings
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
    
    return similarities

def plot_and_save(similarities, model_name, rosbag_name, topic_name):
    """
    Plot similarities and save the image to ADJACENT_SIMILARITIES/model_name/rosbag_name/topic_name/
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

    # Replace / with __ for folder name (matching the structure)
    topic_folder = topic_name.replace("/", "__")
    SAVE_DIR = os.path.join(ADJACENT_SIMILARITIES, model_name, rosbag_name, topic_folder)
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_filename = f"{topic_folder}.png"
    output_path = os.path.join(SAVE_DIR, output_filename)
    plt.savefig(output_path)
    similarity_filename = f"{rosbag_name}.npy"
    similarity_path = os.path.join(SAVE_DIR, similarity_filename)
    np.save(similarity_path, similarities)
    plt.close()

def main():
    if not EMBEDDINGS or not EMBEDDINGS.exists():
        print(f"❌ EMBEDDINGS directory not found: {EMBEDDINGS}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(ADJACENT_SIMILARITIES, exist_ok=True)

    # Iterate through model folders
    for model_name in sorted(EMBEDDINGS.iterdir()):
        if not model_name.is_dir():
            continue
        
        model_path = model_name
        
        # Iterate through rosbag folders
        for rosbag_name in sorted(model_path.iterdir()):
            if not rosbag_name.is_dir():
                continue
            
            rosbag_path = rosbag_name
            
            # Check for manifest and shards directory
            manifest_path = rosbag_path / "manifest.parquet"
            shards_dir = rosbag_path / "shards"
            
            if not manifest_path.exists():
                print(f"⚠️ Skipping {model_name.name}/{rosbag_name.name}: manifest not found")
                continue
            
            if not shards_dir.exists() or not shards_dir.is_dir():
                print(f"⚠️ Skipping {model_name.name}/{rosbag_name.name}: shards directory not found")
                continue
            
            # Read manifest to get all topics
            try:
                manifest = pd.read_parquet(manifest_path)
                topics = manifest["topic"].unique()
            except Exception as e:
                print(f"❌ Error reading manifest {manifest_path}: {e}")
                continue
            
            # Process each topic
            for topic in sorted(topics):
                # Replace / with __ for folder name (matching the structure)
                topic_folder = topic.replace("/", "__")
                
                print(f"Processing: {model_name.name}/{rosbag_name.name}/{topic}")
                
                # Load embeddings from shards for this topic (sorted by timestamp_ns)
                embeddings, manifest_ordered = load_embeddings_from_shards_for_topic(manifest_path, shards_dir, topic)
                
                if len(embeddings) == 0:
                    print(f"  ⚠️ No embeddings found for topic {topic}")
                    continue
                
                print(f"  📊 Loaded {len(embeddings)} embeddings from {len(manifest_ordered['shard_id'].unique())} shard(s)")
                
                # Compute pairwise similarities between adjacent embeddings
                similarities = compute_adjacent_similarities(embeddings)
                
                if similarities is None or len(similarities) == 0:
                    print(f"  ⚠️ Could not compute similarities (need at least 2 embeddings)")
                    continue
                
                # Save plot and similarities
                plot_and_save(similarities, model_name.name, rosbag_name.name, topic)
                print(f"  ✅ Saved {len(similarities)} pairwise similarities")

if __name__ == "__main__":
    main()
