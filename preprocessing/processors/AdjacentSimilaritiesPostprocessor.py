"""
Step 6: Adjacent Similarities

Computes similarities between adjacent embeddings.
Operates as postprocessor only.
"""
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..abstract import PostProcessor
from ..utils import CompletionTracker, PipelineLogger, get_logger

if TYPE_CHECKING:
    from ..core import RosbagProcessingContext


class AdjacentSimilaritiesPostprocessor(PostProcessor):
    """
    Compute similarities between adjacent embeddings.
    
    Runs after main pipeline - analyzes embedding shards.
    """
    
    def __init__(self, embeddings_dir: Path, output_dir: Path):
        """
        Initialize adjacent similarities postprocessor.
        
        Args:
            embeddings_dir: Directory containing embedding shards
            output_dir: Directory to write similarity analysis results
        """
        super().__init__("adjacent_similarities")
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir)
    
    def _load_embeddings_from_shards_for_topic(
        self, manifest_path: Path, shards_dir: Path, topic: str
    ) -> Tuple[np.ndarray, pd.DataFrame]:
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
            self.logger.warning(f"Manifest not found: {manifest_path}")
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
                self.logger.warning(f"Shard not found: {shard_path}")
                continue
            
            try:
                shard_arr = np.load(shard_path, mmap_mode="r")
                if shard_arr.dtype != np.float32:
                    shard_arr = shard_arr.astype(np.float32, copy=False)
                shard_cache[shard_id] = shard_arr
            except Exception as e:
                self.logger.error(f"Error loading shard {shard_path}: {e}")
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
                self.logger.warning(
                    f"Invalid row_in_shard {row_in_shard} for shard {shard_id} (length={len(shard_arr)})"
                )
        
        if not embeddings_list:
            return np.array([]), topic_df
        
        # Stack embeddings (already in correct order from manifest)
        embeddings = np.vstack(embeddings_list).astype(np.float32)
        
        return embeddings, topic_df
    
    def _compute_adjacent_similarities(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize embeddings and compute cosine similarities between adjacent embeddings.
        
        Args:
            embeddings: Array of embeddings (N x D)
        
        Returns:
            Array of similarities (N-1) or None if insufficient embeddings
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
    
    def _plot_and_save(
        self, similarities: np.ndarray, model_name: str, rosbag_name: str, topic_name: str
    ) -> Optional[Path]:
        """
        Plot similarities and save the image to output_dir/model_name/rosbag_name/topic_folder/
        
        Args:
            similarities: Array of adjacent similarities
            model_name: Name of the model
            rosbag_name: Name of the rosbag
            topic_name: Name of the topic
        
        Returns:
            Path to the saved PNG file, or None if no plot was saved
        """
        if similarities is None or len(similarities) == 0:
            return None

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
        # Convert rosbag_name string to Path to properly handle slashes in multipart rosbags
        rosbag_path_obj = Path(rosbag_name)
        save_dir = self.output_dir / model_name / rosbag_path_obj / topic_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{topic_folder}.jpg"
        output_path = save_dir / output_filename
        plt.savefig(output_path)
        
        # Use just the leaf name (e.g., "Part_2.npy") instead of full path
        similarity_filename = f"{topic_folder}.npy"
        similarity_path = save_dir / similarity_filename
        np.save(similarity_path, similarities)
        plt.close()
        
        return output_path
    
    def process_rosbag(self, context: "RosbagProcessingContext"):
        """
        Process adjacent similarities for a single rosbag.
        
        Called after all MCAPs for this rosbag have been processed.
        Computes adjacent similarities for all models and topics for this rosbag.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            None
        """
        if not self.embeddings_dir.exists():
            self.logger.error(f"EMBEDDINGS directory not found: {self.embeddings_dir}")
            return None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get rosbag name using relative path (handles multipart rosbags correctly)
        rosbag_name = context.get_relative_path().as_posix()
        
        # Iterate through model folders
        for model_path in sorted(self.embeddings_dir.iterdir()):
            if not model_path.is_dir():
                continue
            
            model_name = model_path.name
            
            # Check if embeddings exist for this rosbag
            rosbag_path = model_path / rosbag_name
            manifest_path = rosbag_path / "manifest.parquet"
            shards_dir = rosbag_path / "shards"
            
            if not manifest_path.exists():
                # No embeddings for this rosbag/model combination, skip silently
                continue
            
            if not shards_dir.exists() or not shards_dir.is_dir():
                self.logger.warning(
                    f"Skipping {model_name}/{rosbag_name}: shards directory not found"
                )
                continue
            
            # Read manifest to get all topics
            try:
                manifest = pd.read_parquet(manifest_path)
                topics = manifest["topic"].unique()
            except Exception as e:
                self.logger.error(f"Error reading manifest {manifest_path}: {e}")
                continue
            
            # Process each topic
            for topic in sorted(topics):
                # Check completion before processing
                completion_key = f"{model_name}/{rosbag_name}/{topic}"
                topic_folder = topic.replace("/", "__")
                # Convert rosbag_name to Path for proper directory structure
                rosbag_path_obj = Path(rosbag_name)
                
                expected_output = self.output_dir / model_name / rosbag_path_obj / topic_folder / f"{topic_folder}.jpg"
                
                # The completion check should only check one file (the PNG), not both. The .npy file is secondary data.
                if self.completion_tracker.is_completed(completion_key, output_path=expected_output):
                    self.logger.processor_skip(f"Model {model_name} with rosbag {rosbag_name} and topic {topic}", "already completed")
                    continue
                
                # Load embeddings from shards for this topic (sorted by timestamp_ns)
                embeddings, manifest_ordered = self._load_embeddings_from_shards_for_topic(
                    manifest_path, shards_dir, topic
                )
                
                if len(embeddings) == 0:
                    self.logger.warning(f"No embeddings found for topic {topic}")
                    continue
                
                self.logger.info(
                    f"Loaded {len(embeddings)} embeddings from "
                    f"{len(manifest_ordered['shard_id'].unique())} shard(s)"
                )
                
                # Compute pairwise similarities between adjacent embeddings
                similarities = self._compute_adjacent_similarities(embeddings)
                
                if similarities is None or len(similarities) == 0:
                    self.logger.warning(
                        f"Could not compute similarities (need at least 2 embeddings)"
                    )
                    continue
                
                # Save plot and similarities
                output_path = self._plot_and_save(similarities, model_name, rosbag_name, topic)
                if output_path:
                    # Mark as completed after successful save
                    self.completion_tracker.mark_completed(completion_key, output_path=output_path)
                    self.logger.success(
                        f"Saved {len(similarities)} pairwise similarities"
                    )
                else:
                    self.logger.warning("Failed to save plot and similarities")
        
        return None

