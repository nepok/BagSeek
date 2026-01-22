"""
Configuration management for the rosbag processing pipeline.

This module provides a Config class that loads settings from environment variables
and .env files, with sensible defaults for all configuration values.
"""
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv


class Config:
    """
    Configuration class for the rosbag processing pipeline.
    
    Loads configuration from environment variables (via .env file).
    All paths are converted to Path objects. Relative paths are combined with BASE.
    
    Attributes:
        base_dir: Base directory for all relative paths (from BASE)
        
        # Input/Source
        rosbags_dir: Directory containing rosbag files to process (BASE + ROSBAGS)
        
        # Frontend Output
        representative_previews_dir: Directory for representative preview images (BASE + REPRESENTATIVE_PREVIEWS)
        positional_lookup_table_path: Path to positional lookup table JSON (BASE + POSITIONAL_LOOKUP_TABLE)
        canvases_file_path: Path to canvases configuration file (BASE + CANVASES_FILE)
        
        # Metadata Output
        lookup_tables_dir: Directory for metadata lookup tables (BASE + LOOKUP_TABLES)
        topics_dir: Directory for topic metadata (BASE + TOPICS)
        
        # Processed Output
        adjacent_similarities_dir: Directory for similarity analysis results (BASE + ADJACENT_SIMILARITIES)
        embeddings_dir: Directory for embedding data (BASE + EMBEDDINGS)
        
        # Extracted Data
        images_dir: Directory for extracted images (BASE + IMAGES)
        odom_dir: Directory for extracted odometry data (BASE + ODOM)
        pointclouds_dir: Directory for extracted point clouds (BASE + POINTCLOUDS)
        positions_dir: Directory for extracted position data (BASE + POSITIONS)
        videos_dir: Directory for extracted videos (BASE + VIDEOS)
        
        # Export
        export_dir: Directory for exported data (BASE + EXPORT)
        
        # Models
        open_clip_models_dir: Directory for OpenCLIP model cache (OPEN_CLIP_MODELS)
        other_models_dir: Directory for other models (OTHER_MODELS)
    """
    
    def __init__(self):
        """
        Initialize configuration from environment variables.
        
        Reads from environment variables (loaded from .env file if present).
        All relative paths are combined with BASE to create full paths.
        """

        self.base_dir = Path(os.getenv("BASE"))

        # Input/Source paths
        self.rosbags_dir = Path(os.getenv("ROSBAGS"))
        
        # Frontend output paths
        self.image_topic_previews_dir = Path(os.getenv("BASE") + os.getenv("IMAGE_TOPIC_PREVIEWS"))
        self.positional_lookup_table_path = Path(os.getenv("BASE") + os.getenv("POSITIONAL_LOOKUP_TABLE"))
        self.canvases_file_path = Path(os.getenv("BASE") + os.getenv("CANVASES_FILE"))
        
        # Metadata output paths
        self.lookup_tables_dir = Path(os.getenv("BASE") + os.getenv("LOOKUP_TABLES"))
        self.topics_dir = Path(os.getenv("BASE") + os.getenv("TOPICS"))
        
        # Processed output paths
        self.adjacent_similarities_dir = Path(os.getenv("BASE") + os.getenv("ADJACENT_SIMILARITIES"))
        self.embeddings_dir = Path(os.getenv("BASE") + os.getenv("EMBEDDINGS"))
        
        # Extracted data paths
        self.images_dir = Path(os.getenv("BASE") + os.getenv("IMAGES"))
        self.odom_dir = Path(os.getenv("BASE") + os.getenv("ODOM"))
        self.pointclouds_dir = Path(os.getenv("BASE") + os.getenv("POINTCLOUDS"))
        self.positions_dir = Path(os.getenv("BASE") + os.getenv("POSITIONS"))
        self.videos_dir = Path(os.getenv("BASE") + os.getenv("VIDEOS"))
        
        # Export path
        self.export_dir = Path(os.getenv("BASE") + os.getenv("EXPORT"))
        
        # Model configuration
        self.open_clip_models_dir = Path(os.getenv("BASE") + os.getenv("OPEN_CLIP_MODELS"))
        self.other_models_dir = Path(os.getenv("BASE") + os.getenv("OTHER_MODELS"))
        
        # Processing configuration (with defaults)
        self.positional_grid_resolution= float(os.getenv("POSITIONAL_GRID_RESOLUTION", "0.0001"))
        self.open_clip_models = os.getenv("OPEN_CLIP_MODELS")
        self.other_models = os.getenv("OTHER_MODELS")
        self.shard_size = int(os.getenv("SHARD_SIZE", "100000"))
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Check that BASE is set and valid
        if self.base_dir is None:
            raise ValueError("BASE must be set in .env file")
        
        if not self.base_dir.exists():
            raise ValueError(f"BASE directory does not exist: {self.base_dir}")
        
        # Check that rosbags_dir is set
        if self.rosbags_dir is None:
            raise ValueError("ROSBAGS must be set in .env file")
        
        # Validate processing configuration
        if self.positional_grid_resolution <= 0:
            raise ValueError(f"POSITIONAL_GRID_RESOLUTION must be positive, got {self.positional_grid_resolution}")
        
        if self.shard_size <= 0:
            raise ValueError(f"SHARD_SIZE must be positive, got {self.shard_size}")
    
    @classmethod
    def load_config(cls, env_file: str = None) -> "Config":
        """
        Load configuration from .env file and environment variables.
        
        This is the recommended way to create a Config instance. It will:
        1. Load variables from the .env file (if it exists)
        2. Allow environment variables to override .env values
        3. Fall back to defaults if neither is provided
        
        Args:
            env_file: Path to the .env file (default: None, which uses ../../.env relative to this file)
        
        Returns:
            Config: Configured instance
        
        Example:
            >>> config = Config.load_config()
            >>> print(config.rosbags_dir)
            PosixPath('/path/to/rosbags')
        """
        
        # Default to the .env file in /mnt/data/bagseek/
        if env_file is None:
            # Get path relative to this config.py file
            config_dir = Path(__file__).parent
            env_file = config_dir / ".." / ".." / ".env"
        
        # Load .env file if it exists (doesn't raise error if missing)
        load_dotenv(env_file)
        
        return cls()
    
    def __repr__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"Config(\n"
            f"  # Base\n"
            f"  base_dir={self.base_dir},\n"
            f"\n"
            f"  # Input/Source\n"
            f"  rosbags_dir={self.rosbags_dir},\n"
            f"\n"
            f"  # Frontend Output\n"
            f"  image_topic_previews_dir={self.image_topic_previews_dir},\n"
            f"  positional_lookup_table_path={self.positional_lookup_table_path},\n"
            f"  canvases_file_path={self.canvases_file_path},\n"
            f"\n"
            f"  # Metadata Output\n"
            f"  lookup_tables_dir={self.lookup_tables_dir},\n"
            f"  topics_dir={self.topics_dir},\n"
            f"\n"
            f"  # Processed Output\n"
            f"  adjacent_similarities_dir={self.adjacent_similarities_dir},\n"
            f"  embeddings_dir={self.embeddings_dir},\n"
            f"\n"
            f"  # Extracted Data\n"
            f"  images_dir={self.images_dir},\n"
            f"  odom_dir={self.odom_dir},\n"
            f"  pointclouds_dir={self.pointclouds_dir},\n"
            f"  positions_dir={self.positions_dir},\n"
            f"  videos_dir={self.videos_dir},\n"
            f"\n"
            f"  # Processing Configuration\n"
            f"  positional_grid_resolution={self.positional_grid_resolution},\n"
            f"\n"
            f"  # Export\n"
            f"  export_dir={self.export_dir},\n"
            f"\n"
            f"  # Models\n"
            f"  open_clip_models_dir={self.open_clip_models_dir},\n"
            f"  other_models_dir={self.other_models_dir}\n"
            f")"
        )

