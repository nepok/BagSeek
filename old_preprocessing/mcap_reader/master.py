#!/usr/bin/env python3
"""
Master Pipeline: Unified ROS2 bag preprocessing

This script processes all MCAPs in a single pass, generating:
1. Lookup table CSVs (one per MCAP)
2. Embedding shards with manifests (per topic, per model, per rosbag)

Usage:
1. Set environment variables: ROSBAGS, LOOKUP_TABLES, TOPICS, EMBEDDINGS
2. Run: python3 master.py
"""

import csv
import gc
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Timezone support
try:
    from zoneinfo import ZoneInfo
    BERLIN_TZ = ZoneInfo("Europe/Berlin")
    USE_PYTZ = False
except ImportError:
    # Fallback for Python < 3.9
    try:
        import pytz
        BERLIN_TZ = pytz.timezone("Europe/Berlin")
        USE_PYTZ = True
    except ImportError:
        # No timezone support available, use local time
        BERLIN_TZ = None
        USE_PYTZ = False
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from mcap.reader import SeekingReader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import open_clip
try:
    from open_clip.transform import image_transform
except ImportError:
    image_transform = None

# =========================
# Setup Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thread lock for tqdm progress bars to prevent output conflicts
# Use RLock (reentrant lock) for tqdm thread safety
tqdm_lock = threading.RLock()
tqdm.set_lock(tqdm_lock)

# =========================
# Timestamp Helper Functions
# =========================

def get_timestamp_berlin() -> str:
    """Get current timestamp in Berlin timezone with readable format.
    
    Returns:
        Timestamp string in format "YYYY-MM-DD HH:MM:SS" (e.g., "2025-12-09 15:10:38")
    """
    if BERLIN_TZ:
        if USE_PYTZ:
            # pytz timezone - need to localize naive datetime
            now = datetime.now(BERLIN_TZ)
        else:
            # zoneinfo timezone (Python 3.9+)
            now = datetime.now(BERLIN_TZ)
    else:
        # Fallback to local time if no timezone support
        now = datetime.now()
    
    return now.strftime("%Y-%m-%d %H:%M:%S")

# =========================
# Load environment variables
# =========================

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

ROSBAGS = Path(os.getenv("ROSBAGS"))
BASE_STR = os.getenv("BASE")
LOOKUP_TABLES_STR = os.getenv("LOOKUP_TABLES")
TOPICS_STR = os.getenv("TOPICS")
EMBEDDINGS_STR = os.getenv("EMBEDDINGS")
REPRESENTATIVE_PREVIEWS_STR = os.getenv("REPRESENTATIVE_PREVIEWS")
ADJACENT_SIMILARITIES_STR = os.getenv("ADJACENT_SIMILARITIES")
POSITIONAL_LOOKUP_TABLE_STR = os.getenv("POSITIONAL_LOOKUP_TABLE")

LOOKUP_TABLES = Path(BASE_STR + LOOKUP_TABLES_STR)
TOPICS = Path(BASE_STR + TOPICS_STR)
EMBEDDINGS = Path(BASE_STR + EMBEDDINGS_STR) if BASE_STR and EMBEDDINGS_STR else None
REPRESENTATIVE_PREVIEWS = Path(BASE_STR + REPRESENTATIVE_PREVIEWS_STR) if BASE_STR and REPRESENTATIVE_PREVIEWS_STR else None
ADJACENT_SIMILARITIES = Path(BASE_STR + ADJACENT_SIMILARITIES_STR) if BASE_STR and ADJACENT_SIMILARITIES_STR else None
POSITIONAL_LOOKUP_TABLE = Path(BASE_STR + POSITIONAL_LOOKUP_TABLE_STR) if BASE_STR and POSITIONAL_LOOKUP_TABLE_STR else None

# =========================
# Configuration
# =========================

MODE = "single"  # "single", "all", or "multiple"
SINGLE_BAG_NAME = "rosbag2_2025_07_23-12_58_03"  # Only used when MODE = "single"
MULTIPLE_BAG_NAMES = []  # List of rosbag names, only used when MODE = "multiple"
# Example: MULTIPLE_BAG_NAMES = ["rosbag2_2025_07_23-12_58_03", "rosbag2_2025_07_28-07_29_07"]

BATCH_SIZE = 256  # Fallback default batch size (for GPU embedding generation)
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024
BATCH_SIZE_SAFETY_MARGIN = 0.25  # 25% safety margin to avoid OOM errors
BATCH_SIZE_REDUCTION_FACTOR = 0.8  # Use 80% of found batch size for additional safety

# Image processing batch sizes (for CPU RAM)
MIN_IMAGE_BATCH = 500  # Minimum batch size for image conversion/preprocessing
MAX_IMAGE_BATCH = 5000  # Maximum batch size for image conversion/preprocessing (conservative)
IMAGE_BATCH_SAFETY_MARGIN = 0.60  # 60% safety margin for RAM usage (very conservative)
IMAGE_BATCH_SIZE = None  # Will be calculated adaptively based on available RAM

ENABLE_MULTI_GPU = True  # Flag to enable/disable multi-GPU processing
DEFAULT_SHARD_ROWS = 100_000
OUTPUT_DTYPE = np.float32
CACHE_DIR = "/mnt/data/openclip_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPENCLIP_MODELS = [
    #('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    #('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    #('ViT-bigG-14', 'laion2b_s39b_b160k')
]

CUSTOM_MODELS = [
    {
        "name": "ViT-B-16-finetuned(09.10.25)",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/ViT-B-16-finetuned(09.10.25).pt",
        "model_dir_name": "ViT-B-16-finetuned(09.10.25)",
        "batch_size": 256,
        "enabled": True,
    },
    {
        "name": "agriclip",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/agriclip.pt",
        "model_dir_name": "agriclip",
        "batch_size": 256,
        "enabled": True,
    },
]

# =========================
# Utility Functions
# =========================

def extract_mcap_number(mcap_path: Path) -> int:
    """Extract numeric identifier from MCAP filename."""
    name = mcap_path.stem
    parts = name.split('_')
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return 0

def find_mcap_files(rosbag_path: Path) -> List[Path]:
    """Find all MCAP files in a rosbag directory, sorted numerically."""
    return sorted(rosbag_path.glob("*.mcap"), key=extract_mcap_number)

def get_all_topics_from_metadata(rosbag_path: Path) -> Optional[List[str]]:
    """Extract all topic names from metadata.yaml file."""
    metadata_path = rosbag_path / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        topics_with_counts = bag_info.get("topics_with_message_count", [])
        topic_names = []
        for topic_entry in topics_with_counts:
            topic_metadata = topic_entry.get("topic_metadata", {})
            topic_name = topic_metadata.get("name")
            if topic_name:
                topic_names.append(topic_name)
        return sorted(list(set(topic_names))) if topic_names else None
    except Exception:
        return None

def extract_timestamp(topic: str, ros2_msg: object, log_time: int, topic_type_map: Dict[str, str]) -> Tuple[int, str]:
    """Extract timestamp from message log_time."""
    topic_type = topic_type_map.get(topic)
    return log_time, topic_type

def create_alignment_csv(topic_data: Dict[str, List[int]], topic_types: Dict[str, str], csv_path: Path, all_topics: Optional[List[str]] = None):
    """Create alignment CSV file from topic data."""
    if not topic_data:
        return
    
    # Find reference topic (highest frequency)
    ref_topic = max(topic_data.items(), key=lambda x: len(x[1]))[0]
    ref_timestamps = np.array(topic_data[ref_topic], dtype=np.int64)
    
    # Create refined timeline
    if len(ref_timestamps) < 2:
        ref_ts = ref_timestamps
    else:
        diffs = np.diff(ref_timestamps)
        mean_interval = np.mean(diffs)
        refined_interval = mean_interval / 2.0
        ref_start = ref_timestamps[0]
        ref_end = ref_timestamps[-1]
        ref_ts = np.arange(ref_start, ref_end, refined_interval).astype(np.int64)
    
    # Align all topics to reference
    aligned_data: Dict[str, List[Optional[int]]] = {}
    for topic, timestamps in topic_data.items():
        aligned = []
        for ref_time in ref_ts:
            closest = min(timestamps, key=lambda x: abs(x - ref_time))
            if abs(closest - ref_time) < int(1e8):  # 0.1 second tolerance
                aligned.append(closest)
            else:
                aligned.append(None)
        aligned_data[topic] = aligned
    
    # Add missing topics
    if all_topics:
        for topic in all_topics:
            if topic not in aligned_data:
                aligned_data[topic] = [None] * len(ref_ts)
    
    # Write CSV
    topics = all_topics if all_topics else list(aligned_data.keys())
    header = ['Reference Timestamp'] + topics + ['Max Distance']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, ref_time in enumerate(ref_ts):
            row = [int(ref_time)]
            row_values = []
            for topic in topics:
                aligned_ts = aligned_data.get(topic, [None] * len(ref_ts))[i]
                row.append(int(aligned_ts) if aligned_ts is not None else "")
                if aligned_ts is not None:
                    row_values.append(abs(aligned_ts - ref_time))
            max_dist = max(row_values) if row_values else ""
            row.append(int(max_dist) if max_dist != "" else "")
            writer.writerow(row)

class RosbagPipeline(NamedTuple):
    """Pipeline configuration for a single rosbag, tracking what needs to be done."""
    rosbag_path: Path
    csv_path: Path
    topics_json_path: Optional[Path]
    needs_extraction: bool  # Extract data from rosbag (timestamps, images, positional data)
    needs_lookup_tables: bool
    needs_topics_json: bool
    needs_representative_previews: bool
    needs_embeddings: bool
    needs_adjacent_similarities: bool
    needs_positional_data: bool  # Always True if POSITIONAL_LOOKUP_TABLE is configured
    missing_steps: List[str]  # For logging/debugging

def check_rosbag_completion(rosbag_path: Path, csv_path: Path, topics_json_path: Optional[Path]) -> RosbagPipeline:
    """Check all completion statuses for a rosbag and return pipeline configuration.
    
    Returns:
        RosbagPipeline with flags indicating what needs to be done
    """
    rosbag_name = rosbag_path.name
    
    # Check each completion status
    lookup_done, _ = is_lookup_tables_completed(rosbag_path, csv_path)
    topics_done = is_topics_json_completed(rosbag_name, topics_json_path) if topics_json_path else True
    rep_previews_done, _ = is_representative_previews_completed(rosbag_name) if REPRESENTATIVE_PREVIEWS else (True, [])
    positional_done = is_positional_lookup_table_completed(rosbag_name) if POSITIONAL_LOOKUP_TABLE else True
    
    # Check embeddings completion for all models
    all_embeddings_done = True
    all_adj_sim_done = True
    if EMBEDDINGS:
        # Check all OpenCLIP models
        for model_name, pretrained_name in OPENCLIP_MODELS:
            model_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
            if not is_embeddings_completed(rosbag_name, model_id):
                all_embeddings_done = False
                break
            # Check adjacent similarities for this model
            if ADJACENT_SIMILARITIES:
                adj_sim_done, _ = is_adjacent_similarities_completed(rosbag_name, model_id)
                if not adj_sim_done:
                    all_adj_sim_done = False
        
        # Check all custom models
        if all_embeddings_done:
            for config in CUSTOM_MODELS:
                if not config.get("enabled", True):
                    continue
                model_id = config.get("model_dir_name", config["name"])
                if not is_embeddings_completed(rosbag_name, model_id):
                    all_embeddings_done = False
                    break
                # Check adjacent similarities for this model
                if ADJACENT_SIMILARITIES:
                    adj_sim_done, _ = is_adjacent_similarities_completed(rosbag_name, model_id)
                    if not adj_sim_done:
                        all_adj_sim_done = False
    
    # Determine what needs to be done
    # Extract if we need lookup tables, embeddings, or positional data
    needs_extraction = not lookup_done or not all_embeddings_done or (POSITIONAL_LOOKUP_TABLE is not None and not positional_done)
    needs_lookup_tables = not lookup_done
    needs_topics_json = not topics_done
    needs_representative_previews = not rep_previews_done
    needs_embeddings = not all_embeddings_done
    needs_adjacent_similarities = not all_adj_sim_done
    needs_positional_data = POSITIONAL_LOOKUP_TABLE is not None and not positional_done
    
    # Build missing steps list for logging
    missing_steps = []
    if not lookup_done:
        missing_steps.append("lookup_tables")
    if not topics_done:
        missing_steps.append("topics_json")
    if not rep_previews_done:
        missing_steps.append("representative_previews")
    if not all_embeddings_done:
        missing_steps.append("embeddings")
    if not all_adj_sim_done:
        missing_steps.append("adjacent_similarities")
    if not positional_done and POSITIONAL_LOOKUP_TABLE:
        missing_steps.append("positional_lookup_table")
    
    return RosbagPipeline(
        rosbag_path=rosbag_path,
        csv_path=csv_path,
        topics_json_path=topics_json_path,
        needs_extraction=needs_extraction,
        needs_lookup_tables=needs_lookup_tables,
        needs_topics_json=needs_topics_json,
        needs_representative_previews=needs_representative_previews,
        needs_embeddings=needs_embeddings,
        needs_adjacent_similarities=needs_adjacent_similarities,
        needs_positional_data=needs_positional_data,
        missing_steps=missing_steps
    )

def collect_rosbags(rosbags_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Collect all rosbags to process, filtered by MODE.
    
    Only includes directories that contain at least one MCAP file.
    """
    rosbag_list = []
    all_rosbag_paths = [p for p in rosbags_dir.iterdir() if p.is_dir()]
    
    # Filter out directories without MCAP files (e.g., "EXPORTED" folder)
    valid_rosbag_paths = []
    for rosbag_path in all_rosbag_paths:
        mcap_files = find_mcap_files(rosbag_path)
        if mcap_files:
            valid_rosbag_paths.append(rosbag_path)
        else:
            logger.debug(f"  ⏭️  Skipping {rosbag_path.name}: no MCAP files found")
    
    # Filter based on MODE
    if MODE == "single":
        if not SINGLE_BAG_NAME:
            logger.error("MODE is 'single' but SINGLE_BAG_NAME is not set")
            raise SystemExit("MODE is 'single' but SINGLE_BAG_NAME is not set")
        target_path = rosbags_dir / SINGLE_BAG_NAME
        if target_path not in valid_rosbag_paths:
            logger.error(f"Rosbag '{SINGLE_BAG_NAME}' not found in {rosbags_dir} or has no MCAP files")
            raise SystemExit(f"Rosbag '{SINGLE_BAG_NAME}' not found in {rosbags_dir} or has no MCAP files")
        all_rosbag_paths = [target_path]
    elif MODE == "multiple":
        if not MULTIPLE_BAG_NAMES:
            logger.error("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")
            raise SystemExit("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")
        # Filter to only include rosbags in MULTIPLE_BAG_NAMES
        filtered_paths = []
        for rosbag_path in valid_rosbag_paths:
            if rosbag_path.name in MULTIPLE_BAG_NAMES:
                filtered_paths.append(rosbag_path)
        all_rosbag_paths = filtered_paths
        
        # Warn about missing rosbags
        found_names = {p.name for p in filtered_paths}
        missing = set(MULTIPLE_BAG_NAMES) - found_names
        if missing:
            logger.warning(f"⚠️  Warning: {len(missing)} rosbag(s) not found or have no MCAP files: {sorted(missing)}")
    else:  # MODE == "all"
        all_rosbag_paths = valid_rosbag_paths
    
    # Build rosbag_list from filtered paths
    for rosbag_path in all_rosbag_paths:
        rosbag_name = rosbag_path.name
        parent_name = rosbag_path.parent.name
        
        if parent_name.endswith("_multi_parts"):
            csv_folder = LOOKUP_TABLES / parent_name
            csv_folder.mkdir(parents=True, exist_ok=True)
            csv_path = csv_folder / f"{rosbag_name}.csv"
            topics_folder = TOPICS / parent_name
            topics_folder.mkdir(parents=True, exist_ok=True)
            topics_json_path = topics_folder / f"{rosbag_name}.json"
        else:
            csv_path = LOOKUP_TABLES / f"{rosbag_name}.csv"
            topics_json_path = TOPICS / f"{rosbag_name}.json"
        
        rosbag_list.append((rosbag_path, csv_path, topics_json_path))
    
    return rosbag_list

# =========================
# Image Processing Functions
# =========================

def convert_bytes_to_pil(image_bytes: bytes) -> Optional[Image.Image]:
    """Convert raw image bytes to PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception:
        return None

def extract_mcap_identifier(mcap_path: Path) -> str:
    """Extract MCAP identifier from filename."""
    name = mcap_path.stem
    parts = name.split('_')
    if parts and parts[-1].isdigit():
        return parts[-1]
    return "0"

# =========================
# Preprocessing Functions
# =========================

_PREPROCESS_CACHE: Dict[Tuple[str, str], object] = {}
_CUSTOM_PREPROCESS_CACHE: Dict[str, object] = {}

def get_image_resolution_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract image resolution from a custom CLIP model checkpoint."""
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict: Dict[str, torch.Tensor]
        
        if isinstance(raw, dict):
            candidate = None
            for key in ("state_dict", "model_state_dict", "model", "ema_state_dict"):
                if key in raw and isinstance(raw[key], dict):
                    candidate = raw[key]
                    break
            state_dict = candidate if candidate else {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
        else:
            state_dict = raw
        
        for prefix in ("module.", "model.", "clip."):
            if all(key.startswith(prefix) for key in state_dict.keys()):
                state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
        
        if "visual.conv1.weight" not in state_dict:
            suspected_prefixes = {key.split(".", 1)[0] for key in state_dict.keys()}
            for prefix in suspected_prefixes:
                candidate = {key[len(prefix) + 1:]: value for key, value in state_dict.items() if key.startswith(prefix + ".")}
                if "visual.conv1.weight" in candidate:
                    state_dict = candidate
                    break
        
        vision_patch_size = state_dict["visual.conv1.weight"].shape[2]
        num_pos_tokens = state_dict["visual.positional_embedding"].shape[0]
        grid_size = int((num_pos_tokens - 1) ** 0.5)
        return vision_patch_size * grid_size
    except Exception:
        return 224

def get_custom_preprocess_transform(checkpoint_path: Path) -> transforms.Compose:
    """Get preprocessing transform for a custom CLIP model."""
    checkpoint_str = str(checkpoint_path)
    if checkpoint_str in _CUSTOM_PREPROCESS_CACHE:
        return _CUSTOM_PREPROCESS_CACHE[checkpoint_str]
    
    image_resolution = get_image_resolution_from_checkpoint(checkpoint_path)
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    preprocess = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    _CUSTOM_PREPROCESS_CACHE[checkpoint_str] = preprocess
    return preprocess

def get_preprocess_transform(model_name: str, pretrained_name: str) -> object:
    """Get preprocessing transform for OpenCLIP model."""
    key = (model_name, pretrained_name)
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key]
    
    preprocess = None
    try:
        cfg = open_clip.get_preprocess_cfg(pretrained_name)
        if image_transform is not None:
            preprocess = image_transform(cfg, is_train=False)
    except Exception:
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_name, cache_dir=CACHE_DIR
        )
    
    _PREPROCESS_CACHE[key] = preprocess
    return preprocess

def identify_unique_preprocessings() -> Dict[str, List[str]]:
    """Identify unique preprocessing transforms and group models by them.
    
    Returns:
        Dictionary mapping preprocess_str to list of model_ids
    """
    start_time = time.time()
    
    # Suppress verbose logging from open_clip and other libraries during model loading
    # Save original levels
    openclip_logger = logging.getLogger('open_clip')
    original_openclip_level = openclip_logger.level
    openclip_logger.setLevel(logging.WARNING)
    
    # Suppress INFO level from root logger temporarily (but keep our logger working)
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    # Only suppress if it's currently INFO or lower
    if root_logger.level <= logging.INFO:
        root_logger.setLevel(logging.WARNING)
    
    try:
        preprocess_groups: Dict[str, List[str]] = {}
        
        # Process OpenCLIP models
        for model_name, pretrained_name in OPENCLIP_MODELS:
            try:
                preprocess = get_preprocess_transform(model_name, pretrained_name)
                preprocess_str = str(preprocess)
                model_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
                
                if preprocess_str not in preprocess_groups:
                    preprocess_groups[preprocess_str] = []
                if model_id not in preprocess_groups[preprocess_str]:
                    preprocess_groups[preprocess_str].append(model_id)
            except Exception as e:
                logger.warning(f"   ⚠️  Could not get preprocessing for {model_name}/{pretrained_name}: {e}")
        
        # Process custom models
        for config in CUSTOM_MODELS:
            if not config.get("enabled", True):
                continue
            checkpoint_path = Path(config["checkpoint"])
            if not checkpoint_path.exists():
                logger.warning(f"   ⚠️  Checkpoint not found: {checkpoint_path} (skipping {config['name']})")
                continue
            try:
                preprocess = get_custom_preprocess_transform(checkpoint_path)
                preprocess_str = str(preprocess)
                model_id = config.get("model_dir_name", config["name"])
                
                if preprocess_str not in preprocess_groups:
                    preprocess_groups[preprocess_str] = []
                if model_id not in preprocess_groups[preprocess_str]:
                    preprocess_groups[preprocess_str].append(model_id)
            except Exception as e:
                logger.warning(f"   ⚠️  Could not get preprocessing for {config['name']}: {e}")
        
        elapsed = time.time() - start_time
        total_models = sum(len(models) for models in preprocess_groups.values())
        logger.info(f"   ✓ {len(preprocess_groups)} unique group(s) | {total_models} model(s) ({elapsed:.1f}s)\n")
        
        return preprocess_groups
    finally:
        # Restore original logging levels
        openclip_logger.setLevel(original_openclip_level)
        root_logger.setLevel(original_root_level)

# =========================
# CLIP Model Classes (for custom models)
# =========================

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__(normalized_shape, eps=eps)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def _apply_attn(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        result, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask, need_weights=False)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._apply_attn(x)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        grid = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_embedding = self.class_embedding.to(x.dtype)
        batch_class = class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, vision_width: int, vision_layers: int, vision_patch_size: int,
                 image_resolution: int, context_length: int, vocab_size: int, transformer_width: int,
                 transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, attn_mask=self.build_text_mask(context_length))
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    @staticmethod
    def build_text_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self) -> torch.dtype:
        return self.positional_embedding.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))

def count_layers(prefix: str, state_dict: Dict[str, torch.Tensor]) -> int:
    """Count layers in state dict by prefix."""
    layers = set()
    prefix_parts = prefix.split(".")
    index = len(prefix_parts)
    for key in state_dict.keys():
        if key.startswith(prefix):
            layer_id = key.split(".")[index]
            layers.add(int(layer_id))
    return len(layers)

def build_model_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> CLIP:
    """Build CLIP model from state dict."""
    dtype = state_dict["visual.class_embedding"].dtype
    if dtype == torch.float16 and not torch.cuda.is_available():
        state_dict = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        dtype = torch.float32

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_patch_size = state_dict["visual.conv1.weight"].shape[2]
    num_pos_tokens = state_dict["visual.positional_embedding"].shape[0]
    grid_size = int((num_pos_tokens - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    vision_layers = count_layers("visual.transformer.resblocks", state_dict)

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = count_layers("transformer.resblocks", state_dict)

    model = CLIP(embed_dim, vision_width, vision_layers, vision_patch_size, image_resolution,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

    if dtype == torch.float16:
        model = model.half()
    model.load_state_dict(state_dict, strict=True)
    return model.eval()

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Strip prefix from state dict keys."""
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict

def _first_matching_key(container: dict, keys: List[str]) -> Optional[dict]:
    """Find first matching key in container."""
    for key in keys:
        if key in container and isinstance(container[key], dict):
            return container[key]
    return None

def load_openclip_model(model_name: str, pretrained_name: str, device: str = "cuda") -> nn.Module:
    """Load OpenCLIP model."""
    # Suppress verbose logging from open_clip during model loading
    openclip_logger = logging.getLogger('open_clip')
    original_openclip_level = openclip_logger.level
    openclip_logger.setLevel(logging.WARNING)
    
    # Also suppress root logger INFO messages
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    if root_logger.level <= logging.INFO:
        root_logger.setLevel(logging.WARNING)
    
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name, device=device, cache_dir=CACHE_DIR)
        model.eval()
        return model
    finally:
        # Restore original logging levels
        openclip_logger.setLevel(original_openclip_level)
        root_logger.setLevel(original_root_level)

def load_custom_model(checkpoint_path: Path, device: str = "cuda") -> nn.Module:
    """Load custom CLIP model from checkpoint."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict: Dict[str, torch.Tensor]
    
    if isinstance(raw, dict):
        candidate = _first_matching_key(raw, ["state_dict", "model_state_dict", "model", "ema_state_dict"])
        state_dict = candidate if candidate else {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    else:
        state_dict = raw
    
    for prefix in ("module.", "model.", "clip."):
        state_dict = _strip_prefix(state_dict, prefix)
    
    if "visual.class_embedding" not in state_dict:
        suspected_prefixes = {key.split(".", 1)[0] for key in state_dict.keys()}
        for prefix in suspected_prefixes:
            candidate = _strip_prefix(state_dict, f"{prefix}.")
            if "visual.class_embedding" in candidate:
                state_dict = candidate
                break
    
    if "visual.class_embedding" not in state_dict:
        raise KeyError("Checkpoint does not contain CLIP visual weights under expected keys.")
    
    model = build_model_from_state_dict(state_dict)
    model.to(torch.device(device))
    return model

def cleanup_model_and_gpu(model: Optional[nn.Module], device: str = "cuda"):
    """Clean up model and GPU memory."""
    if model is not None:
        del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

def calculate_optimal_batch_size(model: nn.Module, device: str, sample_input: torch.Tensor, 
                                min_batch: int = None, max_batch: int = None, 
                                safety_margin: float = None) -> int:
    """Calculate optimal batch size based on available GPU memory.
    
    Uses binary search to find the largest batch size that fits in available GPU memory.
    
    Args:
        model: The model to test batch sizes with
        device: Device string (e.g., "cuda:0" or "cuda")
        sample_input: Sample input tensor (single item, will be batched)
        min_batch: Minimum batch size to try (defaults to MIN_BATCH_SIZE)
        max_batch: Maximum batch size to try (defaults to MAX_BATCH_SIZE)
        safety_margin: Safety margin as fraction (defaults to BATCH_SIZE_SAFETY_MARGIN)
    
    Returns:
        Optimal batch size that fits in GPU memory
    """
    if min_batch is None:
        min_batch = MIN_BATCH_SIZE
    if max_batch is None:
        max_batch = MAX_BATCH_SIZE
    if safety_margin is None:
        safety_margin = BATCH_SIZE_SAFETY_MARGIN
    
    # If not using CUDA, return default batch size
    if not device.startswith("cuda"):
        logger.debug(f"      Non-CUDA device {device}, using default batch size {BATCH_SIZE}")
        return BATCH_SIZE
    
    # Get device ID
    if ":" in device:
        device_id = int(device.split(":")[1])
    else:
        device_id = 0
    
    device_obj = torch.device(device)
    
    # Get available GPU memory
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        available_mem = free_mem * (1 - safety_margin)  # Reserve safety margin
        logger.debug(f"      GPU {device_id}: {free_mem / 1024**3:.2f} GB free, "
                    f"using {available_mem / 1024**3:.2f} GB (with {safety_margin*100:.0f}% margin)")
    except Exception as e:
        logger.warning(f"      Could not get GPU memory info for {device}: {e}, using default batch size")
        return BATCH_SIZE
    
    # Clear memory aggressively at the start
    torch.cuda.empty_cache()
    gc.collect()
    
    # Binary search for optimal batch size
    best_batch = min_batch
    left, right = min_batch, max_batch
    
    model.eval()
    
    while left <= right:
        mid = (left + right) // 2
        
        try:
            # Clear cache before test
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test with batch size = mid
            test_batch = torch.stack([sample_input] * mid).to(device_obj)
            
            with torch.no_grad():
                # Try forward pass
                if hasattr(model, 'encode_image'):
                    _ = model.encode_image(test_batch)
                elif hasattr(model, 'visual'):
                    _ = model.visual(test_batch.type(model.dtype) if hasattr(model, 'dtype') else test_batch)
                else:
                    _ = model(test_batch)
            
            # If successful, try larger batch
            best_batch = mid
            left = mid + 1
            
            # Clean up aggressively
            del test_batch
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Batch too large, try smaller
                right = mid - 1
                # Clear cache and run garbage collection on OOM
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # Other error, use current best and break
                logger.warning(f"      Error during batch size test: {e}")
                break
        except Exception as e:
            logger.warning(f"      Unexpected error during batch size test: {e}")
            break
    
    # Apply reduction factor for additional safety
    final_batch = max(min_batch, int(best_batch * BATCH_SIZE_REDUCTION_FACTOR))
    
    # Final memory cleanup after calculation
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"      Calculated optimal batch size: {best_batch} -> {final_batch} (reduced by {BATCH_SIZE_REDUCTION_FACTOR*100:.0f}% for safety, GPU {device_id})")
    return final_batch

def calculate_adaptive_image_batch_size(total_images: int, avg_image_size_mb: float = 0.5) -> int:
    """Calculate adaptive batch size for image processing based on available system RAM.
    
    This function determines a safe batch size for loading, converting, and preprocessing
    images without causing OOM errors. It accounts for multiple copies of data in memory
    (raw bytes, PIL images, and preprocessed tensors).
    
    Args:
        total_images: Total number of images to process
        avg_image_size_mb: Average size of a single image in MB (default: 0.5 MB for compressed images)
    
    Returns:
        Adaptive batch size that fits safely in available RAM
    """
    try:
        # Get system memory info
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)
        total_ram_gb = mem.total / (1024**3)
        used_ram_gb = mem.used / (1024**3)
        
        # Apply aggressive safety margin to available RAM
        usable_ram_gb = available_ram_gb * (1 - IMAGE_BATCH_SAFETY_MARGIN)
        
        # Estimate memory needed per image (VERY CONSERVATIVE)
        # Account for:
        # - Raw bytes: avg_image_size_mb (~0.5 MB compressed)
        # - PIL Image: ~8-10x compressed size (uncompressed RGB + overhead)
        # - Preprocessed tensor: ~1.5 MB per image (float32 + transformations)
        # - System overhead and fragmentation: 2x multiplier
        # Total: ~0.5 * 10 + 1.5 = ~6.5 MB per image, then 2x for safety = 13 MB
        memory_multiplier = 13.0  # Very conservative estimate
        memory_per_image_mb = avg_image_size_mb * memory_multiplier
        
        # Calculate batch size based on available RAM
        # Convert GB to MB: usable_ram_gb * 1024
        calculated_batch = int((usable_ram_gb * 1024) / memory_per_image_mb)
        
        # Apply min/max bounds (MAX_IMAGE_BATCH is now 5000, much safer)
        batch_size = max(MIN_IMAGE_BATCH, min(calculated_batch, MAX_IMAGE_BATCH))
        
        # Don't exceed total number of images
        batch_size = min(batch_size, total_images)
        
        logger.info(f"      💾 RAM: {available_ram_gb:.1f} GB available / {total_ram_gb:.1f} GB total ({used_ram_gb:.1f} GB used)")
        logger.info(f"      📊 Calculated image batch size: {calculated_batch:,} -> {batch_size:,} images/batch (conservative)")
        logger.info(f"         Estimated memory per batch: {(batch_size * memory_per_image_mb / 1024):.2f} GB")
        logger.info(f"         Safety margin: {IMAGE_BATCH_SAFETY_MARGIN*100:.0f}% of available RAM reserved")
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"      Could not calculate adaptive batch size: {e}, using default MIN_IMAGE_BATCH")
        return MIN_IMAGE_BATCH

def calculate_adaptive_tensor_batch_size(total_tensors: int, tensor_size_mb: float = 1.5) -> int:
    """Calculate adaptive batch size for loading cached preprocessed tensors.
    
    This is different from image batch size because:
    - Tensors are already preprocessed (no PIL conversion overhead)
    - Tensors are smaller than raw images in memory
    - Only constraint is RAM for holding tensors before GPU processing
    
    Args:
        total_tensors: Total number of preprocessed tensors to load
        tensor_size_mb: Average size of a preprocessed tensor in MB (default: 1.5 MB for float32)
    
    Returns:
        Adaptive batch size for loading cached tensors
    """
    try:
        # Get system memory info
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)
        
        # Use more aggressive memory allocation since tensors are smaller
        # and we're not doing any heavy processing (just loading from disk)
        tensor_safety_margin = 0.40  # 40% safety margin (less conservative than image processing)
        usable_ram_gb = available_ram_gb * (1 - tensor_safety_margin)
        
        # Estimate memory needed per tensor
        # - Preprocessed tensor: tensor_size_mb (~1.5 MB for float32)
        # - Small overhead for metadata: +0.1 MB
        # Total: ~1.6 MB per tensor
        memory_multiplier = 1.1  # Minimal overhead
        memory_per_tensor_mb = tensor_size_mb * memory_multiplier
        
        # Calculate batch size based on available RAM
        calculated_batch = int((usable_ram_gb * 1024) / memory_per_tensor_mb)
        
        # Apply bounds - allow larger batches for tensors than images
        min_tensor_batch = 5000
        max_tensor_batch = 50000  # Can be much larger than image batches
        batch_size = max(min_tensor_batch, min(calculated_batch, max_tensor_batch))
        
        # Don't exceed total number of tensors
        batch_size = min(batch_size, total_tensors)
        
        logger.info(f"      💾 Tensor batch size: {calculated_batch:,} -> {batch_size:,} tensors/batch")
        logger.info(f"         Estimated memory per batch: {(batch_size * memory_per_tensor_mb / 1024):.2f} GB")
        logger.info(f"         (Larger than image batches - tensors are smaller in memory)")
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"      Could not calculate tensor batch size: {e}, using default 10000")
        return 10000

# =========================
# Completion Tracking Functions
# =========================

# =========================
# Completion File Helper Functions
# =========================

def get_parent_folder_name(completion_file: Path) -> str:
    """Get parent folder name for completion.json structure."""
    return completion_file.parent.name

def migrate_old_completion_format(old_data: Dict, completion_file: Path) -> Dict:
    """Migrate old completion.json format to new unified format.
    
    Args:
        old_data: Old format data (may have 'completed' list or other structures)
        completion_file: Path to completion.json file
        
    Returns:
        New format data: {parent_folder_name: {"rosbags": {...}}}
    """
    parent_name = get_parent_folder_name(completion_file)
    new_data = {parent_name: {"rosbags": {}}}
    
    # Handle old format with 'completed' list
    if 'completed' in old_data:
        for item in old_data.get('completed', []):
            if isinstance(item, dict):
                rosbag_name = item.get("rosbag")
                if rosbag_name:
                    # For metadata format, check nested lookup_tables/topics_json
                    if "lookup_tables" in item or "topics_json" in item:
                        # This is metadata format - extract status from nested structures
                        lookup_status = item.get("lookup_tables", {}).get("status", "success")
                        topics_status = item.get("topics_json", {}).get("status", "success")
                        status = "success" if (lookup_status == "success" and topics_status == "success") else "failed"
                        completed_at = item.get("lookup_tables", {}).get("completed_at") or item.get("topics_json", {}).get("completed_at")
                    else:
                        # Regular format
                        status = item.get("status", "success")
                        completed_at = item.get("completed_at")
                    
                    errors = item.get("errors", [])
                    
                    rosbag_entry = {
                        "status": status,
                        "completed_at": completed_at or get_timestamp_berlin()
                    }
                    if errors:
                        rosbag_entry["errors"] = errors
                    
                    new_data[parent_name]["rosbags"][rosbag_name] = rosbag_entry
    
    # Handle old GPS/positional format
    elif 'positional_lookup_table' in old_data or 'gps_lookup_table' in old_data:
        lookup_key = 'positional_lookup_table' if 'positional_lookup_table' in old_data else 'gps_lookup_table'
        lookup_data = old_data.get(lookup_key, {})
        rosbags = lookup_data.get("rosbags", {})
        
        for rosbag_name, rosbag_info in rosbags.items():
            if isinstance(rosbag_info, dict):
                status = rosbag_info.get("status", "success")
                completed_at = rosbag_info.get("completed_at")
                errors = rosbag_info.get("errors", [])
                
                rosbag_entry = {
                    "status": status,
                    "completed_at": completed_at or datetime.now().isoformat()
                }
                if errors:
                    rosbag_entry["errors"] = errors
                
                new_data[parent_name]["rosbags"][rosbag_name] = rosbag_entry
    
    return new_data

def load_unified_completion(completion_file: Path) -> Dict[str, Dict]:
    """Load completion data in unified format, migrating if needed.
    
    Returns:
        Dict mapping rosbag_name to {status, completed_at, errors?}
    """
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
        
        # Check if already in new format
        parent_name = get_parent_folder_name(completion_file)
        if parent_name in data and "rosbags" in data[parent_name]:
            # New format
            return data[parent_name]["rosbags"]
        
        # Old format - migrate
        new_data = migrate_old_completion_format(data, completion_file)
        # Save migrated format
        with open(completion_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        return new_data.get(parent_name, {}).get("rosbags", {})
    except Exception as e:
        logger.warning(f"Error loading completion file {completion_file}: {e}")
        return {}

def save_unified_completion(completion_file: Path, rosbag_data: Dict[str, Dict]):
    """Save completion data in unified format.
    
    Args:
        completion_file: Path to completion.json file
        rosbag_data: Dict mapping rosbag_name to {status, completed_at, errors?}
    """
    parent_name = get_parent_folder_name(completion_file)
    
    data = {
        parent_name: {
            "rosbags": rosbag_data
        }
    }
    
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    with open(completion_file, 'w') as f:
        json.dump(data, f, indent=2)

# =========================
# Metadata Completion Functions (Split)
# =========================

def get_lookup_tables_completion_file() -> Path:
    """Get completion file path for lookup tables."""
    LOOKUP_TABLES.mkdir(parents=True, exist_ok=True)
    return LOOKUP_TABLES / "completion.json"

def get_topics_completion_file() -> Path:
    """Get completion file path for topics JSON."""
    TOPICS.mkdir(parents=True, exist_ok=True)
    return TOPICS / "completion.json"

def load_lookup_tables_completion() -> Dict[str, Dict]:
    """Load lookup tables completion tracking."""
    completion_file = get_lookup_tables_completion_file()
    return load_unified_completion(completion_file)

def load_topics_completion() -> Dict[str, Dict]:
    """Load topics JSON completion tracking."""
    completion_file = get_topics_completion_file()
    return load_unified_completion(completion_file)

def save_lookup_tables_completion(rosbag_data: Dict[str, Dict]):
    """Save lookup tables completion tracking."""
    completion_file = get_lookup_tables_completion_file()
    save_unified_completion(completion_file, rosbag_data)

def save_topics_completion(rosbag_data: Dict[str, Dict]):
    """Save topics JSON completion tracking."""
    completion_file = get_topics_completion_file()
    save_unified_completion(completion_file, rosbag_data)

def get_embedding_completion_file(model_id: str) -> Path:
    """Get completion file path for a specific model's embeddings."""
    if not EMBEDDINGS:
        raise ValueError("EMBEDDINGS not configured")
    model_dir = EMBEDDINGS / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "completion.json"

# Legacy function for backward compatibility - now uses split files
def load_metadata_completion() -> Dict[str, Dict]:
    """Load metadata completion tracking (legacy - combines lookup_tables and topics)."""
    lookup_data = load_lookup_tables_completion()
    topics_data = load_topics_completion()
    
    # Combine both (for backward compatibility with old code that expects combined structure)
    result = {}
    all_rosbags = set(lookup_data.keys()) | set(topics_data.keys())
    for rosbag_name in all_rosbags:
        result[rosbag_name] = {
            "lookup_tables": lookup_data.get(rosbag_name, {}),
            "topics_json": topics_data.get(rosbag_name, {}),
            "errors": []
        }
        # Combine errors if any
        lookup_errors = lookup_data.get(rosbag_name, {}).get("errors", [])
        topics_errors = topics_data.get(rosbag_name, {}).get("errors", [])
        if lookup_errors or topics_errors:
            result[rosbag_name]["errors"] = lookup_errors + topics_errors
    
    return result

def load_embedding_completion(model_id: str) -> Dict[str, Dict]:
    """Load embedding completion tracking for a specific model."""
    completion_file = get_embedding_completion_file(model_id)
    return load_unified_completion(completion_file)

# Legacy function for backward compatibility - now uses split files
def save_metadata_completion(completed: Dict[str, Dict]):
    """Save metadata completion tracking (legacy - splits into lookup_tables and topics)."""
    lookup_data = {}
    topics_data = {}
    
    for rosbag_name, info in completed.items():
        # Extract lookup_tables info
        lookup_info = info.get("lookup_tables", {})
        if lookup_info or info.get("status") == "success":
            lookup_data[rosbag_name] = {
                "status": lookup_info.get("status", "success"),
                "completed_at": lookup_info.get("completed_at") or get_timestamp_berlin()
            }
            if lookup_info.get("errors"):
                lookup_data[rosbag_name]["errors"] = lookup_info["errors"]
        
        # Extract topics_json info
        topics_info = info.get("topics_json", {})
        if topics_info or info.get("status") == "success":
            topics_data[rosbag_name] = {
                "status": topics_info.get("status", "success"),
                "completed_at": topics_info.get("completed_at") or get_timestamp_berlin()
            }
            if topics_info.get("errors"):
                topics_data[rosbag_name]["errors"] = topics_info["errors"]
    
    save_lookup_tables_completion(lookup_data)
    save_topics_completion(topics_data)

def save_embedding_completion(model_id: str, completed: Dict[str, Dict]):
    """Save embedding completion tracking for a specific model."""
    completion_file = get_embedding_completion_file(model_id)
    # Flatten: remove "steps" nested structure, just keep status, completed_at, errors
    flattened = {}
    for rosbag_name, info in completed.items():
        flattened[rosbag_name] = {
            "status": info.get("status", "success"),
            "completed_at": info.get("completed_at") or get_timestamp_berlin()
        }
        if info.get("errors"):
            flattened[rosbag_name]["errors"] = info["errors"]
    
    save_unified_completion(completion_file, flattened)

def verify_lookup_tables_exist(rosbag_path: Path, csv_path: Path) -> Tuple[bool, List[int]]:
    """Verify if lookup table CSV files exist and are populated.
    
    Returns:
        Tuple of (exists: bool, mcap_numbers: List[int])
    """
    rosbag_name = rosbag_path.name
    parent_name = rosbag_path.parent.name
    
    # Determine CSV directory structure
    if parent_name.endswith("_multi_parts"):
        csv_mcap_dir = csv_path.parent / parent_name / rosbag_name
    else:
        csv_mcap_dir = csv_path.parent / rosbag_name
    
    if not csv_mcap_dir.exists():
        return False, []
    
    # Find all MCAP files to check against
    mcap_files = find_mcap_files(rosbag_path)
    expected_mcap_numbers = [extract_mcap_number(mcap) for mcap in mcap_files]
    
    # Check if CSVs exist for all MCAPs
    found_mcap_numbers = []
    for mcap_number in expected_mcap_numbers:
        csv_file = csv_mcap_dir / f"{mcap_number}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 0:
            found_mcap_numbers.append(mcap_number)
    
    # Consider complete if we have CSVs for all expected MCAPs
    if len(found_mcap_numbers) == len(expected_mcap_numbers) and len(found_mcap_numbers) > 0:
        return True, found_mcap_numbers
    
    return False, found_mcap_numbers

def verify_topics_json_exists(topics_json_path: Optional[Path]) -> bool:
    """Verify if topics JSON file exists and is populated."""
    if not topics_json_path:
        return True  # Not required
    
    if not topics_json_path.exists():
        return False
    
    try:
        # Check file size and valid JSON
        if topics_json_path.stat().st_size == 0:
            return False
        
        with open(topics_json_path, 'r') as f:
            data = json.load(f)
            if "topics" in data and "types" in data:
                return len(data.get("topics", [])) > 0
        return False
    except Exception:
        return False

def verify_embeddings_exist(rosbag_name: str, model_id: str) -> bool:
    """Verify if embedding shards and manifest exist and are populated."""
    if not EMBEDDINGS:
        return False
    
    model_dir = EMBEDDINGS / model_id / rosbag_name
    shards_dir = model_dir / "shards"
    manifest_path = model_dir / "manifest.parquet"
    
    # Check if manifest exists and is valid
    if not manifest_path.exists():
        return False
    
    try:
        # Check manifest is not empty
        if manifest_path.stat().st_size == 0:
            return False
        
        # Try to read manifest
        manifest_df = pd.read_parquet(manifest_path)
        if manifest_df.empty:
            return False
        
        # Check if shards referenced in manifest exist
        if "shard_id" in manifest_df.columns:
            shard_ids = manifest_df["shard_id"].unique()
            if len(shard_ids) == 0:
                return False
            
            # Check if shards directory exists
            if not shards_dir.exists():
                return False
            
            # Verify at least some shards exist (check first few to avoid checking all)
            shards_checked = 0
            shards_found = 0
            for shard_id in shard_ids[:10]:  # Check first 10 shards
                shard_path = shards_dir / shard_id
                if shard_path.exists() and shard_path.stat().st_size > 0:
                    shards_found += 1
                shards_checked += 1
            
            # If we checked shards and none exist, it's incomplete
            if shards_checked > 0 and shards_found == 0:
                return False
            
            # If we found some shards, assume it's complete (could check all but expensive)
            if shards_found > 0:
                return True
        
        # If no shard_id column, check if shards directory has files
        if not shards_dir.exists():
            return False
        
        shard_files = list(shards_dir.glob("shard-*.npy"))
        if len(shard_files) == 0:
            return False
        
        # Check if at least one shard is not empty
        for shard_file in shard_files[:5]:  # Check first 5
            if shard_file.stat().st_size > 0:
                return True
        
        return False
    except Exception as e:
        logger.debug(f"Error verifying embeddings for {model_id}/{rosbag_name}: {e}")
        return False

def is_lookup_tables_completed(rosbag_path: Path, csv_path: Path) -> Tuple[bool, Optional[List[int]]]:
    """Check if lookup tables are completed (via completion.json or file verification).
    
    Returns:
        Tuple of (is_completed: bool, mcap_numbers: Optional[List[int]])
    """
    rosbag_name = rosbag_path.name
    
    # First check completion.json
    completed = load_lookup_tables_completion()
    if rosbag_name in completed:
        rosbag_info = completed[rosbag_name]
        if rosbag_info.get("status") == "success":
            # Try to get mcap_numbers from errors or use empty list
            mcap_numbers = []  # We don't store mcap_numbers in unified format, verify files instead
            logger.debug(f"  ✓ Lookup tables found in completion.json for {rosbag_name}")
            return True, mcap_numbers
    
    # If not in completion.json, verify file structure
    logger.debug(f"  🔍 Checking file structure for lookup tables: {rosbag_name}")
    exists, mcap_numbers = verify_lookup_tables_exist(rosbag_path, csv_path)
    if exists:
        # Files exist but no completion.json - create it
        logger.info(f"  📋 Found existing lookup tables for {rosbag_name} (no completion.json), creating completion.json")
        mark_lookup_tables_completed(rosbag_name, mcap_numbers)
        return True, mcap_numbers
    
    logger.debug(f"  ⚠️  Lookup tables not found for {rosbag_name}, will process")
    return False, None

def is_topics_json_completed(rosbag_name: str, topics_json_path: Optional[Path]) -> bool:
    """Check if topics JSON is completed (via completion.json or file verification)."""
    # First check completion.json
    completed = load_topics_completion()
    if rosbag_name in completed:
        rosbag_info = completed[rosbag_name]
        if rosbag_info.get("status") == "success":
            logger.debug(f"  ✓ Topics JSON found in completion.json for {rosbag_name}")
            return True
    
    # If not in completion.json, verify file structure
    logger.debug(f"  🔍 Checking file structure for topics JSON: {rosbag_name}")
    if verify_topics_json_exists(topics_json_path):
        # File exists but no completion.json - create it
        logger.info(f"  📋 Found existing topics JSON for {rosbag_name} (no completion.json), creating completion.json")
        mark_topics_json_completed(rosbag_name)
        return True
    
    logger.debug(f"  ⚠️  Topics JSON not found for {rosbag_name}, will process")
    return False

def is_embeddings_completed(rosbag_name: str, model_id: str) -> bool:
    """Check if embeddings are completed (via completion.json or file verification)."""
    # First check completion.json
    completed = load_embedding_completion(model_id)
    if rosbag_name in completed:
        if completed[rosbag_name].get("status") == "success":
            logger.debug(f"  ✓ Embeddings found in completion.json for {model_id}/{rosbag_name}")
            return True
    
    # If not in completion.json, verify file structure
    logger.debug(f"  🔍 Checking file structure for embeddings: {model_id}/{rosbag_name}")
    if verify_embeddings_exist(rosbag_name, model_id):
        # Files exist but no completion.json - create it
        logger.info(f"  📋 Found existing embeddings for {model_id}/{rosbag_name} (no completion.json), creating completion.json")
        mark_embeddings_completed(rosbag_name, model_id, "embedding_generation", "success")
        mark_embeddings_completed(rosbag_name, model_id, "shard_creation", "success")
        return True
    
    logger.debug(f"  ⚠️  Embeddings not found for {model_id}/{rosbag_name}, will process")
    return False

def mark_lookup_tables_completed(rosbag_name: str, mcap_numbers: List[int], error: Optional[str] = None):
    """Mark lookup tables as completed."""
    completed = load_lookup_tables_completion()
    timestamp = get_timestamp_berlin()
    
    rosbag_entry = {
        "status": "failed" if error else "success",
        "completed_at": timestamp
    }
    
    if error:
        rosbag_entry["errors"] = [{"error": error, "timestamp": timestamp}]
    
    completed[rosbag_name] = rosbag_entry
    save_lookup_tables_completion(completed)

def mark_topics_json_completed(rosbag_name: str, error: Optional[str] = None):
    """Mark topics JSON as completed."""
    completed = load_topics_completion()
    timestamp = get_timestamp_berlin()
    
    rosbag_entry = {
        "status": "failed" if error else "success",
        "completed_at": timestamp
    }
    
    if error:
        rosbag_entry["errors"] = [{"error": error, "timestamp": timestamp}]
    
    completed[rosbag_name] = rosbag_entry
    save_topics_completion(completed)

def get_positional_completion_file() -> Path:
    """Get completion file path for positional lookup table."""
    # Store in same directory as POSITIONAL_LOOKUP_TABLE
    if not POSITIONAL_LOOKUP_TABLE:
        raise ValueError("POSITIONAL_LOOKUP_TABLE not configured")
    completion_dir = POSITIONAL_LOOKUP_TABLE.parent
    completion_dir.mkdir(parents=True, exist_ok=True)
    return completion_dir / "completion.json"

def is_positional_lookup_table_completed(rosbag_name: str) -> bool:
    """Check if positional lookup table is completed for a specific rosbag (via completion.json or file verification)."""
    if not POSITIONAL_LOOKUP_TABLE:
        return True  # Not configured, consider it "completed"
    
    # First check completion.json
    completion_file = get_positional_completion_file()
    completed = load_unified_completion(completion_file)
    if rosbag_name in completed:
        rosbag_info = completed[rosbag_name]
        if rosbag_info.get("status") == "success":
            logger.debug(f"  ✓ Positional lookup table found in completion.json for {rosbag_name}")
            return True
    
    # If not in completion.json, verify file structure
    logger.debug(f"  🔍 Checking file structure for positional lookup table: {rosbag_name}")
    if POSITIONAL_LOOKUP_TABLE.exists() and POSITIONAL_LOOKUP_TABLE.stat().st_size > 0:
        try:
            with open(POSITIONAL_LOOKUP_TABLE, 'r', encoding='utf-8') as f:
                lookup_data = json.load(f)
                if rosbag_name in lookup_data:
                    # Rosbag exists in file but no completion.json - create it
                    logger.info(f"  📋 Found existing positional lookup table entry for {rosbag_name} (no completion.json), creating completion.json")
                    mark_positional_lookup_table_completed(rosbag_names=[rosbag_name])
                    return True
        except Exception as e:
            logger.debug(f"  ⚠️  Could not check positional lookup table file: {e}")
    
    logger.debug(f"  ⚠️  Positional lookup table not found for {rosbag_name}, will process")
    return False

def mark_positional_lookup_table_completed(rosbag_names: Optional[List[str]] = None, error: Optional[str] = None):
    """Mark positional lookup table as completed for specific rosbags.
    
    Args:
        rosbag_names: List of rosbag names that were processed. If None, marks global completion.
        error: Optional error message if processing failed.
    """
    completion_file = get_positional_completion_file()
    timestamp = get_timestamp_berlin()
    
    # Load existing data
    completed = load_unified_completion(completion_file)
    
    # Update rosbags if provided
    if rosbag_names:
        for rosbag_name in rosbag_names:
            rosbag_entry = {
                "status": "failed" if error else "success",
                "completed_at": timestamp
            }
            if error:
                rosbag_entry["errors"] = [{"error": error, "timestamp": timestamp}]
            
            completed[rosbag_name] = rosbag_entry
    
    save_unified_completion(completion_file, completed)

def mark_embeddings_completed(rosbag_name: str, model_id: str, step: str, status: str = "success", error: Optional[str] = None):
    """Mark an embedding step as completed (flattened - no steps tracking)."""
    completed = load_embedding_completion(model_id)
    timestamp = get_timestamp_berlin()
    
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "status": "success" if status == "success" else "failed",
            "completed_at": timestamp
        }
    else:
        # Update status - if any step fails, mark as failed
        if status == "failed" or error:
            completed[rosbag_name]["status"] = "failed"
        # Update completed_at if marking as success
        if status == "success" and completed[rosbag_name].get("status") != "failed":
            completed[rosbag_name]["status"] = "success"
            completed[rosbag_name]["completed_at"] = timestamp
    
    if error:
        if "errors" not in completed[rosbag_name]:
            completed[rosbag_name]["errors"] = []
        completed[rosbag_name]["errors"].append({"error": error, "timestamp": timestamp})
    
    save_embedding_completion(model_id, completed)

# =========================
# Representative Previews Completion Tracking
# =========================

def get_representative_previews_completion_file() -> Path:
    """Get completion file path for representative previews."""
    if not REPRESENTATIVE_PREVIEWS:
        raise ValueError("REPRESENTATIVE_PREVIEWS not configured")
    REPRESENTATIVE_PREVIEWS.mkdir(parents=True, exist_ok=True)
    return REPRESENTATIVE_PREVIEWS / "completion.json"

def load_representative_previews_completion() -> Dict[str, Dict]:
    """Load representative previews completion tracking."""
    completion_file = get_representative_previews_completion_file()
    return load_unified_completion(completion_file)

def save_representative_previews_completion(completed: Dict[str, Dict]):
    """Save representative previews completion tracking."""
    completion_file = get_representative_previews_completion_file()
    # Flatten: remove "topics" nested structure
    flattened = {}
    for rosbag_name, info in completed.items():
        flattened[rosbag_name] = {
            "status": info.get("status", "success"),
            "completed_at": info.get("completed_at") or get_timestamp_berlin()
        }
        if info.get("errors"):
            flattened[rosbag_name]["errors"] = info["errors"]
    
    save_unified_completion(completion_file, flattened)

def verify_representative_previews_exist(rosbag_name: str) -> Tuple[bool, List[str]]:
    """Verify that representative preview files exist for a rosbag.
    
    Returns:
        Tuple of (all_exist, list of missing topics)
    """
    if not REPRESENTATIVE_PREVIEWS:
        return False, []
    
    output_dir = REPRESENTATIVE_PREVIEWS / rosbag_name
    if not output_dir.exists():
        return False, []
    
    # Find all collage files
    collage_files = list(output_dir.glob("*_collage.webp"))
    if not collage_files:
        return False, []
    
    # Extract topics from filenames
    found_topics = set()
    for collage_file in collage_files:
        topic_safe = collage_file.stem.replace("_collage", "")
        found_topics.add(topic_safe)
    
    return True, list(found_topics)

def is_representative_previews_completed(rosbag_name: str) -> Tuple[bool, List[str]]:
    """Check if representative previews are completed for a rosbag.
    
    Returns:
        Tuple of (is_completed, list of completed topics)
    """
    completed = load_representative_previews_completion()
    
    if rosbag_name in completed:
        rosbag_info = completed[rosbag_name]
        if rosbag_info.get("status") == "success":
            # Get topics from file system since we flattened the structure
            _, found_topics = verify_representative_previews_exist(rosbag_name)
            return True, found_topics
    
    # Fallback: check file system
    all_exist, found_topics = verify_representative_previews_exist(rosbag_name)
    if all_exist and found_topics:
        # Create completion.json entry if files exist but tracking doesn't
        completed[rosbag_name] = {
            "status": "success",
            "completed_at": get_timestamp_berlin()
        }
        save_representative_previews_completion(completed)
        logger.debug(f"  Created completion.json for representative previews: {rosbag_name} ({len(found_topics)} topics)")
        return True, found_topics
    
    return False, []

def mark_representative_previews_completed(rosbag_name: str, topics: List[str], error: Optional[str] = None):
    """Mark representative previews as completed for a rosbag."""
    completed = load_representative_previews_completion()
    timestamp = get_timestamp_berlin()
    
    rosbag_entry = {
        "status": "failed" if error else "success",
        "completed_at": timestamp
    }
    
    if error:
        rosbag_entry["errors"] = [{"error": error, "timestamp": timestamp}]
    
    completed[rosbag_name] = rosbag_entry
    save_representative_previews_completion(completed)

# =========================
# Adjacent Similarities Completion Tracking
# =========================

def get_adjacent_similarities_completion_file(model_id: str) -> Path:
    """Get completion file path for adjacent similarities for a specific model."""
    if not ADJACENT_SIMILARITIES:
        raise ValueError("ADJACENT_SIMILARITIES not configured")
    model_dir = ADJACENT_SIMILARITIES / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "completion.json"

def load_adjacent_similarities_completion(model_id: str) -> Dict[str, Dict]:
    """Load adjacent similarities completion tracking for a specific model."""
    completion_file = get_adjacent_similarities_completion_file(model_id)
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
            completed_list = data.get('completed', [])
            result = {}
            for item in completed_list:
                if isinstance(item, dict):
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        result[rosbag_name] = {
                            "topics": item.get("topics", {}),
                            "completed_at": item.get("completed_at"),
                            "errors": item.get("errors", [])
                        }
            return result
    except Exception as e:
        logger.warning(f"Error loading adjacent similarities completion for {model_id}: {e}")
        return {}

def save_adjacent_similarities_completion(model_id: str, completed: Dict[str, Dict]):
    """Save adjacent similarities completion tracking for a specific model."""
    completion_file = get_adjacent_similarities_completion_file(model_id)
    # Flatten: remove "topics" nested structure
    flattened = {}
    for rosbag_name, info in completed.items():
        flattened[rosbag_name] = {
            "status": info.get("status", "success"),
            "completed_at": info.get("completed_at") or get_timestamp_berlin()
        }
        if info.get("errors"):
            flattened[rosbag_name]["errors"] = info["errors"]
    
    save_unified_completion(completion_file, flattened)

def verify_adjacent_similarities_exist(rosbag_name: str, model_id: str) -> Tuple[bool, List[str]]:
    """Verify that adjacent similarity files exist for a rosbag/model combination.
    
    Returns:
        Tuple of (all_exist, list of found topics)
    """
    if not ADJACENT_SIMILARITIES:
        return False, []
    
    rosbag_dir = ADJACENT_SIMILARITIES / model_id / rosbag_name
    if not rosbag_dir.exists():
        return False, []
    
    # Find all topic directories
    topic_dirs = [d for d in rosbag_dir.iterdir() if d.is_dir()]
    if not topic_dirs:
        return False, []
    
    found_topics = []
    for topic_dir in topic_dirs:
        topic_safe = topic_dir.name
        # Check for both PNG and NPY files
        png_file = topic_dir / f"{topic_safe}.png"
        npy_file = topic_dir / f"{rosbag_name}.npy"
        if png_file.exists() and npy_file.exists():
            found_topics.append(topic_safe.replace("__", "/"))
    
    return len(found_topics) > 0, found_topics

def is_adjacent_similarities_completed(rosbag_name: str, model_id: str, topic: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Check if adjacent similarities are completed for a rosbag/model combination.
    
    Returns:
        Tuple of (is_completed, list of completed topics)
    """
    completed = load_adjacent_similarities_completion(model_id)
    
    if rosbag_name in completed:
        rosbag_info = completed[rosbag_name]
        if rosbag_info.get("status") == "success":
            # Get topics from file system since we flattened the structure
            _, found_topics = verify_adjacent_similarities_exist(rosbag_name, model_id)
            if topic:
                if topic in found_topics:
                    return True, [topic]
                return False, []
            return True, found_topics
    
    # Fallback: check file system
    all_exist, found_topics = verify_adjacent_similarities_exist(rosbag_name, model_id)
    if all_exist and found_topics:
        if topic and topic not in found_topics:
            return False, []
        if topic:
            found_topics = [topic] if topic in found_topics else []
        
        # Create completion.json entry if files exist but tracking doesn't
        completed[rosbag_name] = {
            "status": "success",
            "completed_at": get_timestamp_berlin()
        }
        save_adjacent_similarities_completion(model_id, completed)
        logger.debug(f"  Created completion.json for adjacent similarities: {model_id}/{rosbag_name} ({len(found_topics)} topics)")
        return True, found_topics
    
    return False, []

def mark_adjacent_similarities_completed(rosbag_name: str, model_id: str, topics: List[str], error: Optional[str] = None):
    """Mark adjacent similarities as completed for a rosbag/model combination."""
    completed = load_adjacent_similarities_completion(model_id)
    timestamp = get_timestamp_berlin()
    
    rosbag_entry = {
        "status": "failed" if error else "success",
        "completed_at": timestamp
    }
    
    if error:
        rosbag_entry["errors"] = [{"error": error, "timestamp": timestamp}]
    
    completed[rosbag_name] = rosbag_entry
    save_adjacent_similarities_completion(model_id, completed)

# =========================
# Adjacent Similarities Functions
# =========================

def compute_adjacent_similarities(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize embeddings and compute cosine similarities between adjacent embeddings.
    
    Args:
        embeddings: numpy array of shape (N, D) where N is number of embeddings and D is dimension
        
    Returns:
        numpy array of similarities (length N-1), or None if less than 2 embeddings
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

def create_similarity_heatmap(similarities: np.ndarray, model_id: str, rosbag_name: str, topic_name: str) -> None:
    """
    Create and save a similarity heatmap image.
    
    Args:
        similarities: numpy array of similarity values
        model_id: Model identifier
        rosbag_name: Name of the rosbag
        topic_name: Name of the topic
    """
    if similarities is None or len(similarities) == 0:
        return
    
    if not ADJACENT_SIMILARITIES:
        logger.warning(f"  ⚠️  ADJACENT_SIMILARITIES not configured, skipping heatmap for {topic_name}")
        return
    
    try:
        # Set target aspect ratio and high resolution
        target_aspect_ratio = 112 / 9
        target_width_px = 1000
        target_height_px = int(target_width_px / target_aspect_ratio)
        
        dpi = 200
        height_inches = target_height_px / dpi
        width_inches = target_width_px / dpi
        
        # Interpolate or smooth based on length
        if len(similarities) < target_width_px:
            # Interpolate to stretch
            resized = np.interp(
                np.linspace(0, len(similarities) - 1, target_width_px),
                np.arange(len(similarities)),
                1 - similarities  # invert here
            )
        else:
            # Smooth then interpolate
            window_size = max(5, len(similarities) // 200)
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(similarities, kernel, mode='valid')
            resized = 1 - np.interp(
                np.linspace(0, len(smoothed) - 1, target_width_px),
                np.arange(len(smoothed)),
                smoothed
            )
        
        # Create figure
        plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        plt.imshow(resized[np.newaxis, :], aspect='auto', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Replace / with __ for folder name (matching the structure)
        topic_folder = topic_name.replace("/", "__")
        save_dir = ADJACENT_SIMILARITIES / model_id / rosbag_name / topic_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        output_filename = f"{topic_folder}.png"
        output_path = save_dir / output_filename
        plt.savefig(output_path)
        plt.close()
        
        # Save NPY
        similarity_filename = f"{rosbag_name}.npy"
        similarity_path = save_dir / similarity_filename
        np.save(similarity_path, similarities)
        
        logger.debug(f"      ✓ {topic_name}: Saved similarity heatmap and data")
    
    except Exception as e:
        logger.error(f"      ❌ {topic_name}: Error creating similarity heatmap: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

# =========================
# Representative Previews Functions
# =========================

def create_collage_from_pil_images(pil_images: List[Image.Image], output_path: Path) -> bool:
    """Create a collage from multiple PIL Images and save to output_path.
    
    Args:
        pil_images: List of PIL Image objects
        output_path: Path to save the collage
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not pil_images:
            logger.warning(f"No images provided for collage at {output_path}")
            return False
        
        # Resize all images to the same height (minimum height)
        heights = [img.height for img in pil_images]
        min_height = min(heights)
        resized_images = [
            img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS)
            for img in pil_images
        ]
        
        # Calculate total width
        total_width = sum(img.width for img in resized_images)
        collage = Image.new('RGB', (total_width, min_height))
        
        # Paste images horizontally
        x_offset = 0
        for img in resized_images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width
        
        # Save as WebP
        collage.save(output_path, 'WEBP')
        return True
    except Exception as e:
        logger.error(f"Error creating collage {output_path}: {e}")
        return False

def create_representative_previews(image_data_list: List[Tuple[str, bytes, int, str]], rosbag_name: str) -> None:
    """Create representative preview collages for each topic.
    
    For each topic, splits images into 8 parts and takes the first image from parts 1-7,
    then stitches them together into a horizontal collage.
    
    Args:
        image_data_list: List of (topic, image_bytes, timestamp_ns, mcap_identifier)
        rosbag_name: Name of the rosbag
    """
    if not REPRESENTATIVE_PREVIEWS:
        logger.warning("  ⚠️  REPRESENTATIVE_PREVIEWS not configured, skipping")
        return
    
    if not image_data_list:
        logger.info("  ⏭️  No image data for representative previews")
        return
    
    # Check if already completed
    is_completed, completed_topics = is_representative_previews_completed(rosbag_name)
    if is_completed:
        logger.info(f"  ⏭️  Representative previews already completed for {rosbag_name} ({len(completed_topics)} topics), skipping")
        return
    
    logger.info(f"  🖼️  Creating representative previews for {rosbag_name}...")
    start_time = time.time()
    
    # Group images by topic
    images_by_topic: Dict[str, List[Tuple[bytes, int, str]]] = defaultdict(list)
    for topic, image_bytes, timestamp_ns, mcap_id in image_data_list:
        images_by_topic[topic].append((image_bytes, timestamp_ns, mcap_id))
    
    output_dir = REPRESENTATIVE_PREVIEWS / rosbag_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_topics = []
    failed_topics = []
    error_messages = []
    
    for topic, image_list in images_by_topic.items():
        topic_safe = topic.replace("/", "__")
        collage_path = output_dir / f"{topic_safe}_collage.webp"
        
        try:
            num_images = len(image_list)
            
            if num_images == 0:
                logger.debug(f"    ⚠️  {topic}: No images")
                continue
            
            # Fence post logic: divide by 8, take indices 1-7 (1-indexed)
            # This gives 7 evenly spaced points
            target_count = min(7, num_images)
            if target_count == 0:
                logger.debug(f"    ⚠️  {topic}: Not enough images (need at least 1)")
                continue
            
            if num_images == 1:
                selected_indices = [0]
            else:
                part_size = num_images / 8.0
                selected_indices = []
                for i in range(1, 8):  # 1 to 7 (1-indexed)
                    position = i * part_size
                    index = int(round(position))
                    # Clamp to valid range
                    index = max(0, min(index, num_images - 1))
                    selected_indices.append(index)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_indices = []
                for idx in selected_indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                
                # If we have fewer than target_count unique indices, pad with last image
                while len(unique_indices) < target_count and num_images > 0:
                    if num_images - 1 not in unique_indices:
                        unique_indices.append(num_images - 1)
                    else:
                        break
                
                selected_indices = unique_indices[:target_count]
            
            # Convert selected image bytes to PIL Images
            selected_pil_images = []
            for idx in selected_indices:
                try:
                    image_bytes = image_list[idx][0]
                    pil_image = convert_bytes_to_pil(image_bytes)
                    if pil_image:
                        selected_pil_images.append(pil_image)
                except Exception as e:
                    logger.debug(f"    ⚠️  {topic}: Error converting image at index {idx}: {e}")
                    continue
            
            if not selected_pil_images:
                logger.warning(f"    ⚠️  {topic}: No valid images after conversion")
                failed_topics.append(topic)
                error_messages.append(f"{topic}: No valid images after conversion")
                continue
            
            # Create collage
            success = create_collage_from_pil_images(selected_pil_images, collage_path)
            if success:
                logger.debug(f"    ✓ {topic}: Created collage with {len(selected_pil_images)} images")
                successful_topics.append(topic)
            else:
                logger.warning(f"    ❌ {topic}: Failed to create collage")
                failed_topics.append(topic)
                error_messages.append(f"{topic}: Failed to create collage")
        
        except Exception as e:
            logger.error(f"    ❌ {topic}: Error creating representative preview: {e}")
            failed_topics.append(topic)
            error_messages.append(f"{topic}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Representative previews: {len(successful_topics)} topic(s) succeeded, {len(failed_topics)} failed in {elapsed:.1f}s")
    
    # Mark completion
    all_topics = successful_topics + failed_topics
    if all_topics:
        error_msg = "; ".join(error_messages) if error_messages else None
        mark_representative_previews_completed(rosbag_name, all_topics, error=error_msg)

# =========================
# Rosbag Processing Functions
# =========================

def extract_data_from_rosbag(rosbag_path: Path, load_image_bytes: bool = True) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[str, str], List, List[Tuple[str, float, float, int, str]]]:
    """Extract timestamp data, image data, and positional positional data from all MCAPs in a rosbag.
    
    Args:
        rosbag_path: Path to the rosbag directory
        load_image_bytes: If True, loads full image bytes (memory intensive).
                         If False, only stores metadata (topic, mcap_path, timestamp, mcap_id) for batched loading.
    
    Returns:
        Tuple of (topic_data_by_mcap, topic_types, image_data_list, positional_data_list)
        topic_data_by_mcap: Dict mapping mcap_number to topic_data Dict
        image_data_list: If load_image_bytes=True: List of (topic, bytes, timestamp_ns, mcap_identifier)
                        If load_image_bytes=False: List of (topic, mcap_path, timestamp_ns, mcap_identifier)
        positional_data_list: List of (topic, lat, lon, timestamp_ns, mcap_identifier) tuples
    """
    start_time = time.time()
    mcap_files = find_mcap_files(rosbag_path)
    
    topic_data_by_mcap: Dict[int, Dict[str, List[int]]] = {}
    topic_types: Dict[str, str] = {}
    image_data_list: List = []  # Type depends on load_image_bytes parameter
    positional_data_list: List[Tuple[str, float, float, int, str]] = []
    total_messages = 0
    total_images = 0
    total_positions = 0
    
    for mcap_path in tqdm(mcap_files, desc=f"  Extracting from MCAPs"):
        mcap_number = extract_mcap_number(mcap_path)
        mcap_identifier = str(mcap_number)
        topic_data: Dict[str, List[int]] = defaultdict(list)
        
        try:
            with open(mcap_path, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()], record_size_limit=8 * 1024 * 1024 * 1024)
                
                # Get topic types from MCAP summary
                topic_type_map: Dict[str, str] = {}
                try:
                    summary = reader.get_summary()
                    channels = summary.channels if summary else {}
                    schemas = summary.schemas if summary else {}
                    for channel_id, channel in channels.items():
                        schema_id = channel.schema_id
                        if schema_id in schemas:
                            schema = schemas[schema_id]
                            topic_type_map[channel.topic] = schema.name
                except Exception:
                    pass
                
                # Process all messages
                mcap_messages = 0
                mcap_images = 0
                mcap_positions = 0
                for schema, channel, message, ros2_msg in reader.iter_decoded_messages(log_time_order=True, reverse=False):
                    topic = channel.topic
                    log_time = message.log_time
                    schema_name = schema.name if schema else None
                    mcap_messages += 1
                    
                    # Extract timestamp for lookup table
                    timestamp, resolved_type = extract_timestamp(topic, ros2_msg, log_time, topic_type_map)
                    topic_data[topic].append(timestamp)
                    topic_types[topic] = resolved_type
                    
                    # Extract image data if image topic
                    if schema_name in ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"):
                        if hasattr(ros2_msg, 'data'):
                            if load_image_bytes:
                                # Load full image bytes (original behavior)
                                image_bytes = bytes(ros2_msg.data)
                                if image_bytes:
                                    image_data_list.append((topic, image_bytes, log_time, mcap_identifier))
                                    mcap_images += 1
                                    total_images += 1
                            else:
                                # Store only metadata for batched loading (memory efficient)
                                image_data_list.append((topic, mcap_path, log_time, mcap_identifier))
                                mcap_images += 1
                                total_images += 1
                    
                    # Extract positional positional data if BESTPOS topic
                    if schema_name == "novatel_oem7_msgs/msg/BESTPOS":
                        if hasattr(ros2_msg, 'lat') and hasattr(ros2_msg, 'lon'):
                            try:
                                lat = float(ros2_msg.lat)
                                lon = float(ros2_msg.lon)
                                # Basic validation (rough bounds for Germany/Europe)
                                if 40.0 <= lat <= 60.0 and 0.0 <= lon <= 20.0:
                                    positional_data_list.append((topic, lat, lon, log_time, mcap_identifier))
                                    mcap_positions += 1
                                    total_positions += 1
                            except (ValueError, TypeError):
                                pass
                
                total_messages += mcap_messages
                if mcap_messages > 0:
                    logger.debug(f"    MCAP {mcap_number}: {mcap_messages:,} messages, {mcap_images:,} images, {mcap_positions:,} positions, {len(topic_data)} topics")
        except Exception as e:
            logger.error(f"    ❌ Error processing MCAP {mcap_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        if topic_data:
            topic_data_by_mcap[mcap_number] = topic_data
    
    elapsed = time.time() - start_time
    logger.info(f"      📊 {total_messages:,} msgs | {total_images:,} images | {total_positions:,} positional | {len(topic_types)} topics ({elapsed:.1f}s)")
    
    return topic_data_by_mcap, topic_types, image_data_list, positional_data_list

def create_lookup_tables(rosbag_path: Path, topic_data_by_mcap: Dict[int, Dict[str, List[int]]], topic_types: Dict[str, str], csv_path: Path, all_topics: Optional[List[str]]):
    """Create lookup table CSV files for each MCAP."""
    rosbag_name = rosbag_path.name
    
    # Check if already completed (via completion.json or file verification)
    is_completed, mcap_numbers = is_lookup_tables_completed(rosbag_path, csv_path)
    if is_completed:
        logger.info(f"  ⏭️  Lookup tables already completed for {rosbag_name}, skipping")
        return
    
    logger.info(f"  📊 Creating lookup tables...")
    start_time = time.time()
    
    parent_name = rosbag_path.parent.name
    
    # Determine CSV directory structure
    if parent_name.endswith("_multi_parts"):
        csv_mcap_dir = csv_path.parent / parent_name / rosbag_name
    else:
        csv_mcap_dir = csv_path.parent / rosbag_name
    
    csv_mcap_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"    Output directory: {csv_mcap_dir}")
    
    # Create one CSV per MCAP
    csvs_created = 0
    mcap_numbers = []
    error = None
    
    try:
        for mcap_number, topic_data in topic_data_by_mcap.items():
            csv_path_final = csv_mcap_dir / f"{mcap_number}.csv"
            create_alignment_csv(topic_data, topic_types, csv_path_final, all_topics)
            csvs_created += 1
            mcap_numbers.append(mcap_number)
            logger.debug(f"    Created CSV for MCAP {mcap_number}: {csv_path_final.name}")
        
        elapsed = time.time() - start_time
        logger.info(f"  ✓ Created {csvs_created} lookup table CSV(s) in {elapsed:.1f}s")
        mark_lookup_tables_completed(rosbag_name, mcap_numbers)
    except Exception as e:
        error = str(e)
        logger.error(f"  ❌ Error creating lookup tables: {e}")
        mark_lookup_tables_completed(rosbag_name, mcap_numbers, error=error)
        raise

# =========================
# positional Lookup Table Functions
# =========================

GRID_RESOLUTION = 0.0001  # Grid size in degrees (~11 meters)

def is_in_excluded_folder(path: Path) -> bool:
    """Check if a path is inside an EXCLUDED folder (case-insensitive)."""
    current = path.resolve().parent
    while current != current.parent:
        if current.name.upper() == "EXCLUDED":
            return True
        current = current.parent
    return False

def is_rosbag_excluded(rosbag_name: str) -> bool:
    """Check if a rosbag is in an EXCLUDED folder under ROSBAGS directory."""
    if ROSBAGS is None:
        return False
    
    # Check if rosbag path exists in ROSBAGS and is in EXCLUDED folder
    rosbag_path_in_rosbags = ROSBAGS / rosbag_name
    if rosbag_path_in_rosbags.exists():
        if is_in_excluded_folder(rosbag_path_in_rosbags):
            return True
    
    # Search for rosbag in ROSBAGS (might be in subdirectory)
    found_rosbag = None
    for rosbag_path in ROSBAGS.rglob(rosbag_name):
        if rosbag_path.is_dir():
            found_rosbag = rosbag_path
            break
    
    if found_rosbag:
        if is_in_excluded_folder(found_rosbag):
            return True
    else:
        # Also check if rosbag name appears in any EXCLUDED path (case-insensitive partial match)
        for excluded_path in ROSBAGS.rglob("*EXCLUDED*"):
            if excluded_path.is_dir():
                # Check if rosbag_name is in this EXCLUDED directory
                for item in excluded_path.iterdir():
                    if item.is_dir() and rosbag_name.lower() in item.name.lower():
                        return True
    
    return False

def round_to_grid(lat: float, lon: float) -> Tuple[float, float]:
    """Round positional coordinates to grid cells."""
    lat_grid = round(lat / GRID_RESOLUTION) * GRID_RESOLUTION
    lon_grid = round(lon / GRID_RESOLUTION) * GRID_RESOLUTION
    return lat_grid, lon_grid

def parse_rosbag_timestamp(rosbag_name: str) -> Optional[datetime]:
    """Parse timestamp from rosbag directory name: rosbag2_YYYY_MM_DD-HH_MM_SS"""
    try:
        # Remove 'rosbag2_' prefix if present
        if rosbag_name.startswith('rosbag2_'):
            timestamp_str = rosbag_name[8:]  # Remove 'rosbag2_' prefix
        else:
            timestamp_str = rosbag_name
        
        # Parse format: YYYY_MM_DD-HH_MM_SS
        return datetime.strptime(timestamp_str, "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        return None

def update_positional_lookup_table(rosbag_name: str, positional_data_list: List[Tuple[str, float, float, int, str]]) -> None:
    """Update positional lookup table JSON with data for a single rosbag.
    
    Args:
        rosbag_name: Name of the rosbag being processed
        positional_data_list: List of (topic, lat, lon, timestamp_ns, mcap_identifier) tuples for this rosbag
    """
    if not POSITIONAL_LOOKUP_TABLE:
        logger.warning("  ⚠️  POSITIONAL_LOOKUP_TABLE not configured, skipping positional lookup table update")
        return
    
    if not positional_data_list:
        logger.debug(f"  ⚠️  No positional data for {rosbag_name}, adding empty entry")
        # Still update the file with empty entry
        positional_data_list = []
    
    start_time = time.time()
    
    # Load existing lookup data if file exists
    lookup_data = {}
    if POSITIONAL_LOOKUP_TABLE.exists() and POSITIONAL_LOOKUP_TABLE.stat().st_size > 0:
        try:
            with open(POSITIONAL_LOOKUP_TABLE, 'r', encoding='utf-8') as f:
                lookup_data = json.load(f)
        except Exception as e:
            logger.warning(f"  ⚠️  Could not load existing table, starting fresh: {e}")
            lookup_data = {}
    
    # Check if rosbag is excluded
    if is_rosbag_excluded(rosbag_name):
        logger.debug(f"  ⏭️  Skipping {rosbag_name}: found in EXCLUDED folder")
        return
    
    # Process positional data for this rosbag
    if not positional_data_list:
        lookup_data[rosbag_name] = {}
    else:
        # Count visits per rounded positional location per MCAP
        location_counts = defaultdict(lambda: defaultdict(int))
        for topic, lat, lon, timestamp_ns, mcap_identifier in positional_data_list:
            # Round coordinates to grid
            lat_grid, lon_grid = round_to_grid(lat, lon)
            lat_lon_key = f"{lat_grid:.6f},{lon_grid:.6f}"
            location_counts[lat_lon_key][mcap_identifier] += 1
        
        # Convert to final structure with total and mcaps dict
        final_location_data = {}
        for lat_lon_key, mcap_counts in location_counts.items():
            total_count = sum(mcap_counts.values())
            final_location_data[lat_lon_key] = {
                "total": total_count,
                "mcaps": dict(mcap_counts)
            }
        
        # Add/update entry for this rosbag
        lookup_data[rosbag_name] = final_location_data
    
    # Write JSON file (entire file, extending/updating the shared file)
    POSITIONAL_LOOKUP_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONAL_LOOKUP_TABLE, 'w', encoding='utf-8') as f:
        json.dump(lookup_data, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    num_locations = len(lookup_data.get(rosbag_name, {}))
    logger.info(f"      ✓ Positional lookup table updated: {rosbag_name} ({num_locations} locations, {elapsed:.1f}s)")

def load_image_batch_from_mcaps(image_refs: List[Tuple[str, Path, int, str]], 
                                start_idx: int, 
                                batch_size: int,
                                rosbag_path: Path) -> List[Tuple[str, bytes, int, str]]:
    """Load a batch of images from MCAPs on-demand based on metadata references.
    
    This function loads only the requested batch of images from disk, avoiding
    loading all images into memory at once. It reopens MCAPs as needed and
    extracts only the images in the requested batch range.
    
    Args:
        image_refs: List of (topic, mcap_path, timestamp_ns, mcap_identifier) metadata
        start_idx: Starting index in image_refs for this batch
        batch_size: Number of images to load
        rosbag_path: Path to the rosbag directory (for relative path resolution)
    
    Returns:
        List of (topic, image_bytes, timestamp_ns, mcap_identifier) for the batch
    """
    end_idx = min(start_idx + batch_size, len(image_refs))
    batch_refs = image_refs[start_idx:end_idx]
    
    if not batch_refs:
        return []
    
    # Group references by MCAP file for efficient loading
    refs_by_mcap: Dict[Path, List[Tuple[str, int, str]]] = defaultdict(list)
    for topic, mcap_path, timestamp_ns, mcap_id in batch_refs:
        refs_by_mcap[mcap_path].append((topic, timestamp_ns, mcap_id))
    
    # Create a lookup for quick matching: (topic, timestamp) -> needed
    # We use just topic and timestamp since we're already iterating within a specific MCAP
    needed_images = {(topic, ts): mcap_id for topic, mcap_path, ts, mcap_id in batch_refs}
    
    # Load images from each MCAP
    image_data_batch = []
    
    for mcap_path, mcap_refs in refs_by_mcap.items():
        try:
            with open(mcap_path, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()], 
                                     record_size_limit=8 * 1024 * 1024 * 1024)
                
                mcap_number = extract_mcap_number(mcap_path)
                mcap_identifier = str(mcap_number)
                
                # Iterate through messages and collect matching images
                for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                    log_time_order=True, reverse=False):
                    schema_name = schema.name if schema else None
                    
                    # Check if this is an image message we need
                    if schema_name in ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"):
                        topic = channel.topic
                        log_time = message.log_time
                        
                        # Check if this image is in our needed set (match by topic and timestamp)
                        if (topic, log_time) in needed_images:
                            if hasattr(ros2_msg, 'data'):
                                image_bytes = bytes(ros2_msg.data)
                                if image_bytes:
                                    # Use the mcap_identifier from our metadata
                                    stored_mcap_id = needed_images[(topic, log_time)]
                                    image_data_batch.append((topic, image_bytes, log_time, stored_mcap_id))
                                    # Remove from needed set to avoid duplicates
                                    del needed_images[(topic, log_time)]
                                    
                                    # Early exit if we've found all images from this MCAP
                                    if not any(stored_mcap_id == mcap_id for mcap_id in needed_images.values()):
                                        break
                                        
        except Exception as e:
            logger.warning(f"      Error loading images from {mcap_path.name}: {e}")
            continue
    
    return image_data_batch

def convert_images_to_pil(image_data_list: List[Tuple[str, bytes, int, str]]) -> List[Tuple[str, Image.Image, int, str]]:
    """Convert raw image bytes to PIL Images.
    
    Returns:
        List of (topic, pil_image, timestamp_ns, mcap_identifier)
    """
    logger.info(f"      🖼️  Converting {len(image_data_list):,} images to PIL format...")
    start_time = time.time()
    
    pil_images = []
    failed = 0
    for topic, image_bytes, timestamp_ns, mcap_id in tqdm(image_data_list, desc="    Converting", leave=False):
        pil_image = convert_bytes_to_pil(image_bytes)
        if pil_image is not None:
            pil_images.append((topic, pil_image, timestamp_ns, mcap_id))
        else:
            failed += 1
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Converted {len(pil_images):,}/{len(image_data_list):,} images ({failed} failed) in {elapsed:.1f}s")
    return pil_images

def preprocess_images(pil_images: List[Tuple[str, Image.Image, int, str]], preprocess_transform) -> List[Tuple[str, torch.Tensor, int, str]]:
    """Preprocess PIL images using the given transform.
    
    Returns:
        List of (topic, preprocessed_tensor, timestamp_ns, mcap_identifier)
    """
    logger.info(f"  🔄 Preprocessing {len(pil_images):,} images...")
    start_time = time.time()
    
    preprocessed = []
    failed = 0
    for topic, pil_image, timestamp_ns, mcap_id in tqdm(pil_images, desc="    Preprocessing", leave=False):
        try:
            tensor = preprocess_transform(pil_image)
            preprocessed.append((topic, tensor, timestamp_ns, mcap_id))
        except Exception as e:
            failed += 1
            if failed <= 5:  # Log first 5 failures
                logger.debug(f"      Failed to preprocess image: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Preprocessed {len(preprocessed):,}/{len(pil_images):,} images ({failed} failed) in {elapsed:.1f}s")
    return preprocessed

def generate_embeddings(preprocessed_tensors: List[Tuple[str, torch.Tensor, int, str]], model: nn.Module, device: str, model_id: str, batch_size: Optional[int] = None, gpu_id: Optional[int] = None, position: Optional[int] = None) -> List[Tuple[str, np.ndarray, int, str]]:
    """Generate embeddings from preprocessed tensors.
    
    Args:
        preprocessed_tensors: List of (topic, tensor, timestamp_ns, mcap_identifier)
        model: Model to use for embedding generation
        device: Device string (e.g., "cuda:0")
        model_id: Model identifier for logging
        batch_size: Optional batch size (defaults to BATCH_SIZE if not provided)
        gpu_id: Optional GPU ID for progress bar positioning and logging
        position: Optional position for tqdm progress bar (for parallel processing)
    
    Returns:
        List of (topic, embedding, timestamp_ns, mcap_identifier)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    gpu_str = f" on GPU {gpu_id}" if gpu_id is not None else ""
    logger.info(f"    🧠 Generating embeddings with {model_id}{gpu_str} (batch_size={batch_size})...")
    start_time = time.time()
    
    embeddings = []
    num_batches = (len(preprocessed_tensors) + batch_size - 1) // batch_size
    
    # Create progress bar with position for parallel processing
    desc = f"      Embedding {model_id}{gpu_str}"
    tqdm_kwargs = {"desc": desc, "total": num_batches, "leave": True}
    if position is not None:
        tqdm_kwargs["position"] = position
    
    pbar = tqdm(range(0, len(preprocessed_tensors), batch_size), **tqdm_kwargs)
    try:
        for i in pbar:
            batch = preprocessed_tensors[i:i + batch_size]
            tensors = torch.stack([t[1] for t in batch]).to(device)
            
            with torch.no_grad():
                if hasattr(model, 'encode_image'):
                    emb = model.encode_image(tensors)
                elif hasattr(model, 'visual'):
                    emb = model.visual(tensors.type(model.dtype) if hasattr(model, 'dtype') else tensors)
                else:
                    emb = model(tensors)
                
                emb = F.normalize(emb, dim=-1)
                emb_np = emb.cpu().numpy().astype(OUTPUT_DTYPE)
            
            for j, (topic, _, timestamp_ns, mcap_id) in enumerate(batch):
                embeddings.append((topic, emb_np[j], timestamp_ns, mcap_id))
    finally:
        # Close progress bar before logging to avoid conflicts
        pbar.close()
    
    elapsed = time.time() - start_time
    embedding_dim = embeddings[0][1].shape[0] if embeddings else 0
    # Use tqdm.write() to avoid conflicts with other progress bars that might still be running
    tqdm.write(f"    ✓ Generated {len(embeddings):,} embeddings (dim={embedding_dim}) in {elapsed:.1f}s for {model_id}{gpu_str}")
    return embeddings

# =========================
# Checkpoint Functions for Batched Processing
# =========================

def get_checkpoint_dir(model_id: str, rosbag_name: str) -> Path:
    """Get the checkpoint directory for a specific model and rosbag."""
    if not EMBEDDINGS:
        return None
    checkpoint_dir = EMBEDDINGS / "checkpoints" / rosbag_name / model_id
    return checkpoint_dir

def save_batch_checkpoint(embeddings: List[Tuple[str, np.ndarray, int, str]], 
                         batch_id: int, 
                         model_id: str, 
                         rosbag_name: str) -> bool:
    """Save embeddings for a single batch to a checkpoint file.
    
    Args:
        embeddings: List of (topic, embedding, timestamp_ns, mcap_identifier)
        batch_id: Batch identifier (0-indexed)
        model_id: Model identifier
        rosbag_name: Rosbag name
    
    Returns:
        True if checkpoint saved successfully, False otherwise
    """
    checkpoint_dir = get_checkpoint_dir(model_id, rosbag_name)
    if not checkpoint_dir:
        return False
    
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings to parquet
        if embeddings:
            # Convert to structured format
            records = []
            for topic, embedding, timestamp_ns, mcap_id in embeddings:
                records.append({
                    "topic": topic,
                    "timestamp_ns": timestamp_ns,
                    "mcap_identifier": mcap_id,
                    "embedding": embedding.tolist()  # Convert numpy array to list for parquet
                })
            
            df = pd.DataFrame(records)
            checkpoint_path = checkpoint_dir / f"batch_{batch_id:05d}.parquet"
            df.to_parquet(checkpoint_path, compression="zstd", index=False)
        
        # Update progress.json
        progress_path = checkpoint_dir / "progress.json"
        progress_data = {"last_completed_batch": batch_id}
        
        # Merge with existing progress if available
        if progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    existing = json.load(f)
                    progress_data.update(existing)
            except Exception:
                pass
        
        progress_data["last_completed_batch"] = batch_id
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.warning(f"      Failed to save batch checkpoint {batch_id} for {model_id}: {e}")
        return False

def load_checkpoint_progress(model_id: str, rosbag_name: str) -> int:
    """Load the last completed batch ID from checkpoint.
    
    Args:
        model_id: Model identifier
        rosbag_name: Rosbag name
    
    Returns:
        Last completed batch ID, or -1 if no checkpoint exists
    """
    checkpoint_dir = get_checkpoint_dir(model_id, rosbag_name)
    if not checkpoint_dir:
        return -1
    
    progress_path = checkpoint_dir / "progress.json"
    if not progress_path.exists():
        return -1
    
    try:
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
            return progress_data.get("last_completed_batch", -1)
    except Exception as e:
        logger.warning(f"      Failed to load checkpoint progress for {model_id}: {e}")
        return -1

def merge_batch_checkpoints(model_id: str, rosbag_name: str) -> Dict[str, List[Tuple[np.ndarray, int, str]]]:
    """Merge all batch checkpoints into a single embeddings dictionary.
    
    Args:
        model_id: Model identifier
        rosbag_name: Rosbag name
    
    Returns:
        Dict mapping topic to list of (embedding, timestamp_ns, mcap_identifier)
    """
    checkpoint_dir = get_checkpoint_dir(model_id, rosbag_name)
    if not checkpoint_dir or not checkpoint_dir.exists():
        return {}
    
    # Find all batch checkpoint files
    batch_files = sorted(checkpoint_dir.glob("batch_*.parquet"))
    
    if not batch_files:
        return {}
    
    logger.info(f"      Merging {len(batch_files)} batch checkpoint(s) for {model_id}...")
    
    # Collect all embeddings by topic
    embeddings_by_topic: Dict[str, List[Tuple[np.ndarray, int, str]]] = defaultdict(list)
    
    for batch_file in batch_files:
        try:
            df = pd.read_parquet(batch_file)
            
            for _, row in df.iterrows():
                topic = row["topic"]
                timestamp_ns = row["timestamp_ns"]
                mcap_id = row["mcap_identifier"]
                embedding_list = row["embedding"]
                
                # Convert list back to numpy array
                embedding = np.array(embedding_list, dtype=OUTPUT_DTYPE)
                
                embeddings_by_topic[topic].append((embedding, timestamp_ns, mcap_id))
                
        except Exception as e:
            logger.warning(f"      Failed to load batch checkpoint {batch_file.name}: {e}")
            continue
    
    total_embeddings = sum(len(embs) for embs in embeddings_by_topic.values())
    logger.info(f"      ✓ Merged {total_embeddings:,} embeddings from {len(batch_files)} batch(es)")
    
    return embeddings_by_topic

def load_embeddings_from_shards(model_id: str, rosbag_name: str) -> Dict[str, List[Tuple[np.ndarray, int, str]]]:
    """Load existing embeddings from npy shards using manifest for adjacency computation.
    
    Args:
        model_id: Model identifier
        rosbag_name: Rosbag name
        
    Returns:
        Dict mapping topic to list of (embedding_array, timestamp_ns, mcap_id) tuples
    """
    if not EMBEDDINGS:
        return {}
    
    model_dir = EMBEDDINGS / model_id / rosbag_name
    manifest_path = model_dir / "manifest.parquet"
    shards_dir = model_dir / "shards"
    
    if not manifest_path.exists():
        logger.warning(f"      ⚠️  Manifest not found for {model_id}/{rosbag_name}")
        return {}
    
    if not shards_dir.exists():
        logger.warning(f"      ⚠️  Shards directory not found for {model_id}/{rosbag_name}")
        return {}
    
    try:
        # Load manifest
        manifest_df = pd.read_parquet(manifest_path)
        
        # Load all shard files into memory
        shard_cache = {}
        for shard_id in manifest_df['shard_id'].unique():
            shard_path = shards_dir / shard_id
            if shard_path.exists():
                shard_cache[shard_id] = np.load(shard_path)
            else:
                logger.warning(f"      ⚠️  Shard file not found: {shard_id}")
        
        # Build embeddings by topic
        embeddings_by_topic: Dict[str, List[Tuple[np.ndarray, int, str]]] = defaultdict(list)
        
        for _, row in manifest_df.iterrows():
            topic = row['topic']
            timestamp_ns = row['timestamp_ns']
            mcap_id = row['mcap_identifier']
            shard_id = row['shard_id']
            row_in_shard = row['row_in_shard']
            
            if shard_id in shard_cache:
                embedding = shard_cache[shard_id][row_in_shard]
                embeddings_by_topic[topic].append((embedding, timestamp_ns, mcap_id))
        
        total_embeddings = sum(len(embs) for embs in embeddings_by_topic.values())
        logger.info(f"        ✓ Loaded {total_embeddings:,} embeddings from {len(shard_cache)} shard(s)")
        
        return dict(embeddings_by_topic)
        
    except Exception as e:
        logger.warning(f"      ⚠️  Error loading embeddings: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}

def cleanup_checkpoints(model_id: str, rosbag_name: str):
    """Clean up checkpoint files after successful completion.
    
    Args:
        model_id: Model identifier
        rosbag_name: Rosbag name
    """
    checkpoint_dir = get_checkpoint_dir(model_id, rosbag_name)
    if not checkpoint_dir or not checkpoint_dir.exists():
        return
    
    try:
        import shutil
        shutil.rmtree(checkpoint_dir)
        logger.debug(f"      Cleaned up checkpoints for {model_id}/{rosbag_name}")
    except Exception as e:
        logger.warning(f"      Failed to cleanup checkpoints for {model_id}: {e}")

# =========================
# Preprocessing Cache Functions for Tensor Reuse
# =========================

def get_preprocessing_cache_dir(rosbag_name: str, preprocess_hash: str) -> Path:
    """Get cache directory for preprocessed tensors.
    
    Args:
        rosbag_name: Name of the rosbag
        preprocess_hash: Hash identifying the preprocessing transform
    
    Returns:
        Path to cache directory, or None if EMBEDDINGS not configured
    """
    if not EMBEDDINGS:
        return None
    cache_dir = EMBEDDINGS / "tmp_preprocessing" / rosbag_name / preprocess_hash
    return cache_dir

def save_preprocessed_batch(preprocessed_tensors: List[Tuple[str, torch.Tensor, int, str]], 
                           batch_idx: int, 
                           rosbag_name: str, 
                           preprocess_hash: str) -> bool:
    """Save preprocessed tensors for a batch to cache.
    
    Args:
        preprocessed_tensors: List of (topic, tensor, timestamp_ns, mcap_identifier)
        batch_idx: Batch identifier (0-indexed)
        rosbag_name: Name of the rosbag
        preprocess_hash: Hash identifying the preprocessing transform
    
    Returns:
        True if saved successfully, False otherwise
    """
    cache_dir = get_preprocessing_cache_dir(rosbag_name, preprocess_hash)
    if not cache_dir:
        return False
    
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"batch_{batch_idx:05d}.pt"
        
        # Convert to format suitable for torch.save
        batch_data = {
            'tensors': [t[1] for t in preprocessed_tensors],  # Just the tensors
            'metadata': [(t[0], t[2], t[3]) for t in preprocessed_tensors]  # (topic, timestamp, mcap_id)
        }
        torch.save(batch_data, cache_file)
        return True
        
    except Exception as e:
        logger.warning(f"      Failed to cache batch {batch_idx}: {e}")
        return False

def load_preprocessed_batch(batch_idx: int, 
                           rosbag_name: str, 
                           preprocess_hash: str) -> Optional[List[Tuple[str, torch.Tensor, int, str]]]:
    """Load preprocessed tensors for a batch from cache.
    
    Args:
        batch_idx: Batch identifier (0-indexed)
        rosbag_name: Name of the rosbag
        preprocess_hash: Hash identifying the preprocessing transform
    
    Returns:
        List of (topic, tensor, timestamp_ns, mcap_identifier), or None if not cached
    """
    cache_dir = get_preprocessing_cache_dir(rosbag_name, preprocess_hash)
    if not cache_dir:
        return None
    
    cache_file = cache_dir / f"batch_{batch_idx:05d}.pt"
    if not cache_file.exists():
        return None
    
    try:
        batch_data = torch.load(cache_file, map_location='cpu')
        # Reconstruct the list format
        preprocessed_tensors = []
        for tensor, (topic, timestamp_ns, mcap_id) in zip(batch_data['tensors'], batch_data['metadata']):
            preprocessed_tensors.append((topic, tensor, timestamp_ns, mcap_id))
        return preprocessed_tensors
        
    except Exception as e:
        logger.warning(f"      Failed to load cached batch {batch_idx}: {e}")
        return None

def cleanup_preprocessing_cache(rosbag_name: str):
    """Clean up all preprocessing caches for a rosbag.
    
    Args:
        rosbag_name: Name of the rosbag
    """
    if not EMBEDDINGS:
        return
    
    cache_root = EMBEDDINGS / "tmp_preprocessing" / rosbag_name
    if not cache_root.exists():
        return
    
    try:
        import shutil
        shutil.rmtree(cache_root)
        logger.info(f"      Cleaned up preprocessing cache for {rosbag_name}")
    except Exception as e:
        logger.warning(f"      Failed to cleanup preprocessing cache: {e}")

def create_shards(embeddings_by_model_topic: Dict[str, Dict[str, List[Tuple[np.ndarray, int, str]]]], rosbag_name: str, output_dir: Path):
    """Create embedding shards and manifests."""
    if not embeddings_by_model_topic:
        logger.warning("  ⚠️  No embeddings to shard")
        return
    
    logger.info(f"  💾 Creating shards for {len(embeddings_by_model_topic)} model(s)...")
    start_time = time.time()
    
    def ns_to_minute_of_day(ts_ns: int) -> int:
        dt = datetime.utcfromtimestamp(ts_ns / 1e9)
        return dt.hour * 60 + dt.minute
    
    total_shards = 0
    for model_id, topic_embeddings in embeddings_by_model_topic.items():
        logger.info(f"    Creating shards for {model_id}...")
        model_dir = output_dir / model_id / rosbag_name
        shards_dir = model_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all embeddings sorted by topic and timestamp
        all_records: List[Tuple[str, int, str, np.ndarray]] = []
        for topic, embeddings_list in topic_embeddings.items():
            embeddings_list.sort(key=lambda x: x[1])  # Sort by timestamp
            for embedding, timestamp_ns, mcap_identifier in embeddings_list:
                all_records.append((topic, timestamp_ns, mcap_identifier, embedding))
        
        if not all_records:
            logger.warning(f"      No embeddings for {model_id}")
            continue
        
        # Create shards
        manifest_rows = []
        shard_id = 0
        row_in_shard = 0
        current_shard: List[np.ndarray] = []
        
        for topic, timestamp_ns, mcap_identifier, embedding in all_records:
            current_shard.append(embedding)
            minute_of_day = ns_to_minute_of_day(timestamp_ns)
            shard_filename = f"shard-{shard_id:05d}.npy"
            
            manifest_rows.append({
                "topic": topic,
                "timestamp_ns": timestamp_ns,
                "mcap_identifier": mcap_identifier,
                "minute_of_day": minute_of_day,
                "shard_id": shard_filename,
                "row_in_shard": row_in_shard
            })
            
            row_in_shard += 1
            
            if len(current_shard) >= DEFAULT_SHARD_ROWS:
                shard_array = np.stack(current_shard).astype(OUTPUT_DTYPE)
                shard_path = shards_dir / shard_filename
                np.save(shard_path, shard_array)
                current_shard = []
                shard_id += 1
                row_in_shard = 0
        
        # Write final shard
        if current_shard:
            shard_array = np.stack(current_shard).astype(OUTPUT_DTYPE)
            shard_filename = f"shard-{shard_id:05d}.npy"
            shard_path = shards_dir / shard_filename
            np.save(shard_path, shard_array)
        
        num_shards = shard_id + (1 if current_shard else 0)
        total_shards += num_shards
        
        # Write manifest
        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows)
            manifest_path = model_dir / "manifest.parquet"
            table = pa.Table.from_pandas(manifest_df)
            pq.write_table(table, manifest_path, compression="zstd")
            logger.info(f"      ✓ {model_id}: {len(all_records):,} embeddings in {num_shards} shard(s), manifest saved")
    
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Created {total_shards} total shard(s) in {elapsed:.1f}s")

def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs.
    
    Returns:
        List of GPU device IDs (e.g., [0, 1] for 2 GPUs)
    """
    if not torch.cuda.is_available():
        return []
    
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    
    if num_gpus > 0:
        logger.info(f"  🖥️  Detected {num_gpus} GPU(s): {gpu_ids}")
        for gpu_id in gpu_ids:
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                logger.info(f"      GPU {gpu_id}: {props.name} ({total_mem / 1024**3:.2f} GB total, "
                          f"{free_mem / 1024**3:.2f} GB free)")
            except Exception as e:
                logger.warning(f"      Could not get info for GPU {gpu_id}: {e}")
    
    return gpu_ids

def process_model_on_gpu(model_id: str, preprocessed_tensors: List[Tuple[str, torch.Tensor, int, str]], 
                        gpu_id: int, rosbag_name: str, position: Optional[int] = None) -> Tuple[str, Dict[str, List[Tuple[np.ndarray, int, str]]], Optional[Exception]]:
    """Worker function to process a single model on a specific GPU.
    
    Args:
        model_id: Model identifier
        preprocessed_tensors: Preprocessed image tensors
        gpu_id: GPU device ID to use (e.g., 0, 1, 2, etc.). If invalid or CUDA unavailable, falls back to CPU.
        rosbag_name: Rosbag name for logging and completion tracking
        position: Optional position for tqdm progress bar (for parallel processing)
    
    Returns:
        Tuple of (model_id, model_embeddings_dict, error_if_any)
    """
    # Determine device string - validates gpu_id is within available GPU range
    if torch.cuda.is_available() and gpu_id >= 0 and gpu_id < torch.cuda.device_count():
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    model = None
    error = None
    
    try:
        model_start_time = time.time()
        logger.info(f"    🔄 Processing model: {model_id} on GPU {gpu_id}")
        model_load_start = time.time()
        
        # Load model on assigned GPU
        if "__" in model_id:
            model_name, pretrained_name = model_id.split("__", 1)
            model_name = model_name.replace("_", "/")
            model = load_openclip_model(model_name, pretrained_name, device)
            logger.info(f"      ✓ Model loaded: {model_id} on GPU {gpu_id}")
        else:
            for config in CUSTOM_MODELS:
                if config.get("model_dir_name", config["name"]) == model_id:
                    checkpoint_path = Path(config["checkpoint"])
                    if checkpoint_path.exists():
                        model = load_custom_model(checkpoint_path, device)
                        logger.info(f"      ✓ Model loaded: {model_id} on GPU {gpu_id}")
                    break
        
        if model is None:
            error = Exception(f"Could not load model {model_id}")
            logger.warning(f"      ⚠️  Could not load model {model_id}, skipping")
            return (model_id, {}, error)
        
        model_load_time = time.time() - model_load_start
        logger.debug(f"      Model loaded in {model_load_time:.1f}s on GPU {gpu_id}")
        
        # Clear memory before batch size calculation to ensure clean state
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate optimal batch size for this model on this GPU
        if preprocessed_tensors:
            logger.info(f"      🔍 Calculating optimal batch size for {model_id} on GPU {gpu_id}...")
            sample_input = preprocessed_tensors[0][1]  # Get sample tensor
            optimal_batch_size = calculate_optimal_batch_size(model, device, sample_input)
        else:
            optimal_batch_size = BATCH_SIZE
        
        # Generate embeddings with GPU context for progress bar positioning
        embeddings = generate_embeddings(preprocessed_tensors, model, device, model_id, batch_size=optimal_batch_size, gpu_id=gpu_id, position=position)
        
        # Group by topic
        model_embeddings: Dict[str, List[Tuple[np.ndarray, int, str]]] = defaultdict(list)
        for topic, embedding, timestamp_ns, mcap_id in embeddings:
            model_embeddings[topic].append((embedding, timestamp_ns, mcap_id))
        
        logger.info(f"    ✓ {model_id} on GPU {gpu_id}: {len(embeddings):,} embeddings across {len(model_embeddings)} topic(s)")
        
        # Compute adjacent similarities for each topic (before sharding)
        if ADJACENT_SIMILARITIES:
            logger.info(f"    🔍 Computing adjacent similarities for {model_id} on GPU {gpu_id}...")
            similarity_topics = []
            similarity_errors = []
            
            for topic, embeddings_list in model_embeddings.items():
                # Check if already completed
                is_completed, completed_topics = is_adjacent_similarities_completed(rosbag_name, model_id)
                if is_completed and topic in completed_topics:
                    logger.debug(f"      ⏭️  {topic}: Adjacent similarities already completed, skipping")
                    continue
                
                try:
                    # Sort by timestamp to maintain temporal order
                    embeddings_list.sort(key=lambda x: x[1])
                    
                    if len(embeddings_list) < 2:
                        logger.debug(f"      ⚠️  {topic}: Not enough embeddings (need at least 2), skipping")
                        continue
                    
                    # Extract just the embedding arrays
                    embedding_arrays = np.stack([emb for emb, _, _ in embeddings_list])
                    
                    # Compute similarities
                    similarities = compute_adjacent_similarities(embedding_arrays)
                    
                    if similarities is not None and len(similarities) > 0:
                        # Create and save heatmap
                        create_similarity_heatmap(similarities, model_id, rosbag_name, topic)
                        similarity_topics.append(topic)
                        logger.debug(f"      ✓ {topic}: Computed {len(similarities)} similarities")
                    else:
                        logger.debug(f"      ⚠️  {topic}: Could not compute similarities")
                
                except Exception as e:
                    error_msg = f"{topic}: {str(e)}"
                    logger.error(f"      ❌ {topic}: Error computing adjacent similarities: {e}")
                    similarity_errors.append(error_msg)
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Mark completion
            if similarity_topics or similarity_errors:
                all_sim_topics = similarity_topics + [err.split(":")[0] for err in similarity_errors]
                error_msg = "; ".join(similarity_errors) if similarity_errors else None
                mark_adjacent_similarities_completed(rosbag_name, model_id, all_sim_topics, error=error_msg)
                logger.info(f"    ✓ Adjacent similarities for {model_id} on GPU {gpu_id}: "
                          f"{len(similarity_topics)} topic(s) succeeded, {len(similarity_errors)} failed")
        
        # Mark embedding generation as completed
        mark_embeddings_completed(rosbag_name, model_id, "embedding_generation", "success")
        
        model_elapsed = time.time() - model_start_time
        num_embeddings = sum(len(emb_list) for emb_list in model_embeddings.values())
        logger.info(f"    ✓ {model_id} completed on GPU {gpu_id} in {model_elapsed:.1f}s ({num_embeddings:,} embeddings)")
        
        return (model_id, model_embeddings, None)
        
    except Exception as e:
        error = e
        error_msg = str(e)
        model_elapsed = time.time() - model_start_time if 'model_start_time' in locals() else 0
        logger.error(f"    ❌ Error processing model {model_id} on GPU {gpu_id} after {model_elapsed:.1f}s: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        mark_embeddings_completed(rosbag_name, model_id, "embedding_generation", "failed", error=error_msg)
        return (model_id, {}, error)
    finally:
        cleanup_model_and_gpu(model, device)
        logger.debug(f"      Cleaned up model {model_id} on GPU {gpu_id}")

def process_rosbag_images(image_data_list: List, preprocessing_groups: Dict[str, List[str]], rosbag_name: str, rosbag_path: Path):
    """Process images in batches: convert, preprocess, embed, checkpoint, and shard.
    
    Args:
        image_data_list: Either List[Tuple[str, bytes, int, str]] for legacy mode
                        or List[Tuple[str, Path, int, str]] for metadata-only mode (batched)
        preprocessing_groups: Dict mapping preprocessing string to list of model IDs
        rosbag_name: Name of the rosbag being processed
        rosbag_path: Path to the rosbag directory (for batched loading)
    """
    if not EMBEDDINGS:
        return
    
    # Check which models still need processing (embeddings or adjacencies)
    models_to_process = []
    for preprocess_str, model_ids in preprocessing_groups.items():
        for model_id in model_ids:
            embeddings_done = is_embeddings_completed(rosbag_name, model_id)
            adjacencies_done = False
            if ADJACENT_SIMILARITIES:
                adjacencies_done, _ = is_adjacent_similarities_completed(rosbag_name, model_id)
            
            # Process if embeddings or adjacencies are missing
            if not embeddings_done or (ADJACENT_SIMILARITIES and not adjacencies_done):
                models_to_process.append((preprocess_str, model_id, embeddings_done))
            else:
                logger.info(f"  ⏭️  Embeddings and adjacencies already completed for {model_id}/{rosbag_name}, skipping")
    
    if not models_to_process:
        logger.info(f"  ⏭️  All models already completed for {rosbag_name}")
        return
    
    # Group models by preprocessing and track which need embeddings
    models_by_preprocessing: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    for preprocess_str, model_id, embeddings_done in models_to_process:
        models_by_preprocessing[preprocess_str].append((model_id, embeddings_done))
    
    # Check if any model needs embeddings generated (not just adjacencies)
    needs_embedding_generation = any(not embeddings_done for _, _, embeddings_done in models_to_process)
    
    # Allow empty image_data_list only if all models already have embeddings
    if not image_data_list and needs_embedding_generation:
        return
    
    if image_data_list:
        total_images = len(image_data_list)
        logger.info(f"      📸 Processing {total_images:,} images for {len(models_to_process)} model(s)...")
        
        # Detect if we're in metadata-only mode (Path) or legacy mode (bytes)
        is_metadata_mode = isinstance(image_data_list[0][1], Path)
        
        # Calculate adaptive batch size for image processing (based on available RAM)
        image_batch_size = calculate_adaptive_image_batch_size(total_images)
        num_batches = (total_images + image_batch_size - 1) // image_batch_size
        
        logger.info(f"      🔧 Using batched processing: {num_batches} batch(es) of ~{image_batch_size:,} images")
    else:
        # No image data needed - all models have embeddings, just computing adjacencies
        logger.info(f"      📂 Computing adjacency similarities for {len(models_to_process)} model(s)...")
        total_images = 0
        is_metadata_mode = False
        image_batch_size = 0
        num_batches = 0
    
    # Process each preprocessing group
    all_embeddings: Dict[str, Dict[str, List[Tuple[np.ndarray, int, str]]]] = {}
    
    for preprocess_str, model_list in models_by_preprocessing.items():
        if not model_list:
            continue
        
        logger.info(f"    🔧 Processing preprocessing group for {len(model_list)} model(s)...")
        
        # Get preprocessing transform for this group
        preprocess_transform = None
        first_model_id, _ = model_list[0]  # Extract model_id from tuple
        
        # Find transform for first model
        for model_name, pretrained_name in OPENCLIP_MODELS:
            model_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
            if model_id == first_model_id:
                try:
                    candidate = get_preprocess_transform(model_name, pretrained_name)
                    if str(candidate) == preprocess_str:
                        preprocess_transform = candidate
                        break
                except Exception:
                    continue
        
        if preprocess_transform is None:
            for config in CUSTOM_MODELS:
                if config.get("model_dir_name", config["name"]) == first_model_id:
                    checkpoint_path = Path(config["checkpoint"])
                    if checkpoint_path.exists():
                        try:
                            candidate = get_custom_preprocess_transform(checkpoint_path)
                            if str(candidate) == preprocess_str:
                                preprocess_transform = candidate
                                break
                        except Exception:
                            continue
        
        if preprocess_transform is None:
            logger.warning(f"    ⚠️  Could not find preprocessing transform for {first_model_id}, skipping")
            continue
        
        # Create a hash of the preprocessing transform for cache identification
        preprocess_hash = hashlib.md5(preprocess_str.encode()).hexdigest()[:8]
        logger.info(f"    🔧 Preprocessing group: {len(model_list)} model(s), hash: {preprocess_hash}")
        
        # Check if any model in this group needs embeddings generated
        group_needs_embeddings = any(not embeddings_done for _, embeddings_done in model_list)
        
        # =================================================================
        # PHASE 1: Preprocess all batches for this preprocessing group (once)
        # =================================================================
        if not group_needs_embeddings:
            logger.info(f"    ⏭️  Phase 1: Skipping preprocessing (all models have embeddings)")
        else:
            logger.info(f"    📦 Phase 1: Preprocessing and caching {num_batches} batch(es)")
            batches_to_preprocess = []
            
            # Check which batches need preprocessing
            for batch_idx in range(num_batches):
                cached_tensors = load_preprocessed_batch(batch_idx, rosbag_name, preprocess_hash)
                if cached_tensors is None:
                    batches_to_preprocess.append(batch_idx)
                else:
                    logger.debug(f"      Batch {batch_idx + 1}/{num_batches}: using cached preprocessed tensors")
        
        if group_needs_embeddings and batches_to_preprocess:
            logger.info(f"      {len(batches_to_preprocess)} batch(es) need preprocessing")
            
            for batch_idx in batches_to_preprocess:
                batch_start_idx = batch_idx * image_batch_size
                batch_end_idx = min(batch_start_idx + image_batch_size, total_images)
                
                logger.info(f"      Preprocessing batch {batch_idx + 1}/{num_batches} ({batch_end_idx - batch_start_idx:,} images)")
                
                try:
                    # Load image batch
                    if is_metadata_mode:
                        image_batch = load_image_batch_from_mcaps(
                            image_data_list, batch_start_idx, image_batch_size, rosbag_path
                        )
                    else:
                        image_batch = image_data_list[batch_start_idx:batch_end_idx]
                    
                    if not image_batch:
                        logger.warning(f"      ⚠️  Batch {batch_idx + 1} empty, skipping")
                        continue
                    
                    # Convert to PIL
                    pil_images = convert_images_to_pil(image_batch)
                    del image_batch
                    gc.collect()
                    
                    if not pil_images:
                        logger.warning(f"      ⚠️  No PIL images in batch {batch_idx + 1}, skipping")
                        continue
                    
                    # Preprocess
                    preprocessed_tensors = preprocess_images(pil_images, preprocess_transform)
                    del pil_images
                    gc.collect()
                    
                    if not preprocessed_tensors:
                        logger.warning(f"      ⚠️  No preprocessed tensors in batch {batch_idx + 1}, skipping")
                        continue
                    
                    # Cache preprocessed tensors
                    save_preprocessed_batch(preprocessed_tensors, batch_idx, rosbag_name, preprocess_hash)
                    logger.info(f"      ✓ Cached {len(preprocessed_tensors):,} preprocessed tensors for batch {batch_idx + 1}")
                    
                    del preprocessed_tensors
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"      ❌ Error preprocessing batch {batch_idx + 1}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
        else:
            logger.info(f"      ✓ All batches already cached, skipping preprocessing")
        
        # =================================================================
        # PHASE 2: Generate embeddings for each model using cached preprocessed tensors
        # =================================================================
        logger.info(f"    🧠 Phase 2: Processing {len(model_list)} model(s)")
        
        if group_needs_embeddings:
            # Calculate optimal batch size for loading cached tensors
            # This shows how many tensors we COULD load at once (larger than image batches)
            # but we keep preprocessing batch boundaries for checkpoint compatibility
            tensor_batch_size = calculate_adaptive_tensor_batch_size(total_images)
            potential_speedup = tensor_batch_size / image_batch_size
            
            if potential_speedup > 1.5:
                logger.info(f"      Note: Tensors are {potential_speedup:.1f}x smaller in memory than images")
                logger.info(f"      Could load ~{tensor_batch_size:,} tensors vs {image_batch_size:,} images per batch")
                logger.info(f"      (Keeping preprocessing batch boundaries for checkpoint compatibility)")
        
        for model_id, embeddings_done in model_list:
            logger.info(f"      📊 Model: {model_id}")
            
            if embeddings_done:
                # Load existing embeddings for adjacency computation
                logger.info(f"        📂 Loading existing embeddings from shards...")
                model_embeddings = load_embeddings_from_shards(model_id, rosbag_name)
                
                if model_embeddings:
                    all_embeddings[model_id] = model_embeddings
                    total_emb = sum(len(embs) for embs in model_embeddings.values())
                    logger.info(f"        ✓ Loaded {total_emb:,} embeddings")
                    
                    # Compute adjacent similarities (code continues below after else block)
                    # The adjacency computation will happen after the if/else
                else:
                    logger.warning(f"        ⚠️  Could not load embeddings from shards")
                    continue
            else:
                # Generate embeddings from cached tensors (existing path)
                # Check checkpoint progress
                last_completed_batch = load_checkpoint_progress(model_id, rosbag_name)
                start_batch = last_completed_batch + 1
                
                if start_batch > 0:
                    logger.info(f"        ♻️  Resuming from batch {start_batch}/{num_batches}")
                
                # Initialize optimal batch size
                optimal_batch_size = None
                
                # Process each preprocessing batch
                # Note: We load one preprocessing batch at a time for checkpoint compatibility,
                # but the tensors are much lighter in memory than the original images
                for batch_idx in range(start_batch, num_batches):
                    logger.info(f"        Batch {batch_idx + 1}/{num_batches}: loading cached tensors and generating embeddings")
                    
                    try:
                        # Load cached preprocessed tensors
                        preprocessed_tensors = load_preprocessed_batch(batch_idx, rosbag_name, preprocess_hash)
                        
                        if preprocessed_tensors is None:
                            logger.error(f"        ❌ Cached tensors not found for batch {batch_idx + 1}")
                            continue
                        
                        # Load model
                        available_gpus = get_available_gpus()
                        device = f"cuda:{available_gpus[0]}" if available_gpus else "cpu"
                        gpu_id = available_gpus[0] if available_gpus else 0
                        
                        model = None
                        if "__" in model_id:
                            model_name, pretrained_name = model_id.split("__", 1)
                            model_name = model_name.replace("_", "/")
                            model = load_openclip_model(model_name, pretrained_name, device)
                        else:
                            for config in CUSTOM_MODELS:
                                if config.get("model_dir_name", config["name"]) == model_id:
                                    checkpoint_path = Path(config["checkpoint"])
                                    if checkpoint_path.exists():
                                        model = load_custom_model(checkpoint_path, device)
                                    break
                        
                        if model is None:
                            logger.error(f"        ❌ Could not load model {model_id}")
                            break
                        
                        # Calculate optimal batch size once
                        if optimal_batch_size is None:
                            sample_input = preprocessed_tensors[0][1]
                            optimal_batch_size = calculate_optimal_batch_size(model, device, sample_input)
                            logger.info(f"        📊 Optimal GPU batch size: {optimal_batch_size}")
                        
                        # Generate embeddings
                        batch_embeddings = generate_embeddings(
                            preprocessed_tensors, model, device, model_id, 
                            batch_size=optimal_batch_size, gpu_id=gpu_id
                        )
                        
                        # Cleanup
                        cleanup_model_and_gpu(model, device)
                        del model, preprocessed_tensors
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Save checkpoint
                        if batch_embeddings:
                            save_batch_checkpoint(batch_embeddings, batch_idx, model_id, rosbag_name)
                            logger.info(f"        ✓ Batch {batch_idx + 1}/{num_batches}: {len(batch_embeddings):,} embeddings")
                        
                        del batch_embeddings
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"        ❌ Error on batch {batch_idx + 1}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue
                
                # Merge checkpoints after all batches processed
                logger.info(f"      🔄 Merging checkpoints for {model_id}...")
                model_embeddings = merge_batch_checkpoints(model_id, rosbag_name)
                
                if model_embeddings:
                    all_embeddings[model_id] = model_embeddings
                    total_emb = sum(len(embs) for embs in model_embeddings.values())
                    logger.info(f"      ✓ {model_id}: {total_emb:,} embeddings")
                else:
                    logger.warning(f"      ⚠️  {model_id}: No embeddings")
                    mark_embeddings_completed(rosbag_name, model_id, "embedding_generation", "failed", 
                                            error="No embeddings after checkpoint merge")
                    continue
            
            # Compute adjacent similarities for each topic (for both modes)
            if model_embeddings and ADJACENT_SIMILARITIES:
                logger.info(f"      🔍 Computing adjacent similarities for {model_id}...")
                similarity_topics = []
                similarity_errors = []
                
                for topic, embeddings_list in model_embeddings.items():
                    # Check if already completed
                    is_completed, completed_topics = is_adjacent_similarities_completed(rosbag_name, model_id)
                    if is_completed and topic in completed_topics:
                        logger.debug(f"        ⏭️  {topic}: Adjacent similarities already completed, skipping")
                        continue
                    
                    try:
                        # Sort by timestamp to maintain temporal order
                        embeddings_list.sort(key=lambda x: x[1])
                        
                        if len(embeddings_list) < 2:
                            logger.debug(f"        ⚠️  {topic}: Not enough embeddings (need at least 2), skipping")
                            continue
                        
                        # Extract just the embedding arrays
                        embedding_arrays = np.stack([emb for emb, _, _ in embeddings_list])
                        
                        # Compute similarities
                        similarities = compute_adjacent_similarities(embedding_arrays)
                        
                        if similarities is not None and len(similarities) > 0:
                            # Create and save heatmap
                            create_similarity_heatmap(similarities, model_id, rosbag_name, topic)
                            similarity_topics.append(topic)
                            logger.debug(f"        ✓ {topic}: Computed {len(similarities)} similarities")
                        else:
                            logger.debug(f"        ⚠️  {topic}: Could not compute similarities")
                    
                    except Exception as e:
                        error_msg = f"{topic}: {str(e)}"
                        logger.error(f"        ❌ {topic}: Error computing adjacent similarities: {e}")
                        similarity_errors.append(error_msg)
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # Mark completion
                if similarity_topics or similarity_errors:
                    all_sim_topics = similarity_topics + [err.split(":")[0] for err in similarity_errors]
                    error_msg = "; ".join(similarity_errors) if similarity_errors else None
                    mark_adjacent_similarities_completed(rosbag_name, model_id, all_sim_topics, error=error_msg)
                    logger.info(f"      ✓ Adjacent similarities for {model_id}: "
                              f"{len(similarity_topics)} topic(s) succeeded, {len(similarity_errors)} failed")
            
            # Mark completion and cleanup (only if we generated embeddings)
            if not embeddings_done:
                mark_embeddings_completed(rosbag_name, model_id, "embedding_generation", "success")
                cleanup_checkpoints(model_id, rosbag_name)
    
    # Create shards (only for models that generated embeddings)
    embeddings_to_shard = {}
    for model_id, embs in all_embeddings.items():
        # Check if this model generated embeddings (not just loaded for adjacencies)
        model_generated = False
        for model_list in models_by_preprocessing.values():
            for mid, emb_done in model_list:
                if mid == model_id and not emb_done:
                    model_generated = True
                    break
            if model_generated:
                break
        if model_generated:
            embeddings_to_shard[model_id] = embs
    
    if embeddings_to_shard:
        try:
            create_shards(embeddings_to_shard, rosbag_name, EMBEDDINGS)
            # Mark shard creation as completed for each model
            for model_id in embeddings_to_shard.keys():
                mark_embeddings_completed(rosbag_name, model_id, "shard_creation", "success")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"  ❌ Error creating shards: {e}")
            for model_id in embeddings_to_shard.keys():
                mark_embeddings_completed(rosbag_name, model_id, "shard_creation", "failed", error=error_msg)
            raise

def process_single_rosbag(pipeline: RosbagPipeline, preprocessing_groups: Dict[str, List[str]]) -> List[Tuple[str, float, float, int, str]]:
    """Process a single rosbag based on pipeline configuration.
    
    Args:
        pipeline: RosbagPipeline with flags indicating what needs to be done
        preprocessing_groups: Preprocessing groups for embeddings
        
    Returns:
        positional_data_list: List of (topic, lat, lon, timestamp_ns, mcap_identifier) tuples
    """
    rosbag_name = pipeline.rosbag_path.name
    rosbag_start_time = time.time()
    logger.info(f"  📦 {rosbag_name}")
    
    try:
        # Step 1: Extract data from rosbag if needed (timestamps + images + positional positions)
        positional_data_list = []
        topic_data_by_mcap = {}
        topic_types = {}
        image_data_list = []
        
        if pipeline.needs_extraction:
            # Use metadata-only mode for images when doing embeddings (memory efficient batched processing)
            # Otherwise load full bytes
            load_bytes = not pipeline.needs_embeddings
            topic_data_by_mcap, topic_types, image_data_list, positional_data_list = extract_data_from_rosbag(
                pipeline.rosbag_path, load_image_bytes=load_bytes
            )
            
            if not topic_data_by_mcap and (pipeline.needs_lookup_tables or pipeline.needs_topics_json):
                logger.warning(f"  ⚠️  No topic data found in {rosbag_name}")
                return positional_data_list
        elif pipeline.needs_positional_data:
            # Only need positional data, extract just that
            _, _, _, positional_data_list = extract_data_from_rosbag(pipeline.rosbag_path, load_image_bytes=False)
            return positional_data_list
        else:
            # Try to load topic_types from existing topics JSON if needed
            if pipeline.needs_topics_json and pipeline.topics_json_path and pipeline.topics_json_path.exists():
                try:
                    with open(pipeline.topics_json_path, 'r') as f:
                        topics_data = json.load(f)
                        topic_types = topics_data.get("types", {})
                except Exception:
                    pass
        
        # Step 1.5: Create representative previews (requires separate extraction with full bytes)
        if pipeline.needs_representative_previews:
            if pipeline.needs_embeddings and image_data_list:
                # We extracted metadata for embeddings, need to extract bytes for previews
                logger.info(f"  🖼️  Extracting images for representative previews...")
                _, _, image_bytes_list, _ = extract_data_from_rosbag(pipeline.rosbag_path, load_image_bytes=True)
                if image_bytes_list:
                    create_representative_previews(image_bytes_list, rosbag_name)
                    del image_bytes_list  # Clean up after previews
                    gc.collect()
            elif image_data_list:
                # We already have bytes
                create_representative_previews(image_data_list, rosbag_name)
        
        # Step 2: Create lookup tables
        if pipeline.needs_lookup_tables:
            all_topics = get_all_topics_from_metadata(pipeline.rosbag_path)
            create_lookup_tables(pipeline.rosbag_path, topic_data_by_mcap, topic_types, pipeline.csv_path, all_topics)
        elif (pipeline.needs_embeddings or pipeline.needs_adjacent_similarities) and EMBEDDINGS and not image_data_list:
            # Extract metadata-only for embeddings (batched processing)
            # Note: For adjacency-only mode, we don't actually need image_data_list since we load from shards
            if pipeline.needs_embeddings:
                _, _, image_data_list, _ = extract_data_from_rosbag(pipeline.rosbag_path, load_image_bytes=False)
        
        # Step 3: Save topics JSON
        if pipeline.needs_topics_json and pipeline.topics_json_path and topic_types:
            logger.info(f"  📊 Creating topics.json...")
            try:
                pipeline.topics_json_path.parent.mkdir(parents=True, exist_ok=True)
                topics = sorted(topic_types.keys())
                with open(pipeline.topics_json_path, 'w') as f:
                    json.dump({"topics": topics, "types": topic_types}, f, indent=2)
                logger.info(f"      ✓ Topics JSON ({len(topics)} topics)")
                mark_topics_json_completed(rosbag_name)
            except Exception as e:
                error = str(e)
                logger.error(f"      ❌ Topics JSON error: {e}")
                mark_topics_json_completed(rosbag_name, error=error)
                raise
        
        # Step 3.5: Update positional lookup table
        if pipeline.needs_positional_data and positional_data_list:
            logger.info(f"  📊 Creating positional lookup tables...")
            try:
                update_positional_lookup_table(rosbag_name, positional_data_list)
                mark_positional_lookup_table_completed(rosbag_names=[rosbag_name])
            except Exception as e:
                error = str(e)
                logger.error(f"      ❌ Positional lookup table error: {e}")
                mark_positional_lookup_table_completed(rosbag_names=[rosbag_name], error=error)
                # Don't raise - continue processing other steps
        
        # Step 4: Process images if needed (with batched processing) or compute adjacencies
        if (pipeline.needs_embeddings or pipeline.needs_adjacent_similarities) and EMBEDDINGS:
            if image_data_list:
                logger.info(f"  📸 Processing {len(image_data_list):,} images...")
            else:
                logger.info(f"  📂 Computing adjacency similarities from existing embeddings...")
            process_rosbag_images(image_data_list, preprocessing_groups, rosbag_name, pipeline.rosbag_path)
        
        # Cleanup preprocessing cache after rosbag completes
        cleanup_preprocessing_cache(rosbag_name)
        
        rosbag_elapsed = time.time() - rosbag_start_time
        logger.info(f"      ✅ Done ({rosbag_elapsed:.1f}s)\n")
        
        return positional_data_list
    except Exception as e:
        rosbag_elapsed = time.time() - rosbag_start_time
        error_msg = str(e)
        logger.error(f"      ❌ Error after {rosbag_elapsed:.1f}s: {e}\n")
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(traceback_str)
        
        # Mark as failed in completion tracking
        try:
            completed = load_metadata_completion()
            if rosbag_name not in completed:
                completed[rosbag_name] = {"lookup_tables": {}, "topics_json": {}, "errors": []}
            if "errors" not in completed[rosbag_name]:
                completed[rosbag_name]["errors"] = []
            completed[rosbag_name]["errors"].append({
                "step": "rosbag_processing",
                "error": error_msg,
                "traceback": traceback_str,
                "timestamp": get_timestamp_berlin()
            })
            save_metadata_completion(completed)
        except Exception as save_error:
            logger.warning(f"  ⚠️  Could not save error to completion file: {save_error}")
        
        return []
        
        raise

# =========================
# Main execution
# =========================

def main() -> None:
    """Main function to process all rosbags."""
    start_time = time.time()
    logger.info("="*70)
    logger.info(f"🚀 Master Pipeline Starting")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Device: {DEVICE}")
    
    # Log GPU information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"   GPUs: {num_gpus} available")
        logger.info(f"   Multi-GPU: {'Enabled' if ENABLE_MULTI_GPU else 'Disabled'}")
        for gpu_id in range(num_gpus):
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                logger.info(f"      GPU {gpu_id}: {props.name} ({total_mem / 1024**3:.2f} GB)")
            except Exception:
                pass
    else:
        logger.info(f"   GPUs: None (CPU mode)")
    
    logger.info(f"   EMBEDDINGS configured: {EMBEDDINGS is not None}")
    if EMBEDDINGS:
        logger.info(f"   EMBEDDINGS path: {EMBEDDINGS}")
    logger.info("="*70)
    
    # Validate environment
    logger.info("🔍 Validating environment...")
    if ROSBAGS is None or not ROSBAGS.exists():
        logger.error(f"❌ ROSBAGS directory does not exist: {ROSBAGS}")
        raise SystemExit(f"ROSBAGS directory does not exist: {ROSBAGS}")
    if LOOKUP_TABLES is None:
        logger.error("❌ LOOKUP_TABLES environment variable not set")
        raise SystemExit("LOOKUP_TABLES environment variable not set")
    if TOPICS is None:
        logger.error("❌ TOPICS environment variable not set")
        raise SystemExit("TOPICS environment variable not set")
    
    logger.info(f"   ✓ ROSBAGS: {ROSBAGS}")
    logger.info(f"   ✓ LOOKUP_TABLES: {LOOKUP_TABLES}")
    logger.info(f"   ✓ TOPICS: {TOPICS}\n")
    
    # Validate MODE
    if MODE not in ("single", "all", "multiple"):
        logger.error("MODE must be 'single', 'all', or 'multiple'")
        raise SystemExit("MODE must be 'single', 'all', or 'multiple'")

    if MODE == "single" and not SINGLE_BAG_NAME:
        logger.error("MODE is 'single' but SINGLE_BAG_NAME is not set")
        raise SystemExit("MODE is 'single' but SINGLE_BAG_NAME is not set")

    if MODE == "multiple" and not MULTIPLE_BAG_NAMES:
        logger.error("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")
        raise SystemExit("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")

    # Display mode
    if MODE == "single":
        logger.info(f"🎯 MODE: single → {SINGLE_BAG_NAME}\n")
    elif MODE == "multiple":
        logger.info(f"🎯 MODE: multiple → {len(MULTIPLE_BAG_NAMES)} rosbag(s)\n")
    else:  # MODE == "all"
        logger.info(f"🎯 MODE: all → Processing ALL rosbags\n")
    
    # Collect rosbags
    logger.info(f"📂 Scanning: {ROSBAGS}")
    scan_start = time.time()
    all_rosbags = collect_rosbags(ROSBAGS)
    scan_time = time.time() - scan_start
    logger.info(f"   ✓ Found {len(all_rosbags)} rosbag(s) ({scan_time:.2f}s)\n")
    
    if not all_rosbags:
        logger.warning("No rosbags found to process")
        return
    
    # Step 1: Check completion status for all rosbags and build pipeline
    logger.info(f"🔍 Checking completion status...")
    rosbag_pipelines: List[RosbagPipeline] = []
    needs_embeddings = False  # Track if any rosbag needs embedding processing
    needs_adjacent_similarities = False  # Track if any rosbag needs adjacency processing
    
    for rosbag_path, csv_path, topics_json_path in all_rosbags:
        pipeline = check_rosbag_completion(rosbag_path, csv_path, topics_json_path)
        rosbag_pipelines.append(pipeline)
        
        if pipeline.needs_embeddings:
            needs_embeddings = True
        if pipeline.needs_adjacent_similarities:
            needs_adjacent_similarities = True
    
    already_completed = sum(1 for p in rosbag_pipelines if not p.missing_steps)
    needs_processing_count = sum(1 for p in rosbag_pipelines if p.missing_steps)
    
    logger.info(f"   ✓ {already_completed} completed")
    logger.info(f"   📋 {needs_processing_count} need(s) processing\n")
    
    # Step 2: Identify unique preprocessings (if we need embeddings or adjacencies)
    preprocessing_groups = {}
    if (needs_embeddings or needs_adjacent_similarities) and EMBEDDINGS:
        logger.info(f"🔧 Identifying preprocessing groups...")
        preprocessing_groups = identify_unique_preprocessings()
        logger.info(f"   ✓ {len(preprocessing_groups)} unique group(s)\n")
    
    # Step 3: Process each rosbag sequentially
    total_to_process = len(rosbag_pipelines)
    logger.info(f"🚀 Processing {total_to_process} rosbag(s)...\n")
    succeeded = 0
    failed = 0
    
    for idx, pipeline in enumerate(rosbag_pipelines, 1):
        try:
            rosbag_name = pipeline.rosbag_path.name
            if not pipeline.missing_steps:
                logger.info(f"[{idx}/{total_to_process}] ✓ {rosbag_name} (already completed, skipping)")
                succeeded += 1
            else:
                logger.info(f"[{idx}/{total_to_process}] 🔄 {rosbag_name}")
                process_single_rosbag(pipeline, preprocessing_groups)
                succeeded += 1
        except KeyboardInterrupt:
            logger.warning("\n🛑 Keyboard interrupt received\n")
            break
        except Exception as e:
            logger.error(f"   ❌ Error: {e}\n")
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    logger.info(f"{'='*70}")
    logger.info(f"✅ Processing Complete!")
    logger.info(f"   ⏱️  Time: {total_time/60:.1f} min")
    logger.info(f"   ✓ Completed: {already_completed}")
    logger.info(f"   🔄 Processed: {succeeded}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   📦 Total: {len(all_rosbags)}")
    logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    main()
