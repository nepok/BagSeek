import json
import base64
import os
import sys
from pathlib import Path
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from mcap.reader import SeekingReader
from mcap.writer import Writer, CompressionType, IndexType
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from flask import Flask, jsonify, request, send_from_directory # type: ignore
from flask_cors import CORS  # type: ignore
import logging
import pandas as pd
import struct
import torch
import torch.nn.functional as F
from torch import nn
import faiss
from typing import Iterable, Sequence
import open_clip
import math
from threading import Thread, Lock
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import OrderedDict
from dotenv import load_dotenv
from time import time
import subprocess
import re
from typing import Optional, List

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from open_clip.tokenizer import tokenize as open_clip_tokenize
except ImportError:  # pragma: no cover
    open_clip_tokenize = None


# Load environment variables from .env file
PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Flask API for BagSeek: A tool for exploring Rosbag data and using semantic search via CLIP and FAISS to locate safety critical and relevant scenes.
# This API provides endpoints for loading data, searching via CLIP embeddings, exporting segments, and more.

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend (e.g., React)


# Gemma model for prompt enhancement (optional - requires HuggingFace auth)
try:
    gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
    GEMMA_AVAILABLE = True
except Exception as e:
    print(f"Warning: Gemma model not available ({e}). Prompt enhancement disabled.")
    gemma_tokenizer = None
    gemma_model = None
    GEMMA_AVAILABLE = False

def _require_env(name: str) -> str:
    """Get required environment variable or exit with error."""
    value = os.getenv(name)
    if value is None:
        logging.error(f"Required environment variable '{name}' is not set")
        sys.exit(1)
    return value

# Required environment variables
BASE_STR = _require_env("BASE")
ROSBAGS_STR = _require_env("ROSBAGS")
PRESELECTED_ROSBAG_STR = _require_env("PRESELECTED_ROSBAG")
OPEN_CLIP_MODELS_STR = _require_env("OPEN_CLIP_MODELS")
OTHER_MODELS_STR = _require_env("OTHER_MODELS")

# Relative path environment variables (appended to BASE)
IMAGE_TOPIC_PREVIEWS_STR = _require_env("IMAGE_TOPIC_PREVIEWS")
POSITIONAL_LOOKUP_TABLE_STR = _require_env("POSITIONAL_LOOKUP_TABLE")
LOOKUP_TABLES_STR = _require_env("LOOKUP_TABLES")
TOPICS_STR = _require_env("TOPICS")
CANVASES_FILE_STR = _require_env("CANVASES_FILE")
POLYGONS_DIR_STR = _require_env("POLYGONS_DIR")
ADJACENT_SIMILARITIES_STR = _require_env("ADJACENT_SIMILARITIES")
EMBEDDINGS_STR = _require_env("EMBEDDINGS")
EXPORT_STR = _require_env("EXPORT")


###############################################################################################################

ROSBAGS = Path(ROSBAGS_STR)
PRESELECTED_ROSBAG = Path(PRESELECTED_ROSBAG_STR)

PRESELECTED_MODEL = os.getenv("PRESELECTED_MODEL", "ViT-B-16-quickgelu__openai")
OPEN_CLIP_MODELS = Path(OPEN_CLIP_MODELS_STR)
OTHER_MODELS = Path(OTHER_MODELS_STR)

IMAGE_TOPIC_PREVIEWS = Path(BASE_STR + IMAGE_TOPIC_PREVIEWS_STR)
POSITIONAL_LOOKUP_TABLE = Path(BASE_STR + POSITIONAL_LOOKUP_TABLE_STR)

LOOKUP_TABLES = Path(BASE_STR + LOOKUP_TABLES_STR)
TOPICS = Path(BASE_STR + TOPICS_STR)

CANVASES_FILE = Path(BASE_STR + CANVASES_FILE_STR)
POLYGONS_DIR = Path(BASE_STR + POLYGONS_DIR_STR)

ADJACENT_SIMILARITIES = Path(BASE_STR + ADJACENT_SIMILARITIES_STR)
EMBEDDINGS = Path(BASE_STR + EMBEDDINGS_STR)

EXPORT = Path(BASE_STR + EXPORT_STR)


SELECTED_ROSBAG = PRESELECTED_ROSBAG
SELECTED_MODEL = PRESELECTED_MODEL

# Helper function to extract rosbag name from path (handles multipart rosbags)
def get_camera_position_order(topic_name: str) -> int:
    """
    Get ordering for camera positions within the same priority level.
    Returns order where lower order = appears first.
    
    Camera position order:
    0: side left
    1: side right
    2: rear left
    3: rear mid
    4: rear right
    5: everything else (will be sorted alphabetically)
    """
    topic_lower = topic_name.lower()
    
    # Check for camera position keywords (order matters - check more specific first)
    if "side" in topic_lower and "left" in topic_lower:
        return 0
    elif "side" in topic_lower and "right" in topic_lower:
        return 1
    elif "rear" in topic_lower and "left" in topic_lower:
        return 2
    elif "rear" in topic_lower and "mid" in topic_lower:
        return 3
    elif "rear" in topic_lower and "right" in topic_lower:
        return 4
    else:
        return 5

def get_topic_sort_priority(topic_name: str, topic_type: str = None) -> int:
    """
    Get sorting priority for a topic.
    Returns priority where lower priority = appears first.
    
    Priority order:
    0: Topics containing "zed"
    1: Image topics (sensor_msgs/msg/Image, sensor_msgs/msg/CompressedImage)
    2: PointCloud topics (sensor_msgs/msg/PointCloud2)
    3: Positional topics (NavSatFix, GPS, GNSS, TF, pose, position, odom)
    4: Everything else (alphabetically)
    """
    topic_lower = topic_name.lower()
    
    # Priority 0: Topics containing "zed"
    if "zed" in topic_lower:
        return 0
    
    # Determine topic category
    is_image = False
    is_pointcloud = False
    is_positional = False
    
    if topic_type:
        # Use provided topic type
        type_lower = topic_type.lower()
        is_image = "image" in type_lower and ("sensor_msgs" in type_lower or "compressedimage" in type_lower)
        is_pointcloud = "pointcloud" in type_lower or "point_cloud" in type_lower
        is_positional = any(x in type_lower for x in ["navsatfix", "gps", "gnss", "tf", "odom", "pose"])
    else:
        # Infer from topic name
        is_image = any(x in topic_lower for x in ["image", "camera", "rgb", "color"])
        is_pointcloud = any(x in topic_lower for x in ["pointcloud", "point_cloud", "lidar", "pcl"])
        is_positional = any(x in topic_lower for x in ["gps", "gnss", "navsat", "tf", "odom", "pose", "position"])
    
    if is_image:
        return 1
    elif is_pointcloud:
        return 2
    elif is_positional:
        return 3
    else:
        return 4

def sort_topics(topics: list[str], topic_types: dict[str, str] = None) -> list[str]:
    """
    Sort topics according to the default priority order.
    Within image topics, camera positions are ordered: side left, side right, rear left, rear mid, rear right.
    
    Args:
        topics: List of topic names
        topic_types: Optional dict mapping topic names to their types
    
    Returns:
        Sorted list of topics
    """
    if topic_types is None:
        topic_types = {}
    
    def sort_key(topic: str) -> tuple[int, int, str]:
        topic_type = topic_types.get(topic)
        priority = get_topic_sort_priority(topic, topic_type)
        
        # For image topics (priority 1), apply camera position ordering
        camera_order = 5  # Default (alphabetical)
        if priority == 1:  # Image topics
            camera_order = get_camera_position_order(topic)
        
        return (priority, camera_order, topic.lower())
    
    return sorted(topics, key=sort_key)

def extract_rosbag_name_from_path(rosbag_path: str) -> str:
    """Extract the correct rosbag name from a path, handling multipart rosbags.
    
    For multipart rosbags, the lookup tables are stored in a directory named after
    the parent directory (e.g., 'rosbag2_xxx_multi_parts'), not the individual part.
    
    Args:
        rosbag_path: Full path to the rosbag (can be a multipart rosbag part)
    
    Returns:
        Rosbag name to use for lookup tables (parent directory name for multipart, basename for regular)
    
    Examples:
        - Regular: '/path/to/rosbag2_2025_07_25-12_17_25' -> 'rosbag2_2025_07_25-12_17_25'
        - Multipart: '/path/to/rosbag2_xxx_multi_parts/Part_1' -> 'rosbag2_xxx_multi_parts/Part_1'
    """
    path_obj = Path(rosbag_path)
    basename = path_obj.name
    parent_name = path_obj.parent.name
    
    # Check if this is a multipart rosbag (parent ends with _multi_parts and current is Part_N)
    if parent_name.endswith("_multi_parts") and basename.startswith("Part_"):
        # For multipart rosbags, return parent_name/Part_N (e.g., 'rosbag2_xxx_multi_parts/Part_1')
        return f"{parent_name}/{basename}"
    else:
        # For regular rosbags, use the directory name itself
        return basename

# Cache for lookup tables to avoid reloading on every request
_lookup_table_cache: dict[str, tuple[pd.DataFrame, float]] = {}

# Helper function for loading lookup tables (defined below, initialized after)
def load_lookup_tables_for_rosbag(rosbag_name: str, use_cache: bool = True) -> pd.DataFrame:
    """Load and combine all mcap CSV files for a rosbag.
    
    Args:
        rosbag_name: Name of the rosbag (directory name, not full path)
        use_cache: Whether to use cached data if available
    
    Returns:
        Combined DataFrame with all lookup table data, or empty DataFrame if none found.
    """

    if not LOOKUP_TABLES:
        return pd.DataFrame()
    
    lookup_rosbag_dir = LOOKUP_TABLES / rosbag_name
    if not lookup_rosbag_dir.exists():
        return pd.DataFrame()
    
    # Check cache
    if use_cache and rosbag_name in _lookup_table_cache:
        cached_df, cached_mtime = _lookup_table_cache[rosbag_name]
        # Check if any CSV file was modified
        csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
        if csv_files:
            latest_mtime = max(f.stat().st_mtime for f in csv_files)
            if latest_mtime <= cached_mtime:
                return cached_df
    
    csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    all_dfs = []
    latest_mtime = 0.0
    for csv_path in sorted(csv_files):
        mcap_id = csv_path.stem  # Extract mcap_id from filename (without .csv extension)
        try:
            stat = csv_path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
            df = pd.read_csv(csv_path, dtype=str)
            if len(df) > 0:
                # Add mcap_id as a column to track which mcap this row came from
                df['_mcap_id'] = mcap_id
                all_dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed to load CSV {csv_path}: {e}")
            continue
    
    if not all_dfs:
        return pd.DataFrame()
    
    result_df = pd.concat(all_dfs, ignore_index=True)
    
    # Cache the result
    if use_cache:
        _lookup_table_cache[rosbag_name] = (result_df, latest_mtime)
    
    return result_df

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment

ALIGNED_DATA = load_lookup_tables_for_rosbag(extract_rosbag_name_from_path(str(SELECTED_ROSBAG)))


# EXPORT_PROGRESS: Dictionary to track progress and status of export jobs
EXPORT_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for export..."}
SEARCH_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for search..."}

SEARCHED_ROSBAGS = []

# MAX_K: Number of top results to return for semantic search
MAX_K = 100

# Cache setup for expensive rosbag discovery
FILE_PATH_CACHE_TTL_SECONDS = 60
_matching_rosbag_cache = {"paths": [], "timestamp": 0.0}
_file_path_cache_lock = Lock()

_positional_lookup_cache: dict[str, dict[str, dict[str, int]]] = {"data": None, "mtime": None}  # type: ignore[assignment]

CUSTOM_MODEL_DEFAULTS = {
    "agriclip": OTHER_MODELS / "agriclip.pt",
    "epoch32": OTHER_MODELS / "epoch_32.pt",
}

# Used to track the currently selected reference timestamp and its aligned mappings
# current_reference_timestamp: The reference timestamp selected by the user
# mapped_timestamps: Dictionary mapping topic names to their corresponding timestamps for the selected reference timestamp
current_reference_timestamp = None
mapped_timestamps = {}

def resolve_custom_checkpoint(model_name: str) -> Path:
    """Return a filesystem path to the checkpoint for a custom (non-open_clip) model."""
    candidates = [
        OTHER_MODELS / f"{model_name}.pt",
        OTHER_MODELS / model_name,
        CUSTOM_MODEL_DEFAULTS.get(model_name),
        Path(model_name) if model_name.endswith(".pt") else None,
    ]
    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        tried.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate checkpoint for custom model '{model_name}'. Tried: {', '.join(tried) or 'no candidates'}"
    )

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

# Endpoint to set the current reference timestamp and retrieve its aligned mappings
@app.route('/api/set-reference-timestamp', methods=['POST'])
def set_reference_timestamp():
    global current_reference_timestamp, mapped_timestamps
    try:
        data = request.get_json()
        referenceTimestamp = data.get('referenceTimestamp')
        
        if not referenceTimestamp:
            return jsonify({"error": "Missing referenceTimestamp"}), 400

        row = ALIGNED_DATA[ALIGNED_DATA["Reference Timestamp"] == str(referenceTimestamp)]
        if row.empty:
            return jsonify({"error": "Reference timestamp not found in CSV"}), 404

        current_reference_timestamp = referenceTimestamp
        mapped_timestamps = row.iloc[0].to_dict()
        # Extract mcap_identifier from the row (exclude it from mapped_timestamps)
        mcap_identifier = mapped_timestamps.pop('_mcap_id', None)
        # Convert NaNs to None for safe JSON serialization
        cleaned_mapped_timestamps = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in mapped_timestamps.items()
        }
        return jsonify({"mappedTimestamps": cleaned_mapped_timestamps, "mcapIdentifier": mcap_identifier, "message": "Reference timestamp updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Compute normalized CLIP embedding vector for a text query
def get_text_embedding(text, model, tokenizer, device):
    with torch.no_grad():
        tokens = tokenizer([text])
        tokens = tokens.to(device)
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Minimal CLIP implementation for custom checkpoints (AgriCLIP, epoch32, etc.)
# ---------------------------------------------------------------------------

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
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
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
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
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
        x = self.conv1(x)  # [batch, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [batch, grid, width]

        class_embedding = self.class_embedding.to(x.dtype)
        batch_class = class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])  # CLS token

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_width: int,
        vision_layers: int,
        vision_patch_size: int,
        image_resolution: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(
            transformer_width,
            transformer_layers,
            transformer_heads,
            attn_mask=self.build_text_mask(context_length),
        )
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

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def count_layers(prefix: str, state_dict: dict[str, torch.Tensor]) -> int:
    layers = set()
    prefix_parts = prefix.split(".")
    index = len(prefix_parts)
    for key in state_dict.keys():
        if key.startswith(prefix):
            layer_id = key.split(".")[index]
            layers.add(int(layer_id))
    return len(layers)


def build_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> CLIP:
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

    model = CLIP(
        embed_dim=embed_dim,
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_patch_size=vision_patch_size,
        image_resolution=image_resolution,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )

    if dtype == torch.float16:
        model = model.half()
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _first_matching_key(container: dict, keys: Iterable[str]) -> dict | None:
    for key in keys:
        if key in container and isinstance(container[key], dict):
            return container[key]
    return None


def load_agriclip(checkpoint_path: str, device: str = "cpu") -> CLIP:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict):
        candidate = _first_matching_key(raw, ("state_dict", "model_state_dict", "model", "ema_state_dict"))
        if candidate is not None:
            state_dict = candidate
        else:
            state_dict = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            if not state_dict:
                raise ValueError(f"Unsupported checkpoint structure: keys={list(raw.keys())[:10]}")
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
        raise KeyError(
            "Checkpoint does not contain CLIP visual weights under expected keys. "
            "Verify that the checkpoint was produced from a CLIP-compatible model."
        )

    model = build_model_from_state_dict(state_dict)
    model.to(torch.device(device))
    return model


def tokenize_texts(texts: Sequence[str], context_length: int, device: str) -> torch.Tensor:
    if open_clip_tokenize is None:
        raise ImportError("Tokenization requires open-clip-torch; install it with `pip install open-clip-torch`.")
    tokens = open_clip_tokenize(texts, context_length=context_length)
    return tokens.to(device)

# Canvas config utility functions (UI state persistence)
def load_canvases():
    if os.path.exists(CANVASES_FILE):
        try:
            with open(CANVASES_FILE, "r") as f:
                content = f.read().strip()
                # If file is empty or only whitespace, return empty dict
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # If JSON is invalid, return empty dict and log warning
            logging.warning(f"Invalid JSON in {CANVASES_FILE}, returning empty dict")
            return {}
    return {}

def save_canvases(data):
    with open(CANVASES_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Copy messages from one Rosbag to another within a timestamp range and selected topics
def export_rosbag_with_topics(src: Path, dst: Path, includedTopics, start_timestamp, end_timestamp) -> None:
    global EXPORT_PROGRESS
    
    # Reset export status at the beginning
    EXPORT_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for export..."}
    
    reader = SequentialReader()
    writer = SequentialWriter()
    
    try:
        # Open reader
        reader.open(
            StorageOptions(uri=str(src), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Get topic metadata from reader
        all_topics = reader.get_all_topics_and_types()
        topic_type_map = {topic_meta.name: topic_meta.type for topic_meta in all_topics}
        
        # Create topic map: track which topics we should write
        topic_map = {}  # Maps topic name to bool (whether to include)
        topics_to_create = {}  # Maps topic name to topic type
        
        for topic_meta in all_topics:
            if topic_meta.name in includedTopics:
                topic_map[topic_meta.name] = True
                topics_to_create[topic_meta.name] = topic_meta.type
        
        # Count total messages for progress tracking using MCAP summary
        # This is much faster than iterating through all messages
        total_msgs = 0
        with open(src, "rb") as mcap_file:
            mcap_reader = SeekingReader(mcap_file, decoder_factories=[DecoderFactory()])
            summary = mcap_reader.get_summary()
            
            if summary and summary.channels and summary.statistics:
                channels = summary.channels
                channel_message_counts = summary.statistics.channel_message_counts
                
                # Sum message counts for included topics
                for channel_id, channel in channels.items():
                    if channel.topic in topic_map:
                        message_count = channel_message_counts.get(channel_id, 0)
                        total_msgs += message_count
        
        # Open writer
        writer.open(
            StorageOptions(uri=str(dst), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Create topics in writer
        for topic_name, topic_type in topics_to_create.items():
            writer.create_topic(
                TopicMetadata(
                    name=topic_name,
                    type=topic_type,
                    serialization_format="cdr"
                )
            )
        
        EXPORT_PROGRESS["status"] = "starting"
        EXPORT_PROGRESS["progress"] = -1
        
        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = 0.00
        
        msg_counter = 0
        
        # Second pass: read and write messages
        while reader.has_next():
            read_topic, data, log_time = reader.read_next()
            
            # Skip topics not in included list
            if read_topic not in topic_map:
                continue
            
            msg_counter += 1
            current_progress = (msg_counter / total_msgs) if total_msgs > 0 else 0.0
            EXPORT_PROGRESS["progress"] = round(current_progress, 2)
            
            # Get topic type for deserialization
            topic_type = topic_type_map.get(read_topic)
            if not topic_type:
                continue
            
            msg_type = get_message(topic_type)
            
            # Deserialize message to extract header timestamp
            try:
                msg = deserialize_message(data, msg_type)
                # Extract header timestamp (not log_time) for filtering
                header_timestamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            except Exception as e:
                # If header extraction fails, fallback to log_time
                logging.warning(f"Could not extract header timestamp from {read_topic}: {e}")
                header_timestamp = log_time
                # Deserialize for writing even if header extraction failed
                try:
                    msg = deserialize_message(data, msg_type)
                except Exception as e2:
                    logging.warning(f"Could not deserialize message from {read_topic}: {e2}")
                    continue
            
            # Filter by header timestamp range (not log_time)
            if not (start_timestamp <= header_timestamp <= end_timestamp):
                continue
            
            # Handle header stamp normalization (if needed)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                # Header stamp is already in the message, no need to normalize
                pass
            
            # Write message using log_time as recording timestamp
            serialized_msg = serialize_message(msg)
            writer.write(read_topic, serialized_msg, log_time)
        
        EXPORT_PROGRESS["status"] = "done"
        EXPORT_PROGRESS["progress"] = 1.0
        
        # Clean up
        del reader
        del writer
    except Exception as e:
        # Clean up on error
        try:
            del reader
        except NameError:
            pass
        try:
            del writer
        except NameError:
            pass
        EXPORT_PROGRESS["status"] = "error"
        logging.error(f"Error exporting rosbag: {e}")
        raise

# Return the current export progress and status
@app.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)

# Get all available models for semantisch searching    
@app.route('/api/get-models', methods=['GET'])
def get_models():
    """
    List available CLIP models (based on embeddings directory).
    """
    try:
        models = []
        for model in os.listdir(EMBEDDINGS):
            if not model in ['.DS_Store', 'README.md', 'completion.json']:
                models.append(model)
        return jsonify({"models": models}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500

# Set the correct file paths for the selected rosbag and the lookup table after selecting a rosbag
@app.route('/api/set-file-paths', methods=['POST'])
def post_file_paths():
    """
    Set the current Rosbag file and update alignment data accordingly.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        global SELECTED_ROSBAG
        SELECTED_ROSBAG = path_value

        global ALIGNED_DATA
        rosbag_name = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        ALIGNED_DATA = load_lookup_tables_for_rosbag(rosbag_name)

        global SEARCHED_ROSBAGS
        SEARCHED_ROSBAGS = [path_value]  # Reset searched rosbags to the selected one
        logging.warning(SEARCHED_ROSBAGS)
        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_positional_lookup() -> dict[str, dict[str, dict[str, int | dict[str, int]]]]:
    """
    Load and cache the positional lookup JSON, refreshing when the file changes.
    
    Returns structure: {
        "rosbag_name": {
            "lat,lon": {
                "total": int,
                "mcaps": {"mcap_id": int, ...}
            }
        }
    }
    """
    if not POSITIONAL_LOOKUP_TABLE.exists():
        raise FileNotFoundError(f"Positional lookup file not found at {POSITIONAL_LOOKUP_TABLE}")

    stat = POSITIONAL_LOOKUP_TABLE.stat()
    cached_mtime = _positional_lookup_cache.get("mtime")
    if _positional_lookup_cache.get("data") is None or cached_mtime != stat.st_mtime:
        with POSITIONAL_LOOKUP_TABLE.open("r", encoding="utf-8") as fp:
            _positional_lookup_cache["data"] = json.load(fp)
        _positional_lookup_cache["mtime"] = stat.st_mtime

    return _positional_lookup_cache["data"]  # type: ignore[return-value]


@app.route('/api/positions/rosbags', methods=['GET'])
def get_positions_rosbags():
    """
    Return the list of rosbag names available in the positional lookup table.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_names = sorted(lookup.keys())
        return jsonify({"rosbags": rosbag_names}), 200
    except FileNotFoundError:
        return jsonify({"rosbags": []}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/positions/rosbags/<path:rosbag_name>', methods=['GET'])
def get_positional_rosbag_entries(rosbag_name: str):
    """
    Return the positional lookup entries for a specific rosbag.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        points = []
        for lat_lon, location_data in rosbag_data.items():
            try:
                lat_str, lon_str = lat_lon.split(',')
                count = int(location_data["total"])
                
                points.append({
                    "lat": float(lat_str),
                    "lon": float(lon_str),
                    "count": count
                })
            except (ValueError, TypeError, KeyError):
                continue

        points.sort(key=lambda item: item["count"], reverse=True)

        return jsonify({
            "rosbag": rosbag_name,
            "points": points
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/positions/rosbags/<path:rosbag_name>/mcaps', methods=['GET'])
def get_positional_rosbag_mcaps(rosbag_name: str):
    """
    Return positional lookup entries for a specific rosbag grouped by location with per-mcap breakdown.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        points = []
        for lat_lon, location_data in rosbag_data.items():
            try:
                lat_str, lon_str = lat_lon.split(',')
                mcaps = location_data.get("mcaps", {})
                
                # Group by location, include all mcaps at this location
                if mcaps:
                    points.append({
                        "lat": float(lat_str),
                        "lon": float(lon_str),
                        "mcaps": {mcap_id: int(count) for mcap_id, count in mcaps.items()}
                    })
            except (ValueError, TypeError, KeyError):
                continue

        return jsonify({
            "rosbag": rosbag_name,
            "points": points
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/positions/rosbags/<path:rosbag_name>/mcap-list', methods=['GET'])
def get_positional_rosbag_mcap_list(rosbag_name: str):
    """
    Return positional lookup entries for a specific rosbag grouped by location with per-mcap breakdown.
    """
    try:
        lookup = _load_positional_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        mcap_counts: dict[str, int] = {}
        for location_data in rosbag_data.values():
            mcaps = location_data.get("mcaps", {})
            for mcap_id, count in mcaps.items():
                mcap_counts[mcap_id] = mcap_counts.get(mcap_id, 0) + int(count)

        # Convert to list of dicts and sort by mcap_id (numeric if possible, otherwise alphabetical)
        def sort_key(item):
            mcap_id = item[0]
            # Try to parse as integer for numeric sorting, otherwise use string
            try:
                return (0, int(mcap_id))  # Numeric IDs first
            except ValueError:
                return (1, mcap_id)  # Non-numeric IDs after
        
        mcap_list = [
            {"id": mcap_id, "totalCount": count}
            for mcap_id, count in sorted(mcap_counts.items(), key=sort_key)
        ]

        return jsonify({
            "rosbag": rosbag_name,
            "mcaps": mcap_list
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/polygons/list', methods=['GET'])
def get_polygon_files():
    """
    Return list of available polygon JSON files.
    """
    try:
        if not POLYGONS_DIR.exists():
            return jsonify({"error": "Polygons directory not found"}), 404
        
        json_files = sorted([f.name for f in POLYGONS_DIR.glob("*.json")])
        return jsonify({"files": json_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/polygons/<filename>', methods=['GET'])
def get_polygon_file(filename: str):
    """
    Return polygon JSON file content.
    Validates filename to prevent path traversal.
    """
    try:
        # Validate filename to prevent path traversal
        if not filename.endswith('.json') or '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = POLYGONS_DIR / filename
        if not file_path.exists():
            return jsonify({"error": f"File '{filename}' not found"}), 404
        
        # Ensure file is within POLYGONS_DIR (additional safety check)
        if not file_path.resolve().is_relative_to(POLYGONS_DIR.resolve()):
            return jsonify({"error": "Invalid file path"}), 400
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return jsonify(data), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/positions/all', methods=['GET'])
def get_positions_all():
    """
    Return aggregated positional lookup entries across all rosbags.
    """
    try:
        lookup = _load_positional_lookup()
        aggregated: dict[str, dict[str, float | int]] = {}

        for rosbag_data in lookup.values():
            for lat_lon, location_data in rosbag_data.items():
                try:
                    lat_str, lon_str = lat_lon.split(',')
                    count = int(location_data["total"])
                    
                    key = f"{float(lat_str):.6f},{float(lon_str):.6f}"
                    if key not in aggregated:
                        aggregated[key] = {
                            "lat": float(lat_str),
                            "lon": float(lon_str),
                            "count": count,
                        }
                    else:
                        aggregated[key]["count"] = int(aggregated[key]["count"]) + count  # type: ignore[index]
                except (ValueError, TypeError, KeyError):
                    continue

        points = sorted(
            (
                {
                    "lat": value["lat"],
                    "lon": value["lon"],
                    "count": int(value["count"]),
                }
                for value in aggregated.values()
            ),
            key=lambda item: item["count"],
            reverse=True,
        )

        return jsonify({"points": points}), 200
    except FileNotFoundError:
        return jsonify({"error": "Positional lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Get all available rosbags
@app.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    """
    Return all available Rosbag file paths (excluding those in EXCLUDED).
    Recursively scans ROSBAGS directory and collects all leaf directories.
    """
    try:
        now = time()
        with _file_path_cache_lock:
            cached_paths = list(_matching_rosbag_cache["paths"])
            cached_timestamp = _matching_rosbag_cache["timestamp"]

        cache_age = now - cached_timestamp
        #logging.warning(f"[FILTER] Cache check: cached_paths count={len(cached_paths)}, cache_age={cache_age:.2f}s, TTL={FILE_PATH_CACHE_TTL_SECONDS}s")
        
        if (
            cached_paths
            and cache_age < FILE_PATH_CACHE_TTL_SECONDS
        ):
            #logging.warning(f"[FILTER] Returning cached paths (age: {cache_age:.2f}s < TTL: {FILE_PATH_CACHE_TTL_SECONDS}s)")
            return jsonify({"paths": cached_paths}), 200
        
        #logging.warning(f"[FILTER] Cache expired or empty, rebuilding...")

        rosbag_paths: list[str] = []
        base_path = Path(ROSBAGS)
        
        if not base_path.exists():
            return jsonify({"paths": []}), 200
        
        # Recursively walk through all directories
        for root, dirs, files in os.walk(str(base_path), topdown=True):
            # Skip EXCLUDED and EXPORTED directories
            if "EXCLUDED" in root or "EXPORTED" in root:
                dirs[:] = []  # Don't recurse into excluded directories
                continue
            
            # Remove excluded subdirectories from dirs list to skip them
            dirs[:] = [d for d in dirs if "EXCLUDED" not in d and "EXPORTED" not in d]
            
            # If this directory has no subdirectories (or only excluded ones), it's a leaf
            if not dirs:
                root_path = Path(root)
                # Skip the base directory itself
                if root_path != base_path:
                    relative_path = root_path.relative_to(base_path)
                    rosbag_paths.append(str(relative_path))
        
        rosbag_paths.sort()
        
        # Filter rosbags to only include those that have embeddings in at least one model
        #logging.warning(f"[FILTER] Starting filtering: EMBEDDINGS={EMBEDDINGS}, exists={EMBEDDINGS.exists() if EMBEDDINGS else False}")
        #logging.warning(f"[FILTER] ROSBAGS={ROSBAGS}, total rosbag_paths before filtering: {len(rosbag_paths)}")
        
        if EMBEDDINGS and EMBEDDINGS.exists():
            rosbags_base_path = Path(ROSBAGS)
            #logging.warning(f"[FILTER] rosbags_base_path={rosbags_base_path}")
            
            # Filter: check if rosbag exists in ROSBAGS AND in EMBEDDINGS (any model)
            original_count = len(rosbag_paths)
            filtered_paths = []
            
            for relative_path in rosbag_paths:
                # Check if rosbag exists in ROSBAGS
                rosbag_full_path = rosbags_base_path / relative_path
                rosbag_exists = rosbag_full_path.exists() and rosbag_full_path.is_dir()
                #logging.warning(f"[FILTER] Checking rosbag: {relative_path}")
                #logging.warning(f"[FILTER]   ROSBAGS path: {rosbag_full_path}, exists: {rosbag_exists}")
                
                if not rosbag_exists:
                    #logging.warning(f"[FILTER]   Filtering out: {relative_path} - doesn't exist in ROSBAGS")
                    continue
                
                # Check if this rosbag exists in EMBEDDINGS for any model
                found_in_embeddings = False
                for model_dir in EMBEDDINGS.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    embeddings_rosbag_path = model_dir / relative_path
                    if embeddings_rosbag_path.exists() and embeddings_rosbag_path.is_dir():
                        #logging.warning(f"[FILTER]   Found in EMBEDDINGS: {model_dir.name}/{relative_path}")
                        found_in_embeddings = True
                        break
                
                if found_in_embeddings:
                    filtered_paths.append(relative_path)
            
            rosbag_paths = filtered_paths

        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(rosbag_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": rosbag_paths}), 200
    except Exception as e:
        logging.error(f"Error in get_file_paths: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
# Get the currently selected rosbag
@app.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    """
    Return the currently selected Rosbag file name.
    """
    try:
        selectedRosbag = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500

# Get available topics from pre-generated JSON file for the current Rosbag
@app.route('/api/get-available-topics', methods=['GET'])
def get_available_rosbag_topics():
    try:
        rosbag_name = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopics': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        # Topics is now a dict mapping topic names to types
        # Extract the topic names (keys) as a list
        topics_dict = topics_data.get("topics", {})
        if isinstance(topics_dict, dict):
            topics = list(topics_dict.keys())
            topic_types = topics_dict  # Use the dict itself as topic_types mapping
        else:
            # Fallback for old format where topics was a list
            topics = topics_dict if isinstance(topics_dict, list) else []
            topic_types = {}
        
        # Sort topics using the default priority order
        topics = sort_topics(topics, topic_types)
        
        return jsonify({'availableTopics': topics}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopics': []}), 200
    

@app.route('/api/get-available-image-topics', methods=['GET'])
def get_available_image_topics():
    try:
        model_params = request.args.getlist("models")
        rosbag_params = request.args.getlist("rosbags")

        if not model_params or not rosbag_params:
            return jsonify({'availableTopics': {}}), 200

        results = {}

        for model_param in model_params:
            model_path = os.path.join(ADJACENT_SIMILARITIES, model_param)
            if not os.path.isdir(model_path):
                continue

            model_entry = {}
            for rosbag_param in rosbag_params:
                rosbag_name = extract_rosbag_name_from_path(rosbag_param)
                rosbag_path = os.path.join(model_path, rosbag_name)
                if not os.path.isdir(rosbag_path):
                    continue

                topics = []
                for topic in os.listdir(rosbag_path):
                    topics.append(topic.replace("__", "/"))

                # Try to load topic types from topics JSON file
                topic_types = {}
                try:
                    topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")
                    if os.path.exists(topics_json_path):
                        with open(topics_json_path, 'r') as f:
                            topics_data = json.load(f)
                            topics_dict = topics_data.get("topics", {})
                            if isinstance(topics_dict, dict):
                                topic_types = topics_dict
                except Exception as e:
                    logging.debug(f"Could not load topic types for {rosbag_name}: {e}")

                # Sort topics using the default priority order
                model_entry[rosbag_name] = sort_topics(topics, topic_types)

            if model_entry:
                results[model_param] = model_entry

        return jsonify({'availableTopics': results}), 200

    except Exception as e:
        logging.error(f"Error scanning adjacent similarities: {e}")
        return jsonify({'availableTopics': {}}), 200

# Read list of topic types from generated JSON for current Rosbag
@app.route('/api/get-available-topic-types', methods=['GET'])
def get_available_rosbag_topic_types():
    try:
        rosbag_name = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopicTypes': {}}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        # Topics is now a dict mapping topic names to types
        # Return the topics dict directly as it contains the mapping
        topics_dict = topics_data.get("topics", {})
        if isinstance(topics_dict, dict):
            availableTopicTypes = topics_dict
        else:
            # Fallback for old format where types was in a separate "types" field
            availableTopicTypes = topics_data.get("types", {})
        
        return jsonify({'availableTopicTypes': availableTopicTypes}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopicTypes': {}}), 200
    
# Returns list of all reference timestamps used for data alignment
@app.route('/api/get-available-timestamps', methods=['GET'])
def get_available_timestamps():
    availableTimestamps = ALIGNED_DATA['Reference Timestamp'].astype(str).tolist()
    return jsonify({'availableTimestamps': availableTimestamps}), 200

@app.route('/api/get-timestamp-lengths', methods=['GET'])
def get_timestamp_lengths():
    rosbags = request.args.getlist("rosbags")
    topics = request.args.getlist("topics")  # Optional: topics to get counts for
    timestampLengths = {}

    for rosbag in rosbags:
        rosbag_name = extract_rosbag_name_from_path(rosbag)
        try:
            df = load_lookup_tables_for_rosbag(rosbag_name)
            if df.empty:
                if topics:
                    timestampLengths[rosbag] = {topic: 0 for topic in topics}
                else:
                    timestampLengths[rosbag] = 0
            else:
                if topics:
                    # Return counts per topic
                    topic_counts = {}
                    for topic in topics:
                        if topic in df.columns:
                            count = df[topic].notnull().sum()
                            topic_counts[topic] = int(count)
                        else:
                            topic_counts[topic] = 0
                    timestampLengths[rosbag] = topic_counts
                else:
                    # Backward compatibility: return total count if no topics specified
                    count = df['Reference Timestamp'].notnull().sum() if 'Reference Timestamp' in df.columns else len(df)
                    timestampLengths[rosbag] = int(count)
        except Exception as e:
            if topics:
                timestampLengths[rosbag] = {topic: f"Error: {str(e)}" for topic in topics}
            else:
                timestampLengths[rosbag] = f"Error: {str(e)}"

    return jsonify({'timestampLengths': timestampLengths})

# Returns density of valid data fields per timestamp (used for heatmap intensity visualization)
@app.route('/api/get-timestamp-density', methods=['GET'])
def get_timestamp_density():
    density_array = ALIGNED_DATA.drop(columns=["Reference Timestamp"]).notnull().sum(axis=1).tolist()
    return jsonify({'timestampDensity': density_array})

@app.route('/api/get-topic-mcap-mapping', methods=['GET'])
def get_topic_mcap_mapping():
    """Get mcap_identifier ranges for Reference Timestamp indices.
    
    Returns ranges for ALL topics in the rosbag (mcap_id ranges are the same for all topics).
    Only requires relative_rosbag_path parameter.
    
    Returns contiguous ranges where each range has the same mcap_identifier.
    Each range contains only startIndex and mcap_identifier (endIndex can be derived from next range's startIndex - 1).
    
    Optimized for speed:
    - Cached dataframe loading
    - Minimal data returned (no topicTimestamp, no endIndex)
    - Efficient single-pass iteration
    - One call returns ranges for all topics
    """
    try:
        relative_rosbag_path = request.args.get('relative_rosbag_path')
        
        if not relative_rosbag_path:
            return jsonify({'error': 'Missing required parameter: relative_rosbag_path'}), 400
        
        # Convert relative path to full path, then extract rosbag name
        full_rosbag_path = str(ROSBAGS / relative_rosbag_path)
        rosbag_name = extract_rosbag_name_from_path(full_rosbag_path)
        
        # Load lookup tables for this rosbag (cached)
        df = load_lookup_tables_for_rosbag(rosbag_name, use_cache=True)
        if df.empty:
            return jsonify({'error': 'No lookup table data found'}), 404
        
        # Sort by Reference Timestamp to ensure consistent ordering (only if not already sorted)
        if 'Reference Timestamp' in df.columns:
            # Check if already sorted by checking if first < last
            if len(df) > 1:
                first_ts = df.iloc[0]['Reference Timestamp']
                last_ts = df.iloc[-1]['Reference Timestamp']
                try:
                    if float(first_ts) > float(last_ts):
                        df = df.sort_values('Reference Timestamp').reset_index(drop=True)
                except (ValueError, TypeError):
                    df = df.sort_values('Reference Timestamp').reset_index(drop=True)
        
        total = len(df)  # Number of Reference Timestamps (rows)
        
        if total == 0:
            return jsonify({'ranges': [], 'total': 0}), 200
        
        # Build ranges: group contiguous indices with the same mcap_identifier
        # The mcap_id ranges are the same for all topics (based on _mcap_id column)
        ranges = []
        current_range_start = 0
        current_mcap_id = df.iloc[0].get('_mcap_id')
        
        # Single pass: iterate through dataframe and detect mcap_id changes
        for idx in range(1, total):
            mcap_id = df.iloc[idx].get('_mcap_id')
            
            # If mcap_id changed, close current range and start new one
            if mcap_id != current_mcap_id:
                ranges.append({
                    'startIndex': current_range_start,
                    'mcap_identifier': current_mcap_id
                })
                current_range_start = idx
                current_mcap_id = mcap_id
        
        # Close the last range
        ranges.append({
            'startIndex': current_range_start,
            'mcap_identifier': current_mcap_id
        })
        
        # Sort ranges by numeric MCAP ID (not alphabetically)
        def get_mcap_id_numeric(mcap_id):
            """Extract numeric value from MCAP ID for sorting."""
            if mcap_id is None:
                return float('inf')
            try:
                # Try to convert directly to int
                return int(mcap_id)
            except (ValueError, TypeError):
                # If it's a string, try to extract number from it
                try:
                    # Handle cases like "mcap_10" or "10_mcap"
                    numbers = re.findall(r'\d+', str(mcap_id))
                    if numbers:
                        return int(numbers[0])
                    return float('inf')
                except:
                    return float('inf')
        
        ranges.sort(key=lambda r: (get_mcap_id_numeric(r['mcap_identifier']), r['startIndex']))
        
        return jsonify({
            'ranges': ranges,
            'total': total
        }), 200
        
    except Exception as e:
        logging.error(f"Error in get_topic_mcap_mapping: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-topic-timestamp-at-index', methods=['GET'])
def get_topic_timestamp_at_index():
    """Get topic timestamp for a specific index. Lightweight endpoint for hover previews."""
    try:
        relative_rosbag_path = request.args.get('relative_rosbag_path')
        topic = request.args.get('topic')
        index = request.args.get('index', type=int)
        
        if not relative_rosbag_path or not topic or index is None:
            return jsonify({'error': 'Missing required parameters: relative_rosbag_path, topic, index'}), 400
        
        # Convert relative path to full path, then extract rosbag name
        full_rosbag_path = str(ROSBAGS / relative_rosbag_path)
        rosbag_name = extract_rosbag_name_from_path(full_rosbag_path)
        df = load_lookup_tables_for_rosbag(rosbag_name, use_cache=True)
        
        if df.empty or index < 0 or index >= len(df):
            return jsonify({'error': 'Index out of range'}), 404
        
        # Sort if needed
        if 'Reference Timestamp' in df.columns and len(df) > 1:
            first_ts = df.iloc[0]['Reference Timestamp']
            last_ts = df.iloc[-1]['Reference Timestamp']
            try:
                if float(first_ts) > float(last_ts):
                    df = df.sort_values('Reference Timestamp').reset_index(drop=True)
            except (ValueError, TypeError):
                df = df.sort_values('Reference Timestamp').reset_index(drop=True)
        
        if topic not in df.columns:
            return jsonify({'error': f'Topic column "{topic}" not found'}), 404
        
        topic_ts = df.iloc[index][topic]
        
        # Helper to check if value is non-nan
        def non_nan(v):
            if v is None:
                return False
            if isinstance(v, float):
                return not math.isnan(v)
            s = str(v)
            return s.lower() != 'nan' and s != '' and s != 'None'
        
        # If empty, search nearby (up to 100 rows)
        if not non_nan(topic_ts):
            radius = min(100, len(df) - 1)
            for off in range(1, radius + 1):
                for candidate_idx in [index - off, index + off]:
                    if 0 <= candidate_idx < len(df):
                        candidate_ts = df.iloc[candidate_idx][topic]
                        if non_nan(candidate_ts):
                            topic_ts = candidate_ts
                            break
                if non_nan(topic_ts):
                    break
        
        if not non_nan(topic_ts):
            return jsonify({'error': 'No valid timestamp found'}), 404
        
        try:
            timestamp_ns = topic_ts
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid timestamp value'}), 400
        
        return jsonify({'topicTimestamp': timestamp_ns}), 200
        
    except Exception as e:
        logging.error(f"Error in get_topic_timestamp_at_index: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.route('/adjacency-image/<path:adjacency_image_path>')
def serve_adjacency_image(adjacency_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(ADJACENT_SIMILARITIES, adjacency_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/image-topic-preview/<path:image_topic_preview_path>')
def serve_image_topic_preview(image_topic_preview_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(IMAGE_TOPIC_PREVIEWS, image_topic_preview_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

def normalize_topic(topic: str) -> str:
    """Normalize topic name: replace / with _ and remove leading _"""
    return topic.replace('/', '_').lstrip('_')

def format_message_response(msg, topic_type: str, timestamp: str):
    """
    Format a deserialized ROS2 message into JSON response based on message type.
    
    Args:
        msg: Deserialized ROS2 message object
        topic_type: String type of the message (e.g., 'sensor_msgs/msg/PointCloud2')
        timestamp: Timestamp string to include in response
    
    Returns:
        Flask jsonify response
    """
    match topic_type:
        case 'sensor_msgs/msg/CompressedImage' | 'sensor_msgs/msg/Image':
            # Extract image data
            image_data_bytes = bytes(msg.data) if hasattr(msg, 'data') else None
            if not image_data_bytes:
                return jsonify({'error': 'No image data found'}), 404
            
            # Detect format
            format_str = 'jpeg'  # default
            if hasattr(msg, 'format') and msg.format:
                format_lower = msg.format.lower()
                if 'png' in format_lower:
                    format_str = 'png'
                elif 'jpeg' in format_lower or 'jpg' in format_lower:
                    format_str = 'jpeg'
            
            # Encode to base64
            image_data_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
            
            return jsonify({
                'type': 'image',
                'image': image_data_base64,
                'format': format_str,
                'timestamp': timestamp
            })
        case 'sensor_msgs/msg/PointCloud2':
            # Extract point cloud data, filtering out NaNs, Infs, and zeros
            pointCloud = []
            colors = []
            point_step = msg.point_step
            is_bigendian = msg.is_bigendian
            
            # Check fields to find color data
            has_rgb = False
            has_separate_rgb = False
            rgb_offset = None
            r_offset = None
            g_offset = None
            b_offset = None
            rgb_datatype = None
            r_datatype = None
            g_datatype = None
            b_datatype = None
            
            for field in msg.fields:
                # Debug: log all fields to understand structure
                logging.debug(f"PointCloud2 field: name={field.name}, offset={field.offset}, datatype={field.datatype}, count={field.count}")
                if field.name == 'rgb' or field.name == 'rgba':
                    rgb_offset = field.offset
                    rgb_datatype = field.datatype
                    has_rgb = True
                    logging.debug(f"Found RGB field: offset={rgb_offset}, datatype={rgb_datatype}")
                    break
                elif field.name == 'r':
                    r_offset = field.offset
                    r_datatype = field.datatype
                    has_separate_rgb = True
                elif field.name == 'g':
                    g_offset = field.offset
                    g_datatype = field.datatype
                elif field.name == 'b':
                    b_offset = field.offset
                    b_datatype = field.datatype
            
            logging.debug(f"Color detection: has_rgb={has_rgb}, has_separate_rgb={has_separate_rgb}, rgb_datatype={rgb_datatype}")
            
            # Extract points and colors
            for i in range(0, len(msg.data), point_step):
                # Extract x, y, z coordinates
                if is_bigendian:
                    x, y, z = struct.unpack_from('>fff', msg.data, i)
                else:
                    x, y, z = struct.unpack_from('<fff', msg.data, i)
                
                if all(np.isfinite([x, y, z])) and not (x == 0 and y == 0 and z == 0):
                    pointCloud.extend([x, y, z])
                    
                    # Extract RGB colors if available
                    if has_rgb and rgb_offset is not None:
                        try:
                            # Extract RGB as uint32 (4 bytes) - most common format
                            if is_bigendian:
                                rgb = struct.unpack_from('>I', msg.data, i + rgb_offset)[0]
                                # Big-endian: RRGGBBAA format
                                r = (rgb >> 24) & 0xFF
                                g = (rgb >> 16) & 0xFF
                                b = (rgb >> 8) & 0xFF
                            else:
                                rgb = struct.unpack_from('<I', msg.data, i + rgb_offset)[0]
                                # Little-endian: Try different formats
                                b = rgb & 0xFF
                                g = (rgb >> 8) & 0xFF
                                r = (rgb >> 16) & 0xFF
                            colors.extend([r, g, b])
                        except Exception as e:
                            logging.warning(f"Failed to extract RGB color at offset {i + rgb_offset}: {e}")
                            pass
                    elif has_separate_rgb and r_offset is not None and g_offset is not None and b_offset is not None:
                        # Extract separate r, g, b fields
                        try:
                            if r_datatype == 7:  # FLOAT32
                                if is_bigendian:
                                    r_val = struct.unpack_from('>f', msg.data, i + r_offset)[0]
                                    g_val = struct.unpack_from('>f', msg.data, i + g_offset)[0]
                                    b_val = struct.unpack_from('>f', msg.data, i + b_offset)[0]
                                else:
                                    r_val = struct.unpack_from('<f', msg.data, i + r_offset)[0]
                                    g_val = struct.unpack_from('<f', msg.data, i + g_offset)[0]
                                    b_val = struct.unpack_from('<f', msg.data, i + b_offset)[0]
                                # Convert float (0.0-1.0) to uint8 (0-255)
                                r = int(r_val * 255) if r_val <= 1.0 else int(r_val)
                                g = int(g_val * 255) if g_val <= 1.0 else int(g_val)
                                b = int(b_val * 255) if b_val <= 1.0 else int(b_val)
                            else:  # Assume uint8
                                r = struct.unpack_from('B', msg.data, i + r_offset)[0]
                                g = struct.unpack_from('B', msg.data, i + g_offset)[0]
                                b = struct.unpack_from('B', msg.data, i + b_offset)[0]
                            colors.extend([r, g, b])
                        except Exception as e:
                            logging.warning(f"Failed to extract separate RGB: {e}")
                            pass
            
            # Return with colors if available, otherwise just positions
            if colors:
                return jsonify({
                    'type': 'pointCloud',
                    'pointCloud': {'positions': pointCloud, 'colors': colors},
                    'timestamp': timestamp
                })
            else:
                return jsonify({
                    'type': 'pointCloud',
                    'pointCloud': {'positions': pointCloud, 'colors': []},
                    'timestamp': timestamp
                })
        case 'sensor_msgs/msg/NavSatFix':
            return jsonify({'type': 'position', 'position': {'latitude': msg.latitude, 'longitude': msg.longitude, 'altitude': msg.altitude}, 'timestamp': timestamp})
        case 'novatel_oem7_msgs/msg/BESTPOS':
            return jsonify({'type': 'position', 'position': {'latitude': msg.lat, 'longitude': msg.lon, 'altitude': msg.hgt}, 'timestamp': timestamp})
        case 'tf2_msgs/msg/TFMessage':
            # Assume single transform for simplicity
            if len(msg.transforms) > 0:
                transform = msg.transforms[0]
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                tf_data = {
                    'translation': {
                        'x': translation.x,
                        'y': translation.y,
                        'z': translation.z
                    },
                    'rotation': {
                        'x': rotation.x,
                        'y': rotation.y,
                        'z': rotation.z,
                        'w': rotation.w
                    }
                }
                return jsonify({'type': 'tf', 'tf': tf_data, 'timestamp': timestamp})
        case 'sensor_msgs/msg/Imu':
            imu_data = {
                "orientation": {
                    "x": msg.orientation.x,
                    "y": msg.orientation.y,
                    "z": msg.orientation.z,
                    "w": msg.orientation.w
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x,
                    "y": msg.angular_velocity.y,
                    "z": msg.angular_velocity.z
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x,
                    "y": msg.linear_acceleration.y,
                    "z": msg.linear_acceleration.z
                }
            }
            return jsonify({'type': 'imu', 'imu': imu_data, 'timestamp': timestamp})
        case 'nav_msgs/msg/Odometry':
            # Extract odometry data: pose (position + orientation) and twist (linear + angular velocity)
            odometry_data = {
                "header": {
                    "frame_id": msg.header.frame_id,
                    "stamp": {
                        "sec": msg.header.stamp.sec,
                        "nanosec": msg.header.stamp.nanosec
                    }
                },
                "child_frame_id": msg.child_frame_id,
                "pose": {
                    "position": {
                        "x": msg.pose.pose.position.x,
                        "y": msg.pose.pose.position.y,
                        "z": msg.pose.pose.position.z
                    },
                    "orientation": {
                        "x": msg.pose.pose.orientation.x,
                        "y": msg.pose.pose.orientation.y,
                        "z": msg.pose.pose.orientation.z,
                        "w": msg.pose.pose.orientation.w
                    },
                    "covariance": list(msg.pose.covariance) if hasattr(msg.pose, 'covariance') else []
                },
                "twist": {
                    "linear": {
                        "x": msg.twist.twist.linear.x,
                        "y": msg.twist.twist.linear.y,
                        "z": msg.twist.twist.linear.z
                    },
                    "angular": {
                        "x": msg.twist.twist.angular.x,
                        "y": msg.twist.twist.angular.y,
                        "z": msg.twist.twist.angular.z
                    },
                    "covariance": list(msg.twist.covariance) if hasattr(msg.twist, 'covariance') else []
                }
            }
            return jsonify({'type': 'odometry', 'odometry': odometry_data, 'timestamp': timestamp})
        case _:
            # Fallback for unsupported or unknown message types: return string representation
            return jsonify({'type': 'text', 'text': str(msg), 'timestamp': timestamp})

@app.route('/api/content-mcap', methods=['GET', 'POST'])
def get_content_mcap():
    relative_rosbag_path = request.args.get('relative_rosbag_path')
    topic = request.args.get('topic')
    mcap_identifier = request.args.get('mcap_identifier')
    timestamp = request.args.get('timestamp', type=int)
    
    # Validate required parameters
    if not topic or not mcap_identifier or timestamp is None:
        return jsonify({'error': 'Missing required parameters: topic, mcap_identifier, and timestamp are required'}), 400
    
    # Handle missing relative_rosbag_path - use SELECTED_ROSBAG as fallback
    if not relative_rosbag_path:
        if not SELECTED_ROSBAG:
            return jsonify({'error': 'No rosbag selected and relative_rosbag_path not provided'}), 400
        # Get relative path from SELECTED_ROSBAG
        # SELECTED_ROSBAG might be a string or Path, and might be absolute or relative
        selected_rosbag_str = str(SELECTED_ROSBAG)
        selected_rosbag_path = Path(selected_rosbag_str)
        
        # Check if it's an absolute path (starts with /) or relative
        if selected_rosbag_path.is_absolute():
            # Absolute path - get relative path from ROSBAGS
            try:
                relative_rosbag_path = str(selected_rosbag_path.relative_to(ROSBAGS))
            except ValueError:
                # If not a subpath, try to construct it from the basename
                relative_rosbag_path = selected_rosbag_path.name
        else:
            # Already a relative path - use it directly
            relative_rosbag_path = selected_rosbag_str
    
    # Extract base rosbag name for MCAP filename
    # For multipart rosbags like "rosbag2_2025_07_23-07_29_39_multi_parts/Part_2",
    # we need to get the base name (before _multi_parts) for the MCAP filename
    # MCAP files are named like: rosbag2_2025_07_23-07_29_39_669.mcap
    if '_multi_parts' in relative_rosbag_path:
        # Extract base name before _multi_parts
        base_rosbag_name = relative_rosbag_path.split('_multi_parts')[0]
    else:
        # For regular rosbags, use the full path as base name
        base_rosbag_name = relative_rosbag_path
    
    # Construct MCAP path using Path objects for proper handling
    # MCAP files are in the rosbag directory, named: {base_rosbag_name}_{mcap_identifier}.mcap
    mcap_path = ROSBAGS / relative_rosbag_path / f"{base_rosbag_name}_{mcap_identifier}.mcap"
    
    logging.warning(f"[CONTENT_MCAP] MCAP path: {mcap_path}")
    
    try:        
        with open(mcap_path, "rb") as f:
            reader = SeekingReader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                topics=[topic],
                start_time=timestamp,
                end_time=timestamp + 1,
                log_time_order=True,
                reverse=False
            ):
                # Get schema name for message type
                schema_name = schema.name if schema else None
                if not schema_name:
                    return jsonify({'error': 'No schema found for message'}), 404
                    
                # Use the shared format_message_response function
                # Convert timestamp to string to match the function signature
                return format_message_response(ros2_msg, schema_name, str(timestamp))
        
        return jsonify({'error': 'No message found for the provided timestamp and topic'}), 404
        
    except Exception as e:
        logging.exception("[C] Failed to read mcap file")
        return jsonify({'error': f'Error reading mcap file: {str(e)}'}), 500

@app.route('/api/enhance-prompt', methods=['GET'])
def enhance_prompt_endpoint():
    """Enhance a user prompt using the gemma model."""
    if not GEMMA_AVAILABLE:
        return jsonify({'error': 'Prompt enhancement not available (Gemma model not loaded)'}), 503

    user_prompt = request.args.get('prompt', default=None, type=str)
    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Update search status to show enhancement is in progress
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.0
    SEARCH_PROGRESS["message"] = "Enhancing prompt..."
    
    try:
        messages = [
            {
            "role": "user",
            "content": f"Return: 'A photo of' plus the query. Use 'a' or 'an' only if the query is singular, no article if plural. Add ', a type of' and the category if you know it.\nQuery: {user_prompt}"
            }
        ]
        
        inputs = gemma_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma_model.device)
                
        outputs = gemma_model.generate(**inputs, max_new_tokens=50)
        enhanced_prompt = gemma_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        enhanced_prompt = enhanced_prompt.replace('\n', ' ').replace('<end_of_turn>', '').strip()
        
        return jsonify({'original': user_prompt, 'enhanced': enhanced_prompt}), 200
    except Exception as e:
        logging.exception("[C] Failed to enhance prompt")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["message"] = f"Error enhancing prompt: {str(e)}"
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-status', methods=['GET'])
def get_search_status():
    return jsonify(SEARCH_PROGRESS)

@app.route('/api/search', methods=['GET'])
def search():
    from pathlib import Path
    import traceback

    # ---- Debug logging for paths
    logging.warning("[C] LOOKUP_TABLES=%s (type=%s, exists=%s)", LOOKUP_TABLES, type(LOOKUP_TABLES).__name__, LOOKUP_TABLES.exists() if LOOKUP_TABLES else False)
    logging.warning("[C] EMBEDDINGS=%s (type=%s, exists=%s)", EMBEDDINGS, type(EMBEDDINGS).__name__, EMBEDDINGS.exists() if EMBEDDINGS else False)

    # ---- Initial status
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00

    # ---- Inputs
    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)
    MAX_K = globals().get("MAX_K", 50)

    #logging.warning("[C] /api/search called with raw params: query=%r models=%r rosbags=%r timeRange=%r accuracy=%r MAX_K=%r",
    #                query_text, models, rosbags, timeRange, accuracy, MAX_K)

    # ---- Validate inputs
    if query_text is None:                 return jsonify({'error': 'No query text provided'}), 400
    if models is None:                     return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:                    return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:                  return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:                   return jsonify({'error': 'No accuracy provided'}), 400

    # ---- Parse inputs
    try:
        models_list = [m.strip() for m in models.split(",") if m.strip()]  # Filter out empty model names
        rosbags_list = [extract_rosbag_name_from_path(r.strip()) for r in rosbags.split(",") if r.strip()]
        time_start, time_end = map(int, timeRange.split(","))
        k_subsample = max(1, int(accuracy))
    except Exception as e:
        logging.exception("[C] Failed parsing inputs")
        return jsonify({'error': f'Invalid inputs: {e}'}), 400
    
    # Validate parsed inputs
    if not models_list:
        return jsonify({'error': 'No valid models provided (empty or whitespace-only)'}), 400
    if not rosbags_list:
        return jsonify({'error': 'No valid rosbags provided (empty or whitespace-only)'}), 400

    #logging.warning("[C] Parsed: models=%s rosbags=%s time_start=%d time_end=%d accuracy(subsample k)=%d",
     #               models_list, rosbags_list, time_start, time_end, k_subsample)

    # ---- State
    marks: dict[tuple, set] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results: list[dict] = []

    try:
        total_steps = max(1, len(models_list) * len(rosbags_list))
        step_count = 0
        logging.warning("[C] total_steps=%d device=%s torch.cuda.is_available()=%s", total_steps, device, torch.cuda.is_available())

        for model_name in models_list:
            model = None
            tokenizer = None
            query_embedding = None
            model_kind = "open_clip"

            try:
                if '__' in model_name:
                    name, pretrained = model_name.split('__', 1)
                    model, _, _ = open_clip.create_model_and_transforms(
                        name, pretrained, device=device, cache_dir=OPEN_CLIP_MODELS
                    )
                    tokenizer = open_clip.get_tokenizer(name)
                    query_embedding = get_text_embedding(query_text, model, tokenizer, device)
                else:
                    checkpoint_path = resolve_custom_checkpoint(model_name)
                    logging.warning("[C] Loading custom CLIP model '%s' from %s", model_name, checkpoint_path)
                    model = load_agriclip(str(checkpoint_path), device=device)
                    try:
                        tokens = tokenize_texts([query_text], model.context_length, device)
                    except ImportError as exc:
                        raise RuntimeError(
                            "Custom models require open-clip tokenization utilities. Install open-clip-torch or provide pre-tokenized input."
                        ) from exc
                    with torch.no_grad():
                        features = model.encode_text(tokens)
                        features = F.normalize(features, dim=-1)
                    query_embedding = features.detach().cpu().numpy().flatten()
                    model_kind = "custom"
            except Exception as e:
                logging.exception("[C] Failed to load model %s: %s", model_name, e)
                continue

            if query_embedding is None:
                logging.warning("[C] Query embedding is None for model %s; skipping", model_name)
                continue

            query_embedding = query_embedding.astype("float32", copy=False)

            for rosbag_name in rosbags_list:
                step_count += 1
                SEARCH_PROGRESS["status"] = "running"
                SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 3)
                SEARCH_PROGRESS["message"] = (
                    "Searching consolidated shards...\n\n"
                    f"Model: {model_name}\n"
                    f"Rosbag: {rosbag_name}\n\n"
                    f"(Sampling every {k_subsample}th embedding)"
                )

                # ---- Consolidated base
                base = EMBEDDINGS / model_name / rosbag_name
                manifest_path = base / "manifest.parquet"
                shards_dir = base / "shards"
                logging.warning("[C] EMBEDDINGS base path: %s (exists=%s)", base, base.exists() if base else False)
                logging.warning("[C] Base=%s  manifest=%s (exists=%s)  shards_dir=%s (exists=%s)",
                                base, manifest_path, manifest_path.exists(), shards_dir, shards_dir.exists())

                if not manifest_path.is_file() or not shards_dir.is_dir():
                    logging.warning("[C] SKIP: Missing manifest or shards for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Manifest schema check
                needed_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier"]
                try:
                    import pyarrow.parquet as pq
                    available_cols = set(pq.read_schema(manifest_path).names)
                    #logging.warning("[C] Manifest columns: %s", sorted(available_cols))
                except Exception as e:
                    logging.warning("[C] Could not read schema with pyarrow (%s), fallback to pandas head: %s", type(e).__name__, e)
                    head_df = pd.read_parquet(manifest_path)
                    available_cols = set(head_df.columns)
                    logging.warning("[C] Manifest columns (fallback): %s", sorted(available_cols))

                missing = [c for c in needed_cols if c not in available_cols]
                if missing:
                    msg = f"[C] Manifest missing columns {missing}; present={sorted(available_cols)}"
                    logging.warning(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                # ---- Read needed columns
                mf = pd.read_parquet(manifest_path, columns=needed_cols)
                
                """logging.warning("[C] mf.shape=%s  minute_of_day[min,max]=(%s,%s)  sample:\n%s",
                                mf.shape,
                                mf["minute_of_day"].min() if not mf.empty else None,
                                mf["minute_of_day"].max() if not mf.empty else None,
                                mf.head(3).to_dict(orient="records") if not mf.empty else [])"""

                # ---- Filter by time window
                pre_count = len(mf)
                mf = mf.loc[(mf["minute_of_day"] >= time_start) & (mf["minute_of_day"] <= time_end)]
                #logging.warning("[C] After time filter [%d..%d]: %d → %d rows", time_start, time_end, pre_count, len(mf))
                if mf.empty:
                    logging.warning("[C] SKIP: No rows in time window for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Subsample per topic
                parts, per_topic_counts_before, per_topic_counts_after = [], {}, {}
                for topic, df_t in mf.groupby("topic", sort=False):
                    per_topic_counts_before[topic] = len(df_t)
                    sort_cols = [c for c in ["timestamp_ns"] if c in df_t.columns]
                    if sort_cols:
                        df_t = df_t.sort_values(sort_cols)
                    parts.append(df_t.iloc[::k_subsample])
                mf_sel = pd.concat(parts, ignore_index=True) if parts else mf.iloc[0:0]
                for topic, df_t in mf_sel.groupby("topic", sort=False):
                    per_topic_counts_after[topic] = len(df_t)

                """logging.warning("[C] Topics before subsample: %d, after subsample: %d (k=%d)",
                                len(per_topic_counts_before), len(per_topic_counts_after), k_subsample)
                logging.warning("[C] Counts per topic (before) [first 5]: %s",
                                dict(list(per_topic_counts_before.items())[:5]))
                logging.warning("[C] Counts per topic (after)  [first 5]: %s",
                                dict(list(per_topic_counts_after.items())[:5]))
                logging.warning("[C] mf_sel.shape=%s sample:\n%s", mf_sel.shape,
                                mf_sel.head(3).to_dict(orient="records") if not mf_sel.empty else [])"""

                if mf_sel.empty:
                    logging.warning("[C] SKIP: No rows after subsample for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Load aligned CSV for marks
                lookup_dir = (LOOKUP_TABLES / rosbag_name) if LOOKUP_TABLES else None
                logging.warning("[C] LOOKUP_TABLES dir for %s: %s (exists=%s)", rosbag_name, lookup_dir, lookup_dir.exists() if lookup_dir else False)
                aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
                logging.warning("[C] Loaded lookup tables for %s: shape=%s, columns=%s", rosbag_name, aligned_data.shape if not aligned_data.empty else "empty", list(aligned_data.columns) if not aligned_data.empty else [])
                if aligned_data.empty:
                    logging.warning("[C] WARN: no lookup tables found for %s (marks will be empty)", rosbag_name)

                # ---- Gather vectors by shard
                chunks: list[np.ndarray] = []
                meta_for_row: list[dict] = []
                shards_touched = 0
                bytes_read = 0
                shard_ids = sorted(mf_sel["shard_id"].unique().tolist())
                #logging.warning("[C] Unique shard_ids=%d [first 5]=%s", len(shard_ids), shard_ids[:5])

                for shard_id, df_s in mf_sel.groupby("shard_id", sort=False):
                    shard_path = shards_dir / shard_id
                    if not shard_path.is_file():
                        #logging.warning("[C] Missing shard file: %s (skip)", shard_path)
                        continue

                    rows = df_s["row_in_shard"].to_numpy(np.int64)
                    rows.sort()
                    shards_touched += 1
                    #logging.warning("[C] Shard=%s rows=%d row[min,max]=(%s,%s)", shard_id, len(rows), rows.min(), rows.max())

                    # Build meta map (row_in_shard -> meta)
                    meta_map = {}
                    for r in df_s.itertuples(index=False):
                        topic_str = r.topic
                        topic_folder = topic_str.replace("/", "__")
                        meta_map[int(r.row_in_shard)] = {
                            "timestamp_ns": int(r.timestamp_ns),
                            "topic": topic_str.replace("__", "/"),
                            "topic_folder": topic_folder,
                            "minute_of_day": int(r.minute_of_day),
                            "mcap_identifier": str(r.mcap_identifier),
                            "shard_id": shard_id,
                            "row_in_shard": int(r.row_in_shard),
                        }

                    arr = np.load(shard_path, mmap_mode="r")  # expect float32 shards
                    if arr.dtype != np.float32:
                        logging.warning("[C] Shard %s dtype=%s (casting to float32)", shard_id, arr.dtype)
                        arr = arr.astype("float32", copy=False)

                    # Coalesce contiguous ranges → fewer slices
                    ranges = []
                    start = prev = int(rows[0])
                    for rr in rows[1:]:
                        rr = int(rr)
                        if rr == prev + 1:
                            prev = rr
                        else:
                            ranges.append((start, prev))
                            start = prev = rr
                    ranges.append((start, prev))

                    for a, b in ranges:
                        sl = arr[a:b+1]  # (len, D)
                        chunks.append(sl)
                        bytes_read += sl.nbytes
                        for i in range(a, b + 1):
                            meta_for_row.append(meta_map[i])

                #logging.warning("[C] chunks=%d  shards_touched=%d  bytes_read=%.2f MB", len(chunks), shards_touched, bytes_read / (1024*1024))

                if not chunks:
                    msg = f"[C] No chunks loaded (missing shards/files?) for {model_name}/{rosbag_name}"
                    logging.warning(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                X = np.vstack(chunks).astype("float32", copy=False)
                #logging.warning("[C] X.shape=%s  meta_for_row=%d  X.dtype=%s", X.shape, len(meta_for_row), X.dtype)
                if X.shape[0] != len(meta_for_row):
                    msg = f"[C] Row/meta mismatch: X={X.shape[0]} vs meta={len(meta_for_row)}"
                    logging.warning(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                # ---- FAISS search
                index = faiss.IndexFlatL2(X.shape[1])
                index.add(X)
                #logging.warning("[C] FAISS index added rows=%d dim=%d class=%s", X.shape[0], X.shape[1], index.__class__.__name__)
                D, I = index.search(query_embedding.reshape(1, -1), MAX_K)
                #logging.warning("[C] search returned shapes: D=%s I=%s", getattr(D, "shape", None), getattr(I, "shape", None))
                if D.size == 0 or I.size == 0:
                    logging.warning("[C] Empty FAISS result for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Build API-like results
                model_results: list[dict] = []
                for i, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
                    if idx < 0 or idx >= len(meta_for_row):
                        #logging.warning("[C] Skip invalid idx=%d (meta_for_row=%d)", idx, len(meta_for_row))
                        continue
                    m = meta_for_row[int(idx)]
                    ts_ns = m["timestamp_ns"]
                    ts_str = str(ts_ns)
                    # Convert UTC timestamp to Europe/Berlin timezone
                    berlin_tz = ZoneInfo("Europe/Berlin")
                    minute_str = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(berlin_tz).strftime("%H:%M")
                    
                    # Build embedding path with mcap_identifier
                    pt_path = EMBEDDINGS / model_name / rosbag_name / m["topic_folder"] / m["mcap_identifier"] / f"{ts_ns}.pt"
                    
                    if i == 1:  # Log first result path construction
                        logging.warning("[C] EMBEDDINGS path construction: %s (exists=%s)", pt_path, pt_path.exists() if pt_path else False)
                    
                    model_results.append({
                        'rank': i,
                        'rosbag': rosbag_name,
                        'embedding_path': str(pt_path),
                        'similarityScore': float(dist),
                        'topic': m["topic"],           # slashes
                        'timestamp': ts_str,
                        'minuteOfDay': minute_str,
                        'mcap_identifier': m["mcap_identifier"],
                        'model': model_name
                    })
                    if not aligned_data.empty:
                        matching_reference_timestamps = aligned_data.loc[
                            aligned_data.isin([ts_str]).any(axis=1),
                            'Reference Timestamp'
                        ].tolist()
                        if matching_reference_timestamps:
                            match_indices: list[int] = []
                            for ref_ts in matching_reference_timestamps:
                                idxs = aligned_data.index[aligned_data['Reference Timestamp'] == ref_ts].tolist()
                                match_indices.extend(idxs)
                            if match_indices:
                                key = (model_name, rosbag_name, m["topic"])
                                marks.setdefault(key, set()).update(match_indices)

                #logging.warning("[C] model_results built: %d (show 3): %s",
                #                len(model_results), model_results[:3])

                all_results.extend(model_results)
                #logging.warning("[C] all_results size now: %d", len(all_results))

            # Cleanup GPU per model
            del model
            torch.cuda.empty_cache()
            #logging.warning("[C] Freed model %s", model_name)

        # ---- Post processing
        all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
        for rank, result in enumerate(all_results, 1):
            result['rank'] = rank
        filtered_results = all_results

        #logging.warning("[C] FINAL results=%d (show 5): %s", len(filtered_results), filtered_results[:5])

        # marksPerTopic
        marksPerTopic: dict = {}
        for result in filtered_results:
            model = result['model']
            rosbag = result['rosbag']
            topic = result['topic']

            marksPerTopic \
                .setdefault(model, {}) \
                .setdefault(rosbag, {}) \
                .setdefault(topic, {'marks': []})

        for key, indices in marks.items():
            model_key, rosbag_key, topic_key = key
            if (
                model_key in marksPerTopic
                and rosbag_key in marksPerTopic[model_key]
                and topic_key in marksPerTopic[model_key][rosbag_key]
            ):
                marksPerTopic[model_key][rosbag_key][topic_key]['marks'].extend(
                    {'value': idx} for idx in indices
                )

        #logging.warning("[C] marksPerTopic models=%d", len(marksPerTopic))

        # Check if no results were found
        if not filtered_results:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
            SEARCH_PROGRESS["message"] = "No results found for the given query."
        else:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
        
        return jsonify({
            'query': query_text,
            'results': filtered_results,
            'marksPerTopic': marksPerTopic
        })

    except Exception as e:
        logging.exception("[C] search failed")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Error: {e}\n\n{traceback.format_exc()}"
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Export portion of a Rosbag containing selected topics and time range
@app.route('/api/export-rosbag', methods=['POST'])
def export_rosbag():
    """
    Export MCAP files from a rosbag based on MCAP ID range and time filtering.
    
    Request body:
    {
        "new_rosbag_name": str,
        "topics": List[str],
        "start_timestamp": int (nanoseconds),
        "end_timestamp": int (nanoseconds),
        "start_mcap_id": int,
        "end_mcap_id": int
    }
    """
    global SELECTED_ROSBAG, EXPORT_PROGRESS
    
    # Reset export status at the beginning
    EXPORT_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for export..."}
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters from request
        new_rosbag_name = data.get("new_rosbag_name")
        topics = data.get("topics", [])
        start_timestamp_raw = data.get("start_timestamp")
        end_timestamp_raw = data.get("end_timestamp")
        start_mcap_id_raw = data.get("start_mcap_id")
        end_mcap_id_raw = data.get("end_mcap_id")
        
        # Validate required parameters
        if not new_rosbag_name:
            EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": "new_rosbag_name is required"}
            return jsonify({"error": "new_rosbag_name is required"}), 400
        if start_timestamp_raw is None or end_timestamp_raw is None:
            EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": "start_timestamp and end_timestamp are required"}
            return jsonify({"error": "start_timestamp and end_timestamp are required"}), 400
        if start_mcap_id_raw is None or end_mcap_id_raw is None:
            EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": "start_mcap_id and end_mcap_id are required"}
            return jsonify({"error": "start_mcap_id and end_mcap_id are required"}), 400
        
        # Convert to integers (after validation to allow 0 values)
        try:
            start_timestamp = int(start_timestamp_raw)
            end_timestamp = int(end_timestamp_raw)
            start_mcap_id = int(start_mcap_id_raw)
            end_mcap_id = int(end_mcap_id_raw)
        except (ValueError, TypeError) as e:
            EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": f"Invalid number format: {e}"}
            return jsonify({"error": f"Invalid number format: {e}"}), 400

        # Set status to starting - export is beginning
        EXPORT_PROGRESS = {"status": "starting", "progress": -1, "message": "Validating export parameters..."}

        # Set paths
        # SELECTED_ROSBAG might be absolute or relative, handle both cases
        selected_rosbag_str = str(SELECTED_ROSBAG)
        selected_rosbag_path = Path(selected_rosbag_str)
        
        # Check if it's an absolute path or relative
        if selected_rosbag_path.is_absolute():
            # Absolute path - get relative path from ROSBAGS
            try:
                relative_rosbag_path = str(selected_rosbag_path.relative_to(ROSBAGS))
            except ValueError:
                # If not a subpath, use the path as-is (might be outside ROSBAGS)
                relative_rosbag_path = selected_rosbag_str
            input_rosbag_dir = ROSBAGS / relative_rosbag_path
        else:
            # Already a relative path - use it directly
            input_rosbag_dir = ROSBAGS / selected_rosbag_str
        
        output_rosbag_base = EXPORT
        output_rosbag_dir = output_rosbag_base / new_rosbag_name
        
        # Update export progress
        EXPORT_PROGRESS = {"status": "running", "progress": 0.0, "message": "Starting export..."}
        
        # Helper function to extract MCAP ID from filename
        def extract_mcap_id(mcap_path: Path) -> str:
            """Extract MCAP ID from filename (e.g., 'rosbag2_2025_07_25-10_14_58_1.mcap' -> '1')."""
            stem = mcap_path.stem  # filename without extension
            parts = stem.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[1]
            return "0"  # Fallback
        
        # Get all MCAP files from input rosbag, sorted numerically by MCAP ID
        def get_all_mcaps(rosbag_dir: Path) -> List[Path]:
            """Get all MCAP files from a rosbag, sorted numerically by MCAP ID."""
            if not rosbag_dir.exists():
                raise FileNotFoundError(f"Rosbag directory not found: {rosbag_dir}")
            
            mcaps = list(rosbag_dir.glob("*.mcap"))
            
            def extract_number(path: Path) -> int:
                """Extract the number from the mcap filename for sorting."""
                mcap_id = extract_mcap_id(path)
                try:
                    return int(mcap_id)
                except ValueError:
                    return float('inf')
            
            mcaps.sort(key=extract_number)
            return mcaps
        
        # Export a single MCAP file
        def export_mcap(
            input_mcap_path: Path,
            output_mcap_path: Path,
            mcap_id: str,
            start_time_ns: Optional[int],
            end_time_ns: Optional[int],
            topics: Optional[List[str]],
            compression: CompressionType,
            include_attachments: bool,
            include_metadata: bool,
        ):
            """
            Export messages from an MCAP file with optional time filtering.
            """
            app.logger.info(f"  Processing MCAP {mcap_id}: {input_mcap_path.name}")
            if start_time_ns is not None and end_time_ns is not None:
                app.logger.info(f"    Time range: {start_time_ns} to {end_time_ns}")
            elif start_time_ns is not None:
                app.logger.info(f"    Time range: {start_time_ns} to end of MCAP")
            elif end_time_ns is not None:
                app.logger.info(f"    Time range: beginning of MCAP to {end_time_ns}")
            else:
                app.logger.info(f"    Time range: entire MCAP (no time filtering)")
            
            # Ensure output directory exists
            output_mcap_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open input file and create reader
            with open(input_mcap_path, "rb") as input_file:
                reader = SeekingReader(input_file, decoder_factories=[DecoderFactory()])
                
                # Get summary information
                summary = reader.get_summary()
                if not summary:
                    app.logger.warning(f"    No summary found in MCAP {mcap_id}, skipping")
                    return
                
                # Open output file and create writer
                with open(output_mcap_path, "wb") as output_file:
                    writer = Writer(
                        output=output_file,
                        compression=compression,
                        index_types=IndexType.ALL,
                        use_chunking=True,
                        use_statistics=True,
                    )
                    
                    # Start writing
                    writer.start(profile="ros2", library="mcap-exporter")
                    
                    # Register schemas from input file
                    schema_map = {}  # Map old schema_id -> new schema_id
                    if summary.schemas:
                        for schema_id, schema in summary.schemas.items():
                            new_schema_id = writer.register_schema(
                                name=schema.name,
                                encoding=schema.encoding,
                                data=schema.data,
                            )
                            schema_map[schema_id] = new_schema_id
                    
                    # Register channels from input file (only those we want to export)
                    channel_map = {}  # Map old channel_id -> new channel_id
                    if summary.channels:
                        for channel_id, channel in summary.channels.items():
                            # Filter by topics if specified
                            if topics and channel.topic not in topics:
                                continue
                            
                            # Get new schema_id (or 0 if no schema)
                            new_schema_id = schema_map.get(channel.schema_id, 0)
                            
                            new_channel_id = writer.register_channel(
                                topic=channel.topic,
                                message_encoding=channel.message_encoding,
                                schema_id=new_schema_id,
                                metadata=dict(channel.metadata) if channel.metadata else {},
                            )
                            channel_map[channel_id] = new_channel_id
                    
                    if not channel_map:
                        app.logger.warning(f"    No matching channels found for MCAP {mcap_id}")
                        writer.finish()
                        return
                    
                    # Copy messages with optional time filtering
                    message_count = 0
                    skipped_count = 0
                    
                    for schema, channel, message in reader.iter_messages(
                        topics=topics,
                        start_time=start_time_ns,
                        end_time=end_time_ns,
                        log_time_order=True,
                        reverse=False,
                    ):
                        # Get new channel_id
                        new_channel_id = channel_map.get(channel.id)
                        if new_channel_id is None:
                            skipped_count += 1
                            continue
                        
                        # Write message
                        writer.add_message(
                            channel_id=new_channel_id,
                            log_time=message.log_time,
                            data=message.data,
                            publish_time=message.publish_time,
                            sequence=message.sequence,
                        )
                        message_count += 1
                    
                    app.logger.info(f"    Copied {message_count} messages (skipped {skipped_count})")
                    
                    # Copy attachments if requested
                    if include_attachments:
                        attachment_count = 0
                        for attachment in reader.iter_attachments():
                            # Filter attachments by time range (if specified)
                            include_attachment = True
                            if start_time_ns is not None and attachment.log_time < start_time_ns:
                                include_attachment = False
                            if end_time_ns is not None and attachment.log_time > end_time_ns:
                                include_attachment = False
                            
                            if include_attachment:
                                writer.add_attachment(
                                    create_time=attachment.create_time,
                                    log_time=attachment.log_time,
                                    name=attachment.name,
                                    media_type=attachment.media_type,
                                    data=attachment.data,
                                )
                                attachment_count += 1
                        if attachment_count > 0:
                            app.logger.info(f"    Copied {attachment_count} attachment(s)")
                    
                    # Copy metadata if requested
                    if include_metadata:
                        metadata_count = 0
                        for metadata in reader.iter_metadata():
                            writer.add_metadata(
                                name=metadata.name,
                                data=dict(metadata.metadata) if metadata.metadata else {},
                            )
                            metadata_count += 1
                        if metadata_count > 0:
                            app.logger.info(f"    Copied {metadata_count} metadata record(s)")
                    
                    # Finish writing
                    writer.finish()
            
            app.logger.info(f"    Export complete: {output_mcap_path.name}")
        
        # Main export logic
        app.logger.info("=" * 80)
        app.logger.info("MCAP Rosbag Exporter")
        app.logger.info("=" * 80)
        app.logger.info(f"Input rosbag: {input_rosbag_dir}")
        app.logger.info(f"Output directory: {output_rosbag_base}")
        app.logger.info(f"New rosbag name: {new_rosbag_name}")
        app.logger.info(f"Topics: {len(topics) if topics else 'all'}")
        
        # Get all MCAP files from input rosbag
        app.logger.info(f"\nScanning MCAP files in {input_rosbag_dir}...")
        input_mcaps = get_all_mcaps(input_rosbag_dir)
        app.logger.info(f"Found {len(input_mcaps)} MCAP file(s)")
        
        # Create output directory
        output_rosbag_dir.mkdir(parents=True, exist_ok=True)
        app.logger.info(f"Output directory: {output_rosbag_dir}")
        
        # Convert MCAP IDs to integers for range calculation
        try:
            start_mcap_num = int(start_mcap_id)
            end_mcap_num = int(end_mcap_id)
        except ValueError:
            EXPORT_PROGRESS = {"status": "error", "progress": -1}
            return jsonify({"error": f"Invalid MCAP IDs: start={start_mcap_id}, end={end_mcap_id}"}), 400
        
        if start_mcap_num > end_mcap_num:
            EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": f"Start MCAP ID ({start_mcap_num}) must be <= End MCAP ID ({end_mcap_num})"}
            return jsonify({"error": f"Start MCAP ID ({start_mcap_num}) must be <= End MCAP ID ({end_mcap_num})"}), 400
        
        # Calculate total number of MCAPs to process (after validation)
        total_mcaps_to_process = end_mcap_num - start_mcap_num + 1
        
        app.logger.info(f"  Exporting MCAPs {start_mcap_id} to {end_mcap_id} (inclusive)")
        app.logger.info(f"  Start MCAP {start_mcap_id}: from timestamp {start_timestamp} to end")
        app.logger.info(f"  End MCAP {end_mcap_id}: from beginning to timestamp {end_timestamp}")
        if end_mcap_num > start_mcap_num + 1:
            app.logger.info(f"  Middle MCAPs: complete export (no time filtering)")
        
        # Export all MCAPs in the range
        total_mcaps_exported = 0
        for mcap_index, mcap_num in enumerate(range(start_mcap_num, end_mcap_num + 1)):
            mcap_id = str(mcap_num)
            
            # Update progress: calculate percentage based on MCAP index
            progress = (mcap_index / total_mcaps_to_process) if total_mcaps_to_process > 0 else 0.0
            EXPORT_PROGRESS["progress"] = round(progress, 2)
            EXPORT_PROGRESS["message"] = f"Processing MCAP {mcap_id} ({mcap_index + 1}/{total_mcaps_to_process})"
            
            # Find the MCAP file with this ID
            input_mcap = None
            for mcap_path in input_mcaps:
                if extract_mcap_id(mcap_path) == mcap_id:
                    input_mcap = mcap_path
                    break
            
            if input_mcap is None:
                app.logger.warning(f"  MCAP {mcap_id} not found in input rosbag, skipping")
                continue
            
            # Build output path: OUTPUT_ROSBAG / new_rosbag_name / new_rosbag_name_mcap_id.mcap
            output_mcap_name = f"{new_rosbag_name}_{mcap_id}.mcap"
            output_mcap_path = output_rosbag_dir / output_mcap_name
            
            # Update message with filename
            EXPORT_PROGRESS["message"] = f"Writing {output_mcap_name} ({mcap_index + 1}/{total_mcaps_to_process})"
            
            # Determine time filtering based on MCAP position
            if mcap_num == start_mcap_num:
                # Start MCAP: from start_timestamp to end (no upper bound)
                mcap_start_time = start_timestamp
                mcap_end_time = None
            elif mcap_num == end_mcap_num:
                # End MCAP: from beginning to end_timestamp (no lower bound)
                mcap_start_time = None
                mcap_end_time = end_timestamp
            else:
                # Middle MCAPs: no time filtering (export completely)
                mcap_start_time = None
                mcap_end_time = None
            
            # Export this MCAP with appropriate time filtering
            try:
                export_mcap(
                    input_mcap_path=input_mcap,
                    output_mcap_path=output_mcap_path,
                    mcap_id=mcap_id,
                    start_time_ns=mcap_start_time,
                    end_time_ns=mcap_end_time,
                    topics=topics if topics else None,
                    compression=CompressionType.ZSTD,
                    include_attachments=True,
                    include_metadata=True,
                )
                total_mcaps_exported += 1
                
                # Update progress after successful export
                progress = ((mcap_index + 1) / total_mcaps_to_process) if total_mcaps_to_process > 0 else 1.0
                EXPORT_PROGRESS["progress"] = round(progress, 2)
            except Exception as e:
                app.logger.error(f"  Failed to export MCAP {mcap_id}: {e}", exc_info=True)
        
        app.logger.info(f"\n{'=' * 80}")
        app.logger.info(f"Export complete!")
        app.logger.info(f"  Exported {total_mcaps_exported} MCAP file(s)")
        app.logger.info(f"  Output directory: {output_rosbag_dir}")
        
        # Reindex the exported rosbag using ros2 bag reindex
        if total_mcaps_exported > 0:
            app.logger.info(f"\nReindexing rosbag: {output_rosbag_dir}")
            try:
                result = subprocess.run(
                    ["ros2", "bag", "reindex", str(output_rosbag_dir), "-s", "mcap"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                app.logger.info("Reindexing completed successfully")
                if result.stdout:
                    app.logger.debug(f"Reindex output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                app.logger.error(f"Reindexing failed: {e}")
                if e.stderr:
                    app.logger.error(f"Error output: {e.stderr}")
            except FileNotFoundError:
                app.logger.warning("ros2 command not found. Skipping reindexing. Make sure ROS2 is installed and in PATH.")
            except Exception as e:
                app.logger.error(f"Unexpected error during reindexing: {e}", exc_info=True)
        
        app.logger.info(f"{'=' * 80}")
        
        # Update export progress
        EXPORT_PROGRESS = {"status": "completed", "progress": 1.0, "message": f"Export completed! Exported {total_mcaps_exported} MCAP file(s)"}
        
        return jsonify({
            "message": "Export completed successfully",
            "exported_mcaps": total_mcaps_exported,
            "output_directory": str(output_rosbag_dir)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Export failed: {e}", exc_info=True)
        EXPORT_PROGRESS = {"status": "error", "progress": -1, "message": f"Export failed: {str(e)}"}
        return jsonify({"error": str(e)}), 500

    
# Canvas configuration endpoints for restoring panel layouts
@app.route("/api/load-canvases", methods=["GET"])
def api_load_canvases():
    return jsonify(load_canvases())

# Canvas configuration endpoints for saving panel layouts
@app.route("/api/save-canvas", methods=["POST"])
def api_save_canvas():
    data = request.json
    name = data.get("name")
    canvas_data = data.get("canvas")
    
    # Load existing canvases
    canvases = load_canvases()
    
    # Update or add the single canvas
    canvases[name] = canvas_data
    
    # Save back to file
    save_canvases(canvases)
    
    return jsonify({"message": f"Canvas '{name}' saved successfully"})

# Canvas configuration endpoints for deleting panel layouts
@app.route("/api/delete-canvas", methods=["POST"])
def api_delete_canvas():
    data = request.json
    name = data.get("name")
    canvases = load_canvases()

    if name in canvases:
        del canvases[name]
        save_canvases(canvases)
        return jsonify({"message": f"Canvas '{name}' deleted successfully"})
    
    return jsonify({"error": "Canvas not found"}), 404

# Start Flask server (debug mode enabled for development)
# Note: debug=True is for local development only. In production, use a WSGI server and set debug=False.
if __name__ == '__main__':
    app.run(debug=True)
