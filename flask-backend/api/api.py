from csv import reader
import json
import base64
import os
import glob
from pathlib import Path
from prompt_toolkit import prompt
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from mcap_ros2.reader import read_ros2_messages
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file, abort # type: ignore
from flask_cors import CORS  # type: ignore
import logging
import pandas as pd
import struct
import torch
import torch.nn.functional as F
from torch import nn
import faiss
from typing import TYPE_CHECKING, Iterable, Sequence, cast
import open_clip
import math
from threading import Thread, Lock
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import gc
from collections import defaultdict, OrderedDict
from dotenv import load_dotenv
from time import time
import sys

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


gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

BASE_STR = os.getenv("BASE")
IMAGES_STR = os.getenv("IMAGES")

ODOM_STR = os.getenv("ODOM")
POINTCLOUDS_STR = os.getenv("POINTCLOUDS")
POSITIONS_STR = os.getenv("POSITIONS")
VIDEOS_STR = os.getenv("VIDEOS")

REPRESENTATIVE_PREVIEWS_STR = os.getenv("REPRESENTATIVE_PREVIEWS")

LOOKUP_TABLES_STR = os.getenv("LOOKUP_TABLES")
TOPICS_STR = os.getenv("TOPICS")
CANVASES_FILE_STR = os.getenv("CANVASES_FILE")

ADJACENT_SIMILARITIES_STR = os.getenv("ADJACENT_SIMILARITIES")
EMBEDDINGS_STR = os.getenv("EMBEDDINGS")
POSITIONAL_LOOKUP_TABLE_STR = os.getenv("POSITIONAL_LOOKUP_TABLE")

EXPORT_STR = os.getenv("EXPORT")


###############################################################################################################

ROSBAGS = Path(os.getenv("ROSBAGS"))
PRESELECTED_ROSBAG = Path(os.getenv("PRESELECTED_ROSBAG"))

IMAGES = Path(BASE_STR + IMAGES_STR)
ODOM = Path(BASE_STR + ODOM_STR)
POINTCLOUDS = Path(BASE_STR + POINTCLOUDS_STR)
POSITIONS = Path(BASE_STR + POSITIONS_STR)
VIDEOS = Path(BASE_STR + VIDEOS_STR)
REPRESENTATIVE_PREVIEWS = Path(BASE_STR + REPRESENTATIVE_PREVIEWS_STR)

LOOKUP_TABLES = Path(BASE_STR + LOOKUP_TABLES_STR)
TOPICS = Path(BASE_STR + TOPICS_STR)
CANVASES_FILE = Path(BASE_STR + CANVASES_FILE_STR)

ADJACENT_SIMILARITIES = Path(BASE_STR + ADJACENT_SIMILARITIES_STR)
EMBEDDINGS = Path(BASE_STR + EMBEDDINGS_STR)
POSITIONAL_LOOKUP_TABLE = Path(BASE_STR + POSITIONAL_LOOKUP_TABLE_STR)

EXPORT = Path(BASE_STR + EXPORT_STR)

PRESELECTED_MODEL = str(os.getenv("PRESELECTED_MODEL"))
OPEN_CLIP_MODELS = Path(os.getenv("OPEN_CLIP_MODELS"))
OTHER_MODELS = Path(os.getenv("OTHER_MODELS"))

SELECTED_ROSBAG = PRESELECTED_ROSBAG
SELECTED_MODEL = PRESELECTED_MODEL

# Helper function for loading lookup tables (defined below, initialized after)
def load_lookup_tables_for_rosbag(rosbag_name: str) -> pd.DataFrame:
    """Load and combine all mcap CSV files for a rosbag.
    
    Args:
        rosbag_name: Name of the rosbag (directory name, not full path)
    
    Returns:
        Combined DataFrame with all lookup table data, or empty DataFrame if none found.
    """
    if not LOOKUP_TABLES:
        return pd.DataFrame()
    
    lookup_rosbag_dir = LOOKUP_TABLES / rosbag_name
    if not lookup_rosbag_dir.exists():
        return pd.DataFrame()
    
    csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    all_dfs = []
    for csv_path in sorted(csv_files):
        mcap_id = csv_path.stem  # Extract mcap_id from filename (without .csv extension)
        try:
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
    
    return pd.concat(all_dfs, ignore_index=True)

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment

ALIGNED_DATA = load_lookup_tables_for_rosbag(os.path.basename(SELECTED_ROSBAG))


# EXPORT_PROGRESS: Dictionary to track progress and status of export jobs
EXPORT_PROGRESS = {"status": "idle", "progress": -1}
SEARCH_PROGRESS = {"status": "idle", "progress": -1, "message": "idle"}

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
        with open(CANVASES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_canvases(data):
    with open(CANVASES_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Copy messages from one Rosbag to another within a timestamp range and selected topics
def export_rosbag_with_topics(src: Path, dst: Path, includedTopics, start_timestamp, end_timestamp) -> None:
    reader = SequentialReader()
    writer = SequentialWriter()
    
    try:
        # Open reader
        reader.open(
            StorageOptions(uri=str(src), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Open writer
        writer.open(
            StorageOptions(uri=str(dst), storage_id="mcap"),
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
        
        # Create topics in writer
        for topic_name, topic_type in topics_to_create.items():
            writer.create_topic(
                TopicMetadata(
                    name=topic_name,
                    type=topic_type,
                    serialization_format="cdr"
                )
            )
        
        # First pass: count total messages for progress tracking
        total_msgs = 0
        reader2 = SequentialReader()
        reader2.open(
            StorageOptions(uri=str(src), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        while reader2.has_next():
            read_topic, _, _ = reader2.read_next()
            if read_topic in topic_map:
                total_msgs += 1
        del reader2
        
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
        except:
            pass
        try:
            del writer
        except:
            pass
        EXPORT_PROGRESS["status"] = "error"
        logging.error(f"Error exporting rosbag: {e}")
        raise

# Return the current export progress and status
@app.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)

# Set the model for performing semantic search
@app.route('/api/set-model', methods=['POST'])
def post_model():
    """
    Set the current CLIP model for semantic search.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        model_value = data.get('model')  # The path value from the JSON

        global SELECTED_MODEL
        SELECTED_MODEL = model_value

        # Model reload is handled on demand in the search endpoint
        return jsonify({"message": f"Model {model_value} successfully posted."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        rosbag_name = os.path.basename(SELECTED_ROSBAG)
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


@app.route('/api/positions/rosbags/<string:rosbag_name>', methods=['GET'])
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


@app.route('/api/positions/rosbags/<string:rosbag_name>/mcaps', methods=['GET'])
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


@app.route('/api/positions/rosbags/<string:rosbag_name>/mcap-list', methods=['GET'])
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


# Polygon endpoints
POLYGONS_DIR = Path("/mnt/data/ongoing_projects/positional_filter/polygons")

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
    Simplified: Just scans ROSBAGS directory, no IMAGES scan.
    """
    try:
        now = time()
        with _file_path_cache_lock:
            cached_paths = list(_matching_rosbag_cache["paths"])
            cached_timestamp = _matching_rosbag_cache["timestamp"]

        if (
            cached_paths
            and (now - cached_timestamp) < FILE_PATH_CACHE_TTL_SECONDS
        ):
            return jsonify({"paths": cached_paths}), 200

        rosbag_paths: list[str] = []
        for base_dir in [ROSBAGS]:
            if not base_dir or not base_dir.exists():
                continue
            
            base_path = Path(base_dir)
            stack = [base_path]
            
            while stack:
                current_dir = stack.pop()
                
                # Skip EXCLUDED and EXPORTED directories early
                if "EXCLUDED" in current_dir.parts or "EXPORTED" in current_dir.parts:
                    continue
                
                # Skip the base "rosbags" parent folder itself
                if current_dir == base_path:
                    # Just add its subdirectories to stack, don't add base_path to results
                    try:
                        with os.scandir(str(current_dir)) as entries:
                            subdirs = []
                            for entry in entries:
                                if entry.is_dir():
                                    # Skip EXCLUDED and EXPORTED subdirectories immediately
                                    if "EXCLUDED" not in entry.name and "EXPORTED" not in entry.name:
                                        subdirs.append(Path(entry.path))
                            stack.extend(subdirs)
                    except (PermissionError, OSError) as e:
                        logging.warning(f"Cannot access directory {current_dir}: {e}")
                    continue
                
                try:
                    # Fast: Just list directories, like 'ls'
                    with os.scandir(str(current_dir)) as entries:
                        subdirs = []
                        
                        for entry in entries:
                            if entry.is_dir():
                                # Skip EXCLUDED and EXPORTED subdirectories immediately
                                if "EXCLUDED" not in entry.name and "EXPORTED" not in entry.name:
                                    subdirs.append(Path(entry.path))
                        
                        # Add this directory as a potential rosbag path
                        rosbag_paths.append(str(current_dir))
                        
                        # Recurse into subdirectories
                        stack.extend(subdirs)
                            
                except (PermissionError, OSError) as e:
                    # Skip directories we can't access
                    logging.warning(f"Cannot access directory {current_dir}: {e}")
                    continue
        
        rosbag_paths.sort()

        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(rosbag_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": rosbag_paths}), 200
    except Exception as e:
        logging.error(f"Error in get_file_paths: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/refresh-file-paths', methods=['POST'])
def refresh_file_paths():
    """Force refresh of the cached Rosbag file paths."""
    try:
        rosbag_paths: list[str] = []
        for base_dir in [ROSBAGS]: #ROSBAGS_DIR_MNT, ROSBAGS_DIR_NAS):
            if not base_dir:
                continue
            for root, dirs, files in os.walk(str(base_dir), topdown=True):
                if "metadata.yaml" in files:
                    if "EXCLUDED" in root:
                        dirs[:] = []
                        continue
                    rosbag_paths.append(root)
                    dirs[:] = []
        rosbag_paths.sort()

        image_basenames: set[str] = set()
        if IMAGES and IMAGES.exists() and IMAGES.is_dir():
            for entry in os.scandir(str(IMAGES)):
                if entry.is_dir():
                    image_basenames.add(entry.name)

        matching_paths = [
            path
            for path in rosbag_paths
            if os.path.basename(os.path.normpath(path)) in image_basenames
        ]

        now = time()
        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(matching_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": matching_paths, "message": "Rosbag file paths refreshed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Get the currently selected rosbag
@app.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    """
    Return the currently selected Rosbag file name.
    """
    try:
        selectedRosbag = os.path.basename(SELECTED_ROSBAG)
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500

@app.route('/api/set-searched-rosbags', methods=['POST'])
def set_searched_rosbags():
    """
    Set the currently searched Rosbag file name.
    This is used to filter available topics and images based on the selected Rosbag.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        searchedRosbags = data.get('searchedRosbags')  # The list of searched Rosbags

        if not isinstance(searchedRosbags, list):
            return jsonify({"error": "searchedRosbags must be a list"}), 400

        global SEARCHED_ROSBAGS
        SEARCHED_ROSBAGS = searchedRosbags

        return jsonify({"message": "Searched Rosbags updated successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500 
    
# Get available topics from pre-generated JSON file for the current Rosbag
@app.route('/api/get-available-topics', methods=['GET'])
def get_available_rosbag_topics():
    try:
        rosbag_name = os.path.basename(SELECTED_ROSBAG)
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopics': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        topics = topics_data.get("topics", [])
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
                rosbag_name = os.path.basename(rosbag_param)
                rosbag_path = os.path.join(model_path, rosbag_name)
                if not os.path.isdir(rosbag_path):
                    continue

                topics = []
                for topic in os.listdir(rosbag_path):
                    topics.append(topic.replace("__", "/"))

                model_entry[rosbag_name] = sorted(topics)

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
        rosbag_name = os.path.basename(SELECTED_ROSBAG)
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopicTypes': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        availableTopicTypes = topics_data.get("types", [])
        return jsonify({'availableTopicTypes': availableTopicTypes}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopicTypes': []}), 200
    
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
        rosbag_name = os.path.basename(rosbag)
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
    
@app.route('/images/<path:image_path>')
def serve_image(image_path):
    """
    Serve an extracted image file from the backend image directory.
    New structure: images/rosbag_name/topic_name/mcap_identifier/timestamp.png
    """
    try:
        # Construct full path explicitly
        full_path = IMAGES / image_path
        
        # Check if file exists
        if not full_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Check if it's actually a file
        if not full_path.is_file():
            return jsonify({'error': 'Invalid path'}), 400
        
        # Use send_file with explicit path (convert Path to string)
        response = send_file(str(full_path), mimetype='image/png')
        response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
        return response
        
    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/adjacency-image/<path:adjacency_image_path>')
def serve_adjacency_image(adjacency_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(ADJACENT_SIMILARITIES, adjacency_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/representative-image/<path:representative_image_path>')
def serve_representative_image(representative_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(REPRESENTATIVE_PREVIEWS, representative_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

# Return preview image URL for a topic at a given reference index within a rosbag
# The HeatBar represents all reference timestamps for the rosbag
# This endpoint takes the index (position in reference timestamps array) and returns the corresponding image
@app.route('/api/get-topic-image-preview', methods=['GET'])
def get_topic_image_preview():
    try:
        rosbag_name = request.args.get('rosbag', type=str)
        topic = request.args.get('topic', type=str)
        index = request.args.get('index', type=int)

        if not rosbag_name or not topic:
            return jsonify({'error': 'Missing rosbag or topic'}), 400
        
        if index is None:
            return jsonify({'error': 'Missing index parameter'}), 400

        # Helper function to check if value is non-nan
        def non_nan(v):
            if v is None:
                return False
            if isinstance(v, float):
                return not math.isnan(v)
            s = str(v)
            return s.lower() != 'nan' and s != '' and s != 'None'

        # Load all mcap lookup tables for this rosbag and combine them
        # Structure: lookup_tables/rosbag_name/mcap_id.csv
        lookup_rosbag_dir = LOOKUP_TABLES / rosbag_name
        if not lookup_rosbag_dir.exists():
            return jsonify({'error': 'Lookup table directory not found'}), 404

        # Find all mcap CSV files for this rosbag
        csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
        if not csv_files:
            return jsonify({'error': 'No lookup table CSV files found'}), 404

        # Load all mcap CSVs and combine them, tracking which mcap each row came from
        all_dfs = []
        for csv_path in csv_files:
            mcap_id = csv_path.stem  # Extract mcap_id from filename (without .csv extension)
            try:
                df = pd.read_csv(csv_path, dtype=str)
                if len(df) > 0:
                    # Add mcap_id as a column to track which mcap this row came from
                    df['_mcap_id'] = mcap_id
                    all_dfs.append(df)
            except Exception as e:
                logging.warning(f"Failed to load CSV {csv_path}: {e}")
                continue

        if not all_dfs:
            return jsonify({'error': 'No valid lookup table data found'}), 404

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by Reference Timestamp to ensure consistent ordering
        if 'Reference Timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('Reference Timestamp').reset_index(drop=True)
        
        total = len(combined_df)
        
        if total == 0:
            return jsonify({'error': 'No timestamps available'}), 404

        # Validate index
        if index < 0 or index >= total:
            return jsonify({'error': f'Index {index} out of range [0, {total-1}]'}), 400

        # Check if topic exists in the combined dataframe
        if topic not in combined_df.columns:
            return jsonify({'error': f'Topic column "{topic}" not found in lookup tables'}), 404

        # Get the row at the specified index
        row = combined_df.iloc[index]
        
        # Get the original timestamp for this topic at this reference timestamp index
        topic_ts = row[topic]
        mcap_id = row['_mcap_id']
        
        # If the topic timestamp is empty at this index, search nearby for a valid value
        if not non_nan(topic_ts):
            # Search within a reasonable radius
            radius = min(500, total - 1)
            found = False
            
            for off in range(1, radius + 1):
                # Check left and right
                for candidate_idx in [index - off, index + off]:
                    if 0 <= candidate_idx < total:
                        candidate_row = combined_df.iloc[candidate_idx]
                        candidate_ts = candidate_row[topic]
                        if non_nan(candidate_ts):
                            topic_ts = candidate_ts
                            mcap_id = candidate_row['_mcap_id']
                            found = True
                            break
                if found:
                    break
            
            if not found:
                return jsonify({'error': f'No valid timestamp found for topic "{topic}" near index {index}'}), 404

        # Build the image path: images/rosbag_name/topic_name/mcap_id/timestamp.png
        # Normalize topic name: replace / with _ and remove leading _
        topic_safe = topic.replace('/', '_').lstrip('_')
        file_path = IMAGES / rosbag_name / topic_safe / mcap_id / f"{topic_ts}.png"
        
        # If the exact file doesn't exist, search nearby for an existing image
        if not file_path.exists():
            radius = min(500, total - 1)
            best_path = None
            best_ts = topic_ts
            best_mcap = mcap_id
            
            # Search nearby indices for an existing image file
            for off in range(1, radius + 1):
                for candidate_idx in [index - off, index + off]:
                    if 0 <= candidate_idx < total:
                        candidate_row = combined_df.iloc[candidate_idx]
                        candidate_ts = candidate_row[topic]
                        candidate_mcap = candidate_row['_mcap_id']
                        
                        if non_nan(candidate_ts):
                            candidate_path = IMAGES / rosbag_name / topic_safe / candidate_mcap / f"{candidate_ts}.png"
                            if candidate_path.exists():
                                best_path = candidate_path
                                best_ts = candidate_ts
                                best_mcap = candidate_mcap
                                break
                if best_path:
                    break
            
            if best_path:
                file_path = best_path
                topic_ts = best_ts
                mcap_id = best_mcap
            else:
                return jsonify({'error': f'Image not found for topic "{topic}" at index {index} or nearby'}), 404

        # Construct relative URL: /images/rosbag_name/topic_safe/mcap_id/timestamp.png
        rel_url = f"/images/{rosbag_name}/{topic_safe}/{mcap_id}/{topic_ts}.png"
        return jsonify({
            'imageUrl': rel_url,
            'timestamp': topic_ts,
            'index': index,
            'total': total,
            'mcap_id': mcap_id
        }), 200

    except Exception as e:
        logging.error(f"Error in get_topic_image_preview: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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

# Return deserialized message content (image, TF, IMU, etc.) for currently selected topic and reference timestamp
@app.route('/api/content', methods=['GET'])
def get_ros():
    """
    Return deserialized message content (image, TF, IMU, etc.) for the currently selected topic and reference timestamp.
    The logic:
      - Uses the mapped_timestamps for the current reference timestamp to get the aligned topic timestamp.
      - Opens the Rosbag and iterates messages for the topic, looking for the exact timestamp.
      - Deserializes the message and returns a JSON structure based on the message type (point cloud, position, tf, imu, etc).
    """
    global mapped_timestamps
    topic = request.args.get('topic', default=None, type=str)
    mcap_identifier = request.args.get('mcap_identifier', default=None, type=str)
    timestamp = mapped_timestamps.get(topic) if topic and mcap_identifier else None

    logging.warning(f"{SELECTED_ROSBAG}/{os.path.basename(SELECTED_ROSBAG)}_{mcap_identifier}.mcap")
    if not timestamp:
        return jsonify({'error': 'No mapped timestamp found for the provided topic'})

    # Open the rosbag to find the message at the requested timestamp and topic
    reader = SequentialReader()
    try:
        reader.open(
            StorageOptions(uri=f"{SELECTED_ROSBAG}/{os.path.basename(SELECTED_ROSBAG)}_{mcap_identifier}.mcap", storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Get topic types for message deserialization
        all_topics = reader.get_all_topics_and_types()
        topic_type_map = {topic_meta.name: topic_meta.type for topic_meta in all_topics}
        
        # Match normalized input topic against rosbag topics
        # The input topic is normalized (e.g., "camera_side_right_image_raw_compressed")
        # but rosbag topics are original (e.g., "/camera/side_right/image_raw/compressed")
        normalized_input = normalize_topic(topic)
        matched_original_topic = None
        
        for topic_meta in all_topics:
            normalized_rosbag_topic = normalize_topic(topic_meta.name)
            if normalized_rosbag_topic == normalized_input:
                matched_original_topic = topic_meta.name
                break
        
        if not matched_original_topic:
            return jsonify({'error': f'Topic {topic} (normalized: {normalized_input}) not found in rosbag'}), 404
        
        # Get the topic type for the matched original topic
        topic_type = topic_type_map.get(matched_original_topic)
        if not topic_type:
            return jsonify({'error': f'Topic type not found for {matched_original_topic}'}), 404
        
        msg_type = get_message(topic_type)
        
        # Iterate through messages to find the one matching the timestamp
        while reader.has_next():
            read_topic, data, log_time = reader.read_next()
            
            # Only process messages from the requested topic (use original topic name)
            if read_topic != matched_original_topic:
                continue
            
            # Deserialize message to extract header timestamp
            # Special cases: TF messages and ouster topics use relative timestamps (time since boot),
            # so we use log_time instead of header timestamp to match the preprocessing alignment
            if topic_type == 'tf2_msgs/msg/TFMessage' or matched_original_topic in ['/ouster/imu', '/ouster/points', '/tf', '/tf_static']:
                msg = deserialize_message(data, msg_type)
                header_timestamp = log_time
            else:
                try:
                    msg = deserialize_message(data, msg_type)
                    # Extract header timestamp (not log_time)
                    header_timestamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
                except Exception as e:
                    # If header extraction fails, fallback to log_time
                    logging.warning(f"Could not extract header timestamp from {matched_original_topic}: {e}")
                    header_timestamp = log_time
            
            # Match against the requested timestamp (from CSV alignment table, uses header stamps)
            if str(header_timestamp) == timestamp:
                # Deserialization already done above, now match message type
                match topic_type:
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
        
        # Clean up reader
        del reader
        return jsonify({'error': 'No message found for the provided timestamp and topic'})
    except Exception as e:
        # Clean up reader on error
        try:
            del reader
        except:
            pass
        return jsonify({'error': f'Error reading rosbag: {str(e)}'})

@app.route('/api/content-mcap', methods=['GET', 'POST'])
def get_content_mcap():
    topic = request.args.get('topic')
    mcap_identifier = request.args.get('mcap_identifier')
    timestamp = request.args.get('timestamp', type=int)
    
    mcap_path = f"{SELECTED_ROSBAG}/{os.path.basename(SELECTED_ROSBAG)}_{mcap_identifier}.mcap"
    
    try:        
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                topics=[topic],
                start_time=timestamp,
                end_time=timestamp+1,
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
        rosbags_list = [os.path.basename(r) for r in rosbags.split(",") if r.strip()]
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
    Export a portion of a Rosbag containing selected topics and time range.
    Starts a background thread to perform the export and updates EXPORT_PROGRESS.
    """
    try:
        data = request.json
        new_rosbag_name = data.get('new_rosbag_name')
        topics = data.get('topics')
        start_timestamp = int(data.get('start_timestamp'))
        end_timestamp = int(data.get('end_timestamp'))

        if not new_rosbag_name or not topics:
            return jsonify({"error": "Rosbag name and topics are required"}), 400

        if not os.path.exists(SELECTED_ROSBAG):
            return jsonify({"error": "Rosbag not found"}), 404

        EXPORT_PATH = os.path.join(EXPORT, new_rosbag_name)
        EXPORT_PROGRESS["status"] = "starting"
        EXPORT_PROGRESS["progress"] = -1

        def run_export():
            export_rosbag_with_topics(SELECTED_ROSBAG, EXPORT_PATH, topics, start_timestamp, end_timestamp)

        Thread(target=run_export).start()
        return jsonify({"message": "Export started", "exported_path": str(EXPORT_PATH)})

    except Exception as e:
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
