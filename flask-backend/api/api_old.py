from csv import reader
import json
import os
from pathlib import Path
from prompt_toolkit import prompt
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
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
from datetime import datetime
import gc
from collections import defaultdict, OrderedDict
from dotenv import load_dotenv
from time import time

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from open_clip.tokenizer import tokenize as open_clip_tokenize
except ImportError:  # pragma: no cover
    open_clip_tokenize = None


# Load environment variables from .env file
PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Flask API for BagSeek: A tool for exploring Rosbag data and using semantic search via CLIP and FAISS to locate safety critical and relevant scenes.
# This API provides endpoints for loading data, searching via CLIP embeddings, exporting segments, and more.

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend (e.g., React)


gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

#
# --- Global Variables and Constants ---
#
# ROSBAGS_DIR: Directory where all Rosbag files are stored
# BASE_DIR: Base directory for backend resources (topics, images, embeddings, etc.)
# TOPICS_DIR: Directory for storing available topics per Rosbag
# IMAGES_DIR: Directory for extracted image frames
# LOOKUP_TABLES_DIR: Directory for lookup tables mapping reference timestamps to topic timestamps
# EMBEDDINGS_DIR: Directory for precomputed CLIP embeddings
# CANVASES_FILE: JSON file for UI canvas state persistence
# INDICES_DIR: Directory for FAISS indices for semantic search
# EXPORT_DIR: Directory where exported Rosbags are saved
# SELECTED_ROSBAG: Currently selected Rosbag file path

ROSBAGS_DIR_MNT = os.getenv("ROSBAGS_DIR_MNT")
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
BASE_DIR = os.getenv("BASE_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")
IMAGES_DIR = os.getenv("IMAGES_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
EMBEDDINGS_PER_TOPIC_DIR = os.getenv("EMBEDDINGS_PER_TOPIC_DIR")
EMBEDDINGS_CONSOLIDATED_DIR = os.getenv("EMBEDDINGS_CONSOLIDATED_DIR")
CANVASES_FILE = os.getenv("CANVASES_FILE")
INDICES_DIR = os.getenv("INDICES_DIR")  
EXPORT_DIR = os.getenv("EXPORT_DIR")
ADJACENT_SIMILARITIES_DIR = os.getenv("ADJACENT_SIMILARITIES_DIR")
REPRESENTATIVE_IMAGES_DIR = os.getenv("REPRESENTATIVE_IMAGES_DIR")
GPS_LOOKUP_PATH = Path("/mnt/data/bagseek/flask-backend/src/rosbag_gps_lookup.json")

SELECTED_ROSBAG = '/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_07_24-16_01_22'

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment
ALIGNED_DATA = pd.read_csv(os.path.join(LOOKUP_TABLES_DIR, os.path.basename(SELECTED_ROSBAG) + '.csv'), dtype=str)

# EXPORT_PROGRESS: Dictionary to track progress and status of export jobs
EXPORT_PROGRESS = {"status": "idle", "progress": -1}
SEARCH_PROGRESS = {"status": "idle", "progress": -1, "message": "idle"}

# SELECTED_MODEL: Default CLIP model for semantic search (format: <model_name>__<pretrained_name>)
SELECTED_MODEL = 'ViT-B-16-quickgelu__openai'
SEARCHED_ROSBAGS = []

# MAX_K: Number of top results to return for semantic search
MAX_K = 100

# Cache setup for expensive rosbag discovery
FILE_PATH_CACHE_TTL_SECONDS = 60
_matching_rosbag_cache = {"paths": [], "timestamp": 0.0}
_file_path_cache_lock = Lock()

NEW_MODEL_DIR = Path("/mnt/data/new_models")
_gps_lookup_cache: dict[str, dict[str, dict[str, int]]] = {"data": None, "mtime": None}  # type: ignore[assignment]
CUSTOM_MODEL_DEFAULTS = {
    "agriclip": Path("/mnt/data/bagseek/flask-backend/src/models/agriclip.pt"),
    "epoch32": NEW_MODEL_DIR / "epoch_32.pt",
}

""" IF NOVATEL DOESNT WORK: (OLD CODE)
# Initialize the type system for ROS2 deserialization, including custom Novatel messages
# This is required to correctly deserialize messages from Rosbags, especially for custom message types
typestore = get_typestore(Stores.ROS2_HUMBLE)
novatel_msg_folder = Path('/opt/ros/humble/share/novatel_oem7_msgs/msg')
custom_types = {}
for msg_file in novatel_msg_folder.glob('*.msg'):
    try:
        text = msg_file.read_text()
        typename = f"novatel_oem7_msgs/msg/{msg_file.stem}"
        result = get_types_from_msg(text, typename)
        custom_types.update(result)
    except Exception as e:
        logging.warning(f"Failed to parse {msg_file.name}: {e}")
typestore.register(custom_types)"""

# Used to track the currently selected reference timestamp and its aligned mappings
# current_reference_timestamp: The reference timestamp selected by the user
# mapped_timestamps: Dictionary mapping topic names to their corresponding timestamps for the selected reference timestamp
current_reference_timestamp = None
mapped_timestamps = {}

def resolve_custom_checkpoint(model_name: str) -> Path:
    """Return a filesystem path to the checkpoint for a custom (non-open_clip) model."""
    candidates = [
        NEW_MODEL_DIR / f"{model_name}.pt",
        NEW_MODEL_DIR / model_name,
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
        # Convert NaNs to None for safe JSON serialization
        cleaned_mapped_timestamps = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in mapped_timestamps.items()
        }
        return jsonify({"mappedTimestamps": cleaned_mapped_timestamps, "message": "Reference timestamp updated"}), 200

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
        for model in os.listdir(EMBEDDINGS_PER_TOPIC_DIR):
            if not model in ['.DS_Store', 'README.md']:
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
        csv_path = LOOKUP_TABLES_DIR + "/" + os.path.basename(SELECTED_ROSBAG) + ".csv"
        ALIGNED_DATA = pd.read_csv(csv_path, dtype=str)

        global SEARCHED_ROSBAGS
        SEARCHED_ROSBAGS = [path_value]  # Reset searched rosbags to the selected one
        logging.warning(SEARCHED_ROSBAGS)
        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_gps_lookup() -> dict[str, dict[str, int]]:
    """
    Load and cache the GPS lookup JSON, refreshing when the file changes.
    """
    if not GPS_LOOKUP_PATH.exists():
        raise FileNotFoundError(f"GPS lookup file not found at {GPS_LOOKUP_PATH}")

    stat = GPS_LOOKUP_PATH.stat()
    cached_mtime = _gps_lookup_cache.get("mtime")
    if _gps_lookup_cache.get("data") is None or cached_mtime != stat.st_mtime:
        with GPS_LOOKUP_PATH.open("r", encoding="utf-8") as fp:
            _gps_lookup_cache["data"] = json.load(fp)
        _gps_lookup_cache["mtime"] = stat.st_mtime

    return _gps_lookup_cache["data"]  # type: ignore[return-value]


@app.route('/api/gps/rosbags', methods=['GET'])
def get_gps_rosbags():
    """
    Return the list of rosbag names available in the GPS lookup table.
    """
    try:
        lookup = _load_gps_lookup()
        rosbag_names = sorted(lookup.keys())
        return jsonify({"rosbags": rosbag_names}), 200
    except FileNotFoundError:
        return jsonify({"rosbags": []}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/gps/rosbags/<string:rosbag_name>', methods=['GET'])
def get_gps_rosbag_entries(rosbag_name: str):
    """
    Return the GPS lookup entries for a specific rosbag.
    """
    try:
        lookup = _load_gps_lookup()
        rosbag_data = lookup.get(rosbag_name)
        if rosbag_data is None:
            return jsonify({"error": f"Rosbag '{rosbag_name}' not found"}), 404

        points = []
        for lat_lon, count in rosbag_data.items():
            try:
                lat_str, lon_str = lat_lon.split(',')
                points.append({
                    "lat": float(lat_str),
                    "lon": float(lon_str),
                    "count": int(count)
                })
            except (ValueError, TypeError):
                continue

        points.sort(key=lambda item: item["count"], reverse=True)

        return jsonify({
            "rosbag": rosbag_name,
            "points": points
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "GPS lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/gps/all', methods=['GET'])
def get_gps_all():
    """
    Return aggregated GPS lookup entries across all rosbags.
    """
    try:
        lookup = _load_gps_lookup()
        aggregated: dict[str, dict[str, float | int]] = {}

        for rosbag_data in lookup.values():
            for lat_lon, count in rosbag_data.items():
                try:
                    lat_str, lon_str = lat_lon.split(',')
                    key = f"{float(lat_str):.6f},{float(lon_str):.6f}"
                    if key not in aggregated:
                        aggregated[key] = {
                            "lat": float(lat_str),
                            "lon": float(lon_str),
                            "count": int(count),
                        }
                    else:
                        aggregated[key]["count"] = int(aggregated[key]["count"]) + int(count)  # type: ignore[index]
                except (ValueError, TypeError):
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
        return jsonify({"error": "GPS lookup file not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Get all available rosbags
@app.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    """
    Return all available Rosbag file paths (excluding those in EXCLUDED).
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
        for base_dir in (ROSBAGS_DIR_MNT, ROSBAGS_DIR_NAS):
            if not base_dir:
                continue
            for root, dirs, files in os.walk(base_dir, topdown=True):
                if "metadata.yaml" in files:
                    if "EXCLUDED" in root:
                        dirs[:] = []
                        continue
                    rosbag_paths.append(root)
                    dirs[:] = []
        rosbag_paths.sort()

        image_basenames: set[str] = set()
        if IMAGES_PER_TOPIC_DIR and os.path.isdir(IMAGES_PER_TOPIC_DIR):
            for entry in os.scandir(IMAGES_PER_TOPIC_DIR):
                if entry.is_dir():
                    image_basenames.add(entry.name)

        matching_paths = [
            path
            for path in rosbag_paths
            if os.path.basename(os.path.normpath(path)) in image_basenames
        ]

        with _file_path_cache_lock:
            _matching_rosbag_cache["paths"] = list(matching_paths)
            _matching_rosbag_cache["timestamp"] = now

        return jsonify({"paths": matching_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/refresh-file-paths', methods=['POST'])
def refresh_file_paths():
    """Force refresh of the cached Rosbag file paths."""
    try:
        rosbag_paths: list[str] = []
        for base_dir in (ROSBAGS_DIR_MNT, ROSBAGS_DIR_NAS):
            if not base_dir:
                continue
            for root, dirs, files in os.walk(base_dir, topdown=True):
                if "metadata.yaml" in files:
                    if "EXCLUDED" in root:
                        dirs[:] = []
                        continue
                    rosbag_paths.append(root)
                    dirs[:] = []
        rosbag_paths.sort()

        image_basenames: set[str] = set()
        if IMAGES_PER_TOPIC_DIR and os.path.isdir(IMAGES_PER_TOPIC_DIR):
            for entry in os.scandir(IMAGES_PER_TOPIC_DIR):
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
        topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")

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
            model_path = os.path.join(ADJACENT_SIMILARITIES_DIR, model_param)
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
        topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")

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
    timestampLengths = {}

    for rosbag in rosbags:
        csv_path = os.path.join(LOOKUP_TABLES_DIR, os.path.basename(rosbag) + '.csv')
        try:
            df = pd.read_csv(csv_path, dtype=str)
            count = df['Reference Timestamp'].notnull().sum()
            timestampLengths[rosbag] = int(count)
        except Exception as e:
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
    """
    response = send_from_directory(IMAGES_PER_TOPIC_DIR, image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/adjacency-image/<path:adjacency_image_path>')
def serve_adjacency_image(adjacency_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(ADJACENT_SIMILARITIES_DIR, adjacency_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/representative-image/<path:representative_image_path>')
def serve_representative_image(representative_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(REPRESENTATIVE_IMAGES_DIR, representative_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

# Return preview image URL for a topic at a given reference index or position within a rosbag
@app.route('/api/get-topic-image-preview', methods=['GET'])
def get_topic_image_preview():
    try:
        rosbag_name = request.args.get('rosbag', type=str)
        topic = request.args.get('topic', type=str)
        index = request.args.get('index', type=int)
        pos = request.args.get('pos', type=float)

        if not rosbag_name or not topic:
            return jsonify({'error': 'Missing rosbag or topic'}), 400

        csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Lookup table not found'}), 404

        df = pd.read_csv(csv_path, dtype=str)
        total = len(df)
        if total == 0:
            return jsonify({'error': 'No timestamps available'}), 404

        # Determine reference index from pos or index
        if index is None:
            if pos is None:
                index = 0
            else:
                index = int(max(0, min(total - 1, round(pos * (total - 1)))))
        else:
            index = int(max(0, min(total - 1, index)))

        # Find the per-topic timestamp at or near the requested reference index
        def non_nan(v):
            if v is None:
                return False
            if isinstance(v, float):
                return not math.isnan(v)
            s = str(v)
            return s.lower() != 'nan' and s != '' and s != 'None'

        chosen_idx = index
        topic_ts = None
        if topic in df.columns:
            val = df.iloc[index][topic]
            if non_nan(val):
                topic_ts = str(val)
            else:
                # search nearest non-empty within a window
                radius = min(500, total - 1)
                for off in range(1, radius + 1):
                    left = index - off
                    right = index + off
                    if left >= 0:
                        v = df.iloc[left][topic]
                        if non_nan(v):
                            chosen_idx = left
                            topic_ts = str(v)
                            break
                    if right < total:
                        v = df.iloc[right][topic]
                        if non_nan(v):
                            chosen_idx = right
                            topic_ts = str(v)
                            break
        else:
            return jsonify({'error': 'Topic column not found'}), 404

        if not topic_ts:
            return jsonify({'error': 'No valid topic timestamp found nearby'}), 404

        topic_safe = topic.replace('/', '__')
        filename = f"{topic_ts}.webp"
        file_path = os.path.join(IMAGES_PER_TOPIC_DIR, rosbag_name, topic_safe, filename)
        logging.warning(file_path)

        # If file missing, search outward for a nearby timestamp with existing image
        if not os.path.exists(file_path):
            # try nearby indices on topic column
            radius = min(500, total - 1)
            best_path = None
            best_idx = chosen_idx
            for off in range(1, radius + 1):
                for candidate in (chosen_idx - off, chosen_idx + off):
                    if 0 <= candidate < total:
                        v = df.iloc[candidate][topic]
                        if non_nan(v):
                            fp = os.path.join(IMAGES_DIR, rosbag_name, f"{topic_safe}-{str(v)}.webp")
                            if os.path.exists(fp):
                                best_path = fp
                                best_idx = candidate
                                topic_ts = str(v)
                                break
                if best_path:
                    break
            if best_path:
                file_path = best_path
                chosen_idx = best_idx
            else:
                return jsonify({'error': 'Image not found for nearby timestamps'}), 404

        rel_url = f"/images/{rosbag_name}/{os.path.basename(file_path)}"
        return jsonify({'imageUrl': rel_url, 'timestamp': topic_ts, 'index': chosen_idx, 'total': total}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoint to serve image reference map JSON file for a given rosbag
@app.route('/image-reference-map/<rosbag_name>.json', methods=['GET'])
def get_image_reference_map(rosbag_name):
    map_path = os.path.join(BASE_DIR, "image_reference_maps", f"{rosbag_name}.json")
    if not os.path.exists(map_path):
        return jsonify({"error": f"Image reference map for {rosbag_name} not found."}), 404
    return send_file(map_path, mimetype='application/json')

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
    timestamp = mapped_timestamps.get(topic) if topic else None

    if not timestamp:
        return jsonify({'error': 'No mapped timestamp found for the provided topic'})

    # Open the rosbag to find the message at the requested timestamp and topic
    reader = SequentialReader()
    try:
        reader.open(
            StorageOptions(uri=SELECTED_ROSBAG, storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Get topic types for message deserialization
        all_topics = reader.get_all_topics_and_types()
        topic_type_map = {topic_meta.name: topic_meta.type for topic_meta in all_topics}
        
        # Get the topic type for this specific topic
        topic_type = topic_type_map.get(topic)
        if not topic_type:
            return jsonify({'error': f'Topic {topic} not found in rosbag'})
        
        msg_type = get_message(topic_type)
        
        # Iterate through messages to find the one matching the timestamp
        while reader.has_next():
            read_topic, data, log_time = reader.read_next()
            
            # Only process messages from the requested topic
            if read_topic != topic:
                continue
            
            # Deserialize message to extract header timestamp
            try:
                msg = deserialize_message(data, msg_type)
                # Extract header timestamp (not log_time)
                header_timestamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            except Exception as e:
                # If header extraction fails, fallback to log_time
                logging.warning(f"Could not extract header timestamp from {topic}: {e}")
                header_timestamp = log_time
            
            # Match against the requested timestamp (from CSV alignment table, uses header stamps)
            if str(header_timestamp) == timestamp:
                # Deserialization already done above, now match message type
                match topic_type:
                    case 'sensor_msgs/msg/PointCloud2':
                        # Extract point cloud data, filtering out NaNs, Infs, and zeros
                        pointCloud = []
                        point_step = msg.point_step
                        for i in range(0, len(msg.data), point_step):
                            x, y, z = struct.unpack_from('fff', msg.data, i)
                            if all(np.isfinite([x, y, z])) and not (x == 0 and y == 0 and z == 0):
                                pointCloud.extend([x, y, z])
                        return jsonify({'type': 'pointCloud', 'pointCloud': pointCloud, 'timestamp': timestamp})
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

#@app.route('/api/enhance-prompt', methods=['GET'])
def enhance_prompt(user_prompt):
    try:
        #user_prompt = request.args.get('user_prompt', default=None, type=str)
        if not user_prompt:
            print('No prompt provided')
            #return jsonify({'error': 'No prompt provided'}), 400
        
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
        return enhanced_prompt
        #return jsonify({'enhancedPrompt': enhanced_prompt}), 200

    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        #return jsonify({'error': str(e)}), 500

@app.route('/api/search-status', methods=['GET'])
def get_search_status():
    return jsonify(SEARCH_PROGRESS)

@app.route('/api/search-new', methods=['GET'])
def search_new():
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00
    SEARCH_PROGRESS["message"] = "Enhancing prompt..."

    query_text = request.args.get('query', default=None, type=str)
    query_text = enhance_prompt(query_text)
    logging.warning(f"Enhanced prompt: {query_text}")

    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)

    SEARCH_PROGRESS["message"] = f"Starting search...\n\n(sampling every {accuracy}th embedding)"

    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400
    if models is None:
        return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:
        return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:
        return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:
        return jsonify({'error': 'No accuracy provided'}), 400

    models = models.split(",")
    rosbags = rosbags.split(",")
    rosbags = [os.path.basename(r) for r in rosbags]
    time_start, time_end = map(int, timeRange.split(","))

    # marks is now a dict: {(model, rosbag, topic): set(indices)}
    marks = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = []

    try:

        total_steps = 0
        for model in models:
            model_path = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model)
            if not os.path.isdir(model_path):
                continue
            for rosbag in rosbags:
                rosbag_path = os.path.join(model_path, rosbag)
                if not os.path.isdir(rosbag_path):
                    continue
                for topic in os.listdir(rosbag_path):
                    topic_path = os.path.join(rosbag_path, topic)
                    if os.path.isdir(topic_path):
                        total_steps += 1
        
        #total_steps = len(models) * len(rosbags)
        logging.warning(f"Total steps to process: {total_steps}")
        step_count = 0

        for model_idx, model_name in enumerate(models):
            name, pretrained = model_name.split('__')
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
            tokenizer = open_clip.get_tokenizer(name)
            query_embedding = get_text_embedding(query_text, model, tokenizer, device)

            logging.warning(f"Loaded model {model_name} on {device} and computed query embedding {query_embedding.shape}")

            model_dir = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model_name)
            if not os.path.isdir(model_dir):
                del model
                torch.cuda.empty_cache()
                continue

            # For each rosbag, iterate over topic folders
            for rosbag_name in rosbags:
                rosbag_dir = os.path.join(model_dir, rosbag_name)
                if not os.path.isdir(rosbag_dir):
                    step_count += 1
                    continue

                # --- Load aligned CSV for this rosbag_name ---
                aligned_path = os.path.join(LOOKUP_TABLES_DIR, rosbag_name + ".csv")
                aligned_data = pd.read_csv(aligned_path, dtype=str)
                logging.warning(f"Loaded aligned CSV with {len(aligned_data)} rows for {rosbag_name}")

                # List all topic folders in this rosbag_dir
                topic_folders = [t for t in os.listdir(rosbag_dir) if os.path.isdir(os.path.join(rosbag_dir, t))]
                logging.warning(f"Found {len(topic_folders)} topic folders in {rosbag_name} for model {model_name}")

                for topic_folder in topic_folders:
                    topic_name = topic_folder.replace("__", "/")
                    topic_dir = os.path.join(rosbag_dir, topic_folder)
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Loading embeddings...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Topic: {topic_name}\n\n"
                        f"(Searching every {accuracy}th embedding)"
                    )
                    step_count += 1
                    embedding_files = []
                    for file in os.listdir(topic_dir):
                        if file.endswith(".pt"):
                            try:
                                ts = file.replace("_embedding", "").replace(".pt", "")
                                timestamp = int(ts)
                                dt = datetime.utcfromtimestamp(timestamp / 1e9)
                                minute_of_day = dt.hour * 60 + dt.minute
                                if time_start <= minute_of_day <= time_end:
                                    emb_path = os.path.join(topic_dir, file)
                                    embedding_files.append(emb_path)
                            except Exception as e:
                                logging.warning(f"Skipping file {file} due to error: {e}")
                                continue

                    selected_paths = embedding_files[::accuracy]
                    logging.warning(f"Selected {len(selected_paths)} embeddings after applying accuracy filter of {accuracy}")
                    model_embeddings = []
                    model_paths = []
                    for emb_path in selected_paths:
                        try:
                            embedding = torch.load(emb_path, map_location='cpu')
                            if isinstance(embedding, torch.Tensor):
                                embedding = embedding.cpu().numpy()
                            model_embeddings.append(embedding)
                            model_paths.append(emb_path)
                        except Exception as e:
                            logging.warning(f"Error loading embedding from {emb_path}: {e}")
                            continue
                    if not model_embeddings:
                        continue
                    embeddings = np.vstack(model_embeddings).astype('float32')
                    n, d = embeddings.shape
                    index = faiss.IndexFlatL2(d)
                    logging.warning(f"Created FAISS index with {n} embeddings of dimension {d} and name {index.__class__.__name__}")
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Creating FAISS index using {index.__class__.__name__}) "
                        f"from {len(model_embeddings)} sampled embeddings (every {accuracy}th)."
                    )
                    index.add(embeddings)
                    embedding_paths = np.array(model_paths)
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Searching {len(model_embeddings)} sampled embeddings...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Topic: {topic_folder}\n"
                        f"Index: {index.__class__.__name__}\n\n"
                        f"(Searching every {accuracy}th embedding)"
                    )
                    similarityScores, indices = index.search(query_embedding.reshape(1, -1), MAX_K)
                    if len(indices) == 0 or len(similarityScores) == 0:
                        continue
                    model_results = []
                    topic_name = topic_folder.replace("__", "/")
                    for i, idx in enumerate(indices[0][:MAX_K]):
                        similarityScore = float(similarityScores[0][i])
                        if math.isnan(similarityScore) or math.isinf(similarityScore):
                            continue
                        embedding_path = str(embedding_paths[idx])
                        path_of_interest = str(os.path.basename(embedding_path))
                        # result_timestamp from filename
                        result_timestamp = path_of_interest[-32:-13]
                        dt_result = datetime.utcfromtimestamp(int(result_timestamp) / 1e9)
                        minute_of_day = dt_result.strftime("%H:%M")
                        model_results.append({
                            'rank': i + 1,
                            'rosbag': rosbag_name,
                            'embedding_path': embedding_path,
                            'similarityScore': similarityScore,
                            'topic': topic_name,
                            'timestamp': result_timestamp,
                            'minuteOfDay': minute_of_day,
                            'model': model_name
                        })

                        matching_reference_timestamps = aligned_data.loc[
                            aligned_data.isin([result_timestamp]).any(axis=1),
                            'Reference Timestamp'
                        ].tolist()
                        match_indices = []
                        for ref_ts in matching_reference_timestamps:
                            indices_ = aligned_data.index[aligned_data['Reference Timestamp'] == ref_ts].tolist()
                            match_indices.extend(indices_)
                        key = (model_name, rosbag_name, topic_name)
                        if key not in marks:
                            marks[key] = set()
                        for index_val in match_indices:
                            marks[key].add(index_val)
                    all_results.extend(model_results)
            del model
            torch.cuda.empty_cache()
    except Exception as e:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Error: {e}"
        return jsonify({'error': str(e)}), 500

    # Flatten and sort all results
    all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
    for rank, result in enumerate(all_results, 1):
        result['rank'] = rank
    filtered_results = all_results

    # --- Construct categorizedSearchResults ---
    categorizedSearchResults = {}
    for result in all_results:
        model = result['model']
        rosbag = result['rosbag']
        topic = result['topic']
        minute_of_day = result['minuteOfDay']
        rank = result['rank']
        similarity_score = result['similarityScore']
        timestamp = result['timestamp']
        categorizedSearchResults.setdefault(model, {}).setdefault(rosbag, {}).setdefault(topic, {
            'marks': [],
            'results': []
        })
        categorizedSearchResults[model][rosbag][topic]['results'].append({
            'minuteOfDay': minute_of_day,
            'rank': rank,
            'similarityScore': similarity_score
        })

    # Populate marks per topic only once per mark
    for key, indices in marks.items():
        model, rosbag, topic = key
        if model in categorizedSearchResults and rosbag in categorizedSearchResults[model] and topic in categorizedSearchResults[model][rosbag]:
            for index_val in indices:
                categorizedSearchResults[model][rosbag][topic]['marks'].append({'value': index_val})
    # Flatten marks for response (for compatibility)
    flat_marks = [{'value': idx} for indices in marks.values() for idx in indices]
    SEARCH_PROGRESS["status"] = "done"
    return jsonify({'query': query_text, 'results': filtered_results, 'marks': flat_marks, 'categorizedSearchResults': categorizedSearchResults})

@app.route('/api/search-new-c', methods=['GET'])
def search_new_c():
    from pathlib import Path
    import traceback

    # ---- Initial status
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00

    # ---- Inputs
    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)
    enhancePrompt = request.args.get('enhancePrompt', default='true', type=str).lower() == 'true'
    MAX_K = globals().get("MAX_K", 50)

    #logging.warning("[C] /api/search-new-c called with raw params: query=%r models=%r rosbags=%r timeRange=%r accuracy=%r MAX_K=%r",
    #                query_text, models, rosbags, timeRange, accuracy, MAX_K)

    # ---- Validate inputs
    if query_text is None:                 return jsonify({'error': 'No query text provided'}), 400
    if models is None:                     return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:                    return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:                  return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:                   return jsonify({'error': 'No accuracy provided'}), 400
    if enhancePrompt not in (True, False): return jsonify({'error': 'Invalid enhancePrompt value'}), 400

    # ---- Enhance prompt
    if enhancePrompt:
        SEARCH_PROGRESS["message"] = "Enhancing prompt..."
        query_text = enhance_prompt(query_text)
        logging.warning("[C] Enhanced prompt: %s", query_text)

    # ---- Parse inputs
    try:
        models_list = models.split(",")
        rosbags_list = [os.path.basename(r) for r in rosbags.split(",")]
        time_start, time_end = map(int, timeRange.split(","))
        k_subsample = max(1, int(accuracy))
    except Exception as e:
        logging.exception("[C] Failed parsing inputs")
        return jsonify({'error': f'Invalid inputs: {e}'}), 400

    #logging.warning("[C] Parsed: models=%s rosbags=%s time_start=%d time_end=%d accuracy(subsample k)=%d",
     #               models_list, rosbags_list, time_start, time_end, k_subsample)

    # ---- Paths cast to Path (avoid str / str errors)
    ECD = Path(EMBEDDINGS_CONSOLIDATED_DIR)
    EPT = Path(EMBEDDINGS_PER_TOPIC_DIR)
    LKT = Path(LOOKUP_TABLES_DIR)
    #logging.warning("[C] Roots: ECD=%s (exists=%s)  EPT=%s (exists=%s)  LKT=%s (exists=%s)",
     #               ECD, ECD.exists(), EPT, EPT.exists(), LKT, LKT.exists())

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
                        name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache"
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
                base = ECD / model_name / rosbag_name
                manifest_path = base / "manifest.parquet"
                shards_dir = base / "shards"
                #logging.warning("[C] Base=%s  manifest=%s (exists=%s)  shards_dir=%s (exists=%s)",
                #                base, manifest_path, manifest_path.exists(), shards_dir, shards_dir.exists())

                if not manifest_path.is_file() or not shards_dir.is_dir():
                    logging.warning("[C] SKIP: Missing manifest or shards for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Manifest schema check
                needed_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns"]
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
                aligned_path = LKT / f"{rosbag_name}.csv"
                if not aligned_path.is_file():
                    logging.warning("[C] WARN: aligned CSV not found: %s (marks will be empty)", aligned_path)
                    aligned_data = pd.DataFrame()
                else:
                    aligned_data = pd.read_csv(aligned_path, dtype=str)
                #logging.warning("[C] aligned_data.shape=%s columns=%s", getattr(aligned_data, "shape", None), getattr(aligned_data, "columns", None))

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
                    minute_str = datetime.utcfromtimestamp(ts_ns / 1e9).strftime("%H:%M")
                    pt_path = EPT / model_name / rosbag_name / m["topic_folder"] / f"{ts_ns}.pt"
                    model_results.append({
                        'rank': i,
                        'rosbag': rosbag_name,
                        'embedding_path': str(pt_path),
                        'similarityScore': float(dist),
                        'topic': m["topic"],           # slashes
                        'timestamp': ts_str,
                        'minuteOfDay': minute_str,
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

        # categorizedSearchResults
        categorizedSearchResults: dict = {}
        for result in filtered_results:
            model = result['model']
            rosbag = result['rosbag']
            topic = result['topic']
            minute_of_day = result['minuteOfDay']
            rank = result['rank']
            similarity_score = result['similarityScore']

            categorizedSearchResults \
                .setdefault(model, {}) \
                .setdefault(rosbag, {}) \
                .setdefault(topic, {'marks': [], 'results': []})

            categorizedSearchResults[model][rosbag][topic]['results'].append({
                'minuteOfDay': minute_of_day,
                'rank': rank,
                'similarityScore': similarity_score
            })

        for key, indices in marks.items():
            model_key, rosbag_key, topic_key = key
            if (
                model_key in categorizedSearchResults
                and rosbag_key in categorizedSearchResults[model_key]
                and topic_key in categorizedSearchResults[model_key][rosbag_key]
            ):
                categorizedSearchResults[model_key][rosbag_key][topic_key]['marks'].extend(
                    {'value': idx} for idx in indices
                )

        # marks flatten
        flat_marks = [{'value': idx} for indices in marks.values() for idx in indices]
        #logging.warning("[C] categorizedSearchResults models=%d  flat_marks=%d", len(categorizedSearchResults), len(flat_marks))

        SEARCH_PROGRESS["status"] = "done"
        return jsonify({
            'query': query_text,
            'results': filtered_results,
            'marks': flat_marks,
            'categorizedSearchResults': categorizedSearchResults
        })

    except Exception as e:
        logging.exception("[C] search-new-c failed")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Error: {e}\n\n{traceback.format_exc()}"
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Perform semantic search using CLIP embedding against precomputed image features
@app.route('/api/search', methods=['GET'])
def search():
    """
    Perform semantic search using CLIP embedding against precomputed image features.
    Logic:
      - Loads the selected CLIP model and tokenizer.
      - Computes query embedding for the input query text.
      - Loads the FAISS index and embedding paths for the selected model and Rosbag.
      - If multiple searched rosbags are provided, creates a temporary FAISS index from sampled embeddings.
      - Searches the index for the top MAX_K results.
      - For each result:
          - Extracts the topic and timestamp from the embedding path.
          - Appends search results, including similarity score and model.
          - For each result timestamp, finds all reference timestamps in ALIGNED_DATA that map to it (used for UI marks).
    """
    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=str)

    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400
    """if models is None:
        return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:
        return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:
        return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:
        return jsonify({'error': 'No accuracy provided'}), 400
"""

    # Insert logic for multi-rosbag search using sampled embeddings
    results = []
    marks = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load model, tokenizer, and compute query embedding
        name, pretrained = SELECTED_MODEL.split('__')
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
        tokenizer = open_clip.get_tokenizer(name)
        query_embedding = get_text_embedding(query_text, model, tokenizer, device)

        # --- Refactored searchedRosbags logic ---
        searched_rosbags = SEARCHED_ROSBAGS
        if not searched_rosbags:
            return jsonify({"error": "No rosbags selected"}), 400

        if len(searched_rosbags) == 1:
            rosbag_name = os.path.basename(searched_rosbags[0]).replace('.db3', '')
            subdir = f"{name}__{pretrained}"
            index_path = os.path.join(INDICES_DIR, subdir, rosbag_name, "faiss_index.index")
            embedding_paths_file = os.path.join(INDICES_DIR, subdir, rosbag_name, "embedding_paths.npy")

            if not os.path.exists(index_path) or not os.path.exists(embedding_paths_file):
                del model
                torch.cuda.empty_cache()
                return jsonify({"error": "Index or embedding paths not found"}), 404

            index = faiss.read_index(index_path)
            embedding_paths = np.load(embedding_paths_file)

        else:
            all_embeddings = []
            all_paths = []

            for rosbag_path in searched_rosbags:
                rosbag = os.path.basename(rosbag_path)
                subdir = f"{name}__{pretrained}"
                embedding_folder_path = os.path.join(EMBEDDINGS_DIR, subdir, rosbag)

                for root, _, files in os.walk(embedding_folder_path):
                    for file in files:
                        if file.lower().endswith('_embedding.pt'):
                            input_file_path = os.path.join(root, file)
                            try:
                                embedding = torch.load(input_file_path, weights_only=True)
                                all_embeddings.append(embedding.cpu().numpy())
                                all_paths.append(input_file_path)
                            except Exception as e:
                                print(f"Error loading {input_file_path}: {e}")

            if not all_embeddings:
                del model
                torch.cuda.empty_cache()
                return jsonify({"error": "No embeddings found for selected rosbags"}), 404

            stacked_embeddings = np.vstack(all_embeddings[::10]).astype('float32')
            index = faiss.IndexFlatL2(stacked_embeddings.shape[1])
            index.add(stacked_embeddings)
            embedding_paths = np.array(all_paths[::10])

        # (Removed obsolete loading of FAISS index and embedding paths here)

        # Perform nearest neighbor search in the embedding space
        similarityScores, indices = index.search(query_embedding.reshape(1, -1), MAX_K)
        if len(indices) == 0 or len(similarityScores) == 0:
            del model
            torch.cuda.empty_cache()
            return jsonify({'error': 'No results found for the query'}), 200

        # For each result, extract topic/timestamp and match back to reference timestamps for UI highlighting
        for i, idx in enumerate(indices[0][:MAX_K]):
            similarityScore = float(similarityScores[0][i])
            if math.isnan(similarityScore) or math.isinf(similarityScore):
                continue
            embedding_path = str(embedding_paths[idx])
            path_of_interest = str(os.path.basename(embedding_path))
            relative_path = os.path.relpath(embedding_path, EMBEDDINGS_DIR)
            rosbag_name = os.path.basename(os.path.dirname(relative_path))
            result_timestamp = path_of_interest[-32:-13]
            result_topic = path_of_interest[:-33].replace("__", "/")
            results.append({
                'rank': i + 1,
                'rosbag': rosbag_name,
                'embedding_path': embedding_path,
                'similarityScore': similarityScore,
                'topic': result_topic,
                'timestamp': result_timestamp,
                'model': SELECTED_MODEL
            })

            if rosbag_name != os.path.basename(SELECTED_ROSBAG):
                # If the result is from a different Rosbag, skip alignment
                continue
            # For each result, find all reference timestamps that align to this result timestamp (for UI marks)
            matching_reference_timestamps = ALIGNED_DATA.loc[
                ALIGNED_DATA.isin([result_timestamp]).any(axis=1),
                'Reference Timestamp'
            ].tolist()
            match_indices = []
            for ref_ts in matching_reference_timestamps:
                indices = ALIGNED_DATA.index[ALIGNED_DATA['Reference Timestamp'] == ref_ts].tolist()
                match_indices.extend(indices)
            for index in match_indices:
                marks.append({'value': index})

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        return jsonify({'error': str(e)}), 500

    return jsonify({'query': query_text, 'results': results, 'marks': marks})


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

        EXPORT_PATH = os.path.join(EXPORT_DIR, new_rosbag_name)
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
