#!/usr/bin/env python3
"""
Single embeddings step: Generate embeddings and save them as individual .pt files.

This script generates embeddings for all models and saves them as individual files.
It loads preprocessed images (from 01_preprocess_images.py) and generates embeddings.
"""

import os
import re
import json
import math
import warnings
import shutil
import signal
import sys
import copy
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Sequence, Iterable, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from tqdm import tqdm
import open_clip
import gc
from dotenv import load_dotenv

try:
    from open_clip.transform import image_transform
except ImportError:
    image_transform = None

# =========================
# Environment Setup
# =========================

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

def _require_env_path(var_name: str) -> Path:
    """Require and return an environment variable as a Path."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable {var_name} is not set.")
    return Path(value).expanduser()

# Paths
ROSBAGS = Path(os.getenv("ROSBAGS")) if os.getenv("ROSBAGS") else None
IMAGES = Path(os.getenv("IMAGES"))
PREPROCESSED = Path(os.getenv("PREPROCESSED"))
SINGLE_EMBEDDINGS = Path(os.getenv("SINGLE_EMBEDDINGS"))
EMBEDDINGS = Path(os.getenv("EMBEDDINGS"))

# Validate required paths
if IMAGES is None:
    raise RuntimeError("Environment variable IMAGES is not set.")
if EMBEDDINGS is None:
    raise RuntimeError("Environment variable EMBEDDINGS is not set.")

# =========================
# Configuration Flags
# =========================

SKIP_COMPLETED = True  # Set to False to reprocess completed rosbags

# Global variable to track current rosbag for signal handling
_current_rosbag_name: Optional[str] = None
_interrupted = False

# =========================
# Multiprocessing Configuration
# =========================

PREPROCESS_WORKERS = 8  # Threading workers for preprocessing (I/O-bound, can handle more)
EMBEDDING_SAVE_WORKERS = 4  # Threading workers for parallel embedding file saving (I/O-bound)
BATCH_SIZE = 256  # Embedding batch size (increased for better GPU utilization)
PREPROCESS_CHUNK_SIZE = 2000  # Process images in chunks to avoid OOM (load this many preprocessed tensors at a time)

# =========================
# Model Configurations
# =========================

# OpenCLIP models
OPENCLIP_MODELS = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]

# Custom CLIP models
CUSTOM_MODELS = [
    {
        "name": "ViT-B-16-finetuned(09.10.25)",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/ViT-B-16-finetuned(09.10.25).pt",
        "model_dir_name": "ViT-B-16-finetuned(09.10.25)",
        "batch_size": 64,
        "enabled": True,
    },
    {
        "name": "agriclip",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/agriclip.pt",
        "model_dir_name": "agriclip",
        "batch_size": 64,
        "enabled": True,
    },
]

# =========================
# Constants
# =========================

DEFAULT_SHARD_ROWS = 100_000
OUTPUT_DTYPE = np.float32
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_TS_LEADING_RE = re.compile(r"^(\d+)")
CACHE_DIR = "/mnt/data/openclip_cache"
NUM_GPUS = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 0
USE_MULTI_GPU = NUM_GPUS > 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Silence noisy dependency warnings
warnings.filterwarnings("ignore", message=r".*Failed to load image Python extension.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Unable to find acceptable character detection dependency.*")

# =========================
# Utility Functions
# =========================

def log(msg: str) -> None:
    """Print message with flush."""
    print(msg, flush=True)

def ns_to_minute_of_day(ts_ns: int) -> int:
    """Convert nanosecond timestamp to minute of day."""
    dt = datetime.utcfromtimestamp(ts_ns / 1e9)
    return dt.hour * 60 + dt.minute

def topic_folder_to_topic_name(folder: str) -> str:
    """Convert topic folder name to topic name (replace __ with /)."""
    return folder.replace("__", "/")

def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)

# =========================
# Completion Tracking Functions
# =========================

def get_completion_file() -> Path:
    """Get the completion file path for single embeddings."""
    # Store completion file in SINGLE_EMBEDDINGS directory
    return SINGLE_EMBEDDINGS / "completion.json"


def load_completion() -> Dict[str, Dict]:
    """Load the dictionary of completed rosbag data from the completion file.
    
    Returns:
        Dict mapping rosbag_name to dict with:
        - models: Dict mapping model_id to {"status": "success|failed|skipped", "completed_at": "...", "error": "..."}
        - all_completed: bool indicating if all models completed
    """
    completion_file = get_completion_file()
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
            completed_list = data.get('completed', [])
            result = {}
            
            for item in completed_list:
                if isinstance(item, str):
                    # Old format - just rosbag name, migrate to new format
                    result[item] = {
                        "models": {},
                        "all_completed": False
                    }
                elif isinstance(item, dict):
                    # New format - dict with rosbag name and model tracking
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        # Migrate old format if needed
                        if "completed_at" in item and "models" not in item:
                            # Old format - convert to new
                            result[rosbag_name] = {
                                "models": {},
                                "all_completed": False,
                                "legacy_completed_at": item.get("completed_at", "unknown"),
                                "legacy_errors": item.get("errors", [])
                            }
                        else:
                            # New format
                            result[rosbag_name] = {
                                "models": item.get("models", {}),
                                "all_completed": item.get("all_completed", False)
                            }
            return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return {}


def save_completed_step(rosbag_name: str, model_id: str, step: str, status: str = "success", error: Optional[str] = None):
    """Mark a specific step as completed for a model.
    
    Args:
        rosbag_name: Name of the rosbag
        model_id: Model identifier (e.g., "ViT-B-32-quickgelu__openai" or "agriclip")
        step: Step name ("preprocessing", "individual_embeddings", "sharded_embeddings")
        status: "success", "failed", or "skipped"
        error: Optional error message if status is "failed"
    """
    completed = load_completion()
    timestamp = datetime.now().isoformat()
    
    # Initialize rosbag entry if needed
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "models": {},
            "all_completed": False
        }
    
    # Initialize model entry if needed
    if model_id not in completed[rosbag_name]["models"]:
        completed[rosbag_name]["models"][model_id] = {
            "steps": {},
            "all_steps_completed": False
        }
    
    # Initialize steps dict if needed
    if "steps" not in completed[rosbag_name]["models"][model_id]:
        completed[rosbag_name]["models"][model_id]["steps"] = {}
    
    # Update step status
    completed[rosbag_name]["models"][model_id]["steps"][step] = {
        "status": status,
        "completed_at": timestamp
    }
    if error:
        completed[rosbag_name]["models"][model_id]["steps"][step]["error"] = error
    
    # For this script, we only track "individual_embeddings" step
    # Mark it as completed
    completed[rosbag_name]["models"][model_id]["all_steps_completed"] = True
    
    # Check if all models are completed
    all_model_ids = get_all_model_ids()
    all_models_completed = all(
        completed[rosbag_name]["models"].get(model_id, {}).get("all_steps_completed", False)
        for model_id in all_model_ids
    )
    completed[rosbag_name]["all_completed"] = all_models_completed
    
    # Save to file
    save_completion(completed)


def save_completed_model(rosbag_name: str, model_id: str, status: str = "success", error: Optional[str] = None):
    """Legacy function: Mark a model as completed (for backward compatibility).
    Note: This is deprecated - use save_completed_step instead for per-step tracking.
    
    Args:
        rosbag_name: Name of the rosbag
        model_id: Model identifier
        status: "success", "failed", or "skipped"
        error: Optional error message if status is "failed"
    """
    # For backward compatibility, mark all steps as completed
    completed = load_completion()
    timestamp = datetime.now().isoformat()
    
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "models": {},
            "all_completed": False
        }
    
    if model_id not in completed[rosbag_name]["models"]:
        completed[rosbag_name]["models"][model_id] = {
            "steps": {},
            "all_steps_completed": False
        }
    
    # Mark individual_embeddings step as completed
    if "steps" not in completed[rosbag_name]["models"][model_id]:
        completed[rosbag_name]["models"][model_id]["steps"] = {}
    
    if "individual_embeddings" not in completed[rosbag_name]["models"][model_id]["steps"]:
        completed[rosbag_name]["models"][model_id]["steps"]["individual_embeddings"] = {
            "status": status,
            "completed_at": timestamp
        }
        if error:
            completed[rosbag_name]["models"][model_id]["steps"]["individual_embeddings"]["error"] = error
    
    save_completion(completed)


def save_completed_rosbag(rosbag_name: str, errors: Optional[List[str]] = None):
    """Legacy function: Mark a rosbag as completed (for backward compatibility).
    Note: This is deprecated - use save_completed_model instead for per-model tracking.
    
    Args:
        rosbag_name: Name of the rosbag
        errors: Optional list of error messages
    """
    # For backward compatibility, mark all models as completed
    # This is a fallback for error cases
    completed = load_completion()
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "models": {},
            "all_completed": False
        }
    
    # If we have errors, mark as failed for all models
    if errors:
        all_model_ids = get_all_model_ids()
        timestamp = datetime.now().isoformat()
        for model_id in all_model_ids:
            if model_id not in completed[rosbag_name]["models"]:
                completed[rosbag_name]["models"][model_id] = {
                    "status": "failed",
                    "completed_at": timestamp,
                    "error": "; ".join(errors)
                }
    
    save_completion(completed)


def get_all_model_ids() -> List[str]:
    """Get list of all model IDs that should be processed.
    
    Returns:
        List of model identifiers
    """
    model_ids = []
    
    # Get all open_clip models
    for model_name, pretrained_name in OPENCLIP_MODELS:
        model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
        model_ids.append(model_dir_id)
    
    # Get all custom models
    for config in CUSTOM_MODELS:
        if config.get("enabled", True):
            model_dir_name = config.get("model_dir_name", config["name"])
            model_ids.append(model_dir_name)
    
    return model_ids


def is_step_completed(rosbag_name: str, model_id: str, step: str) -> bool:
    """Check if a specific step has been completed for a model.
    
    Args:
        rosbag_name: Name of the rosbag
        model_id: Model identifier
        step: Step name ("preprocessing", "individual_embeddings", "sharded_embeddings")
    
    Returns:
        True if step is marked as completed (success or skipped, not failed)
    """
    completed = load_completion()
    if rosbag_name not in completed:
        return False
    if model_id not in completed[rosbag_name].get("models", {}):
        return False
    model_data = completed[rosbag_name]["models"][model_id]
    steps = model_data.get("steps", {})
    if step not in steps:
        return False
    step_status = steps[step].get("status", "")
    # Only return True if step is successfully completed or skipped (not failed)
    return step_status in ("success", "skipped")


def is_model_completed(rosbag_name: str, model_id: str) -> bool:
    """Check if a specific model has been fully completed (all required steps done).
    
    Args:
        rosbag_name: Name of the rosbag
        model_id: Model identifier
    
    Returns:
        True if all required steps for the model are completed
    """
    completed = load_completion()
    if rosbag_name not in completed:
        return False
    if model_id not in completed[rosbag_name].get("models", {}):
        return False
    
    model_data = completed[rosbag_name]["models"][model_id]
    return model_data.get("all_steps_completed", False)


def save_completion(completed: Dict[str, Dict]):
    """Save completed rosbags dictionary to completion file.
    
    Args:
        completed: Dictionary of completed rosbags to save
    """
    completion_file = get_completion_file()
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file with proper format (including steps)
    data = {
        'completed': [
            {
                "rosbag": name,
                "models": {
                    model_id: {
                        "steps": model_info.get("steps", {}),
                        "all_steps_completed": model_info.get("all_steps_completed", False)
                    }
                    for model_id, model_info in info.get("models", {}).items()
                },
                "all_completed": info.get("all_completed", False)
            }
            for name, info in sorted(completed.items())
        ],
    }
    
    with open(completion_file, 'w') as f:
        json.dump(data, f, indent=2)


def is_rosbag_completed(rosbag_name: str) -> bool:
    """Check if a rosbag has been fully completed (all models processed).
    
    Returns:
        True if all models for this rosbag are completed
    """
    completed = load_completion()
    if rosbag_name not in completed:
        return False
    return completed[rosbag_name].get("all_completed", False)


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, SIGTERM)."""
    global _interrupted, _current_rosbag_name
    _interrupted = True
    log(f"\n⚠️  Interrupt signal received (signal {signum})")
    if _current_rosbag_name:
        log(f"  ⚠️  Marking current rosbag {_current_rosbag_name} as failed due to interruption")
        save_completed_rosbag(_current_rosbag_name, errors=[f"Process interrupted by signal {signum}"])
    log("⚠️  Exiting...")
    sys.exit(130)  # Standard exit code for SIGINT

# =========================
# Preprocessing Functions (from 03_preprocess_images.py)
# =========================

preprocess_dict: Dict[str, List[str]] = {}
_PREPROCESS_CACHE: Dict[Tuple[str, str], object] = {}
_CUSTOM_PREPROCESS_CACHE: Dict[str, object] = {}

def get_preprocess_transform(model_name: str, pretrained_name: str, cache_dir: Optional[str] = None):
    """Fetch and cache the preprocessing transform for a given model/pretrained pair."""
    key = (model_name, pretrained_name)
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key]

    cache_dir = cache_dir or CACHE_DIR
    preprocess = None
    try:
        cfg = open_clip.get_preprocess_cfg(pretrained_name)
        if image_transform is None:
            raise RuntimeError("open_clip.transform.image_transform unavailable")
        preprocess = image_transform(cfg, is_train=False)
    except Exception:
        # Fallback to full model creation if direct cfg access fails
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            cache_dir=cache_dir,
        )

    _PREPROCESS_CACHE[key] = preprocess
    return preprocess

def collect_preprocess_data(model_name: str, pretrained_name: str, cache_dir: Optional[str] = None) -> None:
    """Collect unique preprocessing transforms and associate them with model identifiers."""
    preprocess = get_preprocess_transform(model_name, pretrained_name, cache_dir)
    preprocess_str = str(preprocess)
    model_id = f"{model_name} ({pretrained_name})"

    if preprocess_str not in preprocess_dict:
        preprocess_dict[preprocess_str] = []
    if model_id not in preprocess_dict[preprocess_str]:
        preprocess_dict[preprocess_str].append(model_id)

def get_preprocess_id(preprocess_str: str) -> str:
    """Generate a concise identifier string summarizing the preprocessing steps."""
    summary_parts = []

    resize_match = re.search(r"Resize\(size=(\d+)", preprocess_str)
    if resize_match:
        summary_parts.append(f"resize{resize_match.group(1)}")

    crop_match = re.search(r"CenterCrop\(size=\(?(\d+)", preprocess_str)
    if crop_match:
        summary_parts.append(f"crop{crop_match.group(1)}")

    if "ToTensor" in preprocess_str:
        summary_parts.append("tensor")

    if "Normalize" in preprocess_str:
        summary_parts.append("norm")

    return "_".join(summary_parts) if summary_parts else "default"

def preprocess_image(image_path: Path, preprocess_transform) -> Optional[torch.Tensor]:
    """Preprocess a single image file."""
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = preprocess_transform(image)
        return image_tensor
    except Exception as e:
        log(f"      ⚠️  Error preprocessing {image_path}: {e}")
        return None

def get_image_resolution_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract image resolution from a custom CLIP model checkpoint."""
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict: Dict[str, torch.Tensor]
        
        if isinstance(raw, dict):
            # Try to find state dict in common keys
            candidate = None
            for key in ("state_dict", "model_state_dict", "model", "ema_state_dict"):
                if key in raw and isinstance(raw[key], dict):
                    candidate = raw[key]
                    break
            if candidate is None:
                # Use raw dict if it contains tensors
                state_dict = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            else:
                state_dict = candidate
        else:
            state_dict = raw
        
        # Strip common prefixes
        for prefix in ("module.", "model.", "clip."):
            if all(key.startswith(prefix) for key in state_dict.keys()):
                state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
        
        # Try to find visual weights under different prefixes
        if "visual.conv1.weight" not in state_dict:
            suspected_prefixes = {key.split(".", 1)[0] for key in state_dict.keys()}
            for prefix in suspected_prefixes:
                candidate = {key[len(prefix) + 1:]: value for key, value in state_dict.items() if key.startswith(prefix + ".")}
                if "visual.conv1.weight" in candidate:
                    state_dict = candidate
                    break
        
        if "visual.conv1.weight" not in state_dict:
            raise KeyError("Checkpoint does not contain CLIP visual weights")
        
        vision_patch_size = state_dict["visual.conv1.weight"].shape[2]
        num_pos_tokens = state_dict["visual.positional_embedding"].shape[0]
        grid_size = int((num_pos_tokens - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        return image_resolution
    except Exception as e:
        log(f"      ⚠️  Error extracting image resolution from {checkpoint_path}: {e}")
        # Default to 224 if we can't determine
        return 224

def get_custom_preprocess_transform(checkpoint_path: Path) -> transforms.Compose:
    """Get preprocessing transform for a custom CLIP model."""
    checkpoint_str = str(checkpoint_path)
    if checkpoint_str in _CUSTOM_PREPROCESS_CACHE:
        return _CUSTOM_PREPROCESS_CACHE[checkpoint_str]
    
    image_resolution = get_image_resolution_from_checkpoint(checkpoint_path)
    
    # Custom CLIP models use: resize to image_resolution, normalize with CLIP mean/std
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    preprocess = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    _CUSTOM_PREPROCESS_CACHE[checkpoint_str] = preprocess
    return preprocess

# =========================
# Custom CLIP Model Classes (from 04B_create_not_open_clip_embeddings.py)
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
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model)),
            ])
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
            output_dim=embed_dim,
        )

        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads,
                                       attn_mask=self.build_text_mask(context_length))
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

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

def count_layers(prefix: str, state_dict: dict[str, torch.Tensor]) -> int:
    """Count layers in state dict by prefix."""
    layers = set()
    prefix_parts = prefix.split(".")
    index = len(prefix_parts)
    for key in state_dict.keys():
        if key.startswith(prefix):
            layer_id = key.split(".")[index]
            layers.add(int(layer_id))
    return len(layers)

def build_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> CLIP:
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

    dtype = state_dict["visual.class_embedding"].dtype
    if dtype == torch.float16:
        model = model.half()
    model.load_state_dict(state_dict, strict=True)
    return model.eval()

def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Strip prefix from state dict keys."""
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict

def _first_matching_key(container: dict, keys: Iterable[str]) -> dict | None:
    """Find first matching key in container."""
    for key in keys:
        if key in container and isinstance(container[key], dict):
            return container[key]
    return None

def load_agriclip(checkpoint_path: str, device: str = "cpu") -> CLIP:
    """Load custom CLIP model from checkpoint."""
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, torch.Tensor]

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
        raise KeyError("Checkpoint does not contain CLIP visual weights under expected keys.")

    model = build_model_from_state_dict(state_dict)
    model.to(torch.device(device))
    return model

def load_image_tensor(image_path: Path, image_resolution: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load and preprocess image for custom CLIP model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_resolution, image_resolution), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device=device, dtype=dtype)

# =========================
# Shard and Manifest Functions (from 05_create_manifests.py)
# =========================

def load_meta(meta_path: Path) -> Dict:
    """Load metadata JSON file."""
    if meta_path.is_file():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

def save_meta(meta_path: Path, meta: Dict) -> None:
    """Save metadata JSON file."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def read_manifest(manifest_path: Path) -> pd.DataFrame:
    """Read manifest parquet file."""
    if manifest_path.is_file():
        return pd.read_parquet(manifest_path)
    return pd.DataFrame(columns=["id", "topic", "timestamp_ns", "minute_of_day", "shard_id", "row_in_shard", "pt_path"])

def write_manifest(manifest_path: Path, df: pd.DataFrame) -> None:
    """Write manifest parquet file."""
    table = pa.Table.from_pandas(df.reset_index(drop=True))
    pq.write_table(table, manifest_path, compression="zstd")

def next_shard_name(shards_dir: Path) -> str:
    """Get next shard filename."""
    existing = sorted(shards_dir.glob("shard-*.npy"))
    if not existing:
        return "shard-00000.npy"
    last = existing[-1].stem
    try:
        idx = int(last.split("-")[1])
    except Exception:
        idx = len(existing) - 1
    return f"shard-{idx+1:05d}.npy"

def _parse_timestamp_ns_from_stem(stem: str) -> Optional[int]:
    """Parse timestamp from filename stem."""
    m = _TS_LEADING_RE.match(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def process_batch_on_gpu(
    batch_data: List[Tuple[Path, str, int, torch.Tensor]],
    model: torch.nn.Module,
    device: str,
    gpu_id: int
) -> List[Tuple[Path, torch.Tensor, str, int]]:
    """
    Process a batch of preprocessed images on a specific GPU.
    
    Args:
        batch_data: List of (image_path, topic_name, timestamp_ns, preprocessed_tensor)
        model: The model to use for encoding
        device: Device string (e.g., "cuda:0")
        gpu_id: GPU ID for logging
    
    Returns:
        List of (image_path, embedding, topic_name, timestamp_ns)
    """
    try:
        # Build batch tensor on specified GPU
        batch_tensors = []
        batch_paths = []
        batch_topics = []
        batch_timestamps = []
        
        for image_path, topic_name, timestamp_ns, preprocessed_tensor in batch_data:
            # Move tensor to specified GPU
            tensor_gpu = preprocessed_tensor.to(device)
            if tensor_gpu.dim() == 3:
                tensor_gpu = tensor_gpu.unsqueeze(0)
            batch_tensors.append(tensor_gpu)
            batch_paths.append(image_path)
            batch_topics.append(topic_name)
            batch_timestamps.append(timestamp_ns)
        
        if not batch_tensors:
            return []
        
        # Concatenate and process batch within device context
        batch_tensor = torch.cat(batch_tensors, dim=0)
        with torch.cuda.device(gpu_id):
            with torch.no_grad():
                batch_embeddings = model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
        
        # Move to CPU and store
        batch_embeddings_cpu = batch_embeddings.cpu()
        results = []
        for i, embedding in enumerate(batch_embeddings_cpu):
            results.append((batch_paths[i], embedding, batch_topics[i], batch_timestamps[i]))
        
        return results
    except Exception as e:
        log(f"      ⚠️  Error processing batch on GPU {gpu_id}: {e}")
        return []

def process_batch_on_gpu_custom(
    batch_data: List[Tuple[Path, str, int]],
    model: torch.nn.Module,
    device: str,
    gpu_id: int,
    image_resolution: int,
    dtype: torch.dtype
) -> List[Tuple[Path, torch.Tensor, str, int]]:
    """
    Process a batch of images on a specific GPU for custom models (loads images directly).
    
    Args:
        batch_data: List of (image_path, topic_name, timestamp_ns)
        model: The model to use for encoding
        device: Device string (e.g., "cuda:0")
        gpu_id: GPU ID for logging
        image_resolution: Image resolution for the model
        dtype: Data type for tensors
    
    Returns:
        List of (image_path, embedding, topic_name, timestamp_ns)
    """
    try:
        # Build batch tensor on specified GPU
        batch_tensors = []
        batch_paths = []
        batch_topics = []
        batch_timestamps = []
        
        for image_path, topic_name, timestamp_ns in batch_data:
            try:
                image_tensor = load_image_tensor(image_path, image_resolution, device, dtype)
                batch_tensors.append(image_tensor)
                batch_paths.append(image_path)
                batch_topics.append(topic_name)
                batch_timestamps.append(timestamp_ns)
            except Exception as e:
                log(f"      ⚠️  Failed to load {image_path}: {e}")
                continue
        
        if not batch_tensors:
            return []
        
        # Concatenate and process batch within device context
        batch_tensor = torch.cat(batch_tensors, dim=0)
        with torch.cuda.device(gpu_id):
            with torch.no_grad():
                batch_embeddings = model.encode_image(batch_tensor)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
        
        # Move to CPU and store
        batch_embeddings_cpu = batch_embeddings.cpu()
        results = []
        for i, embedding in enumerate(batch_embeddings_cpu):
            results.append((batch_paths[i], embedding, batch_topics[i], batch_timestamps[i]))
        
        return results
    except Exception as e:
        log(f"      ⚠️  Error processing batch on GPU {gpu_id}: {e}")
        return []

def is_in_excluded_folder(path: Path) -> bool:
    """
    Check if a path is inside an EXCLUDED folder by checking if any parent directory is named 'EXCLUDED'.
    
    Args:
        path: Path to check
    
    Returns:
        True if path is inside an EXCLUDED folder, False otherwise
    """
    if path is None:
        return False
    
    # Check all parent directories (not the path itself)
    current = path.resolve().parent
    while current != current.parent:
        if current.name.upper() == "EXCLUDED":
            return True
        current = current.parent
    
    return False

def collect_embedding_records(embeddings: List[Tuple[Path, torch.Tensor, str, int]], already_seen: Set[str]) -> List[Tuple[str, int, int, torch.Tensor]]:
    """
    Collect embedding records from in-memory embeddings.
    Returns list of (topic, timestamp_ns, minute_of_day, embedding_tensor).
    """
    records: List[Tuple[str, int, int, torch.Tensor]] = []

    for image_path, embedding, topic_name, timestamp_ns in embeddings:
        # Create a unique identifier for this embedding
        embedding_id = f"{topic_name}/{timestamp_ns}"
        if embedding_id in already_seen:
            continue

        minute = ns_to_minute_of_day(timestamp_ns)
        records.append((topic_name, timestamp_ns, minute, embedding))

    records.sort(key=lambda r: (r[0], r[1]))
    return records

def write_shards_from_embeddings(
    out_dir: Path,
    records: List[Tuple[str, int, int, torch.Tensor]],
    meta: Dict,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    dtype = OUTPUT_DTYPE,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Write embeddings directly to shard .npy files.
    Appends to the last shard if it's not full, otherwise creates new shards.
    Returns DataFrame of manifest rows and updated meta.
    """
    shards_dir = out_dir / "shards"
    ensure_dir(shards_dir)

    # Determine D
    D = int(meta.get("D", 0))
    if D == 0 and records:
        D = int(records[0][3].shape[-1])
    if D == 0:
        raise RuntimeError("Could not determine embedding dimension D from records.")

    start_id = int(meta.get("total_count", 0))

    manifest_rows = []
    batch_vecs: List[np.ndarray] = []
    batch_count = 0
    current = 0

    # Check if there's an existing last shard we can append to
    existing_shards = sorted(shards_dir.glob("shard-*.npy"))
    last_shard_path = None
    last_shard_name = None
    last_shard_rows = 0
    last_shard_start_id = 0
    
    if existing_shards:
        last_shard_path = existing_shards[-1]
        last_shard_name = last_shard_path.name
        try:
            last_shard_data = np.load(last_shard_path)
            last_shard_rows = last_shard_data.shape[0]
            # If last shard is not full, we'll append to it
            if last_shard_rows < shard_rows:
                batch_vecs = [last_shard_data[i] for i in range(last_shard_rows)]
                batch_count = last_shard_rows
                # Calculate the start_id for the last shard (needed for manifest row_in_shard)
                last_shard_start_id = start_id - last_shard_rows
        except Exception as e:
            # If we can't load it, start fresh
            log(f"      ⚠️  Could not load last shard {last_shard_name}: {e}, starting fresh")
            last_shard_path = None
            last_shard_name = None

    for (topic, ts_ns, minute, embedding) in tqdm(records, desc="Writing shards", unit="rec", leave=False):
        vec = embedding.detach().cpu().numpy()
        if vec.shape[-1] != D:
            log(f"      ⚠️  D mismatch: got {vec.shape[-1]}, expected {D}; skipping")
            continue

        v = vec.astype(dtype, copy=False)
        batch_vecs.append(v)
        batch_count += 1

        if batch_count >= shard_rows:
            # Use existing shard name if we were appending, otherwise get new one
            if last_shard_name and last_shard_path and last_shard_path.exists():
                shard_name = last_shard_name
                shard_path = last_shard_path
                # Adjust row_in_shard to account for existing rows in the shard
                row_offset = last_shard_rows
            else:
                shard_name = next_shard_name(shards_dir)
                shard_path = shards_dir / shard_name
                row_offset = 0
            
            X = np.stack(batch_vecs, axis=0)
            np.save(shard_path, X)
            for i in range(X.shape[0]):
                manifest_rows.append({
                    "id": start_id + current,
                    "topic": records[current][0],
                    "timestamp_ns": records[current][1],
                    "minute_of_day": records[current][2],
                    "shard_id": shard_name,
                    "row_in_shard": row_offset + i,
                    "pt_path": "",  # No individual .pt file
                })
                current += 1
            batch_vecs.clear()
            batch_count = 0
            last_shard_path = None
            last_shard_name = None
            last_shard_rows = 0

    # Flush remainder
    if batch_vecs:
        # Use existing shard if we were appending, otherwise create new one
        if last_shard_name and last_shard_path and last_shard_path.exists():
            shard_name = last_shard_name
            shard_path = last_shard_path
            # Adjust row_in_shard to account for existing rows in the shard
            row_offset = last_shard_rows
        else:
            shard_name = next_shard_name(shards_dir)
            shard_path = shards_dir / shard_name
            row_offset = 0
        
        X = np.stack(batch_vecs, axis=0)
        np.save(shard_path, X)
        for i in range(X.shape[0]):
            manifest_rows.append({
                "id": start_id + current,
                "topic": records[current][0],
                "timestamp_ns": records[current][1],
                "minute_of_day": records[current][2],
                "shard_id": shard_name,
                "row_in_shard": row_offset + i,
                "pt_path": "",
            })
            current += 1
        batch_vecs.clear()

    meta["D"] = D
    meta["dtype"] = "float32"
    meta["total_count"] = start_id + current

    df_append = pd.DataFrame.from_records(manifest_rows)
    return df_append, meta

# =========================
# Main Processing Functions
# =========================

def collect_images_from_rosbag(rosbag_path: Path) -> List[Tuple[Path, str, int]]:
    """
    Collect all images from a rosbag directory.
    Returns list of (image_path, topic_name, timestamp_ns).
    """
    images = []
    for topic_dir in rosbag_path.iterdir():
        if not topic_dir.is_dir():
            continue

        topic_name = topic_dir.name.replace("__", "/")
        for mcap_dir in topic_dir.iterdir():
            if not mcap_dir.is_dir():
                continue

            for image_file in mcap_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    # Extract timestamp from filename
                    timestamp_ns = _parse_timestamp_ns_from_stem(image_file.stem)
                    if timestamp_ns is not None:
                        images.append((image_file, topic_name, timestamp_ns))

    return sorted(images, key=lambda x: (x[1], x[2]))

def collect_all_preprocessed_files() -> Dict[str, List[Tuple[Path, str, str, str, int]]]:
    """
    Collect all preprocessed .pt files from PREPROCESSED directory tree.
    
    Returns:
        Dictionary mapping preprocess_id to list of (preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns)
    """
    preprocessed_files: Dict[str, List[Tuple[Path, str, str, str, int]]] = {}
    
    if not PREPROCESSED.exists():
        log(f"  ⚠️  PREPROCESSED directory does not exist: {PREPROCESSED}")
        return preprocessed_files
    
    # Structure: PREPROCESSED / preprocess_id / rosbag_name / topic_name / mcap_identifier / timestamp_ns.pt
    for preprocess_id_dir in PREPROCESSED.iterdir():
        if not preprocess_id_dir.is_dir():
            continue
        
        preprocess_id = preprocess_id_dir.name
        
        # Skip completion.json and other non-directory files
        if preprocess_id == "completion.json":
            continue
        
        files_list = []
        
        # Iterate through rosbag directories
        for rosbag_dir in preprocess_id_dir.iterdir():
            if not rosbag_dir.is_dir():
                continue
            
            rosbag_name = rosbag_dir.name
            
            # Iterate through topic directories
            for topic_dir in rosbag_dir.iterdir():
                if not topic_dir.is_dir():
                    continue
                
                topic_name = topic_dir.name.replace("__", "/")
                
                # Iterate through mcap directories
                for mcap_dir in topic_dir.iterdir():
                    if not mcap_dir.is_dir():
                        continue
                    
                    mcap_identifier = mcap_dir.name
                    
                    # Collect all .pt files in this mcap directory
                    pt_count = 0
                    for pt_file in mcap_dir.glob("*.pt"):
                        try:
                            timestamp_ns = int(pt_file.stem)
                            files_list.append((pt_file, rosbag_name, topic_name, mcap_identifier, timestamp_ns))
                            pt_count += 1
                        except ValueError:
                            # Skip files that don't have numeric timestamps
                            continue
                    
                    if pt_count > 0:
                        log(f"      📁 {preprocess_id}/{rosbag_name}/{topic_name}/{mcap_identifier}: {pt_count} file(s)")
        
        if files_list:
            # Group by topic to show summary
            topic_counts = {}
            for _, rosbag_name, topic_name, _, _ in files_list:
                key = f"{rosbag_name}/{topic_name}"
                topic_counts[key] = topic_counts.get(key, 0) + 1
            
            log(f"    📊 {preprocess_id}: {len(files_list)} total file(s) across {len(topic_counts)} topic(s)")
            for key, count in sorted(topic_counts.items()):
                log(f"      - {key}: {count} file(s)")
            
            preprocessed_files[preprocess_id] = sorted(files_list, key=lambda x: (x[1], x[2], x[3], x[4]))
    
    return preprocessed_files

def get_models_for_preprocess_id(preprocess_id: str) -> List[Tuple[str, str, Optional[Path]]]:
    """
    Get all models (open_clip and custom) that use a given preprocess_id.
    
    Returns:
        List of (model_dir_id, model_type, checkpoint_path) tuples
        model_type: "open_clip" or "custom"
        checkpoint_path: None for open_clip, Path for custom models
    """
    models = []
    
    # Group open_clip models by preprocessing
    preprocess_dict.clear()
    for model_name, pretrained_name in OPENCLIP_MODELS:
        collect_preprocess_data(model_name, pretrained_name, CACHE_DIR)
    
    # Find open_clip models with matching preprocess_id
    for preprocess_str, model_list in preprocess_dict.items():
        if get_preprocess_id(preprocess_str) == preprocess_id:
            for model_id_str in model_list:
                if " (" in model_id_str and ")" in model_id_str:
                    model_name, pretrained_name = model_id_str.split(" (", 1)
                    pretrained_name = pretrained_name.rstrip(")")
                    model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
                    models.append((model_dir_id, "open_clip", None))
    
    # Find custom models with matching preprocess_id
    for config in CUSTOM_MODELS:
        if not config.get("enabled", True):
            continue
        
        checkpoint_path = Path(config["checkpoint"])
        if not checkpoint_path.exists():
            continue
        
        # Get preprocess transform for this custom model
        try:
            preprocess = get_custom_preprocess_transform(checkpoint_path)
            preprocess_str = str(preprocess)
            custom_preprocess_id = get_preprocess_id(preprocess_str)
            
            if custom_preprocess_id == preprocess_id:
                model_dir_name = config.get("model_dir_name", config["name"])
                models.append((model_dir_name, "custom", checkpoint_path))
        except Exception as e:
            log(f"      ⚠️  Error checking preprocess for {config['name']}: {e}")
            continue
    
    return models

def process_all_preprocessed_files_for_model(
    model_dir_id: str,
    model_type: str,
    checkpoint_path: Optional[Path],
    preprocessed_files: List[Tuple[Path, str, str, str, int]]
) -> None:
    """
    Process all preprocessed files for a single model.
    
    Args:
        model_dir_id: Model identifier (directory name)
        model_type: "open_clip" or "custom"
        checkpoint_path: Path to checkpoint (None for open_clip)
        preprocessed_files: List of (preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns)
    """
    log(f"  🔄 Processing {model_dir_id} ({len(preprocessed_files)} preprocessed file(s))")
    
    # Check if already completed (check all rosbags)
    all_rosbags = set(rosbag_name for _, rosbag_name, _, _, _ in preprocessed_files)
    all_completed = True
    for rosbag_name in all_rosbags:
        if not is_step_completed(rosbag_name, model_dir_id, "individual_embeddings"):
            all_completed = False
            break
    
    if SKIP_COMPLETED and all_completed:
        log(f"    ⏭️  {model_dir_id} - already completed for all rosbags (skipping)")
        return
    
    # Check for existing individual embedding files (with mcap structure)
    already_seen = set()
    for rosbag_name in all_rosbags:
        single_emb_dir = SINGLE_EMBEDDINGS / model_dir_id / rosbag_name
        if single_emb_dir.exists():
            for topic_dir in single_emb_dir.iterdir():
                if not topic_dir.is_dir():
                    continue
                topic_name = topic_dir.name.replace("__", "/")
                for mcap_dir in topic_dir.iterdir():
                    if not mcap_dir.is_dir():
                        continue
                    mcap_identifier = mcap_dir.name
                    for emb_file in mcap_dir.glob("*.pt"):
                        try:
                            timestamp_ns = int(emb_file.stem)
                            embedding_id = f"{rosbag_name}/{topic_name}/{mcap_identifier}/{timestamp_ns}"
                            already_seen.add(embedding_id)
                        except ValueError:
                            continue
    
    # Filter out already processed files
    files_to_process = [
        (preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns)
        for preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns in preprocessed_files
        if f"{rosbag_name}/{topic_name}/{mcap_identifier}/{timestamp_ns}" not in already_seen
    ]
    
    # Show breakdown by topic
    topic_counts = {}
    for _, rosbag_name, topic_name, _, _ in files_to_process:
        key = f"{rosbag_name}/{topic_name}"
        topic_counts[key] = topic_counts.get(key, 0) + 1
    
    if topic_counts:
        log(f"    📊 Files to process by topic:")
        for key, count in sorted(topic_counts.items()):
            log(f"      - {key}: {count} file(s)")
    
    if already_seen:
        log(f"    ⏭️  Skipping {len(already_seen)} already processed embedding(s)")
    
    if not files_to_process:
        log(f"    ⏭️  All files already processed (skipping)")
        # Mark as completed for all rosbags
        for rosbag_name in all_rosbags:
            save_completed_step(rosbag_name, model_dir_id, "individual_embeddings", status="skipped", error="All files already processed")
        return
    
    # Load model
    if model_type == "open_clip":
        # Extract model_name and pretrained_name from model_dir_id
        if "__" not in model_dir_id:
            log(f"    ⚠️  Invalid model_dir_id format: {model_dir_id}")
            return
        model_name = model_dir_id.split("__")[0].replace("_", "/")
        pretrained_name = model_dir_id.split("__", 1)[1]
        
        if USE_MULTI_GPU:
            base_model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_name, cache_dir=CACHE_DIR
            )
            models = []
            for gpu_id in range(NUM_GPUS):
                model_copy = copy.deepcopy(base_model)
                model_copy = model_copy.to(f"cuda:{gpu_id}")
                model_copy.eval()
                models.append(model_copy)
            del base_model
            model = models[0]
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_name, cache_dir=CACHE_DIR
            )
            model = model.to(DEVICE)
            model.eval()
            models = [model]
    else:  # custom
        if checkpoint_path is None:
            log(f"    ⚠️  Checkpoint path required for custom model: {model_dir_id}")
            return
        
        if USE_MULTI_GPU:
            base_model = load_agriclip(str(checkpoint_path), "cpu")
            models = []
            for gpu_id in range(NUM_GPUS):
                model_copy = copy.deepcopy(base_model)
                model_copy = model_copy.to(f"cuda:{gpu_id}")
                model_copy.eval()
                models.append(model_copy)
            del base_model
            model = models[0]
        else:
            model = load_agriclip(str(checkpoint_path), DEVICE)
            models = [model]
    
    # Process files in chunks
    total_saved = 0
    total_skipped = 0
    batch_size = BATCH_SIZE if model_type == "open_clip" else CUSTOM_MODELS[0].get("batch_size", BATCH_SIZE)
    
    num_chunks = math.ceil(len(files_to_process) / PREPROCESS_CHUNK_SIZE)
    log(f"    📊 Processing {len(files_to_process)} file(s) in {num_chunks} chunk(s)")
    
    from concurrent.futures import ThreadPoolExecutor as TPE
    
    for chunk_idx in tqdm(range(num_chunks), desc=f"Chunks ({model_dir_id})", leave=False):
        start_idx = chunk_idx * PREPROCESS_CHUNK_SIZE
        end_idx = min(start_idx + PREPROCESS_CHUNK_SIZE, len(files_to_process))
        chunk_files = files_to_process[start_idx:end_idx]
        
        # Load preprocessed tensors
        # Format: (preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns, tensor)
        chunk_data: List[Tuple[Path, str, str, str, int, torch.Tensor]] = []
        for preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns in chunk_files:
            try:
                tensor = torch.load(preprocessed_path, map_location="cpu", weights_only=True)
                chunk_data.append((preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns, tensor))
            except Exception as e:
                log(f"      ⚠️  Failed to load {preprocessed_path}: {e}")
                continue
        
        if not chunk_data:
            continue
        
        # Generate embeddings
        chunk_embeddings: List[Tuple[Path, torch.Tensor, str, str, int]] = []
        num_batches = math.ceil(len(chunk_data) / batch_size)
        
        if USE_MULTI_GPU and num_batches > 1:
            def process_batch_with_gpu(batch_info):
                batch_idx, batch_data = batch_info
                gpu_id = batch_idx % NUM_GPUS
                device = f"cuda:{gpu_id}"
                # Both open_clip and custom models use preprocessed tensors
                # Custom models' preprocessed tensors are in the same format
                return process_batch_on_gpu(batch_data, models[gpu_id], device, gpu_id)
            
            batch_list = []
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(chunk_data))
                # Format: (image_path, topic_name, timestamp_ns, preprocessed_tensor)
                batch_data = [(preprocessed_path, topic_name, timestamp_ns, tensor) 
                             for preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns, tensor in chunk_data[batch_start:batch_end]]
                batch_list.append((batch_idx, batch_data))
            
            with TPE(max_workers=NUM_GPUS) as executor:
                batch_results = list(executor.map(process_batch_with_gpu, batch_list))
            
            # Flatten results
            for batch_result in batch_results:
                chunk_embeddings.extend(batch_result)
            
            for gpu_id in range(NUM_GPUS):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            gc.collect()
        else:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(chunk_data))
                # Format: (image_path, topic_name, timestamp_ns, preprocessed_tensor)
                batch_data = [(preprocessed_path, topic_name, timestamp_ns, tensor) 
                             for preprocessed_path, rosbag_name, topic_name, mcap_identifier, timestamp_ns, tensor in chunk_data[batch_start:batch_end]]
                
                # Both open_clip and custom models use preprocessed tensors
                batch_result = process_batch_on_gpu(batch_data, model, DEVICE, 0)
                
                chunk_embeddings.extend(batch_result)
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Save embeddings
        # chunk_embeddings format: (image_path, embedding, topic_name, timestamp_ns)
        # We need to map back to rosbag_name and mcap_identifier - create mappings from chunk_data
        rosbag_map = {preprocessed_path: rosbag_name for preprocessed_path, rosbag_name, _, _, _, _ in chunk_data}
        mcap_map = {preprocessed_path: mcap_identifier for preprocessed_path, _, _, mcap_identifier, _, _ in chunk_data}
        
        # Track which topics we're saving to (use a list that can be modified from nested function)
        topics_being_saved_list = []
        
        def save_single_embedding(args):
            image_path, embedding, topic_name, timestamp_ns = args
            rosbag_name = rosbag_map.get(image_path, "unknown")
            mcap_identifier = mcap_map.get(image_path, "unknown")
            # Convert topic_name back to directory format (replace / with __)
            topic_dir_name = topic_name.replace("/", "__")
            if topic_dir_name not in topics_being_saved_list:
                topics_being_saved_list.append(topic_dir_name)
            # Structure: single_embeddings/model/rosbag_name/topic_name/mcap_identifier/timestamp.pt
            output_path = SINGLE_EMBEDDINGS / model_dir_id / rosbag_name / topic_dir_name / mcap_identifier / f"{timestamp_ns}.pt"
            if output_path.exists():
                return "skipped"
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embedding, output_path)
                return "saved"
            except Exception as e:
                log(f"      ⚠️  Failed to save {output_path}: {e}")
                return "failed"
        
        with TPE(max_workers=EMBEDDING_SAVE_WORKERS) as executor:
            save_results = list(tqdm(
                executor.map(save_single_embedding, chunk_embeddings),
                total=len(chunk_embeddings),
                desc=f"Saving ({model_dir_id})",
                unit="emb",
                leave=False
            ))
        
        saved_count = sum(1 for r in save_results if r == "saved")
        skipped_count = sum(1 for r in save_results if r == "skipped")
        total_saved += saved_count
        total_skipped += skipped_count
        
        # Log which topics were saved in this chunk
        if topics_being_saved_list:
            log(f"      📁 Saved embeddings for {len(topics_being_saved_list)} topic(s): {', '.join(sorted(topics_being_saved_list))}")
        
        # Clear memory
        del chunk_data, chunk_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # Mark as completed for all rosbags
    if total_saved > 0 or total_skipped > 0:
        log(f"    📝 Saved {total_saved}, skipped {total_skipped}")
        for rosbag_name in all_rosbags:
            save_completed_step(rosbag_name, model_dir_id, "individual_embeddings", status="success")
    
    # Cleanup
    if USE_MULTI_GPU:
        del models
        for gpu_id in range(NUM_GPUS):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
    else:
        del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

def process_rosbag(rosbag_path: Path, rosbag_name: str) -> None:
    """
    Process a single rosbag: preprocess, embed, and create shards.
    DEPRECATED: This function is kept for backward compatibility but is no longer used.
    The new approach processes all preprocessed files directly.
    """
    log(f"\n📦 {rosbag_name} (DEPRECATED - use process_all_preprocessed_files instead)")

    # 1. Collect all images
    images = collect_images_from_rosbag(rosbag_path)
    if not images:
        log(f"  ⚠️  No images found (skipping)")
        return

    log(f"  📝 Found {len(images)} image(s)")

    # 2. Group models by preprocessing (for open_clip models)
    preprocess_dict.clear()
    for model_name, pretrained_name in OPENCLIP_MODELS:
        collect_preprocess_data(model_name, pretrained_name, CACHE_DIR)

    # Process open_clip models
    for preprocess_str, model_list in preprocess_dict.items():
        preprocess_id = get_preprocess_id(preprocess_str)
        shared_model_name, shared_pretrained_name = model_list[0].split(" (")
        shared_pretrained_name = shared_pretrained_name.rstrip(")")

        log(f"  🔄 Processing open_clip models with preprocess: {preprocess_id}")

        # Get preprocessing transform
        preprocess_transform = get_preprocess_transform(shared_model_name, shared_pretrained_name, CACHE_DIR)

        # Load preprocessed images helper function - will be called in chunks to avoid OOM
        from concurrent.futures import ThreadPoolExecutor as TPE
        
        def preprocess_single_image(args):
            image_path, topic_name, timestamp_ns = args
            try:
                # Always load from preprocessed files (created in step 1)
                output_path = PREPROCESSED / preprocess_id / rosbag_name / topic_name.replace("/", "__") / f"{timestamp_ns}.pt"
                if output_path.exists():
                    try:
                        return image_path, torch.load(output_path, weights_only=True)
                    except (EOFError, RuntimeError, OSError) as e:
                        log(f"      ⚠️  Failed to load preprocessed {output_path}: {e}")
                        return None
                else:
                    log(f"      ⚠️  Preprocessed file not found: {output_path}")
                    return None
            except Exception as e:
                log(f"      ⚠️  Failed to load preprocessed {image_path}: {e}")
                return None
        
        def preprocess_chunk(chunk_images):
            """Preprocess a chunk of images and return as dict."""
            chunk_data = {}
            with TPE(max_workers=min(PREPROCESS_WORKERS, len(chunk_images) or 1)) as executor:
                results = list(tqdm(
                    executor.map(preprocess_single_image, chunk_images),
                    total=len(chunk_images),
                    desc="Preprocessing chunk",
                    unit="img",
                    leave=False
                ))
                for result in results:
                    if result is not None:
                        image_path, tensor = result
                        chunk_data[image_path] = tensor
            return chunk_data

        # Process each open_clip model in this preprocessing group
        for model_id_str in model_list:
            # Extract model_name and pretrained_name from model_id_str (format: "model_name (pretrained_name)")
            if " (" not in model_id_str:
                continue
            model_name, pretrained_name = model_id_str.split(" (", 1)
            pretrained_name = pretrained_name.rstrip(")")
            model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
            
            # Check if this model is already completed
            if SKIP_COMPLETED and is_model_completed(rosbag_name, model_dir_id):
                log(f"    ⏭️  {model_dir_id} - already completed (skipping)")
                continue
            
            log(f"    🔄 Generating embeddings with {model_dir_id}")

            # Check for existing individual embedding files
            already_seen = set()
            single_emb_dir = SINGLE_EMBEDDINGS / model_dir_id / rosbag_name
            if single_emb_dir.exists():
                for topic_dir in single_emb_dir.iterdir():
                    if not topic_dir.is_dir():
                        continue
                    topic_name = topic_dir.name.replace("__", "/")
                    for emb_file in topic_dir.glob("*.pt"):
                        try:
                            timestamp_ns = int(emb_file.stem)
                            embedding_id = f"{topic_name}/{timestamp_ns}"
                            already_seen.add(embedding_id)
                        except ValueError:
                            continue
            
            # Filter out already processed images
            images_to_process = [
                (image_path, topic_name, timestamp_ns)
                for image_path, topic_name, timestamp_ns in images
                if f"{topic_name}/{timestamp_ns}" not in already_seen
            ]
            
            if already_seen:
                log(f"      ⏭️  Skipping {len(already_seen)} already processed embedding(s)")
            
            if not images_to_process:
                log(f"      ⏭️  All images already processed (skipping)")
                save_completed_step(rosbag_name, model_dir_id, "individual_embeddings", status="skipped", error="All images already processed")
                continue

            # Load model(s) for multi-GPU
            if USE_MULTI_GPU:
                # Load model once, then create independent copies on each GPU
                base_model, _, _ = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained_name,
                    cache_dir=CACHE_DIR
                )
                models = []
                for gpu_id in range(NUM_GPUS):
                    # Create a deep copy to ensure complete independence
                    model_copy = copy.deepcopy(base_model)
                    model_copy = model_copy.to(f"cuda:{gpu_id}")
                    model_copy.eval()
                    models.append(model_copy)
                # Delete base model to free memory
                del base_model
                # Keep reference to first model for compatibility
                model = models[0]
            else:
                model, _, _ = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained_name,
                    cache_dir=CACHE_DIR
                )
                model = model.to(DEVICE)
                model.eval()
                models = [model]

            # Check if individual embeddings step is already completed
            individual_embeddings_done = is_step_completed(rosbag_name, model_dir_id, "individual_embeddings")
            
            # If already done, skip this model
            if individual_embeddings_done:
                log(f"      ⏭️  Individual embeddings already completed for {model_dir_id}")
                continue
                log(f"      🔄 Individual embeddings exist but shards don't - loading embeddings to create shards")
                single_emb_dir = SINGLE_EMBEDDINGS / model_dir_id / rosbag_name
                if single_emb_dir.exists():
                    # Build set of embeddings already in shards (from manifest)
                    shard_embedding_ids = set()
                    if not manifest.empty:
                        for _, row in manifest.iterrows():
                            embedding_id = f"{row['topic']}/{row['timestamp_ns']}"
                            shard_embedding_ids.add(embedding_id)
                    
                    # Load all individual embeddings and create shards from them
                    embedding_records = []
                    for topic_dir in single_emb_dir.iterdir():
                        if not topic_dir.is_dir():
                            continue
                        topic_name = topic_dir.name.replace("__", "/")
                        for emb_file in topic_dir.glob("*.pt"):
                            try:
                                timestamp_ns = int(emb_file.stem)
                                embedding_id = f"{topic_name}/{timestamp_ns}"
                                # Only add if not already in shards
                                if embedding_id not in shard_embedding_ids:
                                    embedding = torch.load(emb_file, map_location="cpu")
                                    embedding_records.append((topic_name, timestamp_ns, ns_to_minute_of_day(timestamp_ns), embedding))
                            except (ValueError, Exception) as e:
                                log(f"      ⚠️  Failed to load {emb_file}: {e}")
                                continue
                    
                    if embedding_records:
                        log(f"      📊 Loading {len(embedding_records)} embedding(s) from individual files to create shards")
                        df_append, meta = write_shards_from_embeddings(out_dir, embedding_records, meta, DEFAULT_SHARD_ROWS, OUTPUT_DTYPE)
                        if not df_append.empty:
                            manifest = pd.concat([manifest, df_append], ignore_index=True)
                            manifest.sort_values("id", inplace=True, kind="mergesort")
                            write_manifest(manifest_path, manifest)
                            save_meta(meta_path, meta)
                            new_rows = meta.get("total_count", 0) - initial_count
                            log(f"      ✅ {model_dir_id}: +{new_rows} row(s), total={meta['total_count']}")
                            save_completed_step(rosbag_name, model_dir_id, "sharded_embeddings", status="success")
                        else:
                            log(f"      ⏭️  No new embeddings to add to shards")
                    else:
                        log(f"      ⏭️  No new embeddings found (all already in shards)")
                        # Check if shards already exist
                        if (out_dir / "shards").exists() and any((out_dir / "shards").glob("shard-*.npy")):
                            save_completed_step(rosbag_name, model_dir_id, "sharded_embeddings", status="success")
                    continue  # Skip to next model since we've handled shard creation
            
            # Generate embeddings - process in chunks to avoid OOM
            total_saved = 0
            total_skipped = 0
            
            # Process images in chunks to avoid loading all into memory
            num_chunks = math.ceil(len(images_to_process) / PREPROCESS_CHUNK_SIZE)
            log(f"      📊 Processing {len(images_to_process)} image(s) in {num_chunks} chunk(s) (chunk size: {PREPROCESS_CHUNK_SIZE})")
            
            for chunk_idx in tqdm(range(num_chunks), desc=f"Chunks ({model_dir_id})", leave=False):
                start_idx = chunk_idx * PREPROCESS_CHUNK_SIZE
                end_idx = min(start_idx + PREPROCESS_CHUNK_SIZE, len(images_to_process))
                chunk_images = images_to_process[start_idx:end_idx]
                
                # Load preprocessed files (created in step 1)
                chunk_preprocessed = {}
                for image_path, topic_name, timestamp_ns in chunk_images:
                    preprocessed_path = PREPROCESSED / preprocess_id / rosbag_name / topic_name.replace("/", "__") / f"{timestamp_ns}.pt"
                    if preprocessed_path.exists():
                        try:
                            chunk_preprocessed[image_path] = torch.load(preprocessed_path, map_location="cpu")
                        except Exception as e:
                            log(f"      ⚠️  Failed to load preprocessed {preprocessed_path}: {e}")
                    else:
                        log(f"      ⚠️  Preprocessed file not found: {preprocessed_path}")
                
                if not chunk_preprocessed:
                    continue
                
                # Build valid data from this chunk
                chunk_valid_data = [
                    (image_path, topic_name, timestamp_ns, chunk_preprocessed[image_path])
                    for image_path, topic_name, timestamp_ns in chunk_images
                    if image_path in chunk_preprocessed
                ]
                
                if not chunk_valid_data:
                    # Clear chunk and continue
                    del chunk_preprocessed
                    gc.collect()
                    continue
                
                # Process embedding batches for this chunk
                chunk_embeddings: List[Tuple[Path, torch.Tensor, str, int]] = []
                num_batches = math.ceil(len(chunk_valid_data) / BATCH_SIZE)
                
                if USE_MULTI_GPU and num_batches > 1:
                    # Parallel batch processing across GPUs
                    def process_batch_with_gpu(batch_info):
                        batch_idx, batch_data = batch_info
                        gpu_id = batch_idx % NUM_GPUS
                        device = f"cuda:{gpu_id}"
                        return process_batch_on_gpu(batch_data, models[gpu_id], device, gpu_id)
                    
                    # Prepare batch data
                    batch_list = []
                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * BATCH_SIZE
                        batch_end = min(batch_start + BATCH_SIZE, len(chunk_valid_data))
                        batch_data = chunk_valid_data[batch_start:batch_end]
                        batch_list.append((batch_idx, batch_data))
                    
                    # Process batches in parallel
                    with TPE(max_workers=NUM_GPUS) as executor:
                        batch_results = list(executor.map(process_batch_with_gpu, batch_list))
                    
                    # Flatten and collect results (maintain order)
                    for batch_result in batch_results:
                        chunk_embeddings.extend(batch_result)
                    
                    # Clear GPU cache for all GPUs
                    for gpu_id in range(NUM_GPUS):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # Sequential processing (single GPU or fallback)
                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * BATCH_SIZE
                        batch_end = min(batch_start + BATCH_SIZE, len(chunk_valid_data))
                        batch_data = chunk_valid_data[batch_start:batch_end]
                        
                        batch_result = process_batch_on_gpu(batch_data, model, DEVICE, 0)
                        chunk_embeddings.extend(batch_result)
                        
                        # Clear GPU cache periodically
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                
                # Save individual embeddings for this chunk (parallelized)
                    def save_single_embedding(args):
                        image_path, embedding, topic_name, timestamp_ns = args
                        output_path = SINGLE_EMBEDDINGS / model_dir_id / rosbag_name / topic_name.replace("/", "__") / f"{timestamp_ns}.pt"
                        if output_path.exists():
                            return "skipped"
                        try:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(embedding, output_path)
                            return "saved"
                        except Exception as e:
                            log(f"      ⚠️  Failed to save {output_path}: {e}")
                            return "failed"
                    
                    with TPE(max_workers=EMBEDDING_SAVE_WORKERS) as executor:
                        save_results = list(tqdm(
                            executor.map(save_single_embedding, chunk_embeddings),
                            total=len(chunk_embeddings),
                            desc=f"Saving embeddings ({model_dir_id})",
                            unit="emb",
                            leave=False
                        ))
                        saved_count = sum(1 for r in save_results if r == "saved")
                        skipped_count = sum(1 for r in save_results if r == "skipped")
                        total_saved += saved_count
                        total_skipped += skipped_count
                
                # Clear chunk from memory
                del chunk_preprocessed, chunk_valid_data, chunk_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # Mark individual embeddings step as completed
            if total_saved > 0 or total_skipped > 0:
                log(f"      📝 Individual embeddings: saved {total_saved}, skipped {total_skipped}")
            save_completed_step(rosbag_name, model_dir_id, "individual_embeddings", status="success")

            # Cleanup model(s) (embeddings are already cleaned up per chunk)
            if USE_MULTI_GPU:
                del models
                # Clear cache for all GPUs
                for gpu_id in range(NUM_GPUS):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            else:
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # Memory is already cleared after each chunk, no need to clear preprocessed_data

    # Process custom models
    for config in CUSTOM_MODELS:
        if not config.get("enabled", True):
            continue

        model_name = config["name"]
        checkpoint_path = Path(config["checkpoint"])
        model_dir_name = config.get("model_dir_name", model_name)
        
        if not checkpoint_path.exists():
            log(f"  ⚠️  {model_name}: Checkpoint not found: {checkpoint_path}")
            save_completed_step(rosbag_name, model_dir_name, "individual_embeddings", status="failed", error=f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Check if individual embeddings step is already completed
        individual_embeddings_done = is_step_completed(rosbag_name, model_dir_name, "individual_embeddings")
        
        # If already done, skip this model
        if individual_embeddings_done:
            log(f"  ⏭️  {model_dir_name} - individual embeddings already completed (skipping)")
            continue
        
        log(f"  🔄 Processing custom model: {model_name}")
        
        # Check for existing individual embedding files
        already_seen = set()
        single_emb_dir = SINGLE_EMBEDDINGS / model_dir_name / rosbag_name
        if single_emb_dir.exists():
                for topic_dir in single_emb_dir.iterdir():
                    if not topic_dir.is_dir():
                        continue
                    topic_name = topic_dir.name.replace("__", "/")
                    for emb_file in topic_dir.glob("*.pt"):
                        try:
                            timestamp_ns = int(emb_file.stem)
                            embedding_id = f"{topic_name}/{timestamp_ns}"
                            already_seen.add(embedding_id)
                        except ValueError:
                            continue
        
        # Filter out already processed images
        images_to_process = [
            (image_path, topic_name, timestamp_ns)
            for image_path, topic_name, timestamp_ns in images
            if f"{topic_name}/{timestamp_ns}" not in already_seen
        ]
        
        if already_seen:
            log(f"    ⏭️  Skipping {len(already_seen)} already processed embedding(s)")
        
        if not images_to_process:
            log(f"    ⏭️  All images already processed (skipping)")
            save_completed_step(rosbag_name, model_dir_name, "individual_embeddings", status="skipped", error="All images already processed")
            continue

        # Load model(s) for multi-GPU
        if USE_MULTI_GPU:
            # Load model once, then create independent copies on each GPU
            base_model = load_agriclip(str(checkpoint_path), "cpu")
            models = []
            for gpu_id in range(NUM_GPUS):
                # Create a deep copy to ensure complete independence
                model_copy = copy.deepcopy(base_model)
                model_copy = model_copy.to(f"cuda:{gpu_id}")
                model_copy.eval()
                models.append(model_copy)
            # Delete base model to free memory
            del base_model
            # Keep reference to first model for compatibility
            model = models[0]
        else:
            model = load_agriclip(str(checkpoint_path), DEVICE)
            models = [model]
        custom_batch_size = config.get("batch_size", BATCH_SIZE)

        # Generate embeddings - process in chunks to avoid OOM
        total_saved = 0
        total_skipped = 0
        
        # Process images in chunks to avoid loading all into memory
        num_chunks = math.ceil(len(images_to_process) / PREPROCESS_CHUNK_SIZE)
        log(f"    📊 Processing {len(images_to_process)} image(s) in {num_chunks} chunk(s) (chunk size: {PREPROCESS_CHUNK_SIZE})")
        
        for chunk_idx in tqdm(range(num_chunks), desc=f"Chunks ({model_name})", leave=False):
            start_idx = chunk_idx * PREPROCESS_CHUNK_SIZE
            end_idx = min(start_idx + PREPROCESS_CHUNK_SIZE, len(images_to_process))
            chunk_images = images_to_process[start_idx:end_idx]
            
            # Generate embeddings for this chunk
            chunk_embeddings: List[Tuple[Path, torch.Tensor, str, int]] = []
            num_batches = math.ceil(len(chunk_images) / custom_batch_size)
            
            if USE_MULTI_GPU and num_batches > 1:
                # Parallel batch processing across GPUs
                def process_batch_with_gpu_custom(batch_info):
                    batch_idx, batch_data = batch_info
                    gpu_id = batch_idx % NUM_GPUS
                    device = f"cuda:{gpu_id}"
                    return process_batch_on_gpu_custom(
                        batch_data, 
                        models[gpu_id], 
                        device, 
                        gpu_id,
                        model.visual.input_resolution,
                        model.dtype
                    )
                
                # Prepare batch data
                batch_list = []
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * custom_batch_size
                    batch_end = min(batch_start + custom_batch_size, len(chunk_images))
                    batch_data = chunk_images[batch_start:batch_end]
                    batch_list.append((batch_idx, batch_data))
                
                # Process batches in parallel
                with TPE(max_workers=NUM_GPUS) as executor:
                    batch_results = list(executor.map(process_batch_with_gpu_custom, batch_list))
                
                # Flatten and collect results (maintain order)
                for batch_result in batch_results:
                    chunk_embeddings.extend(batch_result)
                
                # Clear GPU cache for all GPUs
                for gpu_id in range(NUM_GPUS):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                gc.collect()
            else:
                # Sequential processing (single GPU or fallback)
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * custom_batch_size
                    batch_end = min(batch_start + custom_batch_size, len(chunk_images))
                    batch_data = chunk_images[batch_start:batch_end]
                    
                    batch_result = process_batch_on_gpu_custom(
                        batch_data,
                        model,
                        DEVICE,
                        0,
                        model.visual.input_resolution,
                        model.dtype
                    )
                    chunk_embeddings.extend(batch_result)
                    
                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Save individual embeddings for this chunk (parallelized)
                def save_single_embedding(args):
                    image_path, embedding, topic_name, timestamp_ns = args
                    output_path = SINGLE_EMBEDDINGS / model_dir_name / rosbag_name / topic_name.replace("/", "__") / f"{timestamp_ns}.pt"
                    if output_path.exists():
                        return "skipped"
                    try:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(embedding, output_path)
                        return "saved"
                    except Exception as e:
                        log(f"[WARN] Failed to save {output_path}: {e}")
                        return "failed"
                
                with TPE(max_workers=EMBEDDING_SAVE_WORKERS) as executor:
                    save_results = list(tqdm(
                        executor.map(save_single_embedding, chunk_embeddings),
                        total=len(chunk_embeddings),
                        desc=f"Saving embeddings ({model_name})",
                        unit="emb",
                        leave=False
                    ))
                    saved_count = sum(1 for r in save_results if r == "saved")
                    skipped_count = sum(1 for r in save_results if r == "skipped")
                    total_saved += saved_count
                    total_skipped += skipped_count
            
            # Clear chunk from memory
            del chunk_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Mark individual embeddings step as completed
        if total_saved > 0 or total_skipped > 0:
            log(f"      📝 Individual embeddings: saved {total_saved}, skipped {total_skipped}")
        save_completed_step(rosbag_name, model_dir_name, "individual_embeddings", status="success")

        # Cleanup model(s) (embeddings are already cleaned up per chunk)
        if USE_MULTI_GPU:
            del models
            # Clear cache for all GPUs
            for gpu_id in range(NUM_GPUS):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        else:
            del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    log(f"  ✅ Completed processing {rosbag_name}")

# =========================
# Main Entry Point
# =========================

def main():
    """Main function to process all rosbags."""
    global _current_rosbag_name, _interrupted
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log("🚀 Starting single embeddings generation pipeline")
    
    # Log GPU information
    if torch.cuda.is_available():
        log(f"🔧 CUDA available: {torch.cuda.device_count()} GPU(s) detected")
        if USE_MULTI_GPU:
            log(f"  🔄 Using multi-GPU mode with {NUM_GPUS} GPU(s)")
            for i in range(NUM_GPUS):
                log(f"    📝 GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            log(f"  📝 Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("  ⚠️  CUDA not available, using CPU")

    # Ensure output directories exist
    ensure_dir(SINGLE_EMBEDDINGS)

    # Collect all preprocessed files grouped by preprocess_id
    log("🔍 Scanning PREPROCESSED directory for all preprocessed files...")
    all_preprocessed_files = collect_all_preprocessed_files()
    
    if not all_preprocessed_files:
        log("  ⚠️  No preprocessed files found in PREPROCESSED directory")
        return
    
    total_files = sum(len(files) for files in all_preprocessed_files.values())
    log(f"📋 Found {total_files:,} preprocessed file(s) across {len(all_preprocessed_files)} preprocessing group(s)")
    log(f"🔧 Processing with {PREPROCESS_WORKERS} preprocessing worker(s) and {EMBEDDING_SAVE_WORKERS} embedding save worker(s)")
    
    # Show completion status
    if SKIP_COMPLETED:
        completed = load_completion()
        if completed:
            log(f"⏭️  SKIP_COMPLETED: True (will skip previously completed models)")
        else:
            log(f"⏭️  SKIP_COMPLETED: True (no previously completed models)")
    else:
        log(f"🔄 SKIP_COMPLETED: False (will reprocess all models)")

    # Process by preprocess_id
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    try:
        for preprocess_id, preprocessed_files in tqdm(all_preprocessed_files.items(), desc="Processing preprocessing groups", unit="group"):
            if _interrupted:
                log("  ⚠️  Processing interrupted, stopping")
                break
            
            log(f"\n🔄 Processing preprocessing group: {preprocess_id} ({len(preprocessed_files)} file(s))")
            
            # Get all models that use this preprocess_id
            models = get_models_for_preprocess_id(preprocess_id)
            
            # If no models found, fall back to all custom models
            if not models:
                log(f"  ⚠️  No matching models found for preprocess_id: {preprocess_id}")
                log(f"  🔄 Falling back to all custom models")
                models = []
                for config in CUSTOM_MODELS:
                    if not config.get("enabled", True):
                        continue
                    checkpoint_path = Path(config["checkpoint"])
                    if not checkpoint_path.exists():
                        continue
                    model_dir_name = config.get("model_dir_name", config["name"])
                    models.append((model_dir_name, "custom", checkpoint_path))
                
                if not models:
                    log(f"  ⚠️  No custom models available (skipping)")
                    continue
            
            log(f"  📝 Found {len(models)} model(s) for this preprocessing group")
            
            # Process each model
            for model_dir_id, model_type, checkpoint_path in models:
                try:
                    process_all_preprocessed_files_for_model(
                        model_dir_id,
                        model_type,
                        checkpoint_path,
                        preprocessed_files
                    )
                    processed_count += 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error_msg = str(e)
                    log(f"  ❌ Failed to process {model_dir_id}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
                    continue
    
    except KeyboardInterrupt:
        log("\n⚠️  Keyboard interrupt received")
        if _current_rosbag_name:
            log(f"  ⚠️  Marking current rosbag {_current_rosbag_name} as failed due to interruption")
            save_completed_rosbag(_current_rosbag_name, errors=["Keyboard interrupt"])
        raise
    except Exception as e:
        log(f"❌ Fatal error in main loop: {e}")
        if _current_rosbag_name:
            log(f"  ⚠️  Marking current rosbag {_current_rosbag_name} as failed due to fatal error")
            save_completed_rosbag(_current_rosbag_name, errors=[f"Fatal error: {str(e)}"])
        raise
    finally:
        _current_rosbag_name = None

    log(f"\n✅ Single embeddings generation pipeline complete")
    log(f"📊 Summary: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")

if __name__ == "__main__":
    main()

