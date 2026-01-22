"""
Step 5: Embeddings

Generates embeddings from image messages.
MCAP processor that collects images, preprocesses them, and generates embeddings.
"""
import io
import json
import math
import os
import re
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import open_clip
import gc

try:
    from open_clip.transform import image_transform
except ImportError:
    image_transform = None

from ..abstract import McapProcessor
from ..core import McapProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger

# Silence noisy dependency warnings
warnings.filterwarnings("ignore", message=r".*Failed to load image Python extension.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Unable to find acceptable character detection dependency.*")

# =========================
# Model Configurations
# =========================

# OpenCLIP models
OPENCLIP_MODELS = [
    #('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    #('ViT-L-14-quickgelu', 'openai'),
    #('ViT-B-32', 'laion2b_s34b_b79k'),
    #('ViT-H-14', 'laion2b_s32b_b79k'),
    #('ViT-bigG-14', 'laion2b_s39b_b160k')
]

# Custom CLIP models
CUSTOM_MODELS = [
    {
        "name": "ViT-B-16-finetuned(09.10.25)",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/ViT-B-16-finetuned(09.10.25).pt",
        "model_dir_name": "ViT-B-16-finetuned(09.10.25)",
        "batch_size": 64,
        "enabled": False,
    },
    {
        "name": "agriclip",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/agriclip.pt",
        "model_dir_name": "agriclip",
        "batch_size": 64,
        "enabled": False,
    },
]

# =========================
# Constants
# =========================

DEFAULT_SHARD_ROWS = 100_000
OUTPUT_DTYPE = np.float32
CACHE_DIR = "/mnt/data/openclip_cache"
BATCH_SIZE = 256  # Default batch size for open_clip models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utility Functions
# =========================

def ns_to_minute_of_day(ts_ns: int) -> int:
    """Convert nanosecond timestamp to minute of day."""
    dt = datetime.utcfromtimestamp(ts_ns / 1e9)
    return dt.hour * 60 + dt.minute

def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)

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
            if candidate is None:
                state_dict = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            else:
                state_dict = candidate
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
        
        if "visual.conv1.weight" not in state_dict:
            raise KeyError("Checkpoint does not contain CLIP visual weights")
        
        vision_patch_size = state_dict["visual.conv1.weight"].shape[2]
        num_pos_tokens = state_dict["visual.positional_embedding"].shape[0]
        grid_size = int((num_pos_tokens - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        return image_resolution
    except Exception as e:
        return 224  # Default

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

# =========================
# Custom CLIP Model Classes (from 04_create_single_embeddings.py)
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

def _first_matching_key(container: dict, keys) -> dict | None:
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

# =========================
# Shard and Manifest Functions (from 05_create_embedding_shards.py)
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
    return pd.DataFrame(columns=["id", "topic", "timestamp_ns", "minute_of_day", "mcap_identifier", "shard_id", "row_in_shard", "pt_path"])

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

def write_shards_from_embeddings(
    out_dir: Path,
    records: List[Tuple[str, int, int, str, torch.Tensor]],
    meta: Dict,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    dtype = OUTPUT_DTYPE,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Write embeddings directly to shard .npy files.
    Appends to the last shard if it's not full, otherwise creates new shards.
    
    Args:
        records: List of (topic, timestamp_ns, minute_of_day, mcap_identifier, embedding_tensor)
    
    Returns:
        DataFrame of manifest rows and updated meta.
    """
    shards_dir = out_dir / "shards"
    ensure_dir(shards_dir)

    # Determine D
    D = int(meta.get("D", 0))
    if D == 0 and records:
        D = int(records[0][4].shape[-1])  # embedding is at index 4
    if D == 0:
        raise RuntimeError("Could not determine embedding dimension D from records.")

    start_id = int(meta.get("total_count", 0))

    manifest_rows = []
    batch_vecs: List[np.ndarray] = []
    batch_records: List[Tuple[str, int, int, str]] = []  # Track metadata for new records only
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
            if last_shard_rows < shard_rows:
                # Prepend existing shard data (these already have manifest entries, don't create new ones)
                batch_vecs = [last_shard_data[i] for i in range(last_shard_rows)]
                batch_count = last_shard_rows
                last_shard_start_id = start_id - last_shard_rows
        except Exception as e:
            last_shard_path = None
            last_shard_name = None

    for (topic, ts_ns, minute, mcap_identifier, embedding) in records:
        vec = embedding.detach().cpu().numpy()
        if vec.shape[-1] != D:
            continue

        v = vec.astype(dtype, copy=False)
        batch_vecs.append(v)
        batch_records.append((topic, ts_ns, minute, mcap_identifier))  # Track only new records
        batch_count += 1

        if batch_count >= shard_rows:
            if last_shard_name and last_shard_path and last_shard_path.exists():
                shard_name = last_shard_name
                shard_path = last_shard_path
                row_offset = last_shard_rows
            else:
                shard_name = next_shard_name(shards_dir)
                shard_path = shards_dir / shard_name
                row_offset = 0
            
            X = np.stack(batch_vecs, axis=0)
            np.save(shard_path, X)
            
            # Only create manifest rows for NEW records (skip prepended ones)
            # The first last_shard_rows vectors are from existing shard (already have manifest entries)
            new_records_start_idx = last_shard_rows
            for i in range(new_records_start_idx, X.shape[0]):
                record = batch_records[i - new_records_start_idx]
                manifest_rows.append({
                    "id": start_id + current,
                    "topic": record[0],
                    "timestamp_ns": record[1],
                    "minute_of_day": record[2],
                    "mcap_identifier": record[3],
                    "shard_id": shard_name,
                    "row_in_shard": i,  # i is the index in X, which matches the position in the shard file
                    "pt_path": "",
                })
                current += 1
            batch_vecs.clear()
            batch_records.clear()
            batch_count = 0
            last_shard_path = None
            last_shard_name = None
            last_shard_rows = 0

    # Flush remainder
    if batch_vecs:
        if last_shard_name and last_shard_path and last_shard_path.exists():
            shard_name = last_shard_name
            shard_path = last_shard_path
            row_offset = last_shard_rows
        else:
            shard_name = next_shard_name(shards_dir)
            shard_path = shards_dir / shard_name
            row_offset = 0
        
        X = np.stack(batch_vecs, axis=0)
        np.save(shard_path, X)
        
        # Only create manifest rows for NEW records (skip prepended ones)
        new_records_start_idx = last_shard_rows
        for i in range(new_records_start_idx, X.shape[0]):
            record = batch_records[i - new_records_start_idx]
            manifest_rows.append({
                "id": start_id + current,
                "topic": record[0],
                "timestamp_ns": record[1],
                "minute_of_day": record[2],
                "mcap_identifier": record[3],
                "shard_id": shard_name,
                "row_in_shard": i,  # i is the index in X, which matches the position in the shard file
                "pt_path": "",
            })
            current += 1
        batch_vecs.clear()
        batch_records.clear()

    meta["D"] = D
    meta["dtype"] = "float32"
    meta["total_count"] = start_id + current

    df_append = pd.DataFrame.from_records(manifest_rows)
    return df_append, meta

# =========================
# Main Processor Class
# =========================

class EmbeddingsProcessor(McapProcessor):
    """
    Generate embeddings for image messages in an MCAP.
    
    Collects images during MCAP iteration, preprocesses them by type,
    generates embeddings for all models, and saves to shards.
    
    Operates at MCAP level - processes each MCAP independently.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize embeddings processor.
        
        Args:
            output_dir: Directory to write embedding shards
        """
        super().__init__("embeddings_processor")
        self.output_dir = Path(output_dir)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir)
        
        # Collected images per MCAP: List of {"topic": str, "timestamp_ns": int, "mcap_id": str, "image": PIL.Image}
        self.collected_images: List[Dict[str, Any]] = []
        
        # Model cache: {model_id: model}
        self.models_cache: Dict[str, torch.nn.Module] = {}
        
        # Current MCAP ID (set by main.py before message iteration)
        self.current_mcap_id: Optional[str] = None
        
        # Group models by preprocessing type
        self._group_models_by_preprocess()
    
    def _group_models_by_preprocess(self) -> None:
        """Group models by their preprocessing type."""
        # Clear preprocess dict
        preprocess_dict.clear()
        
        # Collect preprocessing for open_clip models
        for model_name, pretrained_name in OPENCLIP_MODELS:
            collect_preprocess_data(model_name, pretrained_name, CACHE_DIR)
        
        # Build mapping: preprocess_id -> [(model_id, model_type, checkpoint_path, batch_size), ...]
        self.models_by_preprocess: Dict[str, List[Tuple[str, str, Optional[Path], int]]] = {}
        
        # Group open_clip models
        for preprocess_str, model_list in preprocess_dict.items():
            preprocess_id = get_preprocess_id(preprocess_str)
            if preprocess_id not in self.models_by_preprocess:
                self.models_by_preprocess[preprocess_id] = []
            
            for model_id_str in model_list:
                if " (" in model_id_str and ")" in model_id_str:
                    model_name, pretrained_name = model_id_str.split(" (", 1)
                    pretrained_name = pretrained_name.rstrip(")")
                    model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
                    self.models_by_preprocess[preprocess_id].append((model_dir_id, "open_clip", None, BATCH_SIZE))
        
        # Group custom models
        for config in CUSTOM_MODELS:
            if not config.get("enabled", True):
                continue
            
            checkpoint_path = Path(config["checkpoint"])
            if not checkpoint_path.exists():
                continue
            
            try:
                preprocess = get_custom_preprocess_transform(checkpoint_path)
                preprocess_str = str(preprocess)
                preprocess_id = get_preprocess_id(preprocess_str)
                
                if preprocess_id not in self.models_by_preprocess:
                    self.models_by_preprocess[preprocess_id] = []
                
                model_dir_name = config.get("model_dir_name", config["name"])
                batch_size = config.get("batch_size", BATCH_SIZE)
                self.models_by_preprocess[preprocess_id].append((model_dir_name, "custom", checkpoint_path, batch_size))
            except Exception as e:
                self.logger.warning(f"Error checking preprocess for {config['name']}: {e}")
                continue
    
    def reset(self) -> None:
        """Reset collector state before each MCAP iteration."""
        self.collected_images = []
        self.current_mcap_id = None
    
    def wants_message(self, topic: str, msg_type: str) -> bool:
        """Filter which messages to collect - only image topics."""
        return self._is_image_topic(topic)
    
    def _is_image_topic(self, topic: str) -> bool:
        """Check if a topic is an image topic."""
        return "image" in topic.lower() or "camera" in topic.lower()
    
    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any) -> None:
        """
        Collect a single image message and convert to PIL Image.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
            ros2_msg: Decoded ROS2 message
        """
        topic = channel.topic
        if not self._is_image_topic(topic):
            return
        
        try:
            pil_image = self._ros2_image_to_pil(ros2_msg)
            if pil_image is not None:
                mcap_id = self.current_mcap_id if self.current_mcap_id else "unknown"
                self.collected_images.append({
                    "topic": topic,
                    "timestamp_ns": message.log_time,
                    "mcap_id": mcap_id,
                    "image": pil_image
                })
        except Exception as e:
            self.logger.warning(f"Failed to convert image from {topic}: {e}")
    
    def _ros2_image_to_pil(self, ros2_msg: Any) -> Optional[Image.Image]:
        """
        Convert ROS2 image message to PIL Image.
        
        Supports:
        - sensor_msgs/msg/CompressedImage
        - sensor_msgs/msg/Image
        
        Args:
            ros2_msg: Decoded ROS2 message
        
        Returns:
            PIL Image or None if conversion fails
        """
        msg_type = type(ros2_msg).__name__
        
        try:
            # Handle CompressedImage
            if "CompressedImage" in msg_type:
                img_data = bytes(ros2_msg.data)
                pil_image = Image.open(io.BytesIO(img_data))
                return pil_image.convert('RGB')
            
            # Handle raw Image
            elif "Image" in msg_type and hasattr(ros2_msg, 'encoding'):
                if not all(hasattr(ros2_msg, attr) for attr in ['encoding', 'width', 'height', 'data']):
                    return None
                
                encoding = ros2_msg.encoding
                width = ros2_msg.width
                height = ros2_msg.height
                data = bytes(ros2_msg.data)
                
                channels_map = {
                    "mono8": 1,
                    "rgb8": 3,
                    "bgr8": 3,
                    "rgba8": 4,
                    "bgra8": 4
                }
                
                channels = channels_map.get(encoding)
                if channels is None:
                    return None
                
                img_data = np.frombuffer(data, dtype=np.uint8)
                img_data = img_data.reshape((height, width, channels))
                
                if encoding == "bgr8":
                    img_data = img_data[:, :, ::-1]
                elif encoding == "bgra8":
                    img_data = img_data[:, :, [2, 1, 0, 3]]
                
                if channels == 1:
                    pil_image = Image.fromarray(img_data, mode='L')
                elif channels == 3:
                    pil_image = Image.fromarray(img_data, mode='RGB')
                elif channels == 4:
                    pil_image = Image.fromarray(img_data, mode='RGBA').convert('RGB')
                else:
                    return None
                
                return pil_image
            
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to convert ROS2 image to PIL: {e}")
            return None
    
    def _process_batch_on_gpu(
        self,
        batch_tensors: List[torch.Tensor],
        model: torch.nn.Module,
        device: str
    ) -> torch.Tensor:
        """
        Process a batch of preprocessed tensors on GPU.
        
        Args:
            batch_tensors: List of preprocessed image tensors
            model: The model to use for encoding
            device: Device string (e.g., "cuda")
        
        Returns:
            Batch of normalized embeddings
        """
        if not batch_tensors:
            return torch.empty((0, 0))
        
        # Stack tensors and move to device
        batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
        
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
        
        return batch_embeddings.cpu()
    
    def process_mcap(self, context: McapProcessingContext) -> Dict:
        """
        Process collected images: preprocess, embed, and save to shards.
        
        Args:
            context: McapProcessingContext
        
        Returns:
            Processing info dictionary
        """
        if not self.collected_images:
            return {}
        
        rosbag_name = context.get_relative_path().as_posix()
        mcap_id = context.get_mcap_id()
        
        # Check if this MCAP is already processed (reuse the same logic)
        if self.is_mcap_completed(context):
            self.logger.processor_skip(f"embeddings for MCAP {mcap_id}", "already completed")
            # Clear collected images since we're skipping
            self.collected_images = []
            return {}
        
        self.logger.info(f"Processing {len(self.collected_images)} image(s) from MCAP {mcap_id}")
        
        # Process by preprocessing type
        for preprocess_id, models in self.models_by_preprocess.items():
            if not models:
                continue
            
            # Get preprocessing transform (use first model's transform as representative)
            first_model_id, first_model_type, first_checkpoint, _ = models[0]
            
            if first_model_type == "open_clip":
                # Extract model_name and pretrained_name from model_id
                if "__" not in first_model_id:
                    continue
                model_name = first_model_id.split("__")[0].replace("_", "/")
                pretrained_name = first_model_id.split("__", 1)[1]
                preprocess_transform = get_preprocess_transform(model_name, pretrained_name, CACHE_DIR)
            else:
                # Custom model
                if first_checkpoint is None:
                    continue
                preprocess_transform = get_custom_preprocess_transform(first_checkpoint)
            
            # Preprocess all PIL images
            preprocessed_tensors = []
            preprocessed_metadata = []
            
            for img_data in self.collected_images:
                try:
                    pil_image = img_data["image"]
                    tensor = preprocess_transform(pil_image)
                    preprocessed_tensors.append(tensor)
                    preprocessed_metadata.append({
                        "topic": img_data["topic"],
                        "timestamp_ns": img_data["timestamp_ns"],
                        "mcap_id": img_data["mcap_id"]
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to preprocess image: {e}")
                    continue
            
            if not preprocessed_tensors:
                continue
            
            # Process each model of this preprocessing type
            for model_dir_id, model_type, checkpoint_path, batch_size in models:
                try:
                    # Load or get cached model
                    if model_dir_id not in self.models_cache:
                        if model_type == "open_clip":
                            if "__" not in model_dir_id:
                                continue
                            model_name = model_dir_id.split("__")[0].replace("_", "/")
                            pretrained_name = model_dir_id.split("__", 1)[1]
                            model, _, _ = open_clip.create_model_and_transforms(
                                model_name,
                                pretrained=pretrained_name,
                                cache_dir=CACHE_DIR
                            )
                            model = model.to(DEVICE)
                            model.eval()
                        else:
                            if checkpoint_path is None:
                                continue
                            model = load_agriclip(str(checkpoint_path), DEVICE)
                            model.eval()
                        
                        self.models_cache[model_dir_id] = model
                    else:
                        model = self.models_cache[model_dir_id]
                    
                    # Get output directory for this model (load once)
                    out_dir = self.output_dir / model_dir_id / rosbag_name
                    ensure_dir(out_dir)
                    ensure_dir(out_dir / "shards")
                    manifest_path = out_dir / "manifest.parquet"
                    meta_path = out_dir / "meta.json"
                    manifest = read_manifest(manifest_path)
                    meta = load_meta(meta_path)
                    
                    # Process preprocessed tensors in chunks to avoid OOM
                    # Process 1000 images at a time, write to shards incrementally
                    chunk_size = 1000  # Process this many images at a time
                    num_chunks = math.ceil(len(preprocessed_tensors) / chunk_size)
                    total_processed = 0
                    
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, len(preprocessed_tensors))
                        chunk_tensors = preprocessed_tensors[chunk_start:chunk_end]
                        chunk_metadata = preprocessed_metadata[chunk_start:chunk_end]
                        
                        # Generate embeddings for this chunk in batches
                        chunk_embeddings = []
                        num_batches = math.ceil(len(chunk_tensors) / batch_size)
                        
                        for batch_idx in range(num_batches):
                            batch_start = batch_idx * batch_size
                            batch_end = min(batch_start + batch_size, len(chunk_tensors))
                            batch_tensors = chunk_tensors[batch_start:batch_end]
                            
                            batch_embeddings = self._process_batch_on_gpu(batch_tensors, model, DEVICE)
                            chunk_embeddings.append(batch_embeddings)
                            
                            # Clear GPU cache after each batch
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Concatenate chunk embeddings
                        if chunk_embeddings:
                            embeddings_tensor = torch.cat(chunk_embeddings, dim=0)
                            
                            # Prepare records for this chunk
                            records = []
                            for i, (emb, meta_item) in enumerate(zip(embeddings_tensor, chunk_metadata)):
                                records.append((
                                    meta_item["topic"],
                                    meta_item["timestamp_ns"],
                                    ns_to_minute_of_day(meta_item["timestamp_ns"]),
                                    meta_item["mcap_id"],
                                    emb
                                ))
                            
                            # Write this chunk to shards immediately
                            df_append, meta = write_shards_from_embeddings(
                                out_dir, records, meta, DEFAULT_SHARD_ROWS, OUTPUT_DTYPE
                            )
                            
                            # Update manifest
                            if not df_append.empty:
                                manifest = pd.concat([manifest, df_append], ignore_index=True)
                                manifest.sort_values("id", inplace=True, kind="mergesort")
                                write_manifest(manifest_path, manifest)
                                save_meta(meta_path, meta)
                                total_processed += len(df_append)
                            
                            # Clear chunk embeddings from memory
                            del embeddings_tensor, chunk_embeddings, records
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            gc.collect()
                    
                    self.logger.info(f"  {model_dir_id}: Added {total_processed} embedding(s) to shards")
                    
                    # Mark this MCAP as completed for this model
                    self.completion_tracker.mark_mcap_completed_with_model(
                        model_name=model_dir_id,
                        rosbag_name=rosbag_name,
                        mcap_id=mcap_id,
                        output_path=manifest_path
                    )
                    
                    # Unload model from GPU after processing
                    if model_dir_id in self.models_cache:
                        model = self.models_cache[model_dir_id]
                        model = model.cpu()  # Move to CPU
                        del model
                        del self.models_cache[model_dir_id]  # Remove from cache
                    
                    # Clear GPU cache after each model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"Error processing model {model_dir_id}: {e}")
                    # Unload model on error too
                    if model_dir_id in self.models_cache:
                        try:
                            model = self.models_cache[model_dir_id]
                            model = model.cpu()
                            del model
                            del self.models_cache[model_dir_id]
                        except:
                            pass
                    # Clear GPU cache even on error
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    continue
            
            # Clear preprocessed tensors after all models of this type
            del preprocessed_tensors
            gc.collect()
        
        # Store count before clearing
        processed_count = len(self.collected_images)
        
        # Clear collected images
        self.collected_images = []
        gc.collect()
        
        return {"processed": processed_count}
    
    def is_mcap_completed(self, context: McapProcessingContext) -> bool:
        """
        Check if this MCAP is already processed for all models.
        
        This method can be called before message collection to avoid
        unnecessary processing. Checks manifest content to verify completion.
        
        Args:
            context: McapProcessingContext
            
        Returns:
            True if all models are already processed for this MCAP, False otherwise
        """
        rosbag_name = context.get_relative_path().as_posix()
        mcap_id = context.get_mcap_id()
        
        # Check if this MCAP is already processed for all models
        all_models_complete = True
        for preprocess_id, models in self.models_by_preprocess.items():
            if not models:
                continue
            for model_dir_id, _, _, _ in models:
                out_dir = self.output_dir / model_dir_id / rosbag_name
                manifest_path = out_dir / "manifest.parquet"
                
                # Fast check: use completion tracker if available
                completion_marked = self.completion_tracker.is_mcap_completed_with_model(
                    model_name=model_dir_id,
                    rosbag_name=rosbag_name,
                    mcap_id=mcap_id
                )
                
                # Always verify with manifest content (more reliable)
                if manifest_path.exists():
                    manifest = read_manifest(manifest_path)
                    # Check if this MCAP's embeddings are already in the manifest
                    if not manifest.empty and "mcap_identifier" in manifest.columns:
                        mcap_in_manifest = (manifest["mcap_identifier"] == mcap_id).any()
                        if mcap_in_manifest:
                            # Verified in manifest - if not in completion.json, mark it
                            if not completion_marked:
                                self.completion_tracker.mark_mcap_completed_with_model(
                                    model_name=model_dir_id,
                                    rosbag_name=rosbag_name,
                                    mcap_id=mcap_id,
                                    output_path=manifest_path
                                )
                            continue  # This model is complete
                        else:
                            # Not in manifest - need to process
                            all_models_complete = False
                            break
                    else:
                        # Manifest exists but is empty or missing column - need to process
                        all_models_complete = False
                        break
                else:
                    # Manifest doesn't exist - need to process
                    all_models_complete = False
                    break
            if not all_models_complete:
                break
        
        return all_models_complete
    
    def get_output_path(self, context: McapProcessingContext) -> Optional[Path]:
        """Get the expected output path for this context."""
        # Return None to disable file-based completion checking.
        # The completion tracker's auto-repair logic would incorrectly mark all MCAPs
        # as completed if we return the manifest path, because the manifest exists
        # after the first MCAP is processed (it's a rosbag-level file, not MCAP-specific).
        # Instead, completion should be checked by examining the manifest content
        # (checking if mcap_identifier exists in the manifest) inside process_mcap.
        return None
