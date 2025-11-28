#!/usr/bin/env python3
"""
Preprocessing step: Preprocess images for all models.

This script preprocesses images and saves them as .pt files for later use in embedding generation.
It groups models by their preprocessing requirements to avoid redundant preprocessing.
"""

import os
import re
import json
import warnings
import signal
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import open_clip
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

# Validate required paths
if IMAGES is None:
    raise RuntimeError("Environment variable IMAGES is not set.")
if PREPROCESSED is None:
    raise RuntimeError("Environment variable PREPROCESSED is not set.")

# =========================
# Configuration Flags
# =========================

SKIP_COMPLETED = True  # Set to False to reprocess completed rosbags
PREPROCESS_WORKERS = 8  # Threading workers for preprocessing (I/O-bound)

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

# Custom CLIP models (for preprocessing grouping)
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

CACHE_DIR = "/mnt/data/openclip_cache"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Silence noisy dependency warnings
warnings.filterwarnings("ignore", message=r".*Failed to load image Python extension.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Unable to find acceptable character detection dependency.*")

# =========================
# Utility Functions
# =========================

def log(msg: str) -> None:
    """Print message with flush."""
    print(msg, flush=True)

def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)

# =========================
# Completion Tracking
# =========================

def get_completion_file() -> Path:
    """Get the path to the completion file."""
    # Store completion file in PREPROCESSED directory
    return PREPROCESSED / "completion.json"

def load_completion() -> Dict[str, Dict]:
    """Load the dictionary of completed rosbag data from the completion file."""
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
                    result[item] = {
                        "models": {},
                        "all_completed": False
                    }
                elif isinstance(item, dict):
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        if "completed_at" in item and "models" not in item:
                            result[rosbag_name] = {
                                "models": {},
                                "all_completed": False,
                                "legacy_completed_at": item.get("completed_at", "unknown"),
                                "legacy_errors": item.get("errors", [])
                            }
                        else:
                            result[rosbag_name] = {
                                "models": item.get("models", {}),
                                "all_completed": item.get("all_completed", False)
                            }
            return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return {}

def save_completed_step(rosbag_name: str, model_id: str, step: str, status: str = "success", error: Optional[str] = None):
    """Mark a specific step as completed for a model."""
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
    
    if "steps" not in completed[rosbag_name]["models"][model_id]:
        completed[rosbag_name]["models"][model_id]["steps"] = {}
    
    completed[rosbag_name]["models"][model_id]["steps"][step] = {
        "status": status,
        "completed_at": timestamp
    }
    if error:
        completed[rosbag_name]["models"][model_id]["steps"][step]["error"] = error
    
    # Mark preprocessing step as completed for this model
    completed[rosbag_name]["models"][model_id]["steps"]["preprocessing"] = {
        "status": status,
        "completed_at": timestamp
    }
    if error:
        completed[rosbag_name]["models"][model_id]["steps"]["preprocessing"]["error"] = error
    
    completed[rosbag_name]["models"][model_id]["all_steps_completed"] = True  # Only preprocessing step
    
    save_completion(completed)

def save_completion(completed: Dict[str, Dict]):
    """Save completed rosbags dictionary to completion file."""
    completion_file = get_completion_file()
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    
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

def is_step_completed(rosbag_name: str, model_id: str, step: str) -> bool:
    """Check if a specific step has been completed for a model."""
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
    return step_status in ("success", "skipped")

def is_in_excluded_folder(path: Path) -> bool:
    """Check if a path is inside an EXCLUDED folder (case-insensitive)."""
    current = path.resolve().parent
    while current != current.parent:
        if current.name.upper() == "EXCLUDED":
            return True
        current = current.parent
    return False

# =========================
# Preprocessing Functions
# =========================

_PREPROCESS_CACHE: Dict[Tuple[str, str], object] = {}
_CUSTOM_PREPROCESS_CACHE: Dict[str, object] = {}
preprocess_dict: Dict[str, List[str]] = OrderedDict()

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

def collect_custom_preprocess_data(config: Dict) -> None:
    """Collect unique preprocessing transforms for custom models."""
    if not config.get("enabled", True):
        return
    
    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        log(f"      ⚠️  Checkpoint not found: {checkpoint_path} (skipping {config['name']})")
        return
    
    preprocess = get_custom_preprocess_transform(checkpoint_path)
    preprocess_str = str(preprocess)
    model_dir_name = config.get("model_dir_name", config["name"])

    if preprocess_str not in preprocess_dict:
        preprocess_dict[preprocess_str] = []
    if model_dir_name not in preprocess_dict[preprocess_str]:
        preprocess_dict[preprocess_str].append(model_dir_name)

def get_preprocess_id(preprocess_str: str) -> str:
    """Generate a concise identifier string summarizing the preprocessing steps."""
    summary_parts = []

    # Match Resize(size=224) or Resize((224, 224)) formats
    resize_match = re.search(r"Resize\([^)]*?(\d+)", preprocess_str)
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

# =========================
# Image Collection
# =========================

def _parse_timestamp_ns_from_stem(stem: str) -> Optional[int]:
    """Parse timestamp from filename stem."""
    import re
    _TS_LEADING_RE = re.compile(r"^(\d+)")
    m = _TS_LEADING_RE.match(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def collect_images_from_rosbag(rosbag_path: Path) -> List[Tuple[Path, str, str, int]]:
    """
    Collect all images from a rosbag directory.
    Returns list of (image_path, topic_name, mcap_identifier, timestamp_ns).
    """
    images = []
    for topic_dir in rosbag_path.iterdir():
        if not topic_dir.is_dir():
            continue

        topic_name = topic_dir.name.replace("__", "/")
        for mcap_dir in topic_dir.iterdir():
            if not mcap_dir.is_dir():
                continue

            mcap_identifier = mcap_dir.name
            for image_file in mcap_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    # Extract timestamp from filename
                    timestamp_ns = _parse_timestamp_ns_from_stem(image_file.stem)
                    if timestamp_ns is not None:
                        images.append((image_file, topic_name, mcap_identifier, timestamp_ns))

    return sorted(images, key=lambda x: (x[1], x[2], x[3]))

# =========================
# Main Processing
# =========================

def process_rosbag(rosbag_path: Path, rosbag_name: str) -> None:
    """Process a single rosbag: preprocess images for all models."""
    log(f"\n📦 {rosbag_name}")

    # Collect all images
    images = collect_images_from_rosbag(rosbag_path)
    if not images:
        log(f"  ⚠️  No images found (skipping)")
        return

    log(f"  📝 Found {len(images)} image(s)")

    # Group models by preprocessing (for open_clip models and custom models)
    preprocess_dict.clear()
    for model_name, pretrained_name in OPENCLIP_MODELS:
        collect_preprocess_data(model_name, pretrained_name, CACHE_DIR)
    
    # Add custom models
    for config in CUSTOM_MODELS:
        collect_custom_preprocess_data(config)

    # Process all models by preprocessing group
    for preprocess_str, model_list in preprocess_dict.items():
        preprocess_id = get_preprocess_id(preprocess_str)

        log(f"  🔄 Preprocessing for group: {preprocess_id} ({len(model_list)} model(s))")

        # Get preprocessing transform
        # Check if this is an open_clip model or custom model
        if " (" in model_list[0] and ")" in model_list[0]:
            # OpenCLIP model format: "model_name (pretrained_name)"
            shared_model_name, shared_pretrained_name = model_list[0].split(" (", 1)
            shared_pretrained_name = shared_pretrained_name.rstrip(")")
            preprocess_transform = get_preprocess_transform(shared_model_name, shared_pretrained_name, CACHE_DIR)
        else:
            # Custom model - find the checkpoint path
            custom_model_name = model_list[0]
            checkpoint_path = None
            for config in CUSTOM_MODELS:
                model_dir_name = config.get("model_dir_name", config["name"])
                if model_dir_name == custom_model_name:
                    checkpoint_path = Path(config["checkpoint"])
                    break
            if checkpoint_path is None or not checkpoint_path.exists():
                log(f"      ⚠️  Checkpoint not found for {custom_model_name} (skipping)")
                continue
            preprocess_transform = get_custom_preprocess_transform(checkpoint_path)

        # Preprocess images
        from concurrent.futures import ThreadPoolExecutor as TPE
        
        def preprocess_single_image(args):
            image_path, topic_name, mcap_identifier, timestamp_ns = args
            try:
                # Structure: preprocessed/preprocess_id/rosbag_name/topic_name/mcap_identifier/timestamp.pt
                topic_dir_name = topic_name.replace("/", "__")
                output_path = PREPROCESSED / preprocess_id / rosbag_name / topic_dir_name / mcap_identifier / f"{timestamp_ns}.pt"
                if output_path.exists():
                    try:
                        torch.load(output_path, weights_only=True)
                        return "skipped"
                    except (EOFError, RuntimeError, OSError) as e:
                        # Corrupted or incomplete file - delete it and reprocess
                        try:
                            output_path.unlink()
                        except Exception:
                            pass
                
                # File doesn't exist or was corrupted - process it
                tensor = preprocess_image(image_path, preprocess_transform)
                if tensor is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        torch.save(tensor, output_path)
                        return "saved"
                    except Exception as e:
                        log(f"      ⚠️  Failed to save preprocessed file {output_path}: {e}")
                        return "failed"
                return "failed"
            except Exception as e:
                log(f"      ⚠️  Failed to preprocess {image_path}: {e}")
                return "failed"
        
        # Process all images
        with TPE(max_workers=PREPROCESS_WORKERS) as executor:
            results = list(tqdm(
                executor.map(preprocess_single_image, images),
                total=len(images),
                desc=f"Preprocessing ({preprocess_id})",
                unit="img",
                leave=False
            ))
        
        saved_count = sum(1 for r in results if r == "saved")
        skipped_count = sum(1 for r in results if r == "skipped")
        failed_count = sum(1 for r in results if r == "failed")
        
        log(f"    ✅ Preprocessing complete: {saved_count} saved, {skipped_count} skipped, {failed_count} failed")
        
        # Mark preprocessing as completed for all models in this group
        for model_id_str in model_list:
            if " (" in model_id_str and ")" in model_id_str:
                # OpenCLIP model
                model_name, pretrained_name = model_id_str.split(" (", 1)
                pretrained_name = pretrained_name.rstrip(")")
                model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
            else:
                # Custom model - use model_dir_name directly
                model_dir_id = model_id_str
            save_completed_step(rosbag_name, model_dir_id, "preprocessing", status="success")

def main():
    """Main function to process all rosbags."""
    log("🚀 Starting image preprocessing pipeline")
    
    # Ensure output directory exists
    ensure_dir(PREPROCESSED)
    
    # Collect rosbags
    rosbags = []
    for rosbag_dir in IMAGES.iterdir():
        if rosbag_dir.is_dir():
            rosbag_name = rosbag_dir.name
            
            # Check if rosbag directory itself is in EXCLUDED folder
            if is_in_excluded_folder(rosbag_dir):
                log(f"  ⏭️  {rosbag_name} - inside EXCLUDED folder (skipping)")
                continue
            
            # Check if rosbag is in EXCLUDED folder by checking ROSBAGS directory
            should_skip = False
            if ROSBAGS is not None:
                rosbag_path_in_rosbags = ROSBAGS / rosbag_name
                # Also check parent directories in case rosbag is in a subdirectory
                if rosbag_path_in_rosbags.exists():
                    if is_in_excluded_folder(rosbag_path_in_rosbags):
                        log(f"  ⏭️  {rosbag_name} - inside EXCLUDED folder in ROSBAGS (skipping)")
                        should_skip = True
                else:
                    # Search for rosbag in ROSBAGS (might be in subdirectory)
                    found_rosbag = None
                    for rosbag_path in ROSBAGS.rglob(rosbag_name):
                        if rosbag_path.is_dir() and (rosbag_path / "metadata.yaml").exists():
                            found_rosbag = rosbag_path
                            break
                    if found_rosbag:
                        if is_in_excluded_folder(found_rosbag):
                            log(f"  ⏭️  {rosbag_name} - inside EXCLUDED folder in ROSBAGS (found at {found_rosbag}) (skipping)")
                            should_skip = True
                    else:
                        # Also check if rosbag name appears in any EXCLUDED path (case-insensitive partial match)
                        for excluded_path in ROSBAGS.rglob("*EXCLUDED*"):
                            if excluded_path.is_dir():
                                # Check if rosbag_name is in this EXCLUDED directory
                                for item in excluded_path.iterdir():
                                    if item.is_dir() and rosbag_name.lower() in item.name.lower():
                                        log(f"  ⏭️  {rosbag_name} - found in EXCLUDED path: {excluded_path} (skipping)")
                                        should_skip = True
                                        break
                                if should_skip:
                                    break
            
            if should_skip:
                continue
            rosbags.append((rosbag_dir, rosbag_name))
    
    if not rosbags:
        log("  ⚠️  No rosbags found in IMAGES directory")
        return
    
    log(f"📋 Found {len(rosbags)} rosbag(s) to process")
    
    # Process rosbags
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    try:
        for rosbag_path, rosbag_name in tqdm(rosbags, desc="Processing rosbags", unit="rosbag"):
            # Check if preprocessing is already completed for all models
            all_completed = True
            for model_name, pretrained_name in OPENCLIP_MODELS:
                model_dir_id = f"{model_name.replace('/', '_')}__{pretrained_name}"
                if not is_step_completed(rosbag_name, model_dir_id, "preprocessing"):
                    all_completed = False
                    break
            
            # Also check custom models
            if all_completed:
                for config in CUSTOM_MODELS:
                    if not config.get("enabled", True):
                        continue
                    model_dir_name = config.get("model_dir_name", config["name"])
                    if not is_step_completed(rosbag_name, model_dir_name, "preprocessing"):
                        all_completed = False
                        break
            
            if SKIP_COMPLETED and all_completed:
                log(f"  ⏭️  {rosbag_name} - preprocessing already completed (skipping)")
                skipped_count += 1
                continue
            
            try:
                process_rosbag(rosbag_path, rosbag_name)
                processed_count += 1
                log(f"  ✅ Successfully completed preprocessing for {rosbag_name}")
            except Exception as e:
                error_msg = str(e)
                log(f"  ❌ Failed to process {rosbag_name}: {error_msg}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
    except KeyboardInterrupt:
        log("\n⚠️  Keyboard interrupt received")
        raise
    
    log(f"\n✅ Preprocessing pipeline complete")
    log(f"📊 Summary: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")

if __name__ == "__main__":
    main()

