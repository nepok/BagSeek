"""Configuration module for Flask API.

Handles environment variable loading and path constants.
"""
import os
import sys
import secrets
import logging
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)


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
OPEN_CLIP_MODELS_STR = _require_env("OPEN_CLIP_MODELS")
OTHER_MODELS_STR = _require_env("OTHER_MODELS")

# Relative path environment variables (appended to BASE)
IMAGE_TOPIC_PREVIEWS_STR = _require_env("IMAGE_TOPIC_PREVIEWS")
POSITIONAL_LOOKUP_TABLE_STR = _require_env("POSITIONAL_LOOKUP_TABLE")
LOOKUP_TABLES_STR = _require_env("LOOKUP_TABLES")
TOPICS_STR = _require_env("TOPICS")
CANVASES_FILE_STR = _require_env("CANVASES_FILE")
POLYGONS_DIR_STR = _require_env("POLYGONS_DIR")
TOPIC_PRESETS_FILE_STR = _require_env("TOPIC_PRESETS_FILE")
ADJACENT_SIMILARITIES_STR = _require_env("ADJACENT_SIMILARITIES")
EMBEDDINGS_STR = _require_env("EMBEDDINGS")
EXPORT_STR = _require_env("EXPORT")
EXPORT_RAW_STR = _require_env("EXPORT_RAW")
# Path constants
ROSBAGS = Path(ROSBAGS_STR)

PRESELECTED_MODEL = os.getenv("PRESELECTED_MODEL", "ViT-B-16-quickgelu__openai")
OPEN_CLIP_MODELS = Path(BASE_STR + OPEN_CLIP_MODELS_STR)
OTHER_MODELS = Path(BASE_STR + OTHER_MODELS_STR)

IMAGE_TOPIC_PREVIEWS = Path(BASE_STR + IMAGE_TOPIC_PREVIEWS_STR)
POSITIONAL_LOOKUP_TABLE = Path(BASE_STR + POSITIONAL_LOOKUP_TABLE_STR)
POSITIONAL_BOUNDARIES = POSITIONAL_LOOKUP_TABLE.parent / "positional_boundaries.json"

LOOKUP_TABLES = Path(BASE_STR + LOOKUP_TABLES_STR)
TOPICS = Path(BASE_STR + TOPICS_STR)

CANVASES_FILE = Path(BASE_STR + CANVASES_FILE_STR)
POLYGONS_DIR = Path(BASE_STR + POLYGONS_DIR_STR)
TOPIC_PRESETS_FILE = Path(BASE_STR + TOPIC_PRESETS_FILE_STR)

ADJACENT_SIMILARITIES = Path(BASE_STR + ADJACENT_SIMILARITIES_STR)
EMBEDDINGS = Path(BASE_STR + EMBEDDINGS_STR)

EXPORT = Path(BASE_STR + EXPORT_STR)
EXPORT_RAW = Path(BASE_STR + EXPORT_RAW_STR)

# Authentication (optional — if APP_PASSWORD is not set, auth is disabled)
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_PASSWORD:
    logging.warning("APP_PASSWORD not set — authentication is disabled")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRY_SECONDS = 3600  # 1 hour

# Model configuration
CUSTOM_MODEL_DEFAULTS = {
    "agriclip": OTHER_MODELS / "agriclip.pt",
    "ViT-B-16-finetuned(09.10.25)": OTHER_MODELS / "ViT-B-16-finetuned(09.10.25).pt",
}

# Search configuration
MAX_K = 100

# Cache configuration
FILE_PATH_CACHE_TTL_SECONDS = 3600  # 1 hour - index file is the source of truth

# Index file for valid rosbags (written by preprocessing, read by API)
VALID_ROSBAGS_INDEX = Path(BASE_STR) / "valid_rosbags.json"

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
