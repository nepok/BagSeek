# BagSeek — Architecture Overview

BagSeek is a semantic search and visualization tool for large-scale ROS 2 bag files.
Users search image topics with natural language (e.g. "cow on the field"), view matching
frames across time, and export relevant data slices.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [End-to-End Data Flow](#2-end-to-end-data-flow)
3. [Configuration & Environment](#3-configuration--environment)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Flask Backend](#5-flask-backend)
6. [React Frontend](#6-react-frontend)
7. [Key Data Formats](#7-key-data-formats)
8. [Adding a New CLIP Model](#8-adding-a-new-clip-model)
9. [Running the Project](#9-running-the-project)

---

## 1. Repository Layout

```
bagseek/
├── .env                        # All configuration (never commit secrets)
├── preprocessing/              # Offline pipeline: bags → indexed data
│   ├── main.py                 # Pipeline entry point (run this)
│   ├── config.py               # Config dataclass, loaded from .env
│   ├── abstract/               # Base processor classes
│   ├── processors/             # The 6 concrete pipeline steps
│   ├── core/                   # RosbagProcessingContext, McapProcessingContext
│   └── utils/                  # CompletionTracker, logger, file helpers
├── flask-backend/
│   └── api/
│       ├── api.py              # Flask app entry point
│       ├── config.py           # Env var constants (mirrors preprocessing/config.py)
│       ├── state.py            # Thread-safe global state & caches
│       ├── routes/             # Blueprint-per-feature route files
│       └── utils/              # CLIP inference, MCAP reading, rosbag helpers
└── react-frontend/
    └── src/
        ├── App.tsx             # Root + routing
        └── components/         # One folder per feature
```

---

## 2. End-to-End Data Flow

```
ROS 2 Bags (MCAP files)
        │
        ▼
┌─────────────────────┐
│ Preprocessing       │  python preprocessing/main.py
│ (6-step pipeline)   │
└─────────┬───────────┘
          │  writes to BASE/ directory
          ▼
┌──────────────────────────────────────────────────┐
│ Indexed data on disk                             │
│  topics/          JSON per rosbag                │
│  lookup_tables/   Parquet per MCAP (timestamps)  │
│  embeddings/      Parquet shards + manifest      │
│  positional/      positional_lookup.json         │
│  previews/        representative JPEG previews   │
│  summaries.json   global timestamp summary       │
│  valid_rosbags.json  index of searchable bags    │
└──────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Flask API           │  flask run --debug (port 5000)
│                     │  Reads disk data at query time.
│                     │  Builds FAISS index in memory per search.
└─────────┬───────────┘
          │  REST/JSON
          ▼
┌─────────────────────┐
│ React Frontend      │  npm start (port 3000, proxied to 5000)
│                     │  Search UI, timeline, map, export
└─────────────────────┘
```

> **Important:** The Flask API never modifies the indexed data. Preprocessing and the API
> are completely decoupled — you can re-run preprocessing while the server is running; the
> API picks up changes via file-mtime cache invalidation.

---

## 3. Configuration & Environment

All configuration lives in a single `.env` file at the project root.
Both the preprocessing pipeline and the Flask API load it automatically at startup.

```
# Absolute paths
BASE=/mnt/data/bagseek_output    # Root for all generated data
ROSBAGS=/mnt/data/rosbags        # Source ROS 2 bags (read-only)
OPEN_CLIP_MODELS=/mnt/data/openclip_cache
OTHER_MODELS=/mnt/data/bagseek/flask-backend/src/models

# Relative paths (appended to BASE)
IMAGE_TOPIC_PREVIEWS=/previews
POSITIONAL_LOOKUP_TABLE=/positional/positional_lookup.json
METADATA_DIR=/metadata
LOOKUP_TABLES=/lookup_tables
TOPICS=/topics
CANVASES_FILE=/canvases.json
POLYGONS_DIR=/polygons
TOPIC_PRESETS_FILE=/topic_presets.json
ADJACENT_SIMILARITIES=/adjacent_similarities
EMBEDDINGS=/embeddings
EXPORT=/exports
EXPORT_RAW=/exports_raw

# Optional
PRESELECTED_MODEL=ViT-B-16-quickgelu__openai
APP_PASSWORD=                    # Leave empty to disable auth
JWT_SECRET=                      # Auto-generated if not set
SHARD_SIZE=100000
```

`preprocessing/config.py` exposes these as a `Config` dataclass.
`flask-backend/api/config.py` exposes them as module-level constants.
Both files follow the same path-construction pattern: `Path(BASE + RELATIVE)`.

---

## 4. Preprocessing Pipeline

Run once (or incrementally) to prepare all data for the API:

```bash
cd /mnt/data/bagseek
conda activate bagseek-gpu
python preprocessing/main.py
```

The pipeline is **resumable**: each step records completion at rosbag and MCAP granularity
via `CompletionTracker` (`preprocessing/utils/completion.py`). Re-running skips already
completed work at every level.

### Rosbag directory conventions

| Type | Disk layout | Example |
|------|-------------|---------|
| Regular | `ROSBAGS/my_bag/` containing `.mcap` files | `rosbag2_2025_07_25-12_17_25/` |
| Multi-part | `ROSBAGS/my_bag_multi_parts/Part_1/`, `Part_2/`, … | `rosbag2_2025_07_23_multi_parts/Part_1/` |

Multi-part rosbags are treated as independent bags for indexing but share a common parent
name. The `get_all_rosbags()` helper in `preprocessing/utils/file_helpers.py` handles
discovery of both types.

### The 6 Steps

```
Step 1  TopicsExtractionProcessor       RosbagProcessor
Step 2  TimestampAlignmentProcessor     McapProcessor
Step 3  PositionalLookupProcessor       HybridProcessor
Step 4  ImageTopicPreviewsProcessor     HybridProcessor
Step 5  EmbeddingsProcessor             McapProcessor
Step 6  AdjacentSimilaritiesPostprocessor  (runs after embeddings per rosbag)
```

#### Step 1 — Topics Extraction (`TopicsExtractionProcessor`)
Calls `ros2 bag info` on each rosbag to discover all topic names and message types.
Output: `topics/<rosbag_name>.json`

#### Step 2 — Timestamp Alignment (`TimestampAlignmentProcessor`)
Opens every MCAP and records, for each message, which topics have data at that timestamp.
Produces one Parquet file per MCAP: a lookup table mapping *reference timestamps* to
per-topic timestamps. The API uses this to find the closest message to a requested time.
Output: `lookup_tables/<rosbag_name>/<mcap_id>.parquet`

#### Step 3 — Positional Lookup (`PositionalLookupProcessor`)
Reads odometry/GPS topics and snaps GPS coordinates to a grid (resolution 0.0001° ≈ 11 m).
Builds a JSON index mapping grid cells to which rosbag/MCAP has data there.
Used by the map view to show heatmaps and filter by drawn polygon.
Output: `positional/positional_lookup.json`, `positional/positional_boundaries.json`

#### Step 4 — Image Topic Previews (`ImageTopicPreviewsProcessor`)
Extracts representative JPEG frames from image topics using a "fencepost" strategy —
evenly spaced frames rather than every frame — to keep storage manageable.
Output: `previews/<rosbag_name>/<topic>/<timestamp>.jpg`

#### Step 5 — Embeddings (`EmbeddingsProcessor`)
The most compute-intensive step. For each image topic message:
1. Decodes the ROS image message to a PIL image
2. Preprocesses with the CLIP transform
3. Runs the CLIP vision encoder
4. Writes embedding vectors to sharded Parquet files (100 K rows per shard)
5. Writes a `manifest.parquet` with metadata (topic, timestamp, mcap id, shard location)

One subdirectory per model: `embeddings/<model_name>/<rosbag_name>/`

Configured models are in `OPENCLIP_MODELS` and `CUSTOM_MODELS` lists at the top of
`EmbeddingsProcessor.py`. Disable a model by setting `"enabled": False` in `CUSTOM_MODELS`
or commenting it out from `OPENCLIP_MODELS`.

#### Step 6 — Adjacent Similarities (`AdjacentSimilaritiesPostprocessor`)
Computes cosine similarity between consecutive frames within a rosbag. Used by the
HeatBar UI component to show visually redundant vs. novel regions of the timeline.
Output: `adjacent_similarities/<rosbag_name>.parquet`

### Processor Base Classes (`preprocessing/abstract/`)

| Class | Use when |
|-------|----------|
| `RosbagProcessor` | Work runs once per rosbag (no MCAP iteration needed) |
| `McapProcessor` | Work requires iterating every MCAP message |
| `HybridProcessor` | Needs a rosbag-level setup phase, then MCAP iteration, then aggregation |
| `PostProcessor` | Final aggregation across all rosbags after the main loop |

The main loop in `main.py` opens each MCAP **exactly once** and fans messages out to all
active processors simultaneously. This avoids repeated disk reads for multi-step processing.

---

## 5. Flask Backend

```bash
cd flask-backend/api
flask run    # port 5000
```

### Route Blueprints

| File | Blueprint | Responsibility |
|------|-----------|----------------|
| `routes/search.py` | `search_bp` | CLIP encoding, FAISS search, streaming results |
| `routes/content.py` | `content_bp` | Fetch raw MCAP content by topic + timestamp |
| `routes/export.py` | `export_bp` | Export MCAP slices, track export progress |
| `routes/topics.py` | `topics_bp` | List topics, fetch timestamp summaries |
| `routes/positions.py` | `positions_bp` | GPS heatmap data |
| `routes/polygons.py` | `polygons_bp` | Save/load drawn map polygons |
| `routes/config.py` | `config_bp` | File path listing, cache management |
| `routes/canvases.py` | `canvases_bp` | Persist panel layout (canvas state) |
| `routes/auth.py` | `auth_bp` | Optional password auth via JWT cookie |

### Search Flow (`/api/search`)

1. Receives `query` (text), `models` (comma-separated), `rosbags` (comma-separated), optional `timeRange`
2. Encodes the query text with the CLIP text encoder → 512-dim float vector
3. For each (model, rosbag) combination:
   - Loads all embedding shards from `embeddings/<model>/<rosbag>/` into memory
   - Builds a FAISS flat-HNSW index (almost exact search, a little bit of approximation for speedup)
   - Queries top-K nearest neighbours
   - Joins results with the manifest to recover topic, timestamp, and MCAP id
4. Streams incremental JSON results back to the frontend as they complete
5. Search can be cancelled at any point via `/api/cancel-search`; the cancellation
   mechanism uses a generation ID in `state.py` (`start_new_search` / `is_search_cancelled`)

> The FAISS index is **rebuilt on every search** — it is not persisted. For the dataset
> sizes this tool targets (tens of millions of frames), building the index typically takes
> a few seconds on GPU. If this becomes a bottleneck, consider persisting the index with
> `faiss.write_index`.

### Content Fetching (`/api/content-mcap`)

Given a rosbag name, MCAP identifier, timestamp, and topic, this endpoint:
1. Uses the lookup table Parquet (Step 2 output) to find the closest actual timestamp
2. Opens the corresponding `.mcap` file with `mcap.SeekingReader`
3. Seeks to the timestamp and decodes the message
4. Returns the payload as JSON (images as base64, point clouds as arrays, etc.)

### State & Caching (`state.py`)

All shared mutable state lives in `state.py`. Thread safety is provided by explicit locks:

| State | Lock | Purpose |
|-------|------|---------|
| `EXPORT_PROGRESS` | `_export_progress_lock` | Export progress for polling |
| `SEARCH_PROGRESS` | `_search_progress_lock` | Search progress for polling |
| `_current_search_id` | `_search_id_lock` | Search cancellation generation ID |
| `_current_reference_timestamp` | `_reference_timestamp_lock` | Current playback position |
| `_matching_rosbag_cache` | `_file_path_cache_lock` | Cached list of valid rosbag paths |
| `_lookup_table_cache` | `_lookup_table_cache_lock` | Cached timestamp lookup DataFrames |
| `_positional_lookup_cache` | `_positional_cache_lock` | Cached positional lookup JSON |

Always use the getter/setter functions (`get_reference_timestamp()`, etc.) rather than
accessing the private `_` variables directly.

### Authentication

Auth is **optional**. Set `APP_PASSWORD` in `.env` to enable it.
When enabled, all API routes except `/api/login`, `/api/logout`, and `/api/auth-check`
require a valid JWT cookie (`auth_token`). The JWT is issued on POST `/api/login` and
expires after 1 hour. Set `secure=True` in `auth.py` if deploying behind HTTPS.

### Custom CLIP Models (`utils/clip.py`)

Standard OpenCLIP models are loaded via `open_clip.create_model_and_transforms()`.
Custom checkpoints (e.g. AgriCLIP) that were not trained with open_clip require a minimal
ViT re-implementation in `utils/clip.py` — the architecture follows the original CLIP paper.
Model name → checkpoint path mappings are defined in `CUSTOM_MODEL_DEFAULTS` in `config.py`.

---

## 6. React Frontend

```bash
cd react-frontend
npm start    # port 3000, proxies /api/* to localhost:5000
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `GlobalSearch` | Text query input, model/rosbag selection, result grid |
| `TimestampPlayer` | Timeline scrubber, playback controls |
| `SplittableCanvas` | Resizable multi-panel layout manager |
| `NodeContent` | Renders a single topic panel (image / point cloud / map / IMU) |
| `PositionalOverview` | Leaflet map with GPS heatmaps and polygon drawing |
| `Export` | Select time range + topics, trigger MCAP export |
| `HeatBar` | Visual timeline showing similarity and search result density |
| `Header` | Rosbag selector, model selector, canvas management |

### Frontend State

There is no Redux or global state library. State flows via props and React context.
Cross-component communication (e.g. map polygon → export panel) uses `sessionStorage`
with prefixed keys (`__BagSeekMapMcapFilter`, `__BagSeekPositionalFilter`, etc.).

---

## 7. Key Data Formats

### Lookup Tables (`lookup_tables/`)
One Parquet file per MCAP. Columns are topic names; each row is a reference timestamp
with the corresponding per-topic timestamp (or NaN if no message at that time).
Column `_mcap_id` is added at read time by `load_lookup_tables_for_rosbag()`.

### Embeddings (`embeddings/<model>/<rosbag>/`)
```
manifest.parquet          — metadata for every embedded frame
shard_0000.parquet        — embedding vectors (float32, 512-dim)
shard_0001.parquet
...
```
`manifest.parquet` columns: `id`, `topic`, `timestamp_ns`, `minute_of_day`,
`mcap_identifier`, `reference_timestamp`, `reference_timestamp_index`, `shard_id`, `row_in_shard`

The manifest links each embedding row back to its source MCAP + timestamp, which the
search route uses to resolve results back to viewable content.

### Model Name Convention
Model names are formatted as `<arch>__<pretrained>` (double underscore) when stored on
disk and passed between frontend and backend. Example: `ViT-B-16-quickgelu__openai`.

---

## 8. Adding a New CLIP Model

### OpenCLIP model
1. Add to `OPENCLIP_MODELS` list in `preprocessing/processors/EmbeddingsProcessor.py`
2. Re-run preprocessing — new embeddings will be written alongside existing ones
3. The API picks up the new model directory automatically on next search

### Custom checkpoint (`.pt` file)
1. Place the `.pt` file in `OTHER_MODELS/` (configured in `.env`)
2. Add an entry to `CUSTOM_MODELS` in `EmbeddingsProcessor.py` with `"enabled": True`
3. Add the same name → path mapping to `CUSTOM_MODEL_DEFAULTS` in `flask-backend/api/config.py`
4. Re-run preprocessing for the new embeddings
5. If the checkpoint uses a non-standard architecture, extend `utils/clip.py`

---

## 9. Running the Project

### First-time setup
```bash
# 1. Copy and fill in .env
cp .env.example .env   # edit paths to match your system

# 2. Create conda environment (GPU recommended)
conda env create -f environment.gpu.yaml
conda activate bagseek-gpu

# 3. Run preprocessing (can take hours on large datasets)
python preprocessing/main.py

# 4. Start the backend
cd flask-backend/api
flask run --debug

# 5. Start the frontend (separate terminal)
cd react-frontend
npm install && npm start
```

### Incremental re-processing
`CompletionTracker` makes re-runs safe and fast — only unprocessed rosbags/MCAPs are
touched. To force a step to re-run for a specific rosbag, delete its entry from the
corresponding `.completion_tracker.json` file in the output directory.

### Docker
```bash
docker-compose up --build   # backend :5000, frontend :3000
```

### Cache invalidation
The API caches lookup tables and positional data in memory by file mtime. After
re-running preprocessing, call `POST /api/clear-cache` or restart the Flask server
to pick up new data immediately.
