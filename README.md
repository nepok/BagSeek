# 📦 BagSeek: Filtering and Semantic Exploration of ROS Data

**BagSeek** is an interactive tool for filtering and exploring large-scale ROS 2 bag files across three complementary views:

- **MAP** — visualize GPS traces across all rosbags, filter by drawn polygon, and narrow down to relevant spatial regions before touching any data
- **SEARCH** — filter and rank image frames using rosbag metadata, topic selection, time ranges and subsampling; semantic CLIP-based search with natural language adds a powerful semantic layer on top
- **EXPLORE** — scrub through synchronized multi-panel sensor streams (images, point clouds, GPS, IMU) at any timestamp, guided by search heatmaps or spatial filter results

All three views feed into each other and into a flexible **Export** pipeline, so the workflow from "I have 200 hours of rosbag data" to "here is the relevant slice" stays entirely local and privacy-preserving.

## 🧭 Key Features

**MAP**
- Interactive Leaflet map showing positional traces and density heatmaps across all rosbags
- Draw polygons to spatially filter rosbags; save and reuse polygon presets
- Open matching MCAP segments directly in Explore or Export with a single click

**SEARCH**
- Filter by rosbag, topic, time range, and frame subsampling before any embedding lookup
- Natural language queries matched against CLIP embeddings (e.g. *"cow on the field"*)
- Switch between OpenCLIP variants (`ViT-B-16` / OpenAI, `ViT-H-14` / LAION2B, custom models); optional LLM-based prompt enhancement
- Results ranked by similarity and visualized as a heatmap

**EXPLORE**
- Resizable, splittable panel layout for synchronized display of image, point cloud, GPS, IMU, and metadata streams (similar to Foxglove)
- Timeline slider with search result heatmap or MAP polygon highlight strip
- GPS panels overlay the full route heatmap of the rosbag alongside the current position
- Save and load named canvas layouts

**EXPORT**
- Export rosbag segments filtered by time range, topic selection, MCAP range, or semantic search results
- Topic presets for quickly reapplying common topic selections
- Supports exporting rosbags in MCAP format or exporting raw data (.jpeg, .pc, .json, ...)

**General**
- All three views feed into Export and into each other via deep-links (e.g. MAP → Explore, Search → Explore)
- Offline-ready and privacy-preserving — all inference runs locally, no data leaves the machine

## 📸 Example Use Case

A researcher analyzing hundreds of hours of agricultural robotics data wants to find and extract footage of cows near a specific field boundary.

1. **MAP** — open the map, inspect GPS density heatmaps across all rosbags, and draw a polygon around the target area. BagSeek identifies the matching rosbags and MCAP segments immediately. Open the most relevant one directly in Explore.
2. **EXPLORE** — configure a panel layout with the front camera, GPS map, and point cloud side by side. Scrub through the pre-filtered segment to get a feel for the data. The GPS panel shows the full route heatmap so spatial context is always visible.
3. **SEARCH** — switch to Search, select the rosbag and image topic, apply a time range and subsampling rate to keep results manageable, then enter *"cow on the field"*. Browse ranked results as a image grid or jump directly to matches on the heatmapped timeline in Explore. Cross-check with adjacent-similarity scores to catch visually interesting frames the query might have missed.
4. **EXPORT** — send the filtered segment — scoped to the relevant topics, time range, and search results — to Export. Choose between MCAP output or raw files (`.jpeg`, `.pcd`, `.json`, ...) and start exporting.

What would have required manually scrubbing through hundreds of hours of recordings is reduced to a few targeted filter steps and a single natural language query.

## 🚀 Installation

### Prerequisites

⚠️ **Linux only** — ROS 2 / MCAP tooling is Linux-native.

- [Docker](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for faster CLIP inference

### Quick Start (Docker — recommended)

```bash
# Clone the repo
git clone https://github.com/nepok/BagSeek.git
cd BagSeek

# Create and fill in your .env (see section below)
cp .env.example .env
# ... edit .env with your paths ...

# Start all services (backend :5000, frontend :3000)
./start.sh

# Or force a fresh image build:
./start.sh --build

# Tail logs after startup:
./start.sh --logs
```

`start.sh` brings down any existing containers, starts them detached, and optionally streams logs. The frontend waits for the backend health check before starting.

To remove GPU support (CPU-only), delete the `deploy.resources` block from `docker-compose.yml` before starting.

### Local Development (without Docker)

**Backend:**
```bash
cd flask-backend/api
pip install -r ../requirements-docker.txt

# ROS 2 must be sourced for MCAP decoding
source /opt/ros/humble/setup.bash

# Start in debug mode for detailed logs
flask run --debug   # http://localhost:5000

# Or start normally
flask run
```

**Frontend:**
```bash
cd react-frontend
npm install
npm start           # http://localhost:3000
```

The frontend proxies `/api/*` to `localhost:5000`.

### Preprocessing

Run the full 6-step pipeline to build all indexes from your rosbag files:

```bash
cd preprocessing
python main.py
```

Steps run in order:
1. **TopicsExtraction** — extract topics and message types from all rosbags
2. **TimestampAlignment** — build per-mcap timestamp lookup tables
3. **PositionalLookup** — create spatial grid index and concave hulls from positional topics
4. **ImageTopicPreviews** — extract preview images for the search UI
5. **Embeddings** — generate CLIP embeddings (sharded Parquet, 100K rows/shard)
6. **AdjacentSimilarities** — compute frame-to-frame similarity scores

The pipeline supports resumable processing via `CompletionTracker` — individual steps are skipped if already complete.

## 🔗 Configuration (.env)

Create a `.env` file in the project root. All paths under `BASE` can use relative subpaths as shown.

```env
# --- Required ---
BASE=/path/to/output/base          # Root directory for all preprocessed output
ROSBAGS=/path/to/rosbags           # Directory containing ROS 2 bag folders

PRESELECTED_ROSBAG_NAME=my_rosbag  # Default rosbag loaded on startup (folder name only)
PRESELECTED_MODEL=my_model         # Default model loaded on startup (folder name only)

OPEN_CLIP_MODELS=/path/to/openclip_cache   # OpenCLIP model weights cache
OTHER_MODELS=/path/to/custom_models        # Custom model weights (non-OpenCLIP)

# --- Output paths (relative to BASE) ---
IMAGE_TOPIC_PREVIEWS=/frontend/image_topic_previews
POSITIONAL_LOOKUP_TABLE=/frontend/positional_lookup_table/positional_lookup_table.json

METADATA_DIR=/metadata
LOOKUP_TABLES=/metadata/lookup_tables
TOPICS=/metadata/topics

ADJACENT_SIMILARITIES=/processed/adjacent_similarities
EMBEDDINGS=/processed/embeddings

CANVASES_FILE=/public/canvases.json
POLYGONS_DIR=/public/positional_polygons
TOPIC_PRESETS_FILE=/public/topic_presets.json

EXPORT=/export
EXPORT_RAW=/export_raw

# --- Optional ---
CORS_ORIGINS=http://localhost:3000  # Allowed CORS origins (comma-separated)
APP_PASSWORD=                       # Set to enable password authentication (implemented, but not used)
```

## 🗂️ Project Structure

```
bagseek/
├── flask-backend/
│   ├── api/
│   │   ├── api.py                  # App factory, blueprint registration
│   │   ├── state.py                # Shared server-side state
│   │   ├── config.py               # Env-based configuration
│   │   ├── routes/                 # Modular API blueprints
│   │   │   ├── search.py           # Semantic search (CLIP + FAISS)
│   │   │   ├── content.py          # Fetch topic content by timestamp
│   │   │   ├── export.py           # Rosbag export
│   │   │   └── ...
│   │   └── utils/                  # Several helper functions
│   │       ├── rosbag.py          
│   │       ├── mcap.py        
│   │       ├── clip.py  
│   │       └── ... 
│   ├── Dockerfile
│   └── requirements-docker.txt
├── preprocessing/
│   ├── abstract/                   # Base processor classes
│   ├── processors/                 # Six pipeline steps
│   │   ├── TopicsExtractionProcessor.py
│   │   ├── TimestampAlignmentProcessor.py
│   │   ├── PositionalLookupProcessor.py
│   │   ├── ImageTopicPreviewsProcessor.py
│   │   ├── EmbeddingsProcessor.py
│   │   └── AdjacentSimilaritiesPostprocessor.py
│   ├── config.py
│   └── main.py
├── react-frontend/
│   └── src/
│       ├── components/
│       │   ├── SplittableCanvas/   # Resizable panel manager
│       │   ├── NodeContent/        # Topic content renderer (image, pointcloud, GPS, IMU)
│       │   ├── GlobalSearch/       # CLIP search interface
│       │   ├── TimestampPlayer/    # Timeline slider with heatmap and MCAP marks
│       │   ├── HeatBar/            # Search heatmap / MCAP highlight strip
│       │   ├── PositionalOverview/ # MAP view with Leaflet heatmaps and polygon tools
│       │   ├── Export/             # Export dialog with preselection support
│       │   ├── McapRangeFilter/    # MCAP range selector for Export and MAP
│       │   ├── Header/             # Navigation and canvas management
│       │   └── ...
│       ├── App.tsx
│       └── index.tsx
├── docker-compose.yml
├── start.sh                        # Convenience startup script
├── .env                            # Your local configuration (not committed)
└── README.md
```

## ✨ Credits

Developed by **Nepomuk Kindermann** with help of several LLMs.
Powered by [ROS 2](https://www.ros.org/), [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss), [Flask](https://flask.palletsprojects.com/), and [React](https://react.dev/).
Developed with support from the **Smart Farming Lab** at the **University of Leipzig**.

## 📄 License

This project is licensed under the MIT License. See `LICENSE.md` for details.
