# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BagSeek is a semantic exploration tool for large-scale ROS 2 bag files. It enables natural language image search using CLIP embeddings and FAISS indexing, with real-time visualization of multimodal sensor data (images, point clouds, IMU, positional data).

**Use case**: Analyze agricultural robotics data - search for "cow on the field" or "tree stump", view matching frames across time, export relevant data slices.

## Tech Stack

- **Frontend**: React 18 + TypeScript, Material-UI, Three.js (3D), Leaflet (maps), Socket.io client
- **Backend**: Python Flask, OpenCLIP, FAISS, PyTorch
- **Data**: ROS 2 bags (Folder with several MCAP files and one metadata.yaml), Parquet (embeddings)
- **Preprocessing**: 6-step pipeline with abstract processor classes

## Development Commands

### Frontend (react-frontend/)
```bash
npm install       # Install dependencies
npm start         # Dev server on port 3000
npm run build     # Production build
npm test          # Run tests
```

### Backend (flask-backend/api/)
```bash
conda env create -f ../../environment.gpu.yaml  # or environment.cpu.yaml
conda activate bagseek-gpu
flask run --debug  # Dev server on port 5000
```

### Preprocessing
```bash
python preprocessing/main.py  # Run full 6-step pipeline
```

### Docker
```bash
docker-compose up --build  # Backend :5000, Frontend :3000
```

## Architecture

### Data Flow
```
ROS Bags → Preprocessing Pipeline → Backend (CLIP/FAISS index) → Frontend (query/visualize)
```

### Preprocessing Pipeline (preprocessing/)
Six sequential steps using abstract base classes in `preprocessing/abstract/`:
1. **TopicsExtractionProcessor** - Extract topics/message types from rosbags
2. **TimestampAlignmentProcessor** - Build timestamp lookup tables
3. **PositionalLookupProcessor** - Create spatial grid index
4. **ImageTopicPreviewsProcessor** - Extract preview images
5. **EmbeddingsProcessor** - Generate CLIP embeddings (sharded Parquet)
6. **AdjacentSimilaritiesPostprocessor** - Compute frame similarities

Processor base classes:
- `RosbagProcessor` - Once per rosbag
- `McapProcessor` - Per MCAP file within rosbag
- `HybridProcessor` - Pre-rosbag + per-MCAP + post-rosbag phases
- `PostProcessor` - Final aggregation across all rosbags

### Key Files
- `flask-backend/api/api.py` - Flask app entry point (44+ routes across 9 blueprint files)
- `react-frontend/src/App.tsx` - Main UI with routing
- `preprocessing/main.py` - Pipeline orchestrator
- `preprocessing/config.py` - Configuration via .env

### Frontend Components (react-frontend/src/components/)
- **SplittableCanvas** - Resizable panel manager
- **NodeContent** - Renders topic content (images, pointclouds, maps)
- **GlobalSearch** - CLIP query interface
- **TimestampPlayer** - Timeline navigation
- **PositionalOverview** - Leaflet map with heatmaps
- **Export** - Time range + topic selection for MCAP export
- **HeatBar** - Timeline similarity/density visualization
- **Header** - Rosbag/model selector, canvas management
- **Login** - JWT auth gate (shown when APP_PASSWORD is set)

### Backend Endpoints (key ones)
- `GET /api/search` - Semantic search with CLIP + FAISS (streaming results)
- `GET /api/search-by-image` - Image-based similarity search
- `GET /api/content-mcap` - Fetch content by timestamp/topic
- `POST /api/export-rosbag` - Export data subset
- `GET /api/get-available-topics` - Topics in current rosbag
- `GET /api/cancel-search` - Cancel in-progress search
- `POST /api/login` / `POST /api/logout` - Optional JWT auth

## Configuration

Create `.env` in project root with required paths:
```
BASE=/path/to/output/base
ROSBAGS=/path/to/rosbags
PRESELECTED_ROSBAG=/path/to/default/rosbag
PRESELECTED_MODEL=ViT-B-16-quickgelu__openai
OPEN_CLIP_MODELS=/path/to/openclip_cache
OTHER_MODELS=/path/to/custom/models
```

Remaining paths are relative to BASE (see README.md for full template).

## Supported CLIP Models

- OpenAI: `ViT-B-16-quickgelu`
- LAION2B: `ViT-H-14`, `ViT-bigG-14`
- Custom agricultural models (via OTHER_MODELS)

## Notes

- Linux only in Docker (installs `ros-humble-*` system packages for ROS message deserialization); the Python stack (`mcap`, `mcap-ros2-support`) is cross-platform via pip, but removing the Docker ROS dependency requires a pip-only message deserialization path
- GPU optional but recommended for CLIP inference
- Frontend proxies API calls to localhost:5000
- Embeddings stored as sharded Parquet (100K rows per shard)
- CompletionTracker enables resumable preprocessing
- Auth is optional: set `APP_PASSWORD` in `.env` to enable JWT cookie auth (1-hour expiry)
