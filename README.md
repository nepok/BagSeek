# 📦 BagSeek: Semantic Exploration of ROS Data

**BagSeek** is an interactive tool for semantically exploring large-scale ROS 2 bag files. It allows users to search and filter image data using natural language queries powered by CLIP-based embeddings and FAISS indexing and export meaningful subsets of the data.

## 🧭 Key Features

- Panel-based layout for synchronized visualization of image, pointcloud, positional, imu and metadata streams
- Semantic image search via natural language prompts (CLIP + FAISS)
- Model selection from multiple OpenCLIP variants (e.g. `ViT-B-16` trained by [*OpenAI*](https://openai.com/index/clip/), `ViT-H-14` trained with the [*LAION2B*](https://laion.ai/blog/laion-5b/) Dataset)
- Export of filtered Rosbag segments based on timespans, topic, or data type
- Color-coded layout configuration management
- Offline-ready, privacy-preserving system

## 📸 Example Use Case

A user analyzing agricultural robotics data can:

1. Select a ROS 2 bag file from local storage.
2. Configure a custom panel-layout for exploring multimodal topic data.
3. Enter a query like "cow on the field" or "tree stump", retrieve matching frames and view their distribution across time.
4. Export relevant data slices (e.g. images containing objects of interest).

## 🚀 Installation

### Prerequisites

⚠️ **Disclaimer**: This install couldn't be tested properly before release, as it only works on linux.

-	[Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install)
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/en/download) (v18+ recommended)
- [ROS 2](https://docs.ros.org/en/humble/Installation.html) (tested with Humble)
- (Optional) GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) for faster inference

### Backend Setup (Flask + Python Environment)

```bash
# Clone the repo
git clone https://github.com/nepok/BagSeek.git
cd bagseek/flask-backend/api

# Create and activate Conda environment with all dependencies
conda env create -f environment.yaml
conda activate bagseek-gpu

# Start the backend server
flask run --debug
```

### Frontend Setup (React + TypeScript)

```bash
cd bagseek/react-frontend
npm install
npm start
```

### Project Structure

```bash
bagseek/
├── flask-backend/                # Python backend (Flask API, CLIP, indexing, etc.)
│   ├── api/                      # Contains api.py and route definitions
│   │   └── api.py
│   ├── src/                      # necessary preprocessing data: extracted images, embeddings, indices, ...
│   │   ├── extracted_images/
│   │   ├── embeddings/
│   │   ├── faiss_indices/
│   │   └── ...
│   └── preprocessing/            # Standalone preprocessing scripts
│       ├── ...
│
├── react-frontend/               # React frontend (TypeScript, React)
│   ├── node_modules/
│   ├── public/
│   └── src/                      # Frontend logic and UI components
│       ├── components/
│       ├── App.tsx
│       ├── index.tsx
│       └── ...
│
├── rosbags/                      # Local or mounted ROS 2 bag files
└── README.md
```

## 🧪 Evaluation Summary

BagSeek was developed as part of a Bachelor's thesis on semantic image retrieval in agricultural robotics. Six pre-trained CLIP models were evaluated on real-world agricultural scenarios. The results are compared in the following summary plots to assess differences in retrieval performance.

## ✨ Credits

Developed by **Nepomuk Kindermann**.  
Powered by [ROS](https://www.ros.org/), [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss), Flask, and React.  
Developed with support from the **Smart Farming Lab** at the **University of Leipzig**.  

## 📄 License

This project is licensed under the MIT License. See `LICENSE.md` for details.
