#!/mnt/data/bagseek/flask-backend/api/venv/bin/python3

# main_preprocessing.py

import subprocess

# Liste deiner Preprocessing-Skripte
scripts = [
    "01_convert_ROS_to_CSV.py",
    "02_extract_images.py",
    "03_preprocess_images.py",
    "04_create_clip_embeddings.py",
    "05_create_faiss_index.py"
]

def run_scripts():
    for script in scripts:
        print(f"Running: {script}")
        result = subprocess.run(["python3", script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error while running {script}:")
            print(result.stderr)
            break

if __name__ == "__main__":
    run_scripts()