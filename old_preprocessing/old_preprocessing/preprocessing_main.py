#!/mnt/data/bagseek/flask-backend/api/venv/bin/python3

# main_preprocessing.py

import subprocess
import os

WORKING_DIR = "/mnt/data/bagseek/preprocessing"

# Lists preprocessing scripts to run in order
scripts = [
    #"01_generate_alignment_and_metadata.py",
    #"02A_extract_images.py",
    #"02B_create_representative_images.py",
    "03_preprocess_images.py",
    "04A_create_open_clip_embeddings.py",
    "04B_create_not_open_clip_embeddings.py"
    "05_create_faiss_indices.py"
]

# Runs each script and prints output, aborts on error
def run_scripts():
    for script in scripts:
        print(f"Running: {script}")
        result = subprocess.run(["python3", os.path.join(WORKING_DIR, script)], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error while running {script}:")
            print(result.stderr)
            break

# Entry point for executing the full preprocessing pipeline
if __name__ == "__main__":
    run_scripts()