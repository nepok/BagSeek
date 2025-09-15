from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
import os
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parent.parent / ".env"
print(PARENT_ENV)

load_dotenv(dotenv_path=PARENT_ENV)

AUTO_ANNOTATION_DIR = os.getenv("AUTO_ANNOTATION_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")

SELECTED_ROSBAG = "output_bag"
SELECTED_ROSBAG_IMAGE_PATH = os.path.join(IMAGES_PER_TOPIC_DIR, SELECTED_ROSBAG)

CSV_OUTPUT_PATH = os.path.join(AUTO_ANNOTATION_DIR, f"{SELECTED_ROSBAG}.csv")

print("CSV will be written to:", CSV_OUTPUT_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    cache_dir="/mnt/data/bagseek/preprocessing/model_cache/BLIP",
    quantization_config=quantization_config, 
    device_map={"": 0}, 
    torch_dtype=torch.float16,
)

def process_image(img_path, writer):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text_no = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    prompt = "Question: What do you see in the picture?. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    full_simple = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text_simple = full_simple.replace(prompt, "").strip()

    #prompt = "Question: Explain everything you see in at least three sentences. Answer:"
    prompt = "This image is recorded by a camera on top of a tractor and you can see agricultural or safety critical objects like"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    full_detailed = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text_detailed = full_detailed.replace(prompt, "").strip()

    writer.writerow([img_path, generated_text_no, generated_text_simple, generated_text_detailed])


def main():
    # open CSV once (append mode) and load already processed image paths
    file_exists = os.path.exists(CSV_OUTPUT_PATH)
    csvfile = open(CSV_OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Image Path", "No prompt", "Question: What do you see in the picture?", "This image is recorded by a camera on top of a tractor and you can see agricultural or safety critical objects like"])

    processed = set()
    if file_exists:
        with open(CSV_OUTPUT_PATH, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                processed.add(row[0])

    for topic_path in os.listdir(SELECTED_ROSBAG_IMAGE_PATH):
        whole_path = os.path.join(SELECTED_ROSBAG_IMAGE_PATH, topic_path)
        image_paths = [os.path.join(whole_path, f) for f in os.listdir(whole_path) if f.endswith((".webp", ".jpg", ".png"))]
        to_process = [p for p in image_paths if p not in processed]
        if not to_process:
            continue
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(tqdm(executor.map(lambda p: process_image(p, writer), to_process), total=len(to_process)))

    csvfile.close()


if __name__ == '__main__':
    main()