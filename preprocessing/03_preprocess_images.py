import os
import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import open_clip
try:
    from open_clip.transform import image_transform
except ImportError:  # pragma: no cover - fallback for older open_clip versions
    image_transform = None
import re
from dotenv import load_dotenv


# Silence noisy dependency warnings that do not impact preprocessing.
warnings.filterwarnings(
    "ignore",
    message=r".*Failed to load image Python extension.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Unable to find acceptable character detection dependency.*",
)

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define constants for paths
BASE_DIR = os.getenv("BASE_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
PREPROCESSED_DIR = os.getenv("PREPROCESSED_DIR")
LOCAL_WORK_DIR = os.getenv("LOCAL_WORK_DIR", "/mnt/data/tmp/preprocess")

model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]

# Create output directory if it doesn't exist
Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

preprocess_dict: Dict[str, List[str]] = {}
_PREPROCESS_CACHE: Dict[Tuple[str, str], object] = {}
_WORKER_PREPROCESS = None

# Collect unique preprocessing transforms and associate them with model identifiers
def collect_preprocess_data(model_name: str, pretrained_name: str, cache_dir: str | None = None) -> None:
    preprocess = get_preprocess_transform(model_name, pretrained_name, cache_dir)
    preprocess_str = str(preprocess)
    model_id = f"{model_name} ({pretrained_name})"

    if preprocess_str not in preprocess_dict:
        preprocess_dict[preprocess_str] = []
    if model_id not in preprocess_dict[preprocess_str]:
        preprocess_dict[preprocess_str].append(model_id)

# Generate a concise identifier string summarizing the preprocessing steps
def get_preprocess_id(preprocess_str: str) -> str:
    summary_parts = []

    # Match Resize(size=224, ...)
    resize_match = re.search(r"Resize\(size=(\d+)", preprocess_str)
    if resize_match:
        summary_parts.append(f"resize{resize_match.group(1)}")

    # Match CenterCrop(size=(224, 224))
    crop_match = re.search(r"CenterCrop\(size=\(?(\d+)", preprocess_str)
    if crop_match:
        summary_parts.append(f"crop{crop_match.group(1)}")

    # Match ToTensor() (as a function call or object)
    if "ToTensor" in preprocess_str:
        summary_parts.append("tensor")

    # Match Normalize(...), regardless of float content
    if "Normalize" in preprocess_str:
        summary_parts.append("norm")

    # Summary name
    summary = "_".join(summary_parts)
  
    return f"{summary}"


def get_preprocess_transform(model_name: str, pretrained_name: str, cache_dir: str | None = None):
    """Fetch and cache the preprocessing transform for a given model/pretrained pair."""
    key = (model_name, pretrained_name)
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key]

    cache_dir = cache_dir or "/mnt/data/openclip_cache"

    preprocess = None
    try:
        cfg = open_clip.get_preprocess_cfg(pretrained_name)
        if image_transform is None:
            raise RuntimeError("open_clip.transform.image_transform unavailable")
        preprocess = image_transform(cfg, is_train=False)
    except Exception:
        # Fallback to full model creation if direct cfg access fails.
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            cache_dir=cache_dir,
        )

    _PREPROCESS_CACHE[key] = preprocess
    return preprocess


def _init_worker(model_name: str, pretrained_name: str, cache_dir: str | None):
    """Initializer for worker processes to build the shared preprocess pipeline once."""
    global _WORKER_PREPROCESS
    torch.set_num_threads(1)
    _WORKER_PREPROCESS = get_preprocess_transform(model_name, pretrained_name, cache_dir)


def _process_single_image(task: Tuple[str, str]) -> Tuple[str, bool, str | None]:
    """Process a single image path pair (input, output).

    Returns a tuple describing (output_path, processed?, error message).
    """
    input_path, output_path = task

    if _WORKER_PREPROCESS is None:  # pragma: no cover - defensive check
        return output_path, False, "Preprocess pipeline not initialized"

    if os.path.exists(output_path):
        return output_path, False, None

    try:
        with Image.open(input_path) as image:
            image = image.convert("RGB")
            image_tensor = _WORKER_PREPROCESS(image)
        torch.save(image_tensor, output_path)
        return output_path, True, None
    except Exception as exc:  # pragma: no cover - best-effort logging
        return output_path, False, str(exc)

def preprocess_images(
    input_dir: str,
    output_dir: str,
    model_name: str,
    pretrained_name: str,
    max_workers: int | None = None,
    cache_dir: str | None = None,
    chunksize: int = 64,
    stage_locally: bool = True,
):
    """
    Process and cache preprocessed tensors for a given image directory.

    Changes:
      - NAS-first scan: decide what needs processing before any staging.
      - Skip staging entirely if everything is up to date.
      - Sparse staging: copy only the needed source images to the local temp.
      - Verbose prints throughout.
    """
    print(f"[preprocess] ==============================")
    print(f"[preprocess] Input dir : {input_dir}  (likely NAS)")
    print(f"[preprocess] Output dir: {output_dir}  (final destination)")
    print(f"[preprocess] Model     : {model_name} ({pretrained_name})")
    print(f"[preprocess] Staging   : {'ON' if stage_locally else 'OFF'}")
    print(f"[preprocess] ===============================================")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1: NAS-first scan (no staging yet) ----------
    print("[preprocess] Scanning NAS for image files and up-to-date outputs...")
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

    # Build tasks against the FINAL output_dir (not staging).
    tasks: List[Tuple[str, str]] = []  # (abs_input_path_on_nas, abs_final_output_path)
    directories_to_create: set[str] = set()

    total_images = 0
    existing_outputs = 0
    up_to_date_outputs = 0
    stale_outputs = 0
    missing_outputs = 0

    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    for root, _, files in os.walk(input_dir_path):
        for file in files:
            if not file.lower().endswith(img_exts):
                continue

            total_images += 1
            input_file_path = Path(root) / file
            relative_dir = Path(root).relative_to(input_dir_path)
            output_file_dir = output_dir_path / relative_dir

            base_name = os.path.splitext(file)[0]
            output_filename = f"{base_name}.pt"
            output_file_path = output_file_dir / output_filename

            if output_file_path.exists():
                existing_outputs += 1
                try:
                    in_mtime = input_file_path.stat().st_mtime
                    out_mtime = output_file_path.stat().st_mtime
                    if out_mtime >= in_mtime:
                        up_to_date_outputs += 1
                        continue  # no work for this file
                    else:
                        stale_outputs += 1
                        directories_to_create.add(str(output_file_dir))
                        tasks.append((str(input_file_path), str(output_file_path)))
                except OSError as e:
                    print(f"[preprocess] Warning: stat failed ({e}); will reprocess: {input_file_path}")
                    directories_to_create.add(str(output_file_dir))
                    tasks.append((str(input_file_path), str(output_file_path)))
            else:
                missing_outputs += 1
                directories_to_create.add(str(output_file_dir))
                tasks.append((str(input_file_path), str(output_file_path)))

    print("[preprocess] NAS scan complete.")
    print(f"[preprocess]   Images found         : {total_images}")
    print(f"[preprocess]   Existing outputs     : {existing_outputs}")
    print(f"[preprocess]   Up-to-date (skip)    : {up_to_date_outputs}")
    print(f"[preprocess]   Stale (reprocess)    : {stale_outputs}")
    print(f"[preprocess]   Missing (to process) : {missing_outputs}")
    print(f"[preprocess]   Enqueued tasks       : {len(tasks)}")

    # Nothing to do? Exit early before any staging.
    if not tasks:
        print("[preprocess] Nothing to do: all outputs are present and up-to-date.")
        print("[preprocess] ✅ Done.")
        return

    # Ensure final output directories exist (even if we stage first).
    print("[preprocess] Creating required final output directories (if missing)...")
    for directory in sorted(directories_to_create):
        Path(directory).mkdir(parents=True, exist_ok=True)

    # ---------- Phase 2: Decide processing location ----------
    # We either:
    #  - process directly from NAS -> final output (no staging), or
    #  - stage ONLY the needed input files, process locally, then move outputs.
    working_input = None
    working_output = None
    temp_input_dir: Path | None = None
    temp_output_dir: Path | None = None

    if stage_locally:
        # Sparse staging: copy only the input files we actually need.
        staging_root = Path(LOCAL_WORK_DIR)
        bag_name = Path(input_dir).name
        temp_input_dir = staging_root / f"{bag_name}__input_sparse"
        temp_output_dir = staging_root / f"{bag_name}__output"

        print(f"[preprocess] Preparing sparse local staging at: {staging_root}")
        for p in (temp_input_dir, temp_output_dir):
            if p.exists():
                print(f"[preprocess] Removing stale staging dir: {p}")
                shutil.rmtree(p, ignore_errors=True)
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[preprocess] Sparse copy: staging only {len(tasks)} needed source files...")
        for src_path_str, _dst_out_str in tqdm(tasks, desc="[preprocess] Sparse staging inputs"):
            src_path = Path(src_path_str)
            rel = src_path.parent.relative_to(input_dir_path)  # keep subdir structure
            tgt_dir = temp_input_dir / rel
            tgt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, tgt_dir / src_path.name)

        print(f"[preprocess] Sparse staging complete.")

        working_input = temp_input_dir
        working_output = temp_output_dir

        # Rewrite tasks to point to staged inputs and staged outputs
        # (we’ll move staged outputs -> final at the end).
        staged_tasks: List[Tuple[str, str]] = []
        for src_path_str, _dst_out_str in tasks:
            src_path = Path(src_path_str)
            rel_dir = src_path.parent.relative_to(input_dir_path)
            staged_src = working_input / rel_dir / src_path.name

            base_name = src_path.stem
            staged_out_dir = working_output / rel_dir
            staged_out_dir.mkdir(parents=True, exist_ok=True)
            staged_out = staged_out_dir / f"{base_name}.pt"

            staged_tasks.append((str(staged_src), str(staged_out)))

        tasks = staged_tasks
        print("[preprocess] Tasks have been rewritten to staged paths.")
    else:
        # Process straight from NAS and write directly to final output_dir.
        working_input = Path(input_dir)   # used for tqdm label only
        working_output = Path(output_dir) # writes go directly here
        print("[preprocess] Processing directly from NAS to final outputs (no staging).")

    # ---------- Phase 3: Process ----------
    # (At this point, `tasks` refers either to staged inputs+outputs or NAS->final)
    # Reduce library thread over-subscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    max_workers = max_workers or (os.cpu_count() or 1)
    print(f"[preprocess] Launching workers: {max_workers} (chunksize={chunksize})")
    mp_context = mp.get_context("spawn")

    processed_count = 0
    errors: List[Tuple[str, str]] = []

    from tqdm import tqdm as _tqdm

    print("[preprocess] Starting parallel processing...")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(model_name, pretrained_name, cache_dir),
    ) as executor:
        for output_path, _did_process, error in _tqdm(
            executor.map(_process_single_image, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=f"Processing images for {Path(working_input).name}",
        ):
            if error:
                errors.append((output_path, error))
            else:
                processed_count += 1

    if errors:
        print(f"[preprocess] ⚠️  Encountered {len(errors)} errors while preprocessing {input_dir}:")
        for output_path, error in errors[:5]:
            print(f"[preprocess]   - {output_path}: {error}")
        if len(errors) > 5:
            print(f"[preprocess]   ... and {len(errors) - 5} more")
    else:
        print("[preprocess] No processing errors reported.")

    # ---------- Phase 4: Move staged outputs -> final ----------
    moved_files = 0
    if stage_locally and temp_output_dir:
        print(f"[preprocess] Moving staged outputs to final destination: {output_dir}")
        for src_dir, _, files in os.walk(temp_output_dir):
            relative_dir = Path(src_dir).relative_to(temp_output_dir)
            target_dir = output_dir_path / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for file in tqdm(files, desc="[preprocess] Moving files"):
                shutil.move(str(Path(src_dir) / file), str(target_dir / file))
                moved_files += 1

        print("[preprocess] Cleaning staging directories...")
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

    # ---------- Final summary ----------
    print("[preprocess] ===============================================")
    print(f"[preprocess] Summary for {input_dir}:")
    print(f"[preprocess]   Total images           : {total_images}")
    print(f"[preprocess]   Up-to-date (skipped)   : {up_to_date_outputs}")
    print(f"[preprocess]   Enqueued tasks         : {len(tasks)}")
    print(f"[preprocess]   Successfully processed : {processed_count}")
    print(f"[preprocess]   Errors                 : {len(errors)}")
    if stage_locally:
        print(f"[preprocess]   Staged input files     : {processed_count if moved_files == 0 else moved_files}")
        print(f"[preprocess]   Moved from stage       : {moved_files}")
    print("[preprocess] ✅ Done.")

# Main function to coordinate preprocessing across all model configurations
def main():
    print("Starting image preprocessing...")
    print("Collecting preprocessing configurations...")
    for model_name, pretrained_name in model_configs:
        collect_preprocess_data(model_name, pretrained_name)

    # Only proceed if all models share the same preprocessing function
    if len(preprocess_dict) == 1:
        preprocess_str, models = next(iter(preprocess_dict.items()))
        shared_model_name, shared_pretrained_name = models[0].split(" (")
        shared_pretrained_name = shared_pretrained_name.rstrip(")")

        print(f"All models share the same preprocess. Proceeding with: {shared_model_name} ({shared_pretrained_name})")

        preprocess_id = get_preprocess_id(preprocess_str)
        print(f"Preprocess ID: {preprocess_id}")

        cache_dir = "/mnt/data/openclip_cache"

        for image_folder in tqdm(os.listdir(IMAGES_PER_TOPIC_DIR), desc="Processing image folders"):
            image_folder_path = os.path.join(IMAGES_PER_TOPIC_DIR, image_folder)
            if os.path.isdir(image_folder_path):
                output_folder_path = os.path.join(PREPROCESSED_DIR, preprocess_id, image_folder)
                preprocess_images(
                    image_folder_path,
                    output_folder_path,
                    shared_model_name,
                    shared_pretrained_name,
                    cache_dir=cache_dir,
                    max_workers=32,
                    chunksize=128,
                    stage_locally=True,
                )
    else:
        print("Warning: Not all model_configs share the same preprocessing function. Please check manually.")


if __name__ == "__main__":
    main()
