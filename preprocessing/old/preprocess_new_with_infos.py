import os
import warnings
import shutil
import platform
import time
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


# ========= Helper printing & timing =========

VERBOSE = True

def _hb(n: int) -> str:
    """Human bytes."""
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} EB"

class _Timer:
    def __init__(self, label: str):
        self.label = label
        self.start = None
    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[⏳] {self.label} ...")
        return self
    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self.start
        print(f"[✅] {self.label} done in {dur:.2f}s")

def _info(msg: str) -> None:
    if VERBOSE:
        print(f"[INFO] {msg}")

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


# ========= Silence noisy warnings =========
warnings.filterwarnings(
    "ignore",
    message=r".*Failed to load image Python extension.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Unable to find acceptable character detection dependency.*",
)

# ========= Env handling =========
PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
if PARENT_ENV.exists():
    load_dotenv(dotenv_path=PARENT_ENV)
    _info(f"Loaded .env from {PARENT_ENV}")
else:
    _warn(f"No .env found at {PARENT_ENV}; relying on process env.")

# Define constants for paths
BASE_DIR = os.getenv("BASE_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
PREPROCESSED_DIR = os.getenv("PREPROCESSED_DIR")
LOCAL_WORK_DIR = os.getenv("LOCAL_WORK_DIR", "/mnt/data/tmp/preprocess")

# Show environment summary
_info(
    "Environment summary:\n"
    f"  - HOST: {platform.node()} | OS: {platform.system()} {platform.release()} | Py: {platform.python_version()}\n"
    f"  - Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}\n"
    f"  - BASE_DIR={BASE_DIR}\n"
    f"  - IMAGES_PER_TOPIC_DIR={IMAGES_PER_TOPIC_DIR}\n"
    f"  - PREPROCESSED_DIR={PREPROCESSED_DIR}\n"
    f"  - LOCAL_WORK_DIR={LOCAL_WORK_DIR}"
)

if not IMAGES_PER_TOPIC_DIR or not PREPROCESSED_DIR:
    _err("IMAGES_PER_TOPIC_DIR and PREPROCESSED_DIR must be set in your environment.")
    # Don't exit hard; allow importers to still use parts of this module.

# ========= Models =========
model_configs = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('ViT-L-14-quickgelu', 'openai'),
    ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')
]
_info(f"Configured {len(model_configs)} model(s) for preprocess comparison.")

# Create output directory if it doesn't exist
if PREPROCESSED_DIR:
    Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

preprocess_dict: Dict[str, List[str]] = {}
_PREPROCESS_CACHE: Dict[Tuple[str, str], object] = {}
_WORKER_PREPROCESS = None


# ========= Preprocess collection & ID =========
def collect_preprocess_data(model_name: str, pretrained_name: str, cache_dir: str | None = None) -> None:
    with _Timer(f"Collect preprocess for {model_name} ({pretrained_name})"):
        preprocess = get_preprocess_transform(model_name, pretrained_name, cache_dir)
    preprocess_str = str(preprocess)
    model_id = f"{model_name} ({pretrained_name})"

    if preprocess_str not in preprocess_dict:
        preprocess_dict[preprocess_str] = []
    if model_id not in preprocess_dict[preprocess_str]:
        preprocess_dict[preprocess_str].append(model_id)

    _info(f"  ↳ Preprocess registered: {model_id}")

def get_preprocess_id(preprocess_str: str) -> str:
    summary_parts = []

    resize_match = re.search(r"Resize\(size=(\d+)", preprocess_str)
    if resize_match:
        summary_parts.append(f"resize{resize_match.group(1)}")

    crop_match = re.search(r"CenterCrop\(size=\(?(\d+)", preprocess_str)
    if crop_match:
        summary_parts.append(f"crop{crop_match.group(1)}")

    if "ToTensor" in preprocess_str:
        summary_parts.append("tensor")

    if "Normalize" in preprocess_str:
        summary_parts.append("norm")

    summary = "_".join(summary_parts) or "unknown_preprocess"
    return f"{summary}"


def get_preprocess_transform(model_name: str, pretrained_name: str, cache_dir: str | None = None):
    """Fetch and cache the preprocessing transform for a given model/pretrained pair."""
    key = (model_name, pretrained_name)
    if key in _PREPROCESS_CACHE:
        _info(f"Using cached preprocess for {model_name} ({pretrained_name})")
        return _PREPROCESS_CACHE[key]

    cache_dir = cache_dir or "/mnt/data/openclip_cache"

    preprocess = None
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name,
        cache_dir=cache_dir,
    )
    _info(f"Preprocess via create_model_and_transforms for {model_name} ({pretrained_name})")

    _PREPROCESS_CACHE[key] = preprocess
    return preprocess


def _init_worker(model_name: str, pretrained_name: str, cache_dir: str | None):
    """Initializer for worker processes to build the shared preprocess pipeline once."""
    global _WORKER_PREPROCESS
    torch.set_num_threads(1)
    _WORKER_PREPROCESS = get_preprocess_transform(model_name, pretrained_name, cache_dir)
    _info(f"[Worker {os.getpid()}] Preprocess ready for {model_name} ({pretrained_name})")


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


def _dir_disk_usage(path: Path) -> tuple[int, int, int]:
    """Return (total, used, free) for the filesystem containing path."""
    try:
        du = shutil.disk_usage(path)
        return du.total, du.used, du.free
    except Exception:
        return (0, 0, 0)


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
    """Process and cache preprocessed tensors for a given image directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    working_input = Path(input_dir)
    working_output = Path(output_dir)
    temp_input_dir: Path | None = None
    temp_output_dir: Path | None = None

    _info(
        f"Preprocess start:\n"
        f"  - input_dir={input_dir}\n"
        f"  - output_dir={output_dir}\n"
        f"  - model={model_name} ({pretrained_name})\n"
        f"  - stage_locally={stage_locally}"
    )

    if stage_locally:
        staging_root = Path(LOCAL_WORK_DIR)
        staging_root.mkdir(parents=True, exist_ok=True)
        bag_name = Path(input_dir).name
        temp_input_dir = staging_root / f"{bag_name}__input"
        temp_output_dir = staging_root / f"{bag_name}__output"

        # Disk status before staging
        total, used, free = _dir_disk_usage(staging_root)
        _info(f"Staging FS disk: total={_hb(total)} used={_hb(used)} free={_hb(free)} @ {staging_root}")

        # Fresh staging dirs
        if temp_input_dir.exists():
            shutil.rmtree(temp_input_dir)
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

        with _Timer(f"Copy images into local staging {temp_input_dir}"):
            shutil.copytree(input_dir, temp_input_dir)

        temp_output_dir.mkdir(parents=True, exist_ok=True)

        working_input = temp_input_dir
        working_output = temp_output_dir

    # Build task list
    tasks: List[Tuple[str, str]] = []
    directories_to_create: set[str] = set()

    # Counters
    total_files_scanned = 0
    eligible_images = 0
    already_present = 0

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    with _Timer(f"Scan & index tasks in {working_input}"):
        for root, _, files in os.walk(working_input):
            for file in files:
                total_files_scanned += 1
                if not file.lower().endswith(img_exts):
                    continue

                eligible_images += 1

                input_file_path = os.path.join(root, file)
                relative_dir = os.path.relpath(root, working_input)
                output_file_dir = os.path.join(working_output, relative_dir)

                base_name = os.path.splitext(file)[0]
                output_filename = f"{base_name}.pt"
                output_file_path = os.path.join(output_file_dir, output_filename)

                if os.path.exists(output_file_path):
                    already_present += 1
                    continue

                directories_to_create.add(output_file_dir)
                tasks.append((input_file_path, output_file_path))

    _info(
        "Task summary:\n"
        f"  - Total files scanned: {total_files_scanned}\n"
        f"  - Eligible images: {eligible_images}\n"
        f"  - Already preprocessed (skipped): {already_present}\n"
        f"  - To process now: {len(tasks)}"
    )

    # Pre-create output dirs (staged or final)
    for directory in directories_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)

    if not tasks:
        _info("No work to do. All outputs already present.")
        if stage_locally and temp_output_dir and temp_input_dir:
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        return

    # Worker config
    max_workers = max_workers or (os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _info(
        "Execution config:\n"
        f"  - max_workers={max_workers}\n"
        f"  - chunksize={chunksize}\n"
        f"  - OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n"
        f"  - MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}\n"
        f"  - OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}"
    )

    mp_context = mp.get_context("spawn")

    processed_count = 0
    errors: List[Tuple[str, str]] = []

    with _Timer(f"Multiprocessing preprocess ({len(tasks)} images)"):
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(model_name, pretrained_name, cache_dir),
        ) as executor:
            for output_path, did_process, error in tqdm(
                executor.map(_process_single_image, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Processing images for {Path(input_dir).name}",
            ):
                if did_process:
                    processed_count += 1
                if error:
                    errors.append((output_path, error))

    # Report
    _info(
        "Batch result:\n"
        f"  - Newly processed: {processed_count}\n"
        f"  - Failed: {len(errors)}\n"
        f"  - Skipped (already present): {already_present}"
    )

    if processed_count > 0:
        # Crude throughput estimate (files/s) would require timing within scope; kept by _Timer.
        pass

    if errors:
        _warn(f"Encountered {len(errors)} errors while preprocessing {input_dir}:")
        for output_path, error in errors[:10]:
            _warn(f"  - {output_path}: {error}")
        if len(errors) > 10:
            _warn(f"  ... and {len(errors) - 10} more")

    # Move staged outputs back to final location
    if stage_locally and temp_output_dir and temp_input_dir:
        with _Timer(f"Move staged outputs → {output_dir}"):
            moved = 0
            for src_dir, _, files in os.walk(temp_output_dir):
                relative_dir = os.path.relpath(src_dir, temp_output_dir)
                target_dir = Path(output_dir) / relative_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                for file in files:
                    shutil.move(os.path.join(src_dir, file), os.path.join(target_dir, file))
                    moved += 1
            _info(f"  - Files moved: {moved}")

        # Cleanup staging 
        with _Timer("Cleanup staging directories"):
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)


# ========= Main driver =========
def main():
    if not IMAGES_PER_TOPIC_DIR or not PREPROCESSED_DIR:
        _err("Missing required env vars. Aborting main().")
        return

    # High-level hardware summary
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            _info(f"GPU[{i}]: {name} | capability={cap}")
    else:
        _warn("No CUDA GPU detected. Preprocessing runs on CPU.")

    # Model preprocess grouping
    _info("Collecting preprocess pipelines for configured models...")
    for model_name, pretrained_name in model_configs:
        collect_preprocess_data(model_name, pretrained_name)

    _info("Preprocess groups:")
    for idx, (pp_str, models) in enumerate(preprocess_dict.items(), start=1):
        _info(f"  Group {idx}: {len(models)} model(s)")
        for m in models:
            _info(f"    - {m}")
        pp_id = get_preprocess_id(pp_str)
        _info(f"    → ID: {pp_id}")
        _info(f"    → Raw: {pp_str}")

    # Only proceed if all models share the same preprocessing function
    if len(preprocess_dict) == 1:
        preprocess_str, models = next(iter(preprocess_dict.items()))
        shared_model_name, shared_pretrained_name = models[0].split(" (")
        shared_pretrained_name = shared_pretrained_name.rstrip(")")

        _info(f"Unified preprocess detected across all models.")
        print(f"[RUN] Using preprocess of: {shared_model_name} ({shared_pretrained_name})")
        preprocess_id = get_preprocess_id(preprocess_str)
        print(f"[RUN] Preprocess ID: {preprocess_id}")

        cache_dir = "/mnt/data/openclip_cache"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _info(f"open_clip cache dir: {cache_dir}")

        topic_dirs = [d for d in os.listdir(IMAGES_PER_TOPIC_DIR) if os.path.isdir(os.path.join(IMAGES_PER_TOPIC_DIR, d))]
        _info(f"Found {len(topic_dirs)} image folder(s) in {IMAGES_PER_TOPIC_DIR}")

        with _Timer("Full preprocessing pass"):
            for image_folder in tqdm(topic_dirs, desc="Processing image folders"):
                image_folder_path = os.path.join(IMAGES_PER_TOPIC_DIR, image_folder)
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
        _info("All folders processed.")
    else:
        _warn("Not all model_configs share the same preprocessing function. Please check the groups above and decide how to proceed.")


if __name__ == "__main__":
    main()
