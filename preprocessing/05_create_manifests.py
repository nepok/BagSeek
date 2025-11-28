#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import torch
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# ====== Pfade ======
def _require_env_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable {var_name} is not set.")
    return Path(value).expanduser()


EMBEDDINGS_PER_TOPIC_DIR = _require_env_path("EMBEDDINGS_PER_TOPIC_DIR")
EMBEDDINGS_PER_TOPIC_CONSOLIDATED_DIR = EMBEDDINGS_PER_TOPIC_DIR
OUTPUT_ROOT = _require_env_path("EMBEDDINGS_CONSOLIDATED_DIR")  # Path("/mnt/data/bagseek/flask-backend/src/embeddings_consolidated")

# ====== Parameter ======
DEFAULT_SHARD_ROWS = 100_000
OUTPUT_DTYPE = np.float32  # keep as stored

# ====== Regex: leading digits in stem (supports 123.pt and 123_embedding.pt) ======
_TS_LEADING_RE = re.compile(r"^(\d+)")

# ====== Utils ======
def log(msg: str) -> None:
    print(msg, flush=True)

def ns_to_minute_of_day(ts_ns: int) -> int:
    dt = datetime.utcfromtimestamp(ts_ns / 1e9)
    return dt.hour * 60 + dt.minute

def topic_folder_to_topic_name(folder: str) -> str:
    return folder.replace("__", "/")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_meta(meta_path: Path) -> Dict:
    if meta_path.is_file():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

def save_meta(meta_path: Path, meta: Dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def read_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.is_file():
        return pd.read_parquet(manifest_path)
    return pd.DataFrame(columns=[
        "id","topic","timestamp_ns","minute_of_day","shard_id","row_in_shard","pt_path"
    ])

def write_manifest(manifest_path: Path, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df.reset_index(drop=True))
    pq.write_table(table, manifest_path, compression="zstd")

def next_shard_name(shards_dir: Path) -> str:
    existing = sorted(shards_dir.glob("shard-*.npy"))
    if not existing:
        return "shard-00000.npy"
    last = existing[-1].stem  # shard-00012
    try:
        idx = int(last.split("-")[1])
    except Exception:
        idx = len(existing) - 1
    return f"shard-{idx+1:05d}.npy"

def list_models(root: Path, only_models: Optional[List[str]] = None) -> List[str]:
    models = [p.name for p in root.iterdir() if p.is_dir()]
    if only_models:
        models = [m for m in models if m in only_models]
    models.sort()
    return models

def list_rosbags(model_dir: Path, only_rosbags: Optional[List[str]] = None) -> List[str]:
    bags = [p.name for p in model_dir.iterdir() if p.is_dir()]
    if only_rosbags:
        bags = [b for b in bags if b in only_rosbags]
    bags.sort()
    return bags

def _parse_timestamp_ns_from_stem(stem: str) -> Optional[int]:
    m = _TS_LEADING_RE.match(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def collect_new_pt_records(model_dir: Path, rosbag: str, already_seen_paths: set) -> List[Tuple[str, int, int, str]]:
    """
    Recursively scans *.pt under <model>/<rosbag>.
    Topic name = first directory under <rosbag>.
    Supports filenames like <timestamp>.pt and <timestamp>_embedding.pt.
    Returns list of (topic, timestamp_ns, minute_of_day, pt_path).
    """
    rosbag_dir = model_dir / rosbag
    if not rosbag_dir.is_dir():
        log(f"[WARN] missing rosbag dir: {rosbag_dir}")
        return []

    records: List[Tuple[str, int, int, str]] = []
    total_pt = 0

    for pt_file in rosbag_dir.rglob("*.pt"):
        total_pt += 1
        spt = str(pt_file)
        if spt in already_seen_paths:
            continue

        try:
            rel = pt_file.relative_to(rosbag_dir)
        except ValueError:
            rel = pt_file
        parts = rel.parts
        topic_folder = parts[0] if len(parts) >= 2 else "_root"
        topic_name = topic_folder_to_topic_name(topic_folder)

        ts_ns = _parse_timestamp_ns_from_stem(pt_file.stem)
        if ts_ns is None:
            # skip non-timestamp files quietly
            continue

        minute = ns_to_minute_of_day(ts_ns)
        records.append((topic_name, ts_ns, minute, spt))

    records.sort(key=lambda r: (r[0], r[1]))
    log(f"[DBG] {rosbag}: collected {len(records)} new .pt (seen total {total_pt})")
    return records

def load_pt_vector(pt_path: str) -> Optional[np.ndarray]:
    try:
        obj = torch.load(pt_path, map_location="cpu")
        if isinstance(obj, dict):
            if "embedding" in obj:
                obj = obj["embedding"]
            else:
                return None
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    except Exception as e:
        log(f"[WARN] failed to load {pt_path}: {e}")
        return None

def write_shards_incremental(
    out_dir: Path,
    new_records: List[Tuple[str, int, int, str]],
    meta: Dict,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    dtype = OUTPUT_DTYPE,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Writes new_records into one or more shard .npy files under out_dir/'shards',
    updates meta (D, total_count), and returns a DataFrame of manifest rows to append.
    """
    shards_dir = out_dir / "shards"
    ensure_dir(shards_dir)

    # Determine D
    D = int(meta.get("D", 0))
    if D == 0:
        for _, _, _, pt in new_records:
            vec = load_pt_vector(pt)
            if vec is not None:
                D = int(vec.shape[-1])
                break
        if D == 0:
            raise RuntimeError("Could not determine embedding dimension D from new records.")

    # Start-ID
    start_id = int(meta.get("total_count", 0))

    manifest_rows = []
    batch_vecs: List[np.ndarray] = []
    batch_count = 0
    current = 0

    for (topic, ts_ns, minute, pt) in tqdm(new_records, desc="Writing shards", unit="rec"):
        vec = load_pt_vector(pt)
        if vec is None:
            continue
        if vec.shape[-1] != D:
            log(f"[WARN] D mismatch for {pt}: got {vec.shape[-1]}, expected {D}; skipping")
            continue

        v = vec.astype(dtype, copy=False)  # float32
        batch_vecs.append(v)
        batch_count += 1

        if batch_count >= shard_rows:
            shard_name = next_shard_name(shards_dir)
            shard_path = shards_dir / shard_name
            X = np.stack(batch_vecs, axis=0)  # (B, D)
            np.save(shard_path, X)
            for i in range(X.shape[0]):
                manifest_rows.append({
                    "id": start_id + current,
                    "topic": new_records[current][0],
                    "timestamp_ns": new_records[current][1],
                    "minute_of_day": new_records[current][2],
                    "shard_id": shard_name,
                    "row_in_shard": i,
                    "pt_path": new_records[current][3],
                })
                current += 1
            batch_vecs.clear()
            batch_count = 0

    # Flush remainder
    if batch_vecs:
        shard_name = next_shard_name(shards_dir)
        shard_path = shards_dir / shard_name
        X = np.stack(batch_vecs, axis=0)
        np.save(shard_path, X)
        for i in range(X.shape[0]):
            manifest_rows.append({
                "id": start_id + current,
                "topic": new_records[current][0],
                "timestamp_ns": new_records[current][1],
                "minute_of_day": new_records[current][2],
                "shard_id": shard_name,
                "row_in_shard": i,
                "pt_path": new_records[current][3],
            })
            current += 1
        batch_vecs.clear()

    # Update meta
    meta["D"] = D
    meta["dtype"] = "float32"
    meta["total_count"] = start_id + current

    df_append = pd.DataFrame.from_records(manifest_rows)
    return df_append, meta

def process_one_model_rosbag(model_name: str, rosbag: str, force: bool = False, shard_rows: int = DEFAULT_SHARD_ROWS) -> None:
    src_model_dir = EMBEDDINGS_PER_TOPIC_CONSOLIDATED_DIR / model_name
    if not src_model_dir.is_dir():
        return
    src_rosbag_dir = src_model_dir / rosbag
    if not src_rosbag_dir.is_dir():
        return

    out_dir = OUTPUT_ROOT / model_name / rosbag
    ensure_dir(out_dir)
    ensure_dir(out_dir / "shards")

    manifest_path = out_dir / "manifest.parquet"
    meta_path = out_dir / "meta.json"

    manifest = read_manifest(manifest_path)
    meta = load_meta(meta_path)

    already = set(manifest["pt_path"].astype(str).tolist()) if (not force and not manifest.empty) else set()
    new_records = collect_new_pt_records(src_model_dir, rosbag, already_seen_paths=already)

    if not new_records:
        if manifest.empty:
            log(f"[DBG] Manifest is empty; found no .pt files under {src_rosbag_dir}.")
        else:
            log(f"[DBG] Manifest has {len(manifest)} rows; no new .pt beyond existing entries.")
        log(f"[SKIP] {model_name}/{rosbag}: nothing new.")
        return

    log(f"[INFO] {model_name}/{rosbag}: {len(new_records)} new embeddings -> shards")
    df_append, meta = write_shards_incremental(
        out_dir=out_dir,
        new_records=new_records,
        meta=meta,
        shard_rows=shard_rows,
        dtype=OUTPUT_DTYPE,
    )

    if not df_append.empty:
        combined = pd.concat([manifest, df_append], ignore_index=True)
        combined.sort_values("id", inplace=True, kind="mergesort")
        write_manifest(manifest_path, combined)
        save_meta(meta_path, meta)
        log(f"[DONE] {model_name}/{rosbag}: +{len(df_append)} rows, total={meta['total_count']}")
    else:
        log(f"[WARN] {model_name}/{rosbag}: no valid vectors appended")

def main():
    parser = argparse.ArgumentParser(description="Consolidate topic .pt embeddings into sharded .npy + manifest (float32).")
    parser.add_argument("--models", nargs="*", default=None, help="Limit to model names")
    parser.add_argument("--rosbags", nargs="*", default=None, help="Limit to rosbag names")
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS, help="Rows per shard .npy")
    parser.add_argument("--force", action="store_true", help="Rescan ignoring existing manifest")
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)

    models = list_models(EMBEDDINGS_PER_TOPIC_CONSOLIDATED_DIR, args.models)
    if not models:
        log("[INFO] No models found.")
        return

    for model in models:
        rosbags = list_rosbags(EMBEDDINGS_PER_TOPIC_CONSOLIDATED_DIR / model, args.rosbags)
        if not rosbags:
            log(f"[INFO] {model}: no rosbags.")
            continue
        for bag in rosbags:
            process_one_model_rosbag(model, bag, force=args.force, shard_rows=args.shard_rows)

if __name__ == "__main__":
    main()
