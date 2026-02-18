"""Search routes."""
import gc
import json
import uuid
import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from flask import Blueprint, jsonify, request
import open_clip
from ..config import EMBEDDINGS, LOOKUP_TABLES, OPEN_CLIP_MODELS, MAX_K, GEMMA_AVAILABLE, gemma_tokenizer, gemma_model
from ..state import SEARCH_PROGRESS, start_new_search, is_search_cancelled, cancel_current_search
from ..utils.rosbag import load_lookup_tables_for_rosbag
from ..utils.clip import get_text_embedding, load_agriclip, resolve_custom_checkpoint, tokenize_texts

search_bp = Blueprint('search', __name__)

# FAISS index build: add vectors in batches to allow progress updates during vstack+add
FAISS_INDEX_BATCH_SIZE = 50_000


def _release_clip_model_from_gpu(model) -> None:
    """Safely move CLIP model off GPU and free CUDA memory. No-op if model is None or on CPU."""
    if model is None:
        return
    try:
        if hasattr(model, "cpu") and next(model.parameters(), None) is not None:
            device = next(model.parameters()).device
            if device.type == "cuda":
                model.cpu()
    except Exception as e:
        logging.debug("[SEARCH] Error moving model to CPU during cleanup: %s", e)
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _get_image_embedding_direct(
    model_name: str,
    rosbag_name: str,
    shard_id: str,
    row_in_shard: int,
) -> np.ndarray | None:
    """
    Load a precomputed image embedding directly from shard by position (no manifest lookup).
    Use when exact shard_id and row_in_shard are known from a search result.
    """
    base = EMBEDDINGS / model_name / rosbag_name
    shards_dir = base / "shards"
    shard_path = shards_dir / shard_id

    if not shard_path.is_file():
        logging.debug("[SEARCH-BY-IMAGE] Shard not found for direct load: %s", shard_path)
        return None

    try:
        arr = np.load(shard_path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        embedding = arr[int(row_in_shard)].copy()
        return embedding
    except Exception as e:
        logging.debug("[SEARCH-BY-IMAGE] Failed direct load from %s row %s: %s", shard_path, row_in_shard, e)
        return None


def _get_image_embedding_from_shards(
    model_name: str,
    rosbag_name: str,
    topic: str,
    mcap_identifier: str,
) -> np.ndarray | None:
    """
    Look up a precomputed image embedding from the manifest and shards.

    Args:
        model_name: Embedding model name (e.g. 'ViT-B-16-quickgelu__openai')
        rosbag_name: Rosbag path containing the image
        topic: Image topic (e.g. '/camera/image_raw')
        mcap_identifier: MCAP identifier for the image

    Returns:
        Embedding vector as float32 numpy array, or None if not found.
    """
    base = EMBEDDINGS / model_name / rosbag_name
    manifest_path = base / "manifest.parquet"
    shards_dir = base / "shards"

    if not manifest_path.is_file() or not shards_dir.is_dir():
        logging.debug("[SEARCH-BY-IMAGE] Missing manifest or shards for %s/%s", model_name, rosbag_name)
        return None

    try:
        mf = pd.read_parquet(
            manifest_path,
            columns=["topic", "shard_id", "row_in_shard", "mcap_identifier"],
        )
    except Exception as e:
        logging.debug("[SEARCH-BY-IMAGE] Failed to read manifest %s: %s", manifest_path, e)
        return None

    mask = (mf["topic"] == topic) & (mf["mcap_identifier"].astype(str) == str(mcap_identifier))
    matches = mf.loc[mask]

    if matches.empty:
        logging.debug(
            "[SEARCH-BY-IMAGE] No matching row for topic=%s mcap_id=%s in %s/%s",
            topic,
            mcap_identifier,
            model_name,
            rosbag_name,
        )
        return None

    row = matches.iloc[0]
    shard_id = row["shard_id"]
    row_in_shard = int(row["row_in_shard"])
    shard_path = shards_dir / shard_id

    if not shard_path.is_file():
        logging.debug("[SEARCH-BY-IMAGE] Shard not found: %s", shard_path)
        return None

    try:
        arr = np.load(shard_path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        embedding = arr[row_in_shard].copy()
        return embedding
    except Exception as e:
        logging.debug("[SEARCH-BY-IMAGE] Failed to load embedding from %s row %d: %s", shard_path, row_in_shard, e)
        return None


@search_bp.route('/api/enhance-prompt', methods=['GET'])
def enhance_prompt_endpoint():
    """Enhance a user prompt using the gemma model."""
    if not GEMMA_AVAILABLE:
        return jsonify({'error': 'Prompt enhancement not available (Gemma model not loaded)'}), 503

    user_prompt = request.args.get('prompt', default=None, type=str)
    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Update search status to show enhancement is in progress
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.0
    SEARCH_PROGRESS["message"] = "Enhancing prompt..."
    
    try:
        messages = [
            {
            "role": "user",
            "content": f"Return: 'A photo of' plus the query. Use 'a' or 'an' only if the query is singular, no article if plural. Add ', a type of' and the category if you know it.\nQuery: {user_prompt}"
            }
        ]
        
        inputs = gemma_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma_model.device)
                
        outputs = gemma_model.generate(**inputs, max_new_tokens=50)
        enhanced_prompt = gemma_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        enhanced_prompt = enhanced_prompt.replace('\n', ' ').replace('<end_of_turn>', '').strip()
        
        return jsonify({'original': user_prompt, 'enhanced': enhanced_prompt}), 200
    except Exception as e:
        logging.exception("[C] Failed to enhance prompt")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["message"] = f"Error enhancing prompt: {str(e)}"
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search-status', methods=['GET'])
def get_search_status():
    return jsonify(SEARCH_PROGRESS)


@search_bp.route('/api/cancel-search', methods=['POST'])
def cancel_search():
    """Cancel the currently running search."""
    cancel_current_search()
    SEARCH_PROGRESS["status"] = "idle"
    SEARCH_PROGRESS["progress"] = -1
    #SEARCH_PROGRESS["message"] = "Search cancelled."
    logging.info("[SEARCH] Search cancelled by user")
    return jsonify({'cancelled': True})


@search_bp.route('/api/search', methods=['GET'])
def search():
    # ---- Debug logging for paths
    logging.debug("[SEARCH] LOOKUP_TABLES=%s (exists=%s)", LOOKUP_TABLES, LOOKUP_TABLES.exists() if LOOKUP_TABLES else False)
    logging.debug("[SEARCH] EMBEDDINGS=%s (exists=%s)", EMBEDDINGS, EMBEDDINGS.exists() if EMBEDDINGS else False)

    # ---- Register this search (supersedes any running search)
    search_id = start_new_search()

    # ---- Initial status
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00

    # ---- Inputs
    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)

    # ---- Validate inputs
    if query_text is None:                 return jsonify({'error': 'No query text provided'}), 400
    if models is None:                     return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:                    return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:                  return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:                   return jsonify({'error': 'No accuracy provided'}), 400

    # ---- Parse inputs
    mcap_filter_raw = request.args.get('mcapFilter', default=None, type=str)
    topics_raw = request.args.get('topics', default=None, type=str)
    try:
        models_list = [m.strip() for m in models.split(",") if m.strip()]  # Filter out empty model names
        # Use rosbag paths directly - frontend sends relative paths that match preprocessing output
        rosbags_list = [r.strip() for r in rosbags.split(",") if r.strip()]
        time_start, time_end = map(int, timeRange.split(","))
        k_subsample = max(1, int(accuracy))
        # Parse optional MCAP filter: { "rosbag_path": [[startId, endId], ...], ... }
        mcap_filter = json.loads(mcap_filter_raw) if mcap_filter_raw else None
        topics_list = [t.strip() for t in topics_raw.split(",") if t.strip()] if topics_raw else None
    except Exception as e:
        logging.exception("[C] Failed parsing inputs")
        return jsonify({'error': f'Invalid inputs: {e}'}), 400
    
    # Validate parsed inputs
    if not models_list:
        return jsonify({'error': 'No valid models provided (empty or whitespace-only)'}), 400
    if not rosbags_list:
        return jsonify({'error': 'No valid rosbags provided (empty or whitespace-only)'}), 400

    # ---- State
    marks: dict[tuple, set] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results: list[dict] = []
    model_load_errors: list[str] = []  # Track per-model load failures

    try:
        total_steps = max(1, len(models_list) * len(rosbags_list))
        step_count = 0
        logging.debug("[SEARCH] total_steps=%d device=%s torch.cuda.is_available()=%s", total_steps, device, torch.cuda.is_available())

        for model_name in models_list:
            if is_search_cancelled(search_id):
                logging.info("[SEARCH] Cancelled before model %s", model_name)
                break

            model = None
            tokenizer = None
            query_embedding = None
            model_kind = "open_clip"

            try:
                if '__' in model_name:
                    name, pretrained = model_name.split('__', 1)
                    model, _, _ = open_clip.create_model_and_transforms(
                        name, pretrained, device=device, cache_dir=OPEN_CLIP_MODELS
                    )
                    tokenizer = open_clip.get_tokenizer(name)
                    query_embedding = get_text_embedding(query_text, model, tokenizer, device)
                else:
                    checkpoint_path = resolve_custom_checkpoint(model_name)
                    logging.debug("[SEARCH] Loading custom CLIP model '%s' from %s", model_name, checkpoint_path)
                    model = load_agriclip(str(checkpoint_path), device=device)
                    try:
                        tokens = tokenize_texts([query_text], model.context_length, device)
                    except ImportError as exc:
                        raise RuntimeError(
                            "Custom models require open-clip tokenization utilities. Install open-clip-torch or provide pre-tokenized input."
                        ) from exc
                    with torch.no_grad():
                        features = model.encode_text(tokens)
                        features = F.normalize(features, dim=-1)
                    query_embedding = features.detach().cpu().numpy().flatten()
                    model_kind = "custom"
            except Exception as e:
                logging.error("[SEARCH] Failed to load model %s: %s", model_name, e)
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e) or "out of memory" in str(e).lower()
                if is_oom:
                    model_load_errors.append(f"GPU VRAM full – could not load model '{model_name}'")
                    logging.error("[SEARCH] CUDA OOM while loading model %s", model_name)
                else:
                    model_load_errors.append(f"Failed to load model '{model_name}': {e}")
                try:
                    _release_clip_model_from_gpu(model)
                except NameError:
                    pass
                continue

            if query_embedding is None:
                logging.debug("[SEARCH] Query embedding is None for model %s; skipping", model_name)
                _release_clip_model_from_gpu(model)
                continue

            query_embedding = query_embedding.astype("float32", copy=False)

            for rosbag_name in rosbags_list:
                if is_search_cancelled(search_id):
                    logging.info("[SEARCH] Cancelled before rosbag %s", rosbag_name)
                    break

                # Progress tracking: each (model, rosbag) pair gets a slice of 0-99%
                base_progress = (step_count / total_steps) * 0.99
                step_range = 0.99 / total_steps

                PHASE_CHUNKS, PHASE_INDEX, PHASE_SEARCH = "chunks", "index", "search"
                phase_weights = {PHASE_CHUNKS: 0.25, PHASE_INDEX: 0.70, PHASE_SEARCH: 0.05}
                phase_starts = {PHASE_CHUNKS: 0.0, PHASE_INDEX: 0.25, PHASE_SEARCH: 0.95}

                def update_progress(
                    current_phase: str,
                    embeddings_processed: int,
                    total_embeddings: int,
                    phase: str = PHASE_CHUNKS,
                ):
                    """Update progress within the current rosbag's range. Phases: chunks (0-70%%), index (70-95%%), search (95-100%%)."""
                    if total_embeddings > 0:
                        frac = embeddings_processed / total_embeddings
                    else:
                        frac = 1.0
                    p = phase_starts.get(phase, 0) + frac * phase_weights.get(phase, 0.25)
                    within_step = p * step_range
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round(base_progress + within_step, 3)
                    SEARCH_PROGRESS["message"] = (
                        f"{current_phase}\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Progress: {embeddings_processed:,} / {total_embeddings:,} embeddings\n"
                        f"(Sampling every {k_subsample}th embedding)"
                    )

                # Initial progress update for this rosbag (before we know total)
                SEARCH_PROGRESS["status"] = "running"
                SEARCH_PROGRESS["progress"] = round(base_progress, 3)
                SEARCH_PROGRESS["message"] = (
                    f"Loading manifest...\n\n"
                    f"Model: {model_name}\n"
                    f"Rosbag: {rosbag_name}"
                )
                step_count += 1

                # ---- Consolidated base
                base = EMBEDDINGS / model_name / rosbag_name
                manifest_path = base / "manifest.parquet"
                shards_dir = base / "shards"
                logging.debug("[SEARCH] EMBEDDINGS base path: %s (exists=%s)", base, base.exists() if base else False)
                logging.debug("[SEARCH] Base=%s  manifest=%s (exists=%s)  shards_dir=%s (exists=%s)",
                              base, manifest_path, manifest_path.exists(), shards_dir, shards_dir.exists())

                if not manifest_path.is_file() or not shards_dir.is_dir():
                    logging.debug("[SEARCH] SKIP: Missing manifest or shards for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Manifest schema check
                needed_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier", "reference_timestamp"]
                try:
                    import pyarrow.parquet as pq
                    available_cols = set(pq.read_schema(manifest_path).names)
                except Exception as e:
                    logging.debug("[SEARCH] Could not read schema with pyarrow (%s), fallback to pandas head: %s", type(e).__name__, e)
                    head_df = pd.read_parquet(manifest_path)
                    available_cols = set(head_df.columns)
                    logging.debug("[SEARCH] Manifest columns (fallback): %s", sorted(available_cols))

                # reference_timestamp and reference_timestamp_index are optional (for backwards compatibility)
                required_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier"]
                missing = [c for c in required_cols if c not in available_cols]
                if missing:
                    msg = f"[SEARCH] Manifest missing columns {missing}; present={sorted(available_cols)}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                has_reference_timestamp = "reference_timestamp" in available_cols
                has_reference_timestamp_index = "reference_timestamp_index" in available_cols
                cols_to_read = required_cols.copy()
                if has_reference_timestamp:
                    cols_to_read.append("reference_timestamp")
                if has_reference_timestamp_index:
                    cols_to_read.append("reference_timestamp_index")

                # ---- Read needed columns
                t_manifest_start = time.perf_counter()
                mf = pd.read_parquet(manifest_path, columns=cols_to_read)
                t_manifest_read = time.perf_counter()
                logging.info("[SEARCH-DEBUG] Manifest read took %.3fs, rows: %d", t_manifest_read - t_manifest_start, len(mf))

                # ---- Filter by time window
                pre_count = len(mf)
                mf = mf.loc[(mf["minute_of_day"] >= time_start) & (mf["minute_of_day"] <= time_end)]
                t_time_filter = time.perf_counter()
                logging.info("[SEARCH-DEBUG] Time filter took %.3fs, rows: %d -> %d", t_time_filter - t_manifest_read, pre_count, len(mf))
                if mf.empty:
                    logging.debug("[SEARCH] SKIP: No rows in time window for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Filter by MCAP ranges (optional)
                if mcap_filter and rosbag_name in mcap_filter and "mcap_identifier" in mf.columns:
                    allowed_ranges = mcap_filter[rosbag_name]
                    pre_mcap_count = len(mf)
                    mcap_ids = mf["mcap_identifier"].astype(int)
                    mask = pd.Series(False, index=mf.index)
                    for range_start, range_end in allowed_ranges:
                        mask |= (mcap_ids >= int(range_start)) & (mcap_ids <= int(range_end))
                    mf = mf.loc[mask]
                    logging.info("[SEARCH-DEBUG] MCAP filter: %d -> %d rows", pre_mcap_count, len(mf))
                    if mf.empty:
                        logging.debug("[SEARCH] SKIP: No rows after MCAP filter for %s/%s", model_name, rosbag_name)
                        continue

                # ---- Filter by topics (optional)
                if topics_list and "topic" in mf.columns:
                    pre_topic_count = len(mf)
                    mf = mf[mf["topic"].isin(topics_list)]
                    logging.info("[SEARCH-DEBUG] Topic filter: %d -> %d rows", pre_topic_count, len(mf))
                    if mf.empty:
                        logging.debug("[SEARCH] SKIP: No rows after topic filter for %s/%s", model_name, rosbag_name)
                        continue

                # ---- Subsample per topic
                parts, per_topic_counts_before, per_topic_counts_after = [], {}, {}
                for topic, df_t in mf.groupby("topic", sort=False):
                    per_topic_counts_before[topic] = len(df_t)
                    sort_cols = [c for c in ["timestamp_ns"] if c in df_t.columns]
                    if sort_cols:
                        df_t = df_t.sort_values(sort_cols)
                    parts.append(df_t.iloc[::k_subsample])
                mf_sel = pd.concat(parts, ignore_index=True) if parts else mf.iloc[0:0]
                for topic, df_t in mf_sel.groupby("topic", sort=False):
                    per_topic_counts_after[topic] = len(df_t)
                t_subsample = time.perf_counter()
                logging.info("[SEARCH-DEBUG] Subsample took %.3fs, rows: %d -> %d (k=%d)",
                            t_subsample - t_time_filter, len(mf), len(mf_sel), k_subsample)

                if mf_sel.empty:
                    logging.debug("[SEARCH] SKIP: No rows after subsample for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Load aligned CSV for marks (only if manifest lacks reference_timestamp_index)
                aligned_data = pd.DataFrame()  # Default empty
                if not has_reference_timestamp_index:
                    SEARCH_PROGRESS["message"] = (
                        f"Loading lookup tables (legacy mode)...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}"
                    )
                    t_lookup_start = time.perf_counter()
                    lookup_dir = (LOOKUP_TABLES / rosbag_name) if LOOKUP_TABLES else None
                    logging.debug("[SEARCH] LOOKUP_TABLES dir for %s: %s (exists=%s)", rosbag_name, lookup_dir, lookup_dir.exists() if lookup_dir else False)
                    aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
                    t_lookup_end = time.perf_counter()
                    logging.info("[SEARCH-DEBUG] Lookup tables load took %.3fs, shape=%s",
                                t_lookup_end - t_lookup_start,
                                aligned_data.shape if not aligned_data.empty else "empty")
                    if aligned_data.empty:
                        logging.debug("[SEARCH] WARN: no lookup tables found for %s (marks will be empty)", rosbag_name)
                else:
                    logging.info("[SEARCH-DEBUG] Manifest has reference_timestamp_index, skipping lookup tables load")

                # ---- Gather vectors by shard
                t_rosbag_start = time.perf_counter()

                chunks: list[np.ndarray] = []
                meta_for_row: list[dict] = []
                shards_touched = 0
                bytes_read = 0
                shard_ids = sorted(mf_sel["shard_id"].unique().tolist())

                # Progress tracking for this rosbag
                total_embeddings = len(mf_sel)
                embeddings_processed = 0
                last_progress_update = 0
                progress_update_interval = 1000  # Update every N embeddings

                logging.info("[SEARCH-DEBUG] === Processing rosbag: %s ===", rosbag_name)
                logging.info("[SEARCH-DEBUG] Manifest rows after filter: %d, unique shards: %d",
                            total_embeddings, len(shard_ids))

                update_progress("Searching embeddings...", embeddings_processed, total_embeddings)
                t_chunks_start = time.perf_counter()

                for shard_id, df_s in mf_sel.groupby("shard_id", sort=False):
                    if is_search_cancelled(search_id):
                        logging.info("[SEARCH] Cancelled during shard processing")
                        break

                    shard_path = shards_dir / shard_id
                    if not shard_path.is_file():
                        continue

                    rows = df_s["row_in_shard"].to_numpy(np.int64)
                    rows.sort()
                    shards_touched += 1

                    # Build meta map (row_in_shard -> meta)
                    meta_map = {}
                    for r in df_s.itertuples(index=False):
                        topic_str = r.topic
                        topic_folder = topic_str.replace("/", "__")
                        meta_entry = {
                            "timestamp_ns": int(r.timestamp_ns),
                            "topic": topic_str.replace("__", "/"),
                            "topic_folder": topic_folder,
                            "minute_of_day": int(r.minute_of_day),
                            "mcap_identifier": str(r.mcap_identifier),
                            "shard_id": shard_id,
                            "row_in_shard": int(r.row_in_shard),
                        }
                        # Add reference_timestamp if available
                        if has_reference_timestamp and hasattr(r, 'reference_timestamp'):
                            ref_ts = r.reference_timestamp
                            meta_entry["reference_timestamp"] = str(ref_ts) if pd.notna(ref_ts) else None
                        # Add reference_timestamp_index if available
                        if has_reference_timestamp_index and hasattr(r, 'reference_timestamp_index'):
                            ref_ts_idx = r.reference_timestamp_index
                            meta_entry["reference_timestamp_index"] = int(ref_ts_idx) if pd.notna(ref_ts_idx) else None
                        meta_map[int(r.row_in_shard)] = meta_entry

                    arr = np.load(shard_path, mmap_mode="r")  # expect float32 shards
                    if arr.dtype != np.float32:
                        logging.debug("[SEARCH] Shard %s dtype=%s (casting to float32)", shard_id, arr.dtype)
                        arr = arr.astype("float32", copy=False)

                    # Coalesce contiguous ranges → fewer slices
                    ranges = []
                    start = prev = int(rows[0])
                    for rr in rows[1:]:
                        rr = int(rr)
                        if rr == prev + 1:
                            prev = rr
                        else:
                            ranges.append((start, prev))
                            start = prev = rr
                    ranges.append((start, prev))

                    for a, b in ranges:
                        sl = arr[a:b+1]  # (len, D)
                        chunks.append(sl)
                        bytes_read += sl.nbytes
                        for i in range(a, b + 1):
                            meta_for_row.append(meta_map[i])

                        # Update progress tracking
                        embeddings_processed += (b - a + 1)
                        if embeddings_processed - last_progress_update >= progress_update_interval:
                            update_progress("Searching embeddings...", embeddings_processed, total_embeddings)
                            last_progress_update = embeddings_processed

                t_chunks_end = time.perf_counter()
                logging.info("[SEARCH-DEBUG] Chunk collection took %.3fs", t_chunks_end - t_chunks_start)
                logging.info("[SEARCH-DEBUG] Shards touched: %d, total bytes read: %.2f MB, chunks count: %d",
                            shards_touched, bytes_read / 1024 / 1024, len(chunks))

                if not chunks:
                    msg = f"[SEARCH] No chunks loaded (missing shards/files?) for {model_name}/{rosbag_name}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                # ---- FAISS: build index in batches for progress updates during vstack+add
                t0 = time.perf_counter()
                dim = chunks[0].shape[1] if chunks else 0
                total_vectors = sum(c.shape[0] for c in chunks)
                if total_vectors != len(meta_for_row):
                    msg = f"[SEARCH] Row/meta mismatch: vectors={total_vectors} vs meta={len(meta_for_row)}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                M = 32  # edges per node
                index = faiss.IndexHNSWFlat(dim, M)
                index.hnsw.efSearch = 64
                index.hnsw.efConstruction = 40
                t_create = time.perf_counter()
                logging.info("[FAISS-DEBUG] Index created, dim=%d, M=%d", dim, M)

                vectors_added = 0
                vectors_in_index = 0
                accumulated = []
                t_vstack_total, t_add_total = 0.0, 0.0
                for i, chunk in enumerate(chunks):
                    if is_search_cancelled(search_id):
                        logging.info("[SEARCH] Cancelled during index build")
                        break
                    accumulated.append(chunk)
                    vectors_added += chunk.shape[0]
                    if vectors_added >= FAISS_INDEX_BATCH_SIZE or (i == len(chunks) - 1):
                        t_v = time.perf_counter()
                        X_batch = np.vstack(accumulated).astype("float32", copy=False)
                        t_vstack_total += time.perf_counter() - t_v
                        t_a = time.perf_counter()
                        index.add(X_batch)
                        t_add_total += time.perf_counter() - t_a
                        vectors_in_index += X_batch.shape[0]
                        update_progress(
                            "Building search index...",
                            vectors_in_index,
                            total_vectors,
                            phase=PHASE_INDEX,
                        )
                        accumulated = []
                        vectors_added = 0

                t_vstack = t_vstack_total
                t_add = t_add_total
                logging.info("[FAISS-DEBUG] Batched vstack+add: vstack=%.3fs, add=%.3fs, total=%d vectors",
                            t_vstack, t_add, index.ntotal)
                logging.info("[FAISS-DEBUG] X total: %.2f MB", total_vectors * dim * 4 / 1024 / 1024)

                update_progress("Running search...", total_vectors, total_vectors, phase=PHASE_SEARCH)
                t3 = time.perf_counter()
                q = query_embedding.reshape(1, -1)
                logging.info("[FAISS-DEBUG] Starting search for k=%d neighbors", MAX_K)
                D, I = index.search(q, MAX_K)
                t_search = time.perf_counter()
                update_progress("Running search...", total_vectors, total_vectors, phase=PHASE_SEARCH)
                logging.info("[FAISS-DEBUG] index.search() took %.3fs", t_search - t3)
                logging.info("[FAISS-DEBUG] Results: D.shape=%s, I.shape=%s", D.shape, I.shape)
                logging.info("[FAISS-DEBUG] Distance range: min=%.4f, max=%.4f", D.min(), D.max())
                logging.info("[FAISS-DEBUG] Index range: min=%d, max=%d", I.min(), I.max())
                logging.info("[FAISS-DEBUG] TOTAL FAISS pipeline: %.3fs (vstack=%.3fs, create=%.3fs, add=%.3fs, search=%.3fs)",
                            t_search - t0, t_vstack, t_create - t0, t_add, t_search - t3)

                if D.size == 0 or I.size == 0:
                    logging.info("[FAISS-DEBUG] Empty FAISS result for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Build API-like results
                t_results_start = time.perf_counter()

                # Build ref_ts_to_indices for marks (only needed for legacy mode without reference_timestamp_index)
                ref_ts_to_indices: dict[str, list[int]] = {}
                value_to_ref_ts: dict[str, set[str]] = {}

                if has_reference_timestamp_index:
                    # FAST PATH: reference_timestamp_index is in manifest, no index building needed
                    logging.info("[SEARCH-DEBUG] Using reference_timestamp_index from manifest (fast path)")
                elif not aligned_data.empty and 'Reference Timestamp' in aligned_data.columns:
                    # LEGACY PATH: Build indices from lookup tables (slow)
                    t_index_start = time.perf_counter()
                    SEARCH_PROGRESS["message"] = (
                        f"Building marks index (legacy)...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Aligned data: {len(aligned_data):,} rows × {len(aligned_data.columns)} cols"
                    )

                    # Build ref_ts_to_indices using pandas groupby
                    ref_ts_col = aligned_data['Reference Timestamp'].astype(str)
                    for ref_ts, group in aligned_data.groupby(ref_ts_col, sort=False):
                        ref_ts_to_indices[ref_ts] = group.index.tolist()

                    # Build value_to_ref_ts using pandas stack + groupby
                    stacked = aligned_data.astype(str).stack()
                    idx_to_ref = ref_ts_col.to_dict()
                    stacked_with_ref = pd.DataFrame({
                        'value': stacked.values,
                        'ref_ts': [idx_to_ref[idx[0]] for idx in stacked.index]
                    })
                    for val, group in stacked_with_ref.groupby('value', sort=False):
                        value_to_ref_ts[val] = set(group['ref_ts'].unique())

                    t_index_end = time.perf_counter()
                    logging.info("[SEARCH-DEBUG] Legacy marks index build: %.3fs (%d value mappings, %d ref_ts groups)",
                                t_index_end - t_index_start, len(value_to_ref_ts), len(ref_ts_to_indices))

                model_results: list[dict] = []
                for i, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
                    if idx < 0 or idx >= len(meta_for_row):
                        continue
                    m = meta_for_row[int(idx)]
                    ts_ns = m["timestamp_ns"]
                    ts_str = str(ts_ns)
                    # Convert UTC timestamp to Europe/Berlin timezone
                    berlin_tz = ZoneInfo("Europe/Berlin")
                    minute_str = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(berlin_tz).strftime("%H:%M")

                    result_entry = {
                        'rank': i,
                        'rosbag': rosbag_name,
                        'similarityScore': float(dist),
                        'topic': m["topic"],           # slashes
                        'timestamp': ts_str,
                        'minuteOfDay': minute_str,
                        'mcap_identifier': m["mcap_identifier"],
                        'shard_id': m["shard_id"],
                        'row_in_shard': m["row_in_shard"],
                        'model': model_name,
                        '_mark_indices': [],
                    }
                    model_results.append(result_entry)

                    # Marks matching
                    if has_reference_timestamp_index:
                        # FAST: Use reference_timestamp_index directly from manifest
                        ref_ts_idx = m.get("reference_timestamp_index")
                        if i <= 3:  # Log first few results for debugging
                            logging.info("[SEARCH-MARKS] result #%d: has_ref_ts_idx=True, ref_ts_idx=%s (type=%s)", i, ref_ts_idx, type(ref_ts_idx).__name__)
                        if ref_ts_idx is not None:
                            key = (model_name, rosbag_name, m["topic"])
                            marks.setdefault(key, set()).add(ref_ts_idx)
                            result_entry['_mark_indices'].append(ref_ts_idx)
                    elif ts_str in value_to_ref_ts:
                        # LEGACY: O(1) lookup from pre-built index
                        matching_ref_timestamps = value_to_ref_ts[ts_str]
                        match_indices: list[int] = []
                        for ref_ts in matching_ref_timestamps:
                            if ref_ts in ref_ts_to_indices:
                                match_indices.extend(ref_ts_to_indices[ref_ts])
                        if match_indices:
                            key = (model_name, rosbag_name, m["topic"])
                            marks.setdefault(key, set()).update(match_indices)
                            result_entry['_mark_indices'].extend(match_indices)

                all_results.extend(model_results)
                marks_count = sum(len(v) for v in marks.values())
                logging.info("[SEARCH-MARKS] After %s/%s: marks keys=%d, total mark indices=%d, has_ref_ts_idx=%s",
                            model_name, rosbag_name, len(marks), marks_count, has_reference_timestamp_index)

                t_results_end = time.perf_counter()
                logging.info("[SEARCH-DEBUG] Results building took %.3fs for %d results", t_results_end - t_results_start, len(model_results))

                t_rosbag_end = time.perf_counter()
                logging.info("[SEARCH-DEBUG] === Rosbag %s COMPLETE: %.3fs total, %d results ===",
                            rosbag_name, t_rosbag_end - t_rosbag_start, len(model_results))

            # Cleanup GPU per model (including on cancellation/break)
            _release_clip_model_from_gpu(model)

        # ---- Check for cancellation before post-processing
        if is_search_cancelled(search_id):
            logging.info("[SEARCH] Search cancelled, skipping post-processing")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            SEARCH_PROGRESS["status"] = "cancelled"
            SEARCH_PROGRESS["progress"] = 0.0
            #SEARCH_PROGRESS["message"] = "Search cancelled."
            return jsonify({'cancelled': True, 'results': [], 'marksPerTopic': {}})

        # ---- Post processing
        all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
        for rank, result in enumerate(all_results, 1):
            result['rank'] = rank
        filtered_results = all_results

        # marksPerTopic – include rank so the frontend can weight heatmap intensity
        # Build best-rank lookup: (model, rosbag, topic, idx) -> lowest rank
        mark_best_rank: dict[tuple, int] = {}
        for result in filtered_results:
            rank = result['rank']
            key_prefix = (result['model'], result['rosbag'], result['topic'])
            for idx in result.get('_mark_indices', []):
                full_key = (*key_prefix, idx)
                if full_key not in mark_best_rank or rank < mark_best_rank[full_key]:
                    mark_best_rank[full_key] = rank
            result.pop('_mark_indices', None)  # strip internal field before response

        marksPerTopic: dict = {}
        for result in filtered_results:
            model = result['model']
            rosbag = result['rosbag']
            topic = result['topic']

            marksPerTopic \
                .setdefault(model, {}) \
                .setdefault(rosbag, {}) \
                .setdefault(topic, {'marks': []})

        for key, indices in marks.items():
            model_key, rosbag_key, topic_key = key
            if (
                model_key in marksPerTopic
                and rosbag_key in marksPerTopic[model_key]
                and topic_key in marksPerTopic[model_key][rosbag_key]
            ):
                marksPerTopic[model_key][rosbag_key][topic_key]['marks'].extend(
                    {'value': idx, 'rank': mark_best_rank.get((model_key, rosbag_key, topic_key, idx), 100)}
                    for idx in indices
                )

        # Check if no results were found
        if not filtered_results:
            if model_load_errors:
                # All models failed to load (likely CUDA OOM)
                is_vram = any("VRAM" in e or "GPU" in e for e in model_load_errors)
                warning_msg = (
                    "Search unavailable: GPU VRAM is full. Try again later or restart the backend."
                    if is_vram
                    else "; ".join(model_load_errors)
                )
                SEARCH_PROGRESS["status"] = "error"
                SEARCH_PROGRESS["progress"] = 0.0
                SEARCH_PROGRESS["message"] = warning_msg
                return jsonify({
                    'query': query_text,
                    'results': [],
                    'marksPerTopic': {},
                    'error': warning_msg,
                }), 503
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
            SEARCH_PROGRESS["message"] = "No results found for the given query."
        else:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0

        response_data = {
            'query': query_text,
            'results': filtered_results,
            'marksPerTopic': marksPerTopic,
        }
        # Include non-fatal warnings if some models failed but others succeeded
        if model_load_errors and filtered_results:
            response_data['warning'] = "; ".join(model_load_errors)
        return jsonify(response_data)

    except Exception as e:
        # Generate error ID for debugging without exposing internals
        error_id = str(uuid.uuid4())[:8]
        logging.exception(f"[SEARCH] Search failed [error_id={error_id}]")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Search failed. Error ID: {error_id}"
        try:
            _release_clip_model_from_gpu(model)
        except NameError:
            pass
        return jsonify({
            'error': 'Search failed. Please try again.',
            'error_id': error_id
        }), 500


@search_bp.route('/api/pipeline-counts', methods=['GET'])
def pipeline_counts():
    """
    Lightweight endpoint that counts embeddings at each filter stage without loading any model.
    Reads only 3 lightweight columns from each manifest parquet.
    Params: models, rosbags, timeRange, accuracy, mcapFilter (optional), topics (optional)
    """
    models_raw = request.args.get('models', default='', type=str)
    rosbags_raw = request.args.get('rosbags', default='', type=str)
    time_range_raw = request.args.get('timeRange', default='0,1440', type=str)
    accuracy_raw = request.args.get('accuracy', default=1, type=int)
    mcap_filter_raw = request.args.get('mcapFilter', default=None, type=str)
    topics_raw = request.args.get('topics', default=None, type=str)

    try:
        models_list = [m.strip() for m in models_raw.split(',') if m.strip()]
        rosbags_list = [r.strip() for r in rosbags_raw.split(',') if r.strip()]
        time_start, time_end = map(int, time_range_raw.split(','))
        k_subsample = max(1, accuracy_raw)
        mcap_filter = json.loads(mcap_filter_raw) if mcap_filter_raw else None
        topics_list = [t.strip() for t in topics_raw.split(',') if t.strip()] if topics_raw else None
    except Exception as e:
        return jsonify({'error': f'Invalid params: {e}'}), 400

    counts = {'total': 0, 'after_time': 0, 'after_mcap': 0, 'after_topic': 0, 'after_sample': 0}

    for model_name in models_list:
        for rosbag_name in rosbags_list:
            manifest_path = EMBEDDINGS / model_name / rosbag_name / 'manifest.parquet'
            if not manifest_path.is_file():
                continue
            try:
                mf = pd.read_parquet(manifest_path, columns=['minute_of_day', 'mcap_identifier', 'topic'])
            except Exception:
                continue

            counts['total'] += len(mf)

            # Order: MCAPs → Topics → Time → Sample (matches display order)
            # Each stage counts rows AFTER applying all filters up to and including that stage.
            # Snapshot the full mf so we can compute MCAP and topic counts independently of time.
            mf_full = mf

            if mcap_filter and rosbag_name in mcap_filter:
                allowed_ranges = mcap_filter[rosbag_name]
                mcap_ids = mf_full['mcap_identifier'].astype(int)
                mask = pd.Series(False, index=mf_full.index)
                for range_start, range_end in allowed_ranges:
                    mask |= (mcap_ids >= int(range_start)) & (mcap_ids <= int(range_end))
                mf = mf_full.loc[mask]
            counts['after_mcap'] += len(mf)

            if topics_list:
                mf = mf[mf['topic'].isin(topics_list)]
            counts['after_topic'] += len(mf)

            if not mf.empty:
                mod = mf['minute_of_day'].astype(int)
                mf = mf.loc[(mod >= time_start) & (mod <= time_end)]
            counts['after_time'] += len(mf)

            after_sample = sum(len(df_t.iloc[::k_subsample]) for _, df_t in mf.groupby('topic', sort=False)) if not mf.empty else 0
            counts['after_sample'] += after_sample

    return jsonify({
        'total': counts['total'],
        'afterMcap': counts['after_mcap'],
        'afterTopic': counts['after_topic'],
        'afterTime': counts['after_time'],
        'afterSample': counts['after_sample'],
    })


@search_bp.route('/api/search-by-image', methods=['GET'])
def search_by_image():
    """
    Same as /api/search but uses a precomputed image embedding as the query vector
    instead of embedding text. The source image is identified by imageRosbag,
    imageTopic, and imageMcapIdentifier.
    """
    # ---- Debug logging for paths
    logging.debug("[SEARCH-BY-IMAGE] LOOKUP_TABLES=%s (exists=%s)", LOOKUP_TABLES, LOOKUP_TABLES.exists() if LOOKUP_TABLES else False)
    logging.debug("[SEARCH-BY-IMAGE] EMBEDDINGS=%s (exists=%s)", EMBEDDINGS, EMBEDDINGS.exists() if EMBEDDINGS else False)

    # ---- Register this search (supersedes any running search)
    search_id = start_new_search()

    # ---- Initial status
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00

    # ---- Inputs
    image_rosbag = request.args.get('imageRosbag', default=None, type=str)
    image_topic = request.args.get('imageTopic', default=None, type=str)
    image_mcap_identifier = request.args.get('imageMcapIdentifier', default=None, type=str)
    image_model = request.args.get('imageModel', default=None, type=str)
    image_shard_id = request.args.get('imageShardId', default=None, type=str)
    image_row_in_shard = request.args.get('imageRowInShard', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)

    # ---- Validate inputs
    if image_rosbag is None:
        return jsonify({'error': 'No imageRosbag provided'}), 400
    has_direct = image_model and image_shard_id and image_row_in_shard
    has_manifest = image_topic and image_mcap_identifier
    if not has_direct and not has_manifest:
        return jsonify({'error': 'Provide either (imageModel, imageShardId, imageRowInShard) or (imageTopic, imageMcapIdentifier)'}), 400
    if models is None:             return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:            return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:          return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:          return jsonify({'error': 'No accuracy provided'}), 400

    # ---- Parse inputs
    mcap_filter_raw = request.args.get('mcapFilter', default=None, type=str)
    topics_raw = request.args.get('topics', default=None, type=str)
    try:
        models_list = [m.strip() for m in models.split(",") if m.strip()]
        rosbags_list = [r.strip() for r in rosbags.split(",") if r.strip()]
        time_start, time_end = map(int, timeRange.split(","))
        k_subsample = max(1, int(accuracy))
        mcap_filter = json.loads(mcap_filter_raw) if mcap_filter_raw else None
        topics_list = [t.strip() for t in topics_raw.split(",") if t.strip()] if topics_raw else None
    except Exception as e:
        logging.exception("[SEARCH-BY-IMAGE] Failed parsing inputs")
        return jsonify({'error': f'Invalid inputs: {e}'}), 400

    if not models_list:
        return jsonify({'error': 'No valid models provided (empty or whitespace-only)'}), 400
    if not rosbags_list:
        return jsonify({'error': 'No valid rosbags provided (empty or whitespace-only)'}), 400

    # When using exact embedding (shard_id, row_in_shard), ensure the source model is in the list
    if has_direct and image_model and image_model not in models_list:
        models_list = [image_model] + [m for m in models_list if m != image_model]

    # ---- State
    marks: dict[tuple, set] = {}
    all_results: list[dict] = []

    try:
        total_steps = max(1, len(models_list) * len(rosbags_list))
        step_count = 0

        for model_name in models_list:
            if is_search_cancelled(search_id):
                logging.info("[SEARCH-BY-IMAGE] Cancelled before model %s", model_name)
                break

            # Load precomputed image embedding: prefer direct (shard_id, row_in_shard) when available
            if model_name == image_model and has_direct:
                try:
                    row_val = int(image_row_in_shard)
                except (TypeError, ValueError):
                    query_embedding = None
                else:
                    query_embedding = _get_image_embedding_direct(
                        model_name, image_rosbag, image_shard_id, row_val
                    )
            else:
                if has_manifest:
                    query_embedding = _get_image_embedding_from_shards(
                        model_name, image_rosbag, image_topic, image_mcap_identifier
                    )
                else:
                    query_embedding = None
            if query_embedding is None:
                logging.debug("[SEARCH-BY-IMAGE] No embedding for source image in model %s; skipping", model_name)
                continue

            query_embedding = query_embedding.astype("float32", copy=False)

            for rosbag_name in rosbags_list:
                if is_search_cancelled(search_id):
                    logging.info("[SEARCH-BY-IMAGE] Cancelled before rosbag %s", rosbag_name)
                    break

                base_progress = (step_count / total_steps) * 0.99
                step_range = 0.99 / total_steps

                PHASE_CHUNKS_I, PHASE_INDEX_I, PHASE_SEARCH_I = "chunks", "index", "search"
                phase_weights_i = {PHASE_CHUNKS_I: 0.25, PHASE_INDEX_I: 0.70, PHASE_SEARCH_I: 0.05}
                phase_starts_i = {PHASE_CHUNKS_I: 0.0, PHASE_INDEX_I: 0.25, PHASE_SEARCH_I: 0.95}

                def update_progress(
                    current_phase: str,
                    embeddings_processed: int,
                    total_embeddings: int,
                    phase: str = PHASE_CHUNKS_I,
                ):
                    if total_embeddings > 0:
                        frac = embeddings_processed / total_embeddings
                    else:
                        frac = 1.0
                    p = phase_starts_i.get(phase, 0) + frac * phase_weights_i.get(phase, 0.25)
                    within_step = p * step_range
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round(base_progress + within_step, 3)
                    SEARCH_PROGRESS["message"] = (
                        f"{current_phase}\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Progress: {embeddings_processed:,} / {total_embeddings:,} embeddings\n"
                        f"(Sampling every {k_subsample}th embedding)"
                    )

                SEARCH_PROGRESS["status"] = "running"
                SEARCH_PROGRESS["progress"] = round(base_progress, 3)
                SEARCH_PROGRESS["message"] = (
                    f"Loading manifest...\n\n"
                    f"Model: {model_name}\n"
                    f"Rosbag: {rosbag_name}"
                )
                step_count += 1

                base = EMBEDDINGS / model_name / rosbag_name
                manifest_path = base / "manifest.parquet"
                shards_dir = base / "shards"

                if not manifest_path.is_file() or not shards_dir.is_dir():
                    logging.debug("[SEARCH-BY-IMAGE] SKIP: Missing manifest or shards for %s/%s", model_name, rosbag_name)
                    continue

                needed_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier", "reference_timestamp"]
                try:
                    import pyarrow.parquet as pq
                    available_cols = set(pq.read_schema(manifest_path).names)
                except Exception as e:
                    logging.debug("[SEARCH-BY-IMAGE] Could not read schema: %s", e)
                    head_df = pd.read_parquet(manifest_path)
                    available_cols = set(head_df.columns)

                required_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier"]
                missing = [c for c in required_cols if c not in available_cols]
                if missing:
                    logging.debug("[SEARCH-BY-IMAGE] Manifest missing columns %s", missing)
                    continue

                has_reference_timestamp = "reference_timestamp" in available_cols
                has_reference_timestamp_index = "reference_timestamp_index" in available_cols
                cols_to_read = required_cols.copy()
                if has_reference_timestamp:
                    cols_to_read.append("reference_timestamp")
                if has_reference_timestamp_index:
                    cols_to_read.append("reference_timestamp_index")

                mf = pd.read_parquet(manifest_path, columns=cols_to_read)

                mf = mf.loc[(mf["minute_of_day"] >= time_start) & (mf["minute_of_day"] <= time_end)]
                if mf.empty:
                    logging.debug("[SEARCH-BY-IMAGE] SKIP: No rows in time window for %s/%s", model_name, rosbag_name)
                    continue

                if mcap_filter and rosbag_name in mcap_filter and "mcap_identifier" in mf.columns:
                    allowed_ranges = mcap_filter[rosbag_name]
                    mcap_ids = mf["mcap_identifier"].astype(int)
                    mask = pd.Series(False, index=mf.index)
                    for range_start, range_end in allowed_ranges:
                        mask |= (mcap_ids >= int(range_start)) & (mcap_ids <= int(range_end))
                    mf = mf.loc[mask]
                    if mf.empty:
                        continue

                if topics_list and "topic" in mf.columns:
                    mf = mf[mf["topic"].isin(topics_list)]
                    if mf.empty:
                        continue

                parts, per_topic_counts_before, per_topic_counts_after = [], {}, {}
                for topic, df_t in mf.groupby("topic", sort=False):
                    per_topic_counts_before[topic] = len(df_t)
                    sort_cols = [c for c in ["timestamp_ns"] if c in df_t.columns]
                    if sort_cols:
                        df_t = df_t.sort_values(sort_cols)
                    parts.append(df_t.iloc[::k_subsample])
                mf_sel = pd.concat(parts, ignore_index=True) if parts else mf.iloc[0:0]
                for topic, df_t in mf_sel.groupby("topic", sort=False):
                    per_topic_counts_after[topic] = len(df_t)

                if mf_sel.empty:
                    continue

                aligned_data = pd.DataFrame()
                if not has_reference_timestamp_index:
                    lookup_dir = (LOOKUP_TABLES / rosbag_name) if LOOKUP_TABLES else None
                    aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
                else:
                    pass

                t_rosbag_start = time.perf_counter()
                chunks: list[np.ndarray] = []
                meta_for_row: list[dict] = []
                shards_touched = 0
                bytes_read = 0

                total_embeddings = len(mf_sel)
                embeddings_processed = 0
                last_progress_update = 0
                progress_update_interval = 1000

                update_progress("Searching by image...", embeddings_processed, total_embeddings)

                for shard_id, df_s in mf_sel.groupby("shard_id", sort=False):
                    if is_search_cancelled(search_id):
                        break

                    shard_path = shards_dir / shard_id
                    if not shard_path.is_file():
                        continue

                    rows = df_s["row_in_shard"].to_numpy(np.int64)
                    rows.sort()
                    shards_touched += 1

                    meta_map = {}
                    for r in df_s.itertuples(index=False):
                        topic_str = r.topic
                        topic_folder = topic_str.replace("/", "__")
                        meta_entry = {
                            "timestamp_ns": int(r.timestamp_ns),
                            "topic": topic_str.replace("__", "/"),
                            "topic_folder": topic_folder,
                            "minute_of_day": int(r.minute_of_day),
                            "mcap_identifier": str(r.mcap_identifier),
                            "shard_id": shard_id,
                            "row_in_shard": int(r.row_in_shard),
                        }
                        if has_reference_timestamp and hasattr(r, 'reference_timestamp'):
                            ref_ts = r.reference_timestamp
                            meta_entry["reference_timestamp"] = str(ref_ts) if pd.notna(ref_ts) else None
                        if has_reference_timestamp_index and hasattr(r, 'reference_timestamp_index'):
                            ref_ts_idx = r.reference_timestamp_index
                            meta_entry["reference_timestamp_index"] = int(ref_ts_idx) if pd.notna(ref_ts_idx) else None
                        meta_map[int(r.row_in_shard)] = meta_entry

                    arr = np.load(shard_path, mmap_mode="r")
                    if arr.dtype != np.float32:
                        arr = arr.astype("float32", copy=False)

                    ranges = []
                    start = prev = int(rows[0])
                    for rr in rows[1:]:
                        rr = int(rr)
                        if rr == prev + 1:
                            prev = rr
                        else:
                            ranges.append((start, prev))
                            start = prev = rr
                    ranges.append((start, prev))

                    for a, b in ranges:
                        sl = arr[a:b+1]
                        chunks.append(sl)
                        bytes_read += sl.nbytes
                        for i in range(a, b + 1):
                            meta_for_row.append(meta_map[i])
                        embeddings_processed += (b - a + 1)
                        if embeddings_processed - last_progress_update >= progress_update_interval:
                            update_progress("Searching by image...", embeddings_processed, total_embeddings)
                            last_progress_update = embeddings_processed

                if not chunks:
                    continue

                total_vectors = sum(c.shape[0] for c in chunks)
                if total_vectors != len(meta_for_row):
                    continue

                dim = chunks[0].shape[1]
                M = 32
                index = faiss.IndexHNSWFlat(dim, M)
                index.hnsw.efSearch = 64
                index.hnsw.efConstruction = 40

                vectors_added = 0
                vectors_in_index = 0
                accumulated = []
                for i, chunk in enumerate(chunks):
                    if is_search_cancelled(search_id):
                        logging.info("[SEARCH-BY-IMAGE] Cancelled during index build")
                        break
                    accumulated.append(chunk)
                    vectors_added += chunk.shape[0]
                    if vectors_added >= FAISS_INDEX_BATCH_SIZE or (i == len(chunks) - 1):
                        X_batch = np.vstack(accumulated).astype("float32", copy=False)
                        index.add(X_batch)
                        vectors_in_index += X_batch.shape[0]
                        update_progress(
                            "Building search index...",
                            vectors_in_index,
                            total_vectors,
                            phase=PHASE_INDEX_I,
                        )
                        accumulated = []
                        vectors_added = 0

                update_progress("Running search...", total_vectors, total_vectors, phase=PHASE_SEARCH_I)
                q = query_embedding.reshape(1, -1)
                D, I = index.search(q, MAX_K)
                update_progress("Running search...", total_vectors, total_vectors, phase=PHASE_SEARCH_I)

                if D.size == 0 or I.size == 0:
                    continue

                ref_ts_to_indices: dict[str, list[int]] = {}
                value_to_ref_ts: dict[str, set[str]] = {}

                if has_reference_timestamp_index:
                    pass
                elif not aligned_data.empty and 'Reference Timestamp' in aligned_data.columns:
                    ref_ts_col = aligned_data['Reference Timestamp'].astype(str)
                    for ref_ts, group in aligned_data.groupby(ref_ts_col, sort=False):
                        ref_ts_to_indices[ref_ts] = group.index.tolist()
                    stacked = aligned_data.astype(str).stack()
                    idx_to_ref = ref_ts_col.to_dict()
                    stacked_with_ref = pd.DataFrame({
                        'value': stacked.values,
                        'ref_ts': [idx_to_ref[idx[0]] for idx in stacked.index]
                    })
                    for val, group in stacked_with_ref.groupby('value', sort=False):
                        value_to_ref_ts[val] = set(group['ref_ts'].unique())

                model_results: list[dict] = []
                for i, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
                    if idx < 0 or idx >= len(meta_for_row):
                        continue
                    m = meta_for_row[int(idx)]
                    ts_ns = m["timestamp_ns"]
                    ts_str = str(ts_ns)
                    berlin_tz = ZoneInfo("Europe/Berlin")
                    minute_str = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(berlin_tz).strftime("%H:%M")

                    result_entry = {
                        'rank': i,
                        'rosbag': rosbag_name,
                        'similarityScore': float(dist),
                        'topic': m["topic"],
                        'timestamp': ts_str,
                        'minuteOfDay': minute_str,
                        'mcap_identifier': m["mcap_identifier"],
                        'shard_id': m["shard_id"],
                        'row_in_shard': m["row_in_shard"],
                        'model': model_name,
                        '_mark_indices': [],
                    }
                    model_results.append(result_entry)

                    if has_reference_timestamp_index:
                        ref_ts_idx = m.get("reference_timestamp_index")
                        if ref_ts_idx is not None:
                            key = (model_name, rosbag_name, m["topic"])
                            marks.setdefault(key, set()).add(ref_ts_idx)
                            result_entry['_mark_indices'].append(ref_ts_idx)
                    elif ts_str in value_to_ref_ts:
                        matching_ref_timestamps = value_to_ref_ts[ts_str]
                        match_indices: list[int] = []
                        for ref_ts in matching_ref_timestamps:
                            if ref_ts in ref_ts_to_indices:
                                match_indices.extend(ref_ts_to_indices[ref_ts])
                        if match_indices:
                            key = (model_name, rosbag_name, m["topic"])
                            marks.setdefault(key, set()).update(match_indices)
                            result_entry['_mark_indices'].extend(match_indices)

                all_results.extend(model_results)

        if is_search_cancelled(search_id):
            logging.info("[SEARCH-BY-IMAGE] Search cancelled, skipping post-processing")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            SEARCH_PROGRESS["status"] = "cancelled"
            SEARCH_PROGRESS["progress"] = 0.0
            #SEARCH_PROGRESS["message"] = "Search cancelled."
            return jsonify({'cancelled': True, 'results': [], 'marksPerTopic': {}})

        all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
        for rank, result in enumerate(all_results, 1):
            result['rank'] = rank
        filtered_results = all_results

        # Build best-rank lookup for marks
        mark_best_rank: dict[tuple, int] = {}
        for result in filtered_results:
            rank = result['rank']
            key_prefix = (result['model'], result['rosbag'], result['topic'])
            for idx in result.get('_mark_indices', []):
                full_key = (*key_prefix, idx)
                if full_key not in mark_best_rank or rank < mark_best_rank[full_key]:
                    mark_best_rank[full_key] = rank
            result.pop('_mark_indices', None)

        marksPerTopic: dict = {}
        for result in filtered_results:
            model = result['model']
            rosbag = result['rosbag']
            topic = result['topic']
            marksPerTopic.setdefault(model, {}).setdefault(rosbag, {}).setdefault(topic, {'marks': []})

        for key, indices in marks.items():
            model_key, rosbag_key, topic_key = key
            if (
                model_key in marksPerTopic
                and rosbag_key in marksPerTopic[model_key]
                and topic_key in marksPerTopic[model_key][rosbag_key]
            ):
                marksPerTopic[model_key][rosbag_key][topic_key]['marks'].extend(
                    {'value': idx, 'rank': mark_best_rank.get((model_key, rosbag_key, topic_key, idx), 100)}
                    for idx in indices
                )

        if not filtered_results:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
            SEARCH_PROGRESS["message"] = "No results found for the given image."
        else:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0

        return jsonify({
            'imageIdentifier': {
                'rosbag': image_rosbag,
                'topic': image_topic,
                'mcapIdentifier': image_mcap_identifier,
            },
            'results': filtered_results,
            'marksPerTopic': marksPerTopic
        })

    except Exception as e:
        error_id = str(uuid.uuid4())[:8]
        logging.exception("[SEARCH-BY-IMAGE] Search failed [error_id=%s]", error_id)
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Search failed. Error ID: {error_id}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return jsonify({
            'error': 'Search by image failed. Please try again.',
            'error_id': error_id
        }), 500
