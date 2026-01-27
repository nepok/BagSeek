"""Search routes."""
import uuid
import logging
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
from ..state import SEARCH_PROGRESS
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag
from ..utils.clip import get_text_embedding, load_agriclip, resolve_custom_checkpoint, tokenize_texts

search_bp = Blueprint('search', __name__)


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


@search_bp.route('/api/search', methods=['GET'])
def search():
    # ---- Debug logging for paths
    logging.debug("[SEARCH] LOOKUP_TABLES=%s (exists=%s)", LOOKUP_TABLES, LOOKUP_TABLES.exists() if LOOKUP_TABLES else False)
    logging.debug("[SEARCH] EMBEDDINGS=%s (exists=%s)", EMBEDDINGS, EMBEDDINGS.exists() if EMBEDDINGS else False)

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
    try:
        models_list = [m.strip() for m in models.split(",") if m.strip()]  # Filter out empty model names
        rosbags_list = [extract_rosbag_name_from_path(r.strip()) for r in rosbags.split(",") if r.strip()]
        time_start, time_end = map(int, timeRange.split(","))
        k_subsample = max(1, int(accuracy))
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

    try:
        total_steps = max(1, len(models_list) * len(rosbags_list))
        step_count = 0
        logging.debug("[SEARCH] total_steps=%d device=%s torch.cuda.is_available()=%s", total_steps, device, torch.cuda.is_available())

        for model_name in models_list:
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
                continue

            if query_embedding is None:
                logging.debug("[SEARCH] Query embedding is None for model %s; skipping", model_name)
                continue

            query_embedding = query_embedding.astype("float32", copy=False)

            for rosbag_name in rosbags_list:
                step_count += 1
                SEARCH_PROGRESS["status"] = "running"
                SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 3)
                SEARCH_PROGRESS["message"] = (
                    "Searching consolidated shards...\n\n"
                    f"Model: {model_name}\n"
                    f"Rosbag: {rosbag_name}\n\n"
                    f"(Sampling every {k_subsample}th embedding)"
                )

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
                needed_cols = ["topic", "minute_of_day", "shard_id", "row_in_shard", "timestamp_ns", "mcap_identifier"]
                try:
                    import pyarrow.parquet as pq
                    available_cols = set(pq.read_schema(manifest_path).names)
                except Exception as e:
                    logging.debug("[SEARCH] Could not read schema with pyarrow (%s), fallback to pandas head: %s", type(e).__name__, e)
                    head_df = pd.read_parquet(manifest_path)
                    available_cols = set(head_df.columns)
                    logging.debug("[SEARCH] Manifest columns (fallback): %s", sorted(available_cols))

                missing = [c for c in needed_cols if c not in available_cols]
                if missing:
                    msg = f"[SEARCH] Manifest missing columns {missing}; present={sorted(available_cols)}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                # ---- Read needed columns
                mf = pd.read_parquet(manifest_path, columns=needed_cols)

                # ---- Filter by time window
                pre_count = len(mf)
                mf = mf.loc[(mf["minute_of_day"] >= time_start) & (mf["minute_of_day"] <= time_end)]
                if mf.empty:
                    logging.debug("[SEARCH] SKIP: No rows in time window for %s/%s", model_name, rosbag_name)
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

                if mf_sel.empty:
                    logging.debug("[SEARCH] SKIP: No rows after subsample for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Load aligned CSV for marks
                lookup_dir = (LOOKUP_TABLES / rosbag_name) if LOOKUP_TABLES else None
                logging.debug("[SEARCH] LOOKUP_TABLES dir for %s: %s (exists=%s)", rosbag_name, lookup_dir, lookup_dir.exists() if lookup_dir else False)
                aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
                logging.debug("[SEARCH] Loaded lookup tables for %s: shape=%s, columns=%s", rosbag_name, aligned_data.shape if not aligned_data.empty else "empty", list(aligned_data.columns) if not aligned_data.empty else [])
                if aligned_data.empty:
                    logging.debug("[SEARCH] WARN: no lookup tables found for %s (marks will be empty)", rosbag_name)

                # ---- Gather vectors by shard
                chunks: list[np.ndarray] = []
                meta_for_row: list[dict] = []
                shards_touched = 0
                bytes_read = 0
                shard_ids = sorted(mf_sel["shard_id"].unique().tolist())

                for shard_id, df_s in mf_sel.groupby("shard_id", sort=False):
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
                        meta_map[int(r.row_in_shard)] = {
                            "timestamp_ns": int(r.timestamp_ns),
                            "topic": topic_str.replace("__", "/"),
                            "topic_folder": topic_folder,
                            "minute_of_day": int(r.minute_of_day),
                            "mcap_identifier": str(r.mcap_identifier),
                            "shard_id": shard_id,
                            "row_in_shard": int(r.row_in_shard),
                        }

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

                if not chunks:
                    msg = f"[SEARCH] No chunks loaded (missing shards/files?) for {model_name}/{rosbag_name}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                X = np.vstack(chunks).astype("float32", copy=False)
                if X.shape[0] != len(meta_for_row):
                    msg = f"[SEARCH] Row/meta mismatch: X={X.shape[0]} vs meta={len(meta_for_row)}"
                    logging.debug(msg)
                    SEARCH_PROGRESS["message"] = msg
                    continue

                # ---- FAISS search
                index = faiss.IndexFlatL2(X.shape[1])
                index.add(X)
                D, I = index.search(query_embedding.reshape(1, -1), MAX_K)
                if D.size == 0 or I.size == 0:
                    logging.debug("[SEARCH] Empty FAISS result for %s/%s", model_name, rosbag_name)
                    continue

                # ---- Build API-like results
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
                    
                    # Build embedding path with mcap_identifier
                    pt_path = EMBEDDINGS / model_name / rosbag_name / m["topic_folder"] / m["mcap_identifier"] / f"{ts_ns}.pt"
                    
                    if i == 1:  # Log first result path construction
                        logging.debug("[SEARCH] EMBEDDINGS path construction: %s (exists=%s)", pt_path, pt_path.exists() if pt_path else False)
                    
                    model_results.append({
                        'rank': i,
                        'rosbag': rosbag_name,
                        'embedding_path': str(pt_path),
                        'similarityScore': float(dist),
                        'topic': m["topic"],           # slashes
                        'timestamp': ts_str,
                        'minuteOfDay': minute_str,
                        'mcap_identifier': m["mcap_identifier"],
                        'model': model_name
                    })
                    if not aligned_data.empty:
                        matching_reference_timestamps = aligned_data.loc[
                            aligned_data.isin([ts_str]).any(axis=1),
                            'Reference Timestamp'
                        ].tolist()
                        if matching_reference_timestamps:
                            match_indices: list[int] = []
                            for ref_ts in matching_reference_timestamps:
                                idxs = aligned_data.index[aligned_data['Reference Timestamp'] == ref_ts].tolist()
                                match_indices.extend(idxs)
                            if match_indices:
                                key = (model_name, rosbag_name, m["topic"])
                                marks.setdefault(key, set()).update(match_indices)

                all_results.extend(model_results)

            # Cleanup GPU per model
            del model
            torch.cuda.empty_cache()

        # ---- Post processing
        all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
        for rank, result in enumerate(all_results, 1):
            result['rank'] = rank
        filtered_results = all_results

        # marksPerTopic
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
                    {'value': idx} for idx in indices
                )

        # Check if no results were found
        if not filtered_results:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
            SEARCH_PROGRESS["message"] = "No results found for the given query."
        else:
            SEARCH_PROGRESS["status"] = "done"
            SEARCH_PROGRESS["progress"] = 1.0
        
        return jsonify({
            'query': query_text,
            'results': filtered_results,
            'marksPerTopic': marksPerTopic
        })

    except Exception as e:
        # Generate error ID for debugging without exposing internals
        error_id = str(uuid.uuid4())[:8]
        logging.exception(f"[SEARCH] Search failed [error_id={error_id}]")
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Search failed. Error ID: {error_id}"
        return jsonify({
            'error': 'Search failed. Please try again.',
            'error_id': error_id
        }), 500
