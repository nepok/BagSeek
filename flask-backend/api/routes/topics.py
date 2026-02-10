"""Topic and timestamp routes."""
import os
import re
import math
import json
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request
import pandas as pd
import pyarrow.parquet as pq
from ..config import TOPICS, ADJACENT_SIMILARITIES, ROSBAGS, LOOKUP_TABLES
from ..state import get_selected_rosbag, get_aligned_data
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag

topics_bp = Blueprint('topics', __name__)


@topics_bp.route('/api/get-available-topics', methods=['GET'])
def get_available_rosbag_topics():
    """Returns unified topics dict: { topic_name: message_type }

    Response: { topics: { "/camera/image": "sensor_msgs/msg/Image", ... } }
    """
    try:
        selected_rosbag = get_selected_rosbag()
        if selected_rosbag is None:
            return jsonify({'topics': {}}), 200

        rosbag_name = extract_rosbag_name_from_path(str(selected_rosbag))
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'topics': {}}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        # Topics is now a dict mapping topic names to types
        topics_dict = topics_data.get("topics", {})
        if isinstance(topics_dict, dict):
            topic_types = topics_dict
        else:
            # Fallback for old format where topics was a list
            topics_list = topics_dict if isinstance(topics_dict, list) else []
            topic_types = {t: "" for t in topics_list}

        # Return topics as-is (sorting is now handled in frontend)
        return jsonify({'topics': topic_types}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'topics': {}}), 200


@topics_bp.route('/api/get-available-image-topics', methods=['GET'])
def get_available_image_topics():
    try:
        model_params = request.args.getlist("models")
        rosbag_params = request.args.getlist("rosbags")

        if not model_params or not rosbag_params:
            return jsonify({'availableTopics': {}}), 200

        results = {}

        for model_param in model_params:
            model_path = os.path.join(ADJACENT_SIMILARITIES, model_param)
            if not os.path.isdir(model_path):
                continue

            model_entry = {}
            for rosbag_param in rosbag_params:
                rosbag_name = extract_rosbag_name_from_path(rosbag_param)
                rosbag_path = os.path.join(model_path, rosbag_name)
                if not os.path.isdir(rosbag_path):
                    continue

                topics = []
                for topic in os.listdir(rosbag_path):
                    topics.append(topic.replace("__", "/"))

                # Try to load topic types from topics JSON file
                topic_types = {}
                try:
                    topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")
                    if os.path.exists(topics_json_path):
                        with open(topics_json_path, 'r') as f:
                            topics_data = json.load(f)
                            topics_dict = topics_data.get("topics", {})
                            if isinstance(topics_dict, dict):
                                topic_types = topics_dict
                except Exception as e:
                    logging.debug(f"Could not load topic types for {rosbag_name}: {e}")

                # Return topics as-is (sorting is now handled in frontend)
                model_entry[rosbag_name] = topics

            if model_entry:
                results[model_param] = model_entry

        return jsonify({'availableTopics': results}), 200

    except Exception as e:
        logging.error(f"Error scanning adjacent similarities: {e}")
        return jsonify({'availableTopics': {}}), 200


@topics_bp.route('/api/get-timestamp-summary', methods=['GET'])
def get_timestamp_summary():
    """Combined endpoint: timestamp count, bounds, and MCAP boundary ranges.

    Fast path: reads precomputed summary.json (written during preprocessing).
    Fallback: reads pyarrow parquet metadata (file footers only, no row data).
    """
    empty_response = {
        'count': 0,
        'firstTimestampNs': None,
        'lastTimestampNs': None,
        'mcapRanges': [],
    }
    try:
        relative_rosbag_path = request.args.get('rosbag')

        if relative_rosbag_path:
            full_rosbag_path = str(ROSBAGS / relative_rosbag_path)
            rosbag_name = extract_rosbag_name_from_path(full_rosbag_path)
        else:
            selected_rosbag = get_selected_rosbag()
            if selected_rosbag is None:
                return jsonify(empty_response), 200
            rosbag_name = extract_rosbag_name_from_path(str(selected_rosbag))

        lookup_dir = Path(LOOKUP_TABLES) / rosbag_name

        # Fast path: precomputed summary.json
        summary_path = lookup_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            mcap_ranges = []
            needs_ts_backfill = False
            for r in summary.get('mcap_ranges', []):
                entry = {
                    'startIndex': r['start_index'],
                    'mcapIdentifier': r['mcap_id'],
                    'count': r.get('count', 0),
                }
                if 'first_timestamp_ns' in r:
                    entry['firstTimestampNs'] = str(r['first_timestamp_ns'])
                if 'last_timestamp_ns' in r:
                    entry['lastTimestampNs'] = str(r['last_timestamp_ns'])
                else:
                    needs_ts_backfill = True
                mcap_ranges.append(entry)

            # Backfill per-MCAP timestamps from parquet metadata if summary.json is old
            if needs_ts_backfill:
                for entry in mcap_ranges:
                    if 'firstTimestampNs' in entry:
                        continue
                    pf_path = lookup_dir / f"{entry['mcapIdentifier']}.parquet"
                    if not pf_path.exists():
                        continue
                    try:
                        pf = pq.ParquetFile(pf_path)
                        schema = pf.schema_arrow
                        col_idx = schema.get_field_index('Reference Timestamp')
                        f_min, f_max = None, None
                        for rg_idx in range(pf.metadata.num_row_groups):
                            stats = pf.metadata.row_group(rg_idx).column(col_idx).statistics
                            if stats and stats.has_min_max:
                                if f_min is None or stats.min < f_min:
                                    f_min = stats.min
                                if f_max is None or stats.max > f_max:
                                    f_max = stats.max
                        if f_min is not None:
                            entry['firstTimestampNs'] = str(int(f_min))
                        if f_max is not None:
                            entry['lastTimestampNs'] = str(int(f_max))
                    except Exception:
                        pass

            first_ts = summary.get('first_timestamp_ns')
            last_ts = summary.get('last_timestamp_ns')
            return jsonify({
                'count': summary.get('total_count', 0),
                'firstTimestampNs': str(first_ts) if first_ts is not None else None,
                'lastTimestampNs': str(last_ts) if last_ts is not None else None,
                'mcapRanges': mcap_ranges,
            }), 200

        # Fallback: read pyarrow parquet metadata (file footers only)
        if not lookup_dir.exists():
            return jsonify(empty_response), 200

        parquet_files = sorted(
            lookup_dir.glob('*.parquet'),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float('inf'),
        )
        if not parquet_files:
            return jsonify(empty_response), 200

        total_count = 0
        global_first = None
        global_last = None
        mcap_ranges = []

        for pf_path in parquet_files:
            try:
                pf = pq.ParquetFile(pf_path)
                metadata = pf.metadata
                n = metadata.num_rows
                if n == 0:
                    continue

                mcap_id = pf_path.stem

                # Try to get min/max from row group statistics
                schema = pf.schema_arrow
                col_idx = schema.get_field_index('Reference Timestamp')
                file_min = None
                file_max = None
                for rg_idx in range(metadata.num_row_groups):
                    col_stats = metadata.row_group(rg_idx).column(col_idx).statistics
                    if col_stats is not None and col_stats.has_min_max:
                        if file_min is None or col_stats.min < file_min:
                            file_min = col_stats.min
                        if file_max is None or col_stats.max > file_max:
                            file_max = col_stats.max

                # Fallback: read the column
                if file_min is None or file_max is None:
                    df = pd.read_parquet(pf_path, columns=['Reference Timestamp'])
                    file_min = int(df['Reference Timestamp'].min())
                    file_max = int(df['Reference Timestamp'].max())

                mcap_ranges.append({
                    'startIndex': total_count,
                    'mcapIdentifier': mcap_id,
                    'count': n,
                    'firstTimestampNs': str(file_min) if file_min is not None else None,
                    'lastTimestampNs': str(file_max) if file_max is not None else None,
                })
                if global_first is None or file_min < global_first:
                    global_first = file_min
                if global_last is None or file_max > global_last:
                    global_last = file_max
                total_count += n
            except Exception as e:
                logging.warning(f"Failed to read parquet metadata from {pf_path}: {e}")
                continue

        return jsonify({
            'count': total_count,
            'firstTimestampNs': str(global_first) if global_first is not None else None,
            'lastTimestampNs': str(global_last) if global_last is not None else None,
            'mcapRanges': mcap_ranges,
        }), 200

    except Exception as e:
        logging.error(f"Error in get_timestamp_summary: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@topics_bp.route('/api/get-topic-timestamp-at-index', methods=['GET'])
def get_topic_timestamp_at_index():
    """Get topic timestamp for a specific index. Lightweight endpoint for hover previews."""
    try:
        relative_rosbag_path = request.args.get('relative_rosbag_path')
        topic = request.args.get('topic')
        index = request.args.get('index', type=int)
        
        if not relative_rosbag_path or not topic or index is None:
            return jsonify({'error': 'Missing required parameters: relative_rosbag_path, topic, index'}), 400
        
        # Convert relative path to full path, then extract rosbag name
        full_rosbag_path = str(ROSBAGS / relative_rosbag_path)
        rosbag_name = extract_rosbag_name_from_path(full_rosbag_path)
        df = load_lookup_tables_for_rosbag(rosbag_name, use_cache=True)
        
        if df.empty or index < 0 or index >= len(df):
            return jsonify({'error': 'Index out of range'}), 404
        
        # Sort if needed
        if 'Reference Timestamp' in df.columns and len(df) > 1:
            first_ts = df.iloc[0]['Reference Timestamp']
            last_ts = df.iloc[-1]['Reference Timestamp']
            try:
                if float(first_ts) > float(last_ts):
                    df = df.sort_values('Reference Timestamp').reset_index(drop=True)
            except (ValueError, TypeError):
                df = df.sort_values('Reference Timestamp').reset_index(drop=True)
        
        if topic not in df.columns:
            return jsonify({'error': f'Topic column "{topic}" not found'}), 404
        
        topic_ts = df.iloc[index][topic]
        
        # Helper to check if value is non-nan
        def non_nan(v):
            if v is None:
                return False
            if isinstance(v, float):
                return not math.isnan(v)
            s = str(v)
            return s.lower() != 'nan' and s != '' and s != 'None'
        
        # If empty, search nearby (up to 100 rows)
        if not non_nan(topic_ts):
            radius = min(100, len(df) - 1)
            for off in range(1, radius + 1):
                for candidate_idx in [index - off, index + off]:
                    if 0 <= candidate_idx < len(df):
                        candidate_ts = df.iloc[candidate_idx][topic]
                        if non_nan(candidate_ts):
                            topic_ts = candidate_ts
                            break
                if non_nan(topic_ts):
                    break
        
        return jsonify({'topicTimestamp': str(topic_ts) if non_nan(topic_ts) else None}), 200
        
    except Exception as e:
        logging.error(f"Error in get_topic_timestamp_at_index: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
