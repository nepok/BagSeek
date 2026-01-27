"""Topic and timestamp routes."""
import os
import re
import math
import json
import logging
from flask import Blueprint, jsonify, request
from ..config import TOPICS, ADJACENT_SIMILARITIES, ROSBAGS
from ..state import SELECTED_ROSBAG, get_aligned_data
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag
from ..utils.topics import sort_topics

topics_bp = Blueprint('topics', __name__)


@topics_bp.route('/api/get-available-topics', methods=['GET'])
def get_available_rosbag_topics():
    try:
        rosbag_name = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopics': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        # Topics is now a dict mapping topic names to types
        # Extract the topic names (keys) as a list
        topics_dict = topics_data.get("topics", {})
        if isinstance(topics_dict, dict):
            topics = list(topics_dict.keys())
            topic_types = topics_dict  # Use the dict itself as topic_types mapping
        else:
            # Fallback for old format where topics was a list
            topics = topics_dict if isinstance(topics_dict, list) else []
            topic_types = {}
        
        # Sort topics using the default priority order
        topics = sort_topics(topics, topic_types)
        
        return jsonify({'availableTopics': topics}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopics': []}), 200


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

                # Sort topics using the default priority order
                model_entry[rosbag_name] = sort_topics(topics, topic_types)

            if model_entry:
                results[model_param] = model_entry

        return jsonify({'availableTopics': results}), 200

    except Exception as e:
        logging.error(f"Error scanning adjacent similarities: {e}")
        return jsonify({'availableTopics': {}}), 200


@topics_bp.route('/api/get-available-topic-types', methods=['GET'])
def get_available_rosbag_topic_types():
    try:
        rosbag_name = extract_rosbag_name_from_path(str(SELECTED_ROSBAG))
        topics_json_path = os.path.join(TOPICS, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopicTypes': {}}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        # Topics is now a dict mapping topic names to types
        # Return the topics dict directly as it contains the mapping
        topics_dict = topics_data.get("topics", {})
        if isinstance(topics_dict, dict):
            availableTopicTypes = topics_dict
        else:
            # Fallback for old format where types was in a separate "types" field
            availableTopicTypes = topics_data.get("types", {})
        
        return jsonify({'availableTopicTypes': availableTopicTypes}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopicTypes': {}}), 200


@topics_bp.route('/api/get-available-timestamps', methods=['GET'])
def get_available_timestamps():
    aligned_data = get_aligned_data()
    availableTimestamps = aligned_data['Reference Timestamp'].astype(str).tolist()
    return jsonify({'availableTimestamps': availableTimestamps}), 200


@topics_bp.route('/api/get-timestamp-lengths', methods=['GET'])
def get_timestamp_lengths():
    rosbags = request.args.getlist("rosbags")
    topics = request.args.getlist("topics")  # Optional: topics to get counts for
    timestampLengths = {}

    for rosbag in rosbags:
        rosbag_name = extract_rosbag_name_from_path(rosbag)
        try:
            df = load_lookup_tables_for_rosbag(rosbag_name)
            if df.empty:
                if topics:
                    timestampLengths[rosbag] = {topic: 0 for topic in topics}
                else:
                    timestampLengths[rosbag] = 0
            else:
                if topics:
                    # Return counts per topic
                    topic_counts = {}
                    for topic in topics:
                        if topic in df.columns:
                            count = df[topic].notnull().sum()
                            topic_counts[topic] = int(count)
                        else:
                            topic_counts[topic] = 0
                    timestampLengths[rosbag] = topic_counts
                else:
                    # Backward compatibility: return total count if no topics specified
                    count = df['Reference Timestamp'].notnull().sum() if 'Reference Timestamp' in df.columns else len(df)
                    timestampLengths[rosbag] = int(count)
        except Exception as e:
            if topics:
                timestampLengths[rosbag] = {topic: f"Error: {str(e)}" for topic in topics}
            else:
                timestampLengths[rosbag] = f"Error: {str(e)}"

    return jsonify({'timestampLengths': timestampLengths})


@topics_bp.route('/api/get-timestamp-density', methods=['GET'])
def get_timestamp_density():
    aligned_data = get_aligned_data()
    density_array = aligned_data.drop(columns=["Reference Timestamp"]).notnull().sum(axis=1).tolist()
    return jsonify({'timestampDensity': density_array})


@topics_bp.route('/api/get-topic-mcap-mapping', methods=['GET'])
def get_topic_mcap_mapping():
    """Get mcap_identifier ranges for Reference Timestamp indices.
    
    Returns ranges for ALL topics in the rosbag (mcap_id ranges are the same for all topics).
    Only requires relative_rosbag_path parameter.
    
    Returns contiguous ranges where each range has the same mcap_identifier.
    Each range contains only startIndex and mcap_identifier (endIndex can be derived from next range's startIndex - 1).
    
    Optimized for speed:
    - Cached dataframe loading
    - Minimal data returned (no topicTimestamp, no endIndex)
    - Efficient single-pass iteration
    - One call returns ranges for all topics
    """
    try:
        relative_rosbag_path = request.args.get('relative_rosbag_path')
        
        if not relative_rosbag_path:
            return jsonify({'error': 'Missing required parameter: relative_rosbag_path'}), 400
        
        # Convert relative path to full path, then extract rosbag name
        full_rosbag_path = str(ROSBAGS / relative_rosbag_path)
        rosbag_name = extract_rosbag_name_from_path(full_rosbag_path)
        
        # Load lookup tables for this rosbag (cached)
        df = load_lookup_tables_for_rosbag(rosbag_name, use_cache=True)
        if df.empty:
            return jsonify({'error': 'No lookup table data found'}), 404
        
        # Sort by Reference Timestamp to ensure consistent ordering (only if not already sorted)
        if 'Reference Timestamp' in df.columns:
            # Check if already sorted by checking if first < last
            if len(df) > 1:
                first_ts = df.iloc[0]['Reference Timestamp']
                last_ts = df.iloc[-1]['Reference Timestamp']
                try:
                    if float(first_ts) > float(last_ts):
                        df = df.sort_values('Reference Timestamp').reset_index(drop=True)
                except (ValueError, TypeError):
                    df = df.sort_values('Reference Timestamp').reset_index(drop=True)
        
        total = len(df)  # Number of Reference Timestamps (rows)
        
        if total == 0:
            return jsonify({'ranges': [], 'total': 0}), 200
        
        # Build ranges: group contiguous indices with the same mcap_identifier
        # The mcap_id ranges are the same for all topics (based on _mcap_id column)
        ranges = []
        current_range_start = 0
        current_mcap_id = df.iloc[0].get('_mcap_id')
        
        # Single pass: iterate through dataframe and detect mcap_id changes
        for idx in range(1, total):
            mcap_id = df.iloc[idx].get('_mcap_id')
            
            # If mcap_id changed, close current range and start new one
            if mcap_id != current_mcap_id:
                ranges.append({
                    'startIndex': current_range_start,
                    'mcap_identifier': current_mcap_id
                })
                current_range_start = idx
                current_mcap_id = mcap_id
        
        # Close the last range
        ranges.append({
            'startIndex': current_range_start,
            'mcap_identifier': current_mcap_id
        })
        
        # Sort ranges by startIndex ONLY - this is critical for correct lookup
        # The frontend iterates through ranges sequentially to find which range contains a given index
        # If sorted by mcap_identifier, the lookup will fail because ranges won't be in index order
        ranges.sort(key=lambda r: r['startIndex'])
        
        return jsonify({
            'ranges': ranges,
            'total': total
        }), 200
        
    except Exception as e:
        logging.error(f"Error in get_topic_mcap_mapping: {e}", exc_info=True)
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
