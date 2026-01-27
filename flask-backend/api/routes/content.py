"""Content routes for MCAP data retrieval."""
import math
import logging
import re
from pathlib import Path
from flask import Blueprint, jsonify, request
from mcap.reader import SeekingReader
from mcap_ros2.decoder import DecoderFactory
from ..config import ROSBAGS
from ..state import SELECTED_ROSBAG, get_aligned_data, get_selected_rosbag
from .. import state
from ..utils.mcap import format_message_response

content_bp = Blueprint('content', __name__)

# Regex for safe mcap identifiers (alphanumeric, underscore, hyphen only)
SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _is_safe_path(path: str) -> bool:
    """Validate path to prevent directory traversal attacks."""
    if not path:
        return False
    return '..' not in path and not path.startswith('/')


@content_bp.route('/api/set-reference-timestamp', methods=['POST'])
def set_reference_timestamp():
    """Set the current reference timestamp and retrieve its aligned mappings."""
    try:
        data = request.get_json()
        referenceTimestamp = data.get('referenceTimestamp')
        
        if not referenceTimestamp:
            return jsonify({"error": "Missing referenceTimestamp"}), 400

        aligned_data = get_aligned_data()
        row = aligned_data[aligned_data["Reference Timestamp"] == str(referenceTimestamp)]
        if row.empty:
            return jsonify({"error": "Reference timestamp not found in CSV"}), 404

        state.set_reference_timestamp(referenceTimestamp)
        mapped_timestamps_dict = row.iloc[0].to_dict()
        # Extract mcap_identifier from the row (exclude it from mapped_timestamps)
        mcap_identifier = mapped_timestamps_dict.pop('_mcap_id', None)
        state.set_mapped_timestamps(mapped_timestamps_dict)
        # Convert NaNs to None for safe JSON serialization
        cleaned_mapped_timestamps = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in mapped_timestamps_dict.items()
        }
        return jsonify({"mappedTimestamps": cleaned_mapped_timestamps, "mcapIdentifier": mcap_identifier, "message": "Reference timestamp updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@content_bp.route('/api/content-mcap', methods=['GET', 'POST'])
def get_content_mcap():
    """Get content from MCAP file for a specific topic, timestamp, and mcap identifier."""
    relative_rosbag_path = request.args.get('relative_rosbag_path')
    topic = request.args.get('topic')
    mcap_identifier = request.args.get('mcap_identifier')
    timestamp = request.args.get('timestamp', type=int)
    
    # Validate required parameters
    if not topic or not mcap_identifier or timestamp is None:
        return jsonify({'error': 'Missing required parameters: topic, mcap_identifier, and timestamp are required'}), 400

    # Security: Validate mcap_identifier to prevent path traversal
    if not SAFE_IDENTIFIER_PATTERN.match(mcap_identifier):
        return jsonify({'error': 'Invalid mcap_identifier format'}), 400
    
    # Handle missing relative_rosbag_path - use SELECTED_ROSBAG as fallback
    # Capture once with thread-safe getter to avoid race conditions
    if not relative_rosbag_path:
        current_selected_rosbag = get_selected_rosbag()
        if not current_selected_rosbag:
            return jsonify({'error': 'No rosbag selected and relative_rosbag_path not provided'}), 400
        # Get relative path from selected rosbag
        # It might be a string or Path, and might be absolute or relative
        selected_rosbag_str = str(current_selected_rosbag)
        selected_rosbag_path = Path(selected_rosbag_str)
        
        # Check if it's an absolute path (starts with /) or relative
        if selected_rosbag_path.is_absolute():
            # Absolute path - get relative path from ROSBAGS
            try:
                relative_rosbag_path = str(selected_rosbag_path.relative_to(ROSBAGS))
            except ValueError:
                # If not a subpath, try to construct it from the basename
                relative_rosbag_path = selected_rosbag_path.name
        else:
            # Already a relative path - use it directly
            relative_rosbag_path = selected_rosbag_str

    # Security: Validate relative_rosbag_path to prevent directory traversal
    if not _is_safe_path(relative_rosbag_path):
        return jsonify({'error': 'Invalid rosbag path'}), 400

    # Extract base rosbag name for MCAP filename
    # For multipart rosbags like "rosbag2_2025_07_23-07_29_39_multi_parts/Part_2",
    # we need to get the base name (before _multi_parts) for the MCAP filename
    # MCAP files are named like: rosbag2_2025_07_23-07_29_39_669.mcap
    if '_multi_parts' in relative_rosbag_path:
        # Extract base name before _multi_parts
        base_rosbag_name = relative_rosbag_path.split('_multi_parts')[0]
    else:
        # For regular rosbags, use the full path as base name
        base_rosbag_name = relative_rosbag_path
    
    # Construct MCAP path using Path objects for proper handling
    # MCAP files are in the rosbag directory, named: {base_rosbag_name}_{mcap_identifier}.mcap
    mcap_path = ROSBAGS / relative_rosbag_path / f"{base_rosbag_name}_{mcap_identifier}.mcap"
    
    logging.warning(f"[CONTENT_MCAP] MCAP path: {mcap_path}")
    
    try:        
        with open(mcap_path, "rb") as f:
            reader = SeekingReader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                topics=[topic],
                start_time=timestamp,
                end_time=timestamp + 1,
                log_time_order=True,
                reverse=False
            ):
                # Get schema name for message type
                schema_name = schema.name if schema else None
                if not schema_name:
                    return jsonify({'error': 'No schema found for message'}), 404
                    
                # Use the shared format_message_response function
                # Convert timestamp to string to match the function signature
                return format_message_response(ros2_msg, schema_name, str(timestamp))
        
        return jsonify({'error': 'No message found for the provided timestamp and topic'}), 404
        
    except Exception as e:
        logging.exception("[C] Failed to read mcap file")
        return jsonify({'error': f'Error reading mcap file: {str(e)}'}), 500
