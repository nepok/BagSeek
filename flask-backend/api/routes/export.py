"""Export routes."""
import subprocess
import logging
import re
from pathlib import Path
from typing import Optional, List
from flask import Blueprint, jsonify, request, current_app
from mcap.reader import SeekingReader
from mcap.writer import Writer, CompressionType, IndexType
from mcap_ros2.decoder import DecoderFactory
from ..config import ROSBAGS, EXPORT
from ..state import get_selected_rosbag, get_aligned_data, EXPORT_PROGRESS
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag

export_bp = Blueprint('export', __name__)

# Regex for safe rosbag names (alphanumeric, underscore, hyphen only)
# Prevents path traversal and shell metacharacter injection
SAFE_ROSBAG_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _export_rosbag_batch(exports_list: list) -> tuple:
    """Process multiple exports sequentially. Each item is a single-export request."""
    total = len(exports_list)
    first_source = exports_list[0].get("source_rosbag") if exports_list else None

    for idx, item in enumerate(exports_list):
        if not isinstance(item, dict):
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = f"Invalid export item at index {idx}"
            return jsonify({"error": "Invalid export item"}), 400

        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = (idx / total) if total > 0 else 0
        EXPORT_PROGRESS["message"] = f"Exporting part {idx + 1}/{total}: {item.get('new_rosbag_name', '?')}"

        req_data = dict(item)
        if not req_data.get("source_rosbag") and first_source:
            req_data["source_rosbag"] = first_source

        err = _run_single_export(req_data)
        if err is not None:
            return err

    EXPORT_PROGRESS["status"] = "completed"
    EXPORT_PROGRESS["progress"] = 1.0
    EXPORT_PROGRESS["message"] = f"Exported {total} rosbag(s)"
    return jsonify({
        "message": f"Batch export completed: {total} rosbag(s)",
        "exported_count": total,
    }), 200


def _run_single_export(data: dict):
    """
    Run a single export. Returns None on success, or (response, status_code) tuple on error.
    """
    new_rosbag_name = data.get("new_rosbag_name")
    topics = data.get("topics", [])
    start_index_raw = data.get("start_index")
    end_index_raw = data.get("end_index")
    start_mcap_id_raw = data.get("start_mcap_id")
    end_mcap_id_raw = data.get("end_mcap_id")
    mcap_ranges_raw = data.get("mcap_ranges")
    source_rosbag = data.get("source_rosbag")

    if not new_rosbag_name:
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "new_rosbag_name is required"
        return (jsonify({"error": "new_rosbag_name is required"}), 400)

    if not SAFE_ROSBAG_NAME_PATTERN.match(new_rosbag_name):
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "Invalid rosbag name format."
        return (jsonify({"error": "Invalid rosbag name format"}), 400)

    if start_index_raw is None or end_index_raw is None:
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "start_index and end_index are required"
        return (jsonify({"error": "start_index and end_index are required"}), 400)

    use_mcap_ranges = False
    mcap_ranges: List[tuple] = []
    if mcap_ranges_raw and isinstance(mcap_ranges_raw, list) and len(mcap_ranges_raw) > 0:
        try:
            for r in mcap_ranges_raw:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    mcap_ranges.append((int(r[0]), int(r[1])))
            use_mcap_ranges = len(mcap_ranges) > 0
        except (ValueError, TypeError):
            pass

    if not use_mcap_ranges and (start_mcap_id_raw is None or end_mcap_id_raw is None):
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "start_mcap_id and end_mcap_id are required (or mcap_ranges)"
        return (jsonify({"error": "start_mcap_id and end_mcap_id are required"}), 400)

    try:
        start_index = int(start_index_raw)
        end_index = int(end_index_raw)
        start_mcap_id = int(start_mcap_id_raw) if start_mcap_id_raw is not None else 0
        end_mcap_id = int(end_mcap_id_raw) if end_mcap_id_raw is not None else 0
        if use_mcap_ranges:
            all_ids = []
            for rs, re in mcap_ranges:
                all_ids.extend(range(rs, re + 1))
            if all_ids:
                start_mcap_id = min(all_ids)
                end_mcap_id = max(all_ids)
    except (ValueError, TypeError) as e:
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = f"Invalid number format: {e}"
        return (jsonify({"error": f"Invalid number format: {e}"}), 400)

    if source_rosbag:
        rosbag_name = extract_rosbag_name_from_path(source_rosbag)
        aligned_data = load_lookup_tables_for_rosbag(rosbag_name)
        selected_rosbag_str = str(source_rosbag)
    else:
        sel_rosbag = get_selected_rosbag()
        if sel_rosbag is None:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "No rosbag selected"
            return (jsonify({"error": "No rosbag selected"}), 400)
        selected_rosbag_str = str(sel_rosbag)
        aligned_data = get_aligned_data()

    if aligned_data is None or aligned_data.empty:
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "No aligned data available"
        return (jsonify({"error": "No aligned data available"}), 400)

    if start_index < 0 or start_index >= len(aligned_data) or end_index < 0 or end_index >= len(aligned_data):
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "Index out of range"
        return (jsonify({"error": "Index out of range"}), 400)

    start_timestamp = int(aligned_data.iloc[start_index]['Reference Timestamp'])
    end_timestamp = int(aligned_data.iloc[end_index]['Reference Timestamp'])

    selected_rosbag_path = Path(selected_rosbag_str)
    if selected_rosbag_path.is_absolute():
        try:
            relative_rosbag_path = str(selected_rosbag_path.relative_to(ROSBAGS))
        except ValueError:
            relative_rosbag_path = selected_rosbag_str
        input_rosbag_dir = ROSBAGS / relative_rosbag_path
    else:
        input_rosbag_dir = ROSBAGS / selected_rosbag_str

    output_rosbag_base = EXPORT
    output_rosbag_dir = output_rosbag_base / new_rosbag_name

    def extract_mcap_id(mcap_path: Path) -> str:
        stem = mcap_path.stem
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[1]
        return "0"

    def get_all_mcaps(rosbag_dir: Path) -> List[Path]:
        if not rosbag_dir.exists():
            raise FileNotFoundError(f"Rosbag directory not found: {rosbag_dir}")
        mcaps = list(rosbag_dir.glob("*.mcap"))
        def extract_number(path: Path) -> int:
            try:
                return int(extract_mcap_id(path))
            except ValueError:
                return 999999
        mcaps.sort(key=extract_number)
        return mcaps

    def export_mcap(
        input_mcap_path: Path, output_mcap_path: Path, mcap_id: str,
        start_time_ns: Optional[int], end_time_ns: Optional[int], topics_: Optional[List[str]],
        compression: CompressionType, include_attachments: bool, include_metadata: bool,
    ):
        output_mcap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_mcap_path, "rb") as input_file:
            reader = SeekingReader(input_file, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            if not summary:
                return
            with open(output_mcap_path, "wb") as output_file:
                writer = Writer(
                    output=output_file, compression=compression,
                    index_types=IndexType.ALL, use_chunking=True, use_statistics=True,
                )
                writer.start(profile="ros2", library="mcap-exporter")
                schema_map = {}
                if summary.schemas:
                    for schema_id, schema in summary.schemas.items():
                        schema_map[schema_id] = writer.register_schema(
                            name=schema.name, encoding=schema.encoding, data=schema.data
                        )
                channel_map = {}
                if summary.channels:
                    for channel_id, channel in summary.channels.items():
                        if topics_ and channel.topic not in topics_:
                            continue
                        new_schema_id = schema_map.get(channel.schema_id, 0)
                        channel_map[channel_id] = writer.register_channel(
                            topic=channel.topic, message_encoding=channel.message_encoding,
                            schema_id=new_schema_id, metadata=dict(channel.metadata) if channel.metadata else {},
                        )
                if not channel_map:
                    writer.finish()
                    return
                for schema, channel, message in reader.iter_messages(
                    topics=topics_, start_time=start_time_ns, end_time=end_time_ns,
                    log_time_order=True, reverse=False,
                ):
                    new_channel_id = channel_map.get(channel.id)
                    if new_channel_id is not None:
                        writer.add_message(
                            channel_id=new_channel_id, log_time=message.log_time,
                            data=message.data, publish_time=message.publish_time, sequence=message.sequence,
                        )
                if include_attachments:
                    for attachment in reader.iter_attachments():
                        include_a = True
                        if start_time_ns is not None and attachment.log_time < start_time_ns:
                            include_a = False
                        if end_time_ns is not None and attachment.log_time > end_time_ns:
                            include_a = False
                        if include_a:
                            writer.add_attachment(
                                create_time=attachment.create_time, log_time=attachment.log_time,
                                name=attachment.name, media_type=attachment.media_type, data=attachment.data,
                            )
                if include_metadata:
                    for metadata in reader.iter_metadata():
                        writer.add_metadata(
                            name=metadata.name, data=dict(metadata.metadata) if metadata.metadata else {},
                        )
                writer.finish()

    input_mcaps = get_all_mcaps(input_rosbag_dir)
    output_rosbag_dir.mkdir(parents=True, exist_ok=True)

    if use_mcap_ranges:
        mcap_nums_to_process = []
        for rs, re in mcap_ranges:
            mcap_nums_to_process.extend(range(rs, re + 1))
        mcap_nums_to_process = sorted(set(mcap_nums_to_process))
        start_mcap_num = mcap_nums_to_process[0] if mcap_nums_to_process else start_mcap_id
        end_mcap_num = mcap_nums_to_process[-1] if mcap_nums_to_process else end_mcap_id
    else:
        start_mcap_num = int(start_mcap_id)
        end_mcap_num = int(end_mcap_id)
        if start_mcap_num > end_mcap_num:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            return (jsonify({"error": "Start MCAP ID must be <= End MCAP ID"}), 400)
        mcap_nums_to_process = list(range(start_mcap_num, end_mcap_num + 1))

    exported_count = 0
    for mcap_index, mcap_num in enumerate(mcap_nums_to_process):
        mcap_id = str(mcap_num)
        input_mcap = None
        for mcap_path in input_mcaps:
            if extract_mcap_id(mcap_path) == mcap_id:
                input_mcap = mcap_path
                break
        if input_mcap is None:
            continue
        output_mcap_name = f"{new_rosbag_name}.mcap"
        output_mcap_path = output_rosbag_dir / output_mcap_name
        if mcap_num == start_mcap_num:
            mcap_start_time, mcap_end_time = start_timestamp, None
        elif mcap_num == end_mcap_num:
            mcap_start_time, mcap_end_time = None, end_timestamp
        else:
            mcap_start_time, mcap_end_time = None, None
        try:
            export_mcap(
                input_mcap_path=input_mcap, output_mcap_path=output_mcap_path, mcap_id=mcap_id,
                start_time_ns=mcap_start_time, end_time_ns=mcap_end_time,
                topics_=topics if topics else None, compression=CompressionType.ZSTD,
                include_attachments=True, include_metadata=True,
            )
            exported_count += 1
        except Exception as e:
            current_app.logger.error(f"Failed to export MCAP {mcap_id}: {e}", exc_info=True)
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = str(e)
            return (jsonify({"error": str(e)}), 500)

    if exported_count > 0:
        try:
            result = subprocess.run(
                ["ros2", "bag", "reindex", str(output_rosbag_dir), "-s", "mcap"],
                capture_output=True, text=True, check=True
            )
            current_app.logger.info(f"Reindexed {output_rosbag_dir}: {result.stdout or 'ok'}")
        except subprocess.CalledProcessError as e:
            current_app.logger.error(f"ros2 bag reindex failed for {output_rosbag_dir}: {e.stderr or e}")
        except FileNotFoundError:
            current_app.logger.warning("ros2 command not found; skipping reindex. Ensure ROS2 is in PATH.")

    return None


@export_bp.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)


@export_bp.route('/api/export-rosbag', methods=['POST'])
def export_rosbag():
    """
    Export MCAP files from a rosbag based on MCAP ID range and time filtering.

    Single export request body:
    {
        "new_rosbag_name": str,
        "topics": List[str],
        "start_index": int,
        "end_index": int,
        "start_mcap_id": int,
        "end_mcap_id": int,
        "mcap_ranges": [[start_id, end_id], ...],
        "source_rosbag": str (optional)
    }

    Batch export (multiple MCAP ranges as separate rosbags):
    {
        "exports": [
            {
                "new_rosbag_name": str,
                "topics": List[str],
                "start_index": int,
                "end_index": int,
                "mcap_ranges": [[start_id, end_id]],
                "source_rosbag": str (optional)
            },
            ...
        ]
    }
    """
    # Reset export status at the beginning
    EXPORT_PROGRESS["status"] = "idle"
    EXPORT_PROGRESS["progress"] = -1
    EXPORT_PROGRESS["message"] = "Waiting for export..."

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Batch mode: multiple exports in one request
        exports_list = data.get("exports")
        if exports_list and isinstance(exports_list, list) and len(exports_list) > 0:
            return _export_rosbag_batch(exports_list)

        # Single export mode – delegate to _run_single_export
        err = _run_single_export(data)
        if err is not None:
            return err
        EXPORT_PROGRESS["status"] = "completed"
        EXPORT_PROGRESS["progress"] = 1.0
        EXPORT_PROGRESS["message"] = "Export completed!"
        return jsonify({
            "message": "Export completed successfully",
            "output_directory": str(EXPORT / data.get("new_rosbag_name", "")),
        }), 200

    except Exception as e:
        current_app.logger.error(f"Export failed: {e}", exc_info=True)
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = f"Export failed: {str(e)}"
        return jsonify({"error": str(e)}), 500


