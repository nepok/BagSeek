"""Export routes."""
import subprocess
import logging
import re
import json
import struct
from pathlib import Path
from typing import Optional, List
from flask import Blueprint, jsonify, request, current_app
from mcap.reader import SeekingReader
from mcap.writer import Writer, CompressionType, IndexType
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from ..config import ROSBAGS, EXPORT, EXPORT_RAW
from ..state import get_selected_rosbag, get_aligned_data, EXPORT_PROGRESS
from ..utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag
from ..utils.mcap import normalize_topic

export_bp = Blueprint('export', __name__)

# Regex for safe rosbag names (alphanumeric, underscore, hyphen only)
# Prevents path traversal and shell metacharacter injection
SAFE_ROSBAG_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _make_export_parent_folder(relative_rosbag_path: str) -> str:
    """Build a safe parent folder name from relative rosbag path: path_export."""
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', relative_rosbag_path.replace('/', '_').strip('_'))
    return f"{safe}_export" if safe else "export"


def _export_rosbag_batch(exports_list: list) -> tuple:
    """Process multiple exports sequentially. Each item is a single-export request."""
    total = len(exports_list)

    # Determine parent folder name based on source rosbags
    unique_sources = list(dict.fromkeys(
        item.get("source_rosbag") for item in exports_list if isinstance(item, dict) and item.get("source_rosbag")
    ))

    output_parent = None
    if total > 1:
        if len(unique_sources) == 1:
            # Single source rosbag: use its name as parent
            try:
                p = Path(str(unique_sources[0]))
                if p.is_absolute():
                    rel = str(p.relative_to(ROSBAGS))
                else:
                    rel = str(unique_sources[0])
                output_parent = _make_export_parent_folder(rel)
            except (ValueError, TypeError):
                pass
        elif len(unique_sources) > 1:
            # Multiple source rosbags: use generic parent
            output_parent = "multi_rosbag_export"
    first_source = unique_sources[0] if unique_sources else None

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
        if output_parent is not None:
            req_data["output_parent"] = output_parent

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
    output_parent = data.get("output_parent")

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

    if output_parent and isinstance(output_parent, str) and '..' not in output_parent and '/' not in output_parent:
        safe_parent = output_parent
    else:
        safe_parent = None
    output_rosbag_base = EXPORT / safe_parent if safe_parent else EXPORT
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
    EXPORT_PROGRESS["status"] = "running"
    EXPORT_PROGRESS["progress"] = 0
    EXPORT_PROGRESS["message"] = "Preparing export..."

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


# ---------------------------------------------------------------------------
# Raw data export
# ---------------------------------------------------------------------------

def _write_raw_message(ros2_msg, schema_name: str, output_dir: Path, timestamp_ns: int):
    """Write a single decoded ROS2 message as a raw file. Returns the written path or None."""
    output_dir.mkdir(parents=True, exist_ok=True)

    match schema_name:
        case 'sensor_msgs/msg/CompressedImage':
            image_data = bytes(ros2_msg.data) if hasattr(ros2_msg, 'data') else None
            if not image_data:
                return None
            fmt = 'jpg'
            if hasattr(ros2_msg, 'format') and ros2_msg.format:
                fl = ros2_msg.format.lower()
                if 'png' in fl:
                    fmt = 'png'
            path = output_dir / f"{timestamp_ns}.{fmt}"
            path.write_bytes(image_data)
            return path

        case 'sensor_msgs/msg/Image':
            raw_data = bytes(ros2_msg.data) if hasattr(ros2_msg, 'data') else None
            if not raw_data:
                return None
            try:
                from PIL import Image as PILImage
                width = ros2_msg.width
                height = ros2_msg.height
                encoding = ros2_msg.encoding if hasattr(ros2_msg, 'encoding') else 'rgb8'
                if encoding in ('rgb8', 'bgr8'):
                    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                    if encoding == 'bgr8':
                        arr = arr[:, :, ::-1]
                    img = PILImage.fromarray(arr, 'RGB')
                elif encoding in ('rgba8', 'bgra8'):
                    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
                    if encoding == 'bgra8':
                        arr = arr[:, :, [2, 1, 0, 3]]
                    img = PILImage.fromarray(arr, 'RGBA')
                elif encoding == 'mono8':
                    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
                    img = PILImage.fromarray(arr, 'L')
                elif encoding == '16UC1' or encoding == 'mono16':
                    arr = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))
                    arr8 = (arr / 256).astype(np.uint8)
                    img = PILImage.fromarray(arr8, 'L')
                else:
                    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, -1))
                    if arr.shape[2] >= 3:
                        img = PILImage.fromarray(arr[:, :, :3], 'RGB')
                    else:
                        img = PILImage.fromarray(arr[:, :, 0], 'L')
                path = output_dir / f"{timestamp_ns}.png"
                img.save(str(path))
                return path
            except Exception as e:
                logging.warning(f"Failed to convert Image message: {e}")
                path = output_dir / f"{timestamp_ns}.raw"
                path.write_bytes(raw_data)
                return path

        case 'sensor_msgs/msg/PointCloud2':
            points = []
            colors = []
            point_step = ros2_msg.point_step
            is_bigendian = ros2_msg.is_bigendian
            has_rgb = False
            has_separate_rgb = False
            rgb_offset = None
            r_offset = g_offset = b_offset = None
            r_datatype = None

            for field in ros2_msg.fields:
                if field.name in ('rgb', 'rgba'):
                    rgb_offset = field.offset
                    has_rgb = True
                    break
                elif field.name == 'r':
                    r_offset = field.offset
                    r_datatype = field.datatype
                    has_separate_rgb = True
                elif field.name == 'g':
                    g_offset = field.offset
                elif field.name == 'b':
                    b_offset = field.offset

            for i in range(0, len(ros2_msg.data), point_step):
                endian = '>' if is_bigendian else '<'
                x, y, z = struct.unpack_from(f'{endian}fff', ros2_msg.data, i)
                if not all(np.isfinite([x, y, z])):
                    continue
                r_val = g_val = b_val = None
                if has_rgb and rgb_offset is not None:
                    try:
                        rgb_int = struct.unpack_from(f'{endian}I', ros2_msg.data, i + rgb_offset)[0]
                        if is_bigendian:
                            r_val = (rgb_int >> 24) & 0xFF
                            g_val = (rgb_int >> 16) & 0xFF
                            b_val = (rgb_int >> 8) & 0xFF
                        else:
                            b_val = rgb_int & 0xFF
                            g_val = (rgb_int >> 8) & 0xFF
                            r_val = (rgb_int >> 16) & 0xFF
                    except Exception:
                        pass
                elif has_separate_rgb and r_offset is not None and g_offset is not None and b_offset is not None:
                    try:
                        if r_datatype == 7:  # FLOAT32
                            r_f = struct.unpack_from(f'{endian}f', ros2_msg.data, i + r_offset)[0]
                            g_f = struct.unpack_from(f'{endian}f', ros2_msg.data, i + g_offset)[0]
                            b_f = struct.unpack_from(f'{endian}f', ros2_msg.data, i + b_offset)[0]
                            r_val = int(r_f * 255) if r_f <= 1.0 else int(r_f)
                            g_val = int(g_f * 255) if g_f <= 1.0 else int(g_f)
                            b_val = int(b_f * 255) if b_f <= 1.0 else int(b_f)
                        else:
                            r_val = struct.unpack_from('B', ros2_msg.data, i + r_offset)[0]
                            g_val = struct.unpack_from('B', ros2_msg.data, i + g_offset)[0]
                            b_val = struct.unpack_from('B', ros2_msg.data, i + b_offset)[0]
                    except Exception:
                        pass
                points.append((x, y, z))
                if r_val is not None:
                    colors.append((r_val, g_val, b_val))

            has_color = len(colors) == len(points)
            path = output_dir / f"{timestamp_ns}.pcd"
            with open(path, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data\n")
                f.write("VERSION 0.7\n")
                if has_color:
                    f.write("FIELDS x y z rgb\n")
                    f.write("SIZE 4 4 4 4\n")
                    f.write("TYPE F F F U\n")
                    f.write("COUNT 1 1 1 1\n")
                else:
                    f.write("FIELDS x y z\n")
                    f.write("SIZE 4 4 4\n")
                    f.write("TYPE F F F\n")
                    f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                for idx, (x, y, z) in enumerate(points):
                    if has_color:
                        r, g, b = colors[idx]
                        rgb_packed = (r << 16) | (g << 8) | b
                        f.write(f"{x} {y} {z} {rgb_packed}\n")
                    else:
                        f.write(f"{x} {y} {z}\n")
            return path

        case 'sensor_msgs/msg/NavSatFix':
            data = {
                'latitude': ros2_msg.latitude,
                'longitude': ros2_msg.longitude,
                'altitude': ros2_msg.altitude,
            }
            path = output_dir / f"{timestamp_ns}.json"
            path.write_text(json.dumps(data, indent=2))
            return path

        case 'novatel_oem7_msgs/msg/BESTPOS':
            data = {
                'lat': ros2_msg.lat,
                'lon': ros2_msg.lon,
                'hgt': ros2_msg.hgt,
            }
            path = output_dir / f"{timestamp_ns}.json"
            path.write_text(json.dumps(data, indent=2))
            return path

        case 'sensor_msgs/msg/Imu':
            data = {
                'orientation': {
                    'x': ros2_msg.orientation.x,
                    'y': ros2_msg.orientation.y,
                    'z': ros2_msg.orientation.z,
                    'w': ros2_msg.orientation.w,
                },
                'angular_velocity': {
                    'x': ros2_msg.angular_velocity.x,
                    'y': ros2_msg.angular_velocity.y,
                    'z': ros2_msg.angular_velocity.z,
                },
                'linear_acceleration': {
                    'x': ros2_msg.linear_acceleration.x,
                    'y': ros2_msg.linear_acceleration.y,
                    'z': ros2_msg.linear_acceleration.z,
                },
            }
            path = output_dir / f"{timestamp_ns}.json"
            path.write_text(json.dumps(data, indent=2))
            return path

        case 'nav_msgs/msg/Odometry':
            data = {
                'header': {
                    'frame_id': ros2_msg.header.frame_id,
                    'stamp': {
                        'sec': ros2_msg.header.stamp.sec,
                        'nanosec': ros2_msg.header.stamp.nanosec,
                    },
                },
                'child_frame_id': ros2_msg.child_frame_id,
                'pose': {
                    'position': {
                        'x': ros2_msg.pose.pose.position.x,
                        'y': ros2_msg.pose.pose.position.y,
                        'z': ros2_msg.pose.pose.position.z,
                    },
                    'orientation': {
                        'x': ros2_msg.pose.pose.orientation.x,
                        'y': ros2_msg.pose.pose.orientation.y,
                        'z': ros2_msg.pose.pose.orientation.z,
                        'w': ros2_msg.pose.pose.orientation.w,
                    },
                },
                'twist': {
                    'linear': {
                        'x': ros2_msg.twist.twist.linear.x,
                        'y': ros2_msg.twist.twist.linear.y,
                        'z': ros2_msg.twist.twist.linear.z,
                    },
                    'angular': {
                        'x': ros2_msg.twist.twist.angular.x,
                        'y': ros2_msg.twist.twist.angular.y,
                        'z': ros2_msg.twist.twist.angular.z,
                    },
                },
            }
            path = output_dir / f"{timestamp_ns}.json"
            path.write_text(json.dumps(data, indent=2))
            return path

        case 'tf2_msgs/msg/TFMessage':
            if len(ros2_msg.transforms) > 0:
                transform = ros2_msg.transforms[0]
                data = {
                    'translation': {
                        'x': transform.transform.translation.x,
                        'y': transform.transform.translation.y,
                        'z': transform.transform.translation.z,
                    },
                    'rotation': {
                        'x': transform.transform.rotation.x,
                        'y': transform.transform.rotation.y,
                        'z': transform.transform.rotation.z,
                        'w': transform.transform.rotation.w,
                    },
                }
                path = output_dir / f"{timestamp_ns}.json"
                path.write_text(json.dumps(data, indent=2))
                return path
            return None

        case _:
            path = output_dir / f"{timestamp_ns}.txt"
            path.write_text(str(ros2_msg))
            return path


def _run_single_raw_export(data: dict):
    """
    Run a single raw export. Returns None on success, or (response, status_code) tuple on error.
    Writes individual files per message instead of MCAP containers.
    """
    new_rosbag_name = data.get("new_rosbag_name")
    topics = data.get("topics", [])
    start_index_raw = data.get("start_index")
    end_index_raw = data.get("end_index")
    start_mcap_id_raw = data.get("start_mcap_id")
    end_mcap_id_raw = data.get("end_mcap_id")
    mcap_ranges_raw = data.get("mcap_ranges")
    source_rosbag = data.get("source_rosbag")
    output_parent = data.get("output_parent")

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
            for rs, re_ in mcap_ranges:
                all_ids.extend(range(rs, re_ + 1))
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

    if output_parent and isinstance(output_parent, str) and '..' not in output_parent and '/' not in output_parent:
        safe_parent = output_parent
    else:
        safe_parent = None
    output_base = EXPORT_RAW / safe_parent if safe_parent else EXPORT_RAW
    output_rosbag_dir = output_base / new_rosbag_name

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

    input_mcaps = get_all_mcaps(input_rosbag_dir)

    if use_mcap_ranges:
        mcap_nums_to_process = []
        for rs, re_ in mcap_ranges:
            mcap_nums_to_process.extend(range(rs, re_ + 1))
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

    # Pre-scan: estimate total message count from MCAP summaries
    total_mcaps = len(mcap_nums_to_process)
    mcap_message_counts: dict[str, int] = {}
    total_messages_estimate = 0
    EXPORT_PROGRESS["status"] = "running"
    EXPORT_PROGRESS["progress"] = 0
    EXPORT_PROGRESS["message"] = f"Scanning {total_mcaps} MCAP file(s)..."

    for mcap_num in mcap_nums_to_process:
        mcap_id_str = str(mcap_num)
        for mcap_path in input_mcaps:
            if extract_mcap_id(mcap_path) == mcap_id_str:
                try:
                    with open(mcap_path, "rb") as f:
                        reader = SeekingReader(f)
                        summary = reader.get_summary()
                        if summary and summary.statistics:
                            count = summary.statistics.message_count or 0
                        else:
                            count = 0
                        mcap_message_counts[mcap_id_str] = count
                        total_messages_estimate += count
                except Exception:
                    mcap_message_counts[mcap_id_str] = 0
                break

    exported_files = 0
    messages_processed = 0
    for mcap_index, mcap_num in enumerate(mcap_nums_to_process):
        mcap_id_str = str(mcap_num)
        input_mcap = None
        for mcap_path in input_mcaps:
            if extract_mcap_id(mcap_path) == mcap_id_str:
                input_mcap = mcap_path
                break
        if input_mcap is None:
            continue

        if mcap_num == start_mcap_num:
            mcap_start_time, mcap_end_time = start_timestamp, None
        elif mcap_num == end_mcap_num:
            mcap_start_time, mcap_end_time = None, end_timestamp
        else:
            mcap_start_time, mcap_end_time = None, None

        EXPORT_PROGRESS["message"] = (
            f"MCAP {mcap_index + 1}/{total_mcaps} (id {mcap_id_str}) | "
            f"{exported_files} files written"
        )

        try:
            with open(input_mcap, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                    topics=topics if topics else None,
                    start_time=mcap_start_time,
                    end_time=mcap_end_time,
                    log_time_order=True,
                ):
                    topic_dir = normalize_topic(channel.topic)
                    out_dir = output_rosbag_dir / mcap_id_str / topic_dir
                    _write_raw_message(ros2_msg, schema.name, out_dir, message.log_time)
                    exported_files += 1
                    messages_processed += 1

                    # Update progress every 50 messages to avoid excessive dict writes
                    if messages_processed % 50 == 0:
                        if total_messages_estimate > 0:
                            EXPORT_PROGRESS["progress"] = min(messages_processed / total_messages_estimate, 0.99)
                        EXPORT_PROGRESS["message"] = (
                            f"MCAP {mcap_index + 1}/{total_mcaps} (id {mcap_id_str}) | "
                            f"{exported_files} files written | {channel.topic}"
                        )
        except Exception as e:
            current_app.logger.error(f"Failed to raw-export MCAP {mcap_id_str}: {e}", exc_info=True)
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = f"Error at MCAP {mcap_id_str}: {e}"
            return (jsonify({"error": str(e)}), 500)

    EXPORT_PROGRESS["_files_written"] = exported_files
    return None


def _export_raw_batch(exports_list: list) -> tuple:
    """Process multiple raw exports sequentially."""
    total = len(exports_list)

    unique_sources = list(dict.fromkeys(
        item.get("source_rosbag") for item in exports_list if isinstance(item, dict) and item.get("source_rosbag")
    ))

    output_parent = None
    if total > 1:
        if len(unique_sources) == 1:
            try:
                p = Path(str(unique_sources[0]))
                if p.is_absolute():
                    rel = str(p.relative_to(ROSBAGS))
                else:
                    rel = str(unique_sources[0])
                output_parent = _make_export_parent_folder(rel)
            except (ValueError, TypeError):
                pass
        elif len(unique_sources) > 1:
            output_parent = "multi_rosbag_raw_export"
    first_source = unique_sources[0] if unique_sources else None

    for idx, item in enumerate(exports_list):
        if not isinstance(item, dict):
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = f"Invalid export item at index {idx}"
            return jsonify({"error": "Invalid export item"}), 400

        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = (idx / total) if total > 0 else 0
        EXPORT_PROGRESS["message"] = f"Raw export part {idx + 1}/{total}: {item.get('new_rosbag_name', '?')}"

        req_data = dict(item)
        if not req_data.get("source_rosbag") and first_source:
            req_data["source_rosbag"] = first_source
        if output_parent is not None:
            req_data["output_parent"] = output_parent

        err = _run_single_raw_export(req_data)
        if err is not None:
            return err

    total_files = EXPORT_PROGRESS.get("_files_written", 0)
    EXPORT_PROGRESS["status"] = "completed"
    EXPORT_PROGRESS["progress"] = 1.0
    EXPORT_PROGRESS["message"] = f"Raw exported {total} part(s) ({total_files} files)"
    return jsonify({
        "message": f"Raw batch export completed: {total} part(s)",
        "exported_count": total,
    }), 200


@export_bp.route('/api/export-raw', methods=['POST'])
def export_raw():
    """
    Export raw data files from a rosbag (individual files per message).

    Accepts the same request format as /api/export-rosbag (single + batch).
    Output structure: export_raw/<name>/<mcap_id>/<topic>/<timestamp>.<ext>
    """
    EXPORT_PROGRESS["status"] = "running"
    EXPORT_PROGRESS["progress"] = 0
    EXPORT_PROGRESS["message"] = "Preparing raw export..."

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        exports_list = data.get("exports")
        if exports_list and isinstance(exports_list, list) and len(exports_list) > 0:
            return _export_raw_batch(exports_list)

        err = _run_single_raw_export(data)
        if err is not None:
            return err
        EXPORT_PROGRESS["status"] = "completed"
        EXPORT_PROGRESS["progress"] = 1.0
        EXPORT_PROGRESS["message"] = f"Raw export completed! ({EXPORT_PROGRESS.get('_files_written', 0)} files)"
        return jsonify({
            "message": "Raw export completed successfully",
            "output_directory": str(EXPORT_RAW / data.get("new_rosbag_name", "")),
        }), 200

    except Exception as e:
        current_app.logger.error(f"Raw export failed: {e}", exc_info=True)
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = f"Raw export failed: {str(e)}"
        return jsonify({"error": str(e)}), 500
