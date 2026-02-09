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
from ..state import get_selected_rosbag, EXPORT_PROGRESS

export_bp = Blueprint('export', __name__)

# Regex for safe rosbag names (alphanumeric, underscore, hyphen only)
# Prevents path traversal and shell metacharacter injection
SAFE_ROSBAG_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


@export_bp.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)


@export_bp.route('/api/export-rosbag', methods=['POST'])
def export_rosbag():
    """
    Export MCAP files from a rosbag based on MCAP ID range and time filtering.
    
    Request body:
    {
        "new_rosbag_name": str,
        "topics": List[str],
        "start_timestamp": int (nanoseconds),
        "end_timestamp": int (nanoseconds),
        "start_mcap_id": int,
        "end_mcap_id": int
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
        
        # Extract parameters from request
        new_rosbag_name = data.get("new_rosbag_name")
        topics = data.get("topics", [])
        start_timestamp_raw = data.get("start_timestamp")
        end_timestamp_raw = data.get("end_timestamp")
        start_mcap_id_raw = data.get("start_mcap_id")
        end_mcap_id_raw = data.get("end_mcap_id")
        
        # Validate required parameters
        if not new_rosbag_name:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "new_rosbag_name is required"
            return jsonify({"error": "new_rosbag_name is required"}), 400

        # Security: Validate new_rosbag_name to prevent path traversal and shell injection
        if not SAFE_ROSBAG_NAME_PATTERN.match(new_rosbag_name):
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "Invalid rosbag name format. Only alphanumeric characters, underscores, and hyphens are allowed."
            return jsonify({"error": "Invalid rosbag name format. Only alphanumeric characters, underscores, and hyphens are allowed."}), 400

        if start_timestamp_raw is None or end_timestamp_raw is None:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "start_timestamp and end_timestamp are required"
            return jsonify({"error": "start_timestamp and end_timestamp are required"}), 400
        if start_mcap_id_raw is None or end_mcap_id_raw is None:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "start_mcap_id and end_mcap_id are required"
            return jsonify({"error": "start_mcap_id and end_mcap_id are required"}), 400
        
        # Convert to integers (after validation to allow 0 values)
        try:
            start_timestamp = int(start_timestamp_raw)
            end_timestamp = int(end_timestamp_raw)
            start_mcap_id = int(start_mcap_id_raw)
            end_mcap_id = int(end_mcap_id_raw)
        except (ValueError, TypeError) as e:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = f"Invalid number format: {e}"
            return jsonify({"error": f"Invalid number format: {e}"}), 400

        # Check if a rosbag is selected
        selected_rosbag = get_selected_rosbag()
        if selected_rosbag is None:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = "No rosbag selected"
            return jsonify({"error": "No rosbag selected"}), 400

        # Set status to starting - export is beginning
        EXPORT_PROGRESS["status"] = "starting"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = "Validating export parameters..."

        # Set paths
        # selected_rosbag might be absolute or relative, handle both cases
        selected_rosbag_str = str(selected_rosbag)
        selected_rosbag_path = Path(selected_rosbag_str)
        
        # Check if it's an absolute path or relative
        if selected_rosbag_path.is_absolute():
            # Absolute path - get relative path from ROSBAGS
            try:
                relative_rosbag_path = str(selected_rosbag_path.relative_to(ROSBAGS))
            except ValueError:
                # If not a subpath, use the path as-is (might be outside ROSBAGS)
                relative_rosbag_path = selected_rosbag_str
            input_rosbag_dir = ROSBAGS / relative_rosbag_path
        else:
            # Already a relative path - use it directly
            input_rosbag_dir = ROSBAGS / selected_rosbag_str
        
        output_rosbag_base = EXPORT
        output_rosbag_dir = output_rosbag_base / new_rosbag_name
        
        # Update export progress
        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = 0.0
        EXPORT_PROGRESS["message"] = "Starting export..."
        
        # Helper function to extract MCAP ID from filename
        def extract_mcap_id(mcap_path: Path) -> str:
            """Extract MCAP ID from filename (e.g., 'rosbag2_2025_07_25-10_14_58_1.mcap' -> '1')."""
            stem = mcap_path.stem  # filename without extension
            parts = stem.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[1]
            return "0"  # Fallback
        
        # Get all MCAP files from input rosbag, sorted numerically by MCAP ID
        def get_all_mcaps(rosbag_dir: Path) -> List[Path]:
            """Get all MCAP files from a rosbag, sorted numerically by MCAP ID."""
            if not rosbag_dir.exists():
                raise FileNotFoundError(f"Rosbag directory not found: {rosbag_dir}")
            
            mcaps = list(rosbag_dir.glob("*.mcap"))
            
            def extract_number(path: Path) -> int:
                """Extract the number from the mcap filename for sorting."""
                mcap_id = extract_mcap_id(path)
                try:
                    return int(mcap_id)
                except ValueError:
                    return float('inf')
            
            mcaps.sort(key=extract_number)
            return mcaps
        
        # Export a single MCAP file
        def export_mcap(
            input_mcap_path: Path,
            output_mcap_path: Path,
            mcap_id: str,
            start_time_ns: Optional[int],
            end_time_ns: Optional[int],
            topics: Optional[List[str]],
            compression: CompressionType,
            include_attachments: bool,
            include_metadata: bool,
        ):
            """
            Export messages from an MCAP file with optional time filtering.
            """
            current_app.logger.info(f"  Processing MCAP {mcap_id}: {input_mcap_path.name}")
            if start_time_ns is not None and end_time_ns is not None:
                current_app.logger.info(f"    Time range: {start_time_ns} to {end_time_ns}")
            elif start_time_ns is not None:
                current_app.logger.info(f"    Time range: {start_time_ns} to end of MCAP")
            elif end_time_ns is not None:
                current_app.logger.info(f"    Time range: beginning of MCAP to {end_time_ns}")
            else:
                current_app.logger.info(f"    Time range: entire MCAP (no time filtering)")
            
            # Ensure output directory exists
            output_mcap_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open input file and create reader
            with open(input_mcap_path, "rb") as input_file:
                reader = SeekingReader(input_file, decoder_factories=[DecoderFactory()])
                
                # Get summary information
                summary = reader.get_summary()
                if not summary:
                    current_app.logger.warning(f"    No summary found in MCAP {mcap_id}, skipping")
                    return
                
                # Open output file and create writer
                with open(output_mcap_path, "wb") as output_file:
                    writer = Writer(
                        output=output_file,
                        compression=compression,
                        index_types=IndexType.ALL,
                        use_chunking=True,
                        use_statistics=True,
                    )
                    
                    # Start writing
                    writer.start(profile="ros2", library="mcap-exporter")
                    
                    # Register schemas from input file
                    schema_map = {}  # Map old schema_id -> new schema_id
                    if summary.schemas:
                        for schema_id, schema in summary.schemas.items():
                            new_schema_id = writer.register_schema(
                                name=schema.name,
                                encoding=schema.encoding,
                                data=schema.data,
                            )
                            schema_map[schema_id] = new_schema_id
                    
                    # Register channels from input file (only those we want to export)
                    channel_map = {}  # Map old channel_id -> new channel_id
                    if summary.channels:
                        for channel_id, channel in summary.channels.items():
                            # Filter by topics if specified
                            if topics and channel.topic not in topics:
                                continue
                            
                            # Get new schema_id (or 0 if no schema)
                            new_schema_id = schema_map.get(channel.schema_id, 0)
                            
                            new_channel_id = writer.register_channel(
                                topic=channel.topic,
                                message_encoding=channel.message_encoding,
                                schema_id=new_schema_id,
                                metadata=dict(channel.metadata) if channel.metadata else {},
                            )
                            channel_map[channel_id] = new_channel_id
                    
                    if not channel_map:
                        current_app.logger.warning(f"    No matching channels found for MCAP {mcap_id}")
                        writer.finish()
                        return
                    
                    # Copy messages with optional time filtering
                    message_count = 0
                    skipped_count = 0
                    
                    for schema, channel, message in reader.iter_messages(
                        topics=topics,
                        start_time=start_time_ns,
                        end_time=end_time_ns,
                        log_time_order=True,
                        reverse=False,
                    ):
                        # Get new channel_id
                        new_channel_id = channel_map.get(channel.id)
                        if new_channel_id is None:
                            skipped_count += 1
                            continue
                        
                        # Write message
                        writer.add_message(
                            channel_id=new_channel_id,
                            log_time=message.log_time,
                            data=message.data,
                            publish_time=message.publish_time,
                            sequence=message.sequence,
                        )
                        message_count += 1
                    
                    current_app.logger.info(f"    Copied {message_count} messages (skipped {skipped_count})")
                    
                    # Copy attachments if requested
                    if include_attachments:
                        attachment_count = 0
                        for attachment in reader.iter_attachments():
                            # Filter attachments by time range (if specified)
                            include_attachment = True
                            if start_time_ns is not None and attachment.log_time < start_time_ns:
                                include_attachment = False
                            if end_time_ns is not None and attachment.log_time > end_time_ns:
                                include_attachment = False
                            
                            if include_attachment:
                                writer.add_attachment(
                                    create_time=attachment.create_time,
                                    log_time=attachment.log_time,
                                    name=attachment.name,
                                    media_type=attachment.media_type,
                                    data=attachment.data,
                                )
                                attachment_count += 1
                        if attachment_count > 0:
                            current_app.logger.info(f"    Copied {attachment_count} attachment(s)")
                    
                    # Copy metadata if requested
                    if include_metadata:
                        metadata_count = 0
                        for metadata in reader.iter_metadata():
                            writer.add_metadata(
                                name=metadata.name,
                                data=dict(metadata.metadata) if metadata.metadata else {},
                            )
                            metadata_count += 1
                        if metadata_count > 0:
                            current_app.logger.info(f"    Copied {metadata_count} metadata record(s)")
                    
                    # Finish writing
                    writer.finish()
            
            current_app.logger.info(f"    Export complete: {output_mcap_path.name}")
        
        # Main export logic
        current_app.logger.info("=" * 80)
        current_app.logger.info("MCAP Rosbag Exporter")
        current_app.logger.info("=" * 80)
        current_app.logger.info(f"Input rosbag: {input_rosbag_dir}")
        current_app.logger.info(f"Output directory: {output_rosbag_base}")
        current_app.logger.info(f"New rosbag name: {new_rosbag_name}")
        current_app.logger.info(f"Topics: {len(topics) if topics else 'all'}")
        
        # Get all MCAP files from input rosbag
        current_app.logger.info(f"\nScanning MCAP files in {input_rosbag_dir}...")
        input_mcaps = get_all_mcaps(input_rosbag_dir)
        current_app.logger.info(f"Found {len(input_mcaps)} MCAP file(s)")
        
        # Create output directory
        output_rosbag_dir.mkdir(parents=True, exist_ok=True)
        current_app.logger.info(f"Output directory: {output_rosbag_dir}")
        
        # Convert MCAP IDs to integers for range calculation
        try:
            start_mcap_num = int(start_mcap_id)
            end_mcap_num = int(end_mcap_id)
        except ValueError:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            return jsonify({"error": f"Invalid MCAP IDs: start={start_mcap_id}, end={end_mcap_id}"}), 400
        
        if start_mcap_num > end_mcap_num:
            EXPORT_PROGRESS["status"] = "error"
            EXPORT_PROGRESS["progress"] = -1
            EXPORT_PROGRESS["message"] = f"Start MCAP ID ({start_mcap_num}) must be <= End MCAP ID ({end_mcap_num})"
            return jsonify({"error": f"Start MCAP ID ({start_mcap_num}) must be <= End MCAP ID ({end_mcap_num})"}), 400
        
        # Calculate total number of MCAPs to process (after validation)
        total_mcaps_to_process = end_mcap_num - start_mcap_num + 1
        
        current_app.logger.info(f"  Exporting MCAPs {start_mcap_id} to {end_mcap_id} (inclusive)")
        current_app.logger.info(f"  Start MCAP {start_mcap_id}: from timestamp {start_timestamp} to end")
        current_app.logger.info(f"  End MCAP {end_mcap_id}: from beginning to timestamp {end_timestamp}")
        if end_mcap_num > start_mcap_num + 1:
            current_app.logger.info(f"  Middle MCAPs: complete export (no time filtering)")
        
        # Export all MCAPs in the range
        total_mcaps_exported = 0
        for mcap_index, mcap_num in enumerate(range(start_mcap_num, end_mcap_num + 1)):
            mcap_id = str(mcap_num)
            
            # Update progress: calculate percentage based on MCAP index
            progress = (mcap_index / total_mcaps_to_process) if total_mcaps_to_process > 0 else 0.0
            EXPORT_PROGRESS["progress"] = round(progress, 2)
            EXPORT_PROGRESS["message"] = f"Processing MCAP {mcap_id} ({mcap_index + 1}/{total_mcaps_to_process})"
            
            # Find the MCAP file with this ID
            input_mcap = None
            for mcap_path in input_mcaps:
                if extract_mcap_id(mcap_path) == mcap_id:
                    input_mcap = mcap_path
                    break
            
            if input_mcap is None:
                current_app.logger.warning(f"  MCAP {mcap_id} not found in input rosbag, skipping")
                continue
            
            # Build output path: OUTPUT_ROSBAG / new_rosbag_name / new_rosbag_name_mcap_id.mcap
            output_mcap_name = f"{new_rosbag_name}_{mcap_id}.mcap"
            output_mcap_path = output_rosbag_dir / output_mcap_name
            
            # Update message with filename
            EXPORT_PROGRESS["message"] = f"Writing {output_mcap_name} ({mcap_index + 1}/{total_mcaps_to_process})"
            
            # Determine time filtering based on MCAP position
            if mcap_num == start_mcap_num:
                # Start MCAP: from start_timestamp to end (no upper bound)
                mcap_start_time = start_timestamp
                mcap_end_time = None
            elif mcap_num == end_mcap_num:
                # End MCAP: from beginning to end_timestamp (no lower bound)
                mcap_start_time = None
                mcap_end_time = end_timestamp
            else:
                # Middle MCAPs: no time filtering (export completely)
                mcap_start_time = None
                mcap_end_time = None
            
            # Export this MCAP with appropriate time filtering
            try:
                export_mcap(
                    input_mcap_path=input_mcap,
                    output_mcap_path=output_mcap_path,
                    mcap_id=mcap_id,
                    start_time_ns=mcap_start_time,
                    end_time_ns=mcap_end_time,
                    topics=topics if topics else None,
                    compression=CompressionType.ZSTD,
                    include_attachments=True,
                    include_metadata=True,
                )
                total_mcaps_exported += 1
                
                # Update progress after successful export
                progress = ((mcap_index + 1) / total_mcaps_to_process) if total_mcaps_to_process > 0 else 1.0
                EXPORT_PROGRESS["progress"] = round(progress, 2)
            except Exception as e:
                current_app.logger.error(f"  Failed to export MCAP {mcap_id}: {e}", exc_info=True)
        
        current_app.logger.info(f"\n{'=' * 80}")
        current_app.logger.info(f"Export complete!")
        current_app.logger.info(f"  Exported {total_mcaps_exported} MCAP file(s)")
        current_app.logger.info(f"  Output directory: {output_rosbag_dir}")
        
        # Reindex the exported rosbag using ros2 bag reindex
        if total_mcaps_exported > 0:
            current_app.logger.info(f"\nReindexing rosbag: {output_rosbag_dir}")
            try:
                result = subprocess.run(
                    ["ros2", "bag", "reindex", str(output_rosbag_dir), "-s", "mcap"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_app.logger.info("Reindexing completed successfully")
                if result.stdout:
                    current_app.logger.debug(f"Reindex output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                current_app.logger.error(f"Reindexing failed: {e}")
                if e.stderr:
                    current_app.logger.error(f"Error output: {e.stderr}")
            except FileNotFoundError:
                current_app.logger.warning("ros2 command not found. Skipping reindexing. Make sure ROS2 is installed and in PATH.")
            except Exception as e:
                current_app.logger.error(f"Unexpected error during reindexing: {e}", exc_info=True)
        
        current_app.logger.info(f"{'=' * 80}")
        
        # Update export progress
        EXPORT_PROGRESS["status"] = "completed"
        EXPORT_PROGRESS["progress"] = 1.0
        EXPORT_PROGRESS["message"] = f"Export completed! Exported {total_mcaps_exported} MCAP file(s)"
        
        return jsonify({
            "message": "Export completed successfully",
            "exported_mcaps": total_mcaps_exported,
            "output_directory": str(output_rosbag_dir)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Export failed: {e}", exc_info=True)
        EXPORT_PROGRESS["status"] = "error"
        EXPORT_PROGRESS["progress"] = -1
        EXPORT_PROGRESS["message"] = f"Export failed: {str(e)}"
        return jsonify({"error": str(e)}), 500
