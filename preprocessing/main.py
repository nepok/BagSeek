"""
Preprocessing - Main Entry Point

This script orchestrates all 6 processing steps.
Uses granular completion checking to skip already-processed work at multiple levels:
- Processor level: Skip entire processor if all rosbags complete
- Model level: Skip entire model if all rosbags complete for that model
- Rosbag level: Skip entire rosbag if all MCAPs complete
- MCAP level: Skip individual MCAPs if already processed
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCAP reading imports
from mcap.reader import SeekingReader
from mcap_ros2.decoder import DecoderFactory

# Import configuration
from preprocessing.config import Config

# Import core context classes
from preprocessing.core import RosbagProcessingContext, McapProcessingContext

# Import utilities
from preprocessing.utils import get_logger, get_all_rosbags, get_all_mcaps, CompletionTracker

# Import all step classes
from preprocessing.processors import (
    # Step 1: Topics
    TopicsExtractionProcessor,
    # Step 2: Timestamp Lookup
    TimestampAlignmentProcessor,
    # Step 3: Positional Lookup
    PositionalLookupProcessor,
    # Step 4: Image Topic Previews
    ImageTopicPreviewsProcessor,
    # Step 5: Embeddings
    EmbeddingsProcessor,
    # Step 6: Adjacent Similarities
    AdjacentSimilaritiesPostprocessor,
)

from preprocessing.abstract import HybridProcessor




def main():
    """
    Main pipeline orchestrating all 6 processing steps.
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Load configuration from .env file
    config = Config.load_config()
    
    # Setup logger
    logger = get_logger()
    logger.section("PREPROCESSING PIPELINE")
    
    # ========================================================================
    # INITIALIZE STEPS
    # ========================================================================
    
    logger.info("\nInitializing processing steps...")
    
    # Step 1: Topics
    topics_extraction_processor = TopicsExtractionProcessor(
        output_dir=config.topics_dir,
        rosbags_dir=config.rosbags_dir
    )
    
    # Step 2: Timestamp Lookup
    timestamp_alignment_processor = TimestampAlignmentProcessor(
        output_dir=config.lookup_tables_dir,
        metadata_dir=config.metadata_dir,
    )
    
    # Step 3: Positional Lookup
    positional_lookup_processor = PositionalLookupProcessor(
        output_dir=config.positional_lookup_table_path,
        positional_grid_resolution=config.positional_grid_resolution
    )

    # Step 4: Image Topic Previews
    image_topic_previews_processor = ImageTopicPreviewsProcessor(
        output_dir=config.image_topic_previews_dir
    )

    # Step 5: Embeddings
    embeddings_processor = EmbeddingsProcessor(
        output_dir=config.embeddings_dir
    )

    # Step 6: Adjacent Similarities
    adjacent_similarities_postprocessor = AdjacentSimilaritiesPostprocessor(
        embeddings_dir=config.embeddings_dir,
        output_dir=config.adjacent_similarities_dir
    )

    logger.success("All processors initialized")

    # Separate processors by type
    mcap_processors = [
        timestamp_alignment_processor,
        embeddings_processor,
    ]
    
    hybrid_processors = [
        positional_lookup_processor,
        image_topic_previews_processor,
    ]
    
    # ========================================================================
    # MAIN PIPELINE: ITERATE THROUGH ROSBAGS
    # ========================================================================

    logger.subsection("Running main pipeline...")

    # Get all rosbag directories
    rosbags = get_all_rosbags(config.rosbags_dir)
    logger.info(f"Found {len(rosbags)} rosbag(s) to process\n")

    # Track statistics for summary
    stats = {
        "rosbags_skipped": 0,
        "rosbags_processed": 0,
        "mcaps_skipped": 0,
        "mcaps_processed": 0,
    }

    for idx, rosbag_path in enumerate(rosbags, 1):
        # Get all MCAP files for this rosbag
        mcap_files = get_all_mcaps(rosbag_path)
        mcap_names = [f.name for f in mcap_files]

        # Create context for this rosbag
        rosbag_context = RosbagProcessingContext(
            rosbag_path=rosbag_path,
            mcap_files=mcap_files,
            config=config
        )

        # Use relative path for logging to properly show multi-part rosbags
        relative_path = rosbag_context.get_relative_path()
        rosbag_name = str(relative_path)

        # ====================================================================
        # ROSBAG-LEVEL COMPLETION CHECK
        # Check which processors need to run for this rosbag BEFORE iterating MCAPs
        # ====================================================================

        # Check completion at rosbag level for each processor type
        topics_complete      = topics_extraction_processor.is_rosbag_complete(rosbag_name, mcap_names)
        timestamp_complete   = timestamp_alignment_processor.is_rosbag_complete(rosbag_name, mcap_names)
        positional_complete  = positional_lookup_processor.is_rosbag_complete(rosbag_name, mcap_names)
        boundaries_complete  = positional_lookup_processor.is_boundaries_complete(rosbag_name)
        previews_complete    = image_topic_previews_processor.is_rosbag_complete(rosbag_name, mcap_names)
        embeddings_complete  = embeddings_processor.is_rosbag_complete(rosbag_name, mcap_names)
        # Similarities needs work when embeddings still need work (runs after embeddings)
        similarities_complete = (
            embeddings_complete and
            adjacent_similarities_postprocessor.is_rosbag_complete(rosbag_name, mcap_names)
        )

        # Check if ALL processors are complete for this rosbag
        all_complete = (
            topics_complete and
            timestamp_complete and
            positional_complete and
            boundaries_complete and
            previews_complete and
            embeddings_complete and
            similarities_complete
        )

        if all_complete:
            logger.info(f"[{idx}/{len(rosbags)}] Rosbag {relative_path}: ALL COMPLETE, skipping")
            stats["rosbags_skipped"] += 1
            continue

        # At least one processor needs work - show header
        logger.subsection(f"Processing rosbag ({idx}/{len(rosbags)}): {relative_path}")
        logger.info(f"Found {len(mcap_files)} mcap file(s) in {relative_path}")

        # Log which processors need work at rosbag level
        needs_work = []
        if not topics_complete:
            needs_work.append("topics")
        if not timestamp_complete:
            needs_work.append("timestamps")
        if not positional_complete:
            needs_work.append("positional")
        if not boundaries_complete:
            needs_work.append("boundaries")
        if not previews_complete:
            needs_work.append("previews")
        if not embeddings_complete:
            incomplete_models = embeddings_processor.get_incomplete_models(rosbag_name, mcap_names)
            if incomplete_models:
                needs_work.append(f"embeddings({len(incomplete_models)} model(s))")
            else:
                needs_work.append("embeddings")
        if not similarities_complete:
            needs_work.append("similarities")

        logger.info(f"Processors needing work: {', '.join(needs_work)}")
        stats["rosbags_processed"] += 1

        # Step 1: Topics extraction (rosbag-level)
        if not topics_complete:
            topics_data = topics_extraction_processor.process_rosbag(rosbag_context)
            all_topics = list(topics_data.get("topics", {}).keys())
        else:
            logger.processor_skip("topics_extraction_processor", "rosbag already complete")
            # Load topics from existing file for timestamp alignment
            topics_file = config.topics_dir / relative_path.with_suffix('.json')
            if topics_file.exists():
                with open(topics_file, 'r') as f:
                    topics_data = json.load(f)
                all_topics = list(topics_data.get("topics", {}).keys())
            else:
                all_topics = []

        # Pass topic list to timestamp processor before MCAP loop
        timestamp_alignment_processor.set_topics(all_topics)

        # Build list of processors that need MCAP iteration for this rosbag
        active_mcap_processors = []
        active_hybrid_processors = []

        if not timestamp_complete:
            active_mcap_processors.append(timestamp_alignment_processor)
        if not embeddings_complete:
            active_mcap_processors.append(embeddings_processor)
        if not positional_complete:
            active_hybrid_processors.append(positional_lookup_processor)
        if not previews_complete:
            active_hybrid_processors.append(image_topic_previews_processor)

        # Call process_rosbag_before_mcaps only for active hybrid processors
        for processor in active_hybrid_processors:
            processor.process_rosbag_before_mcaps(rosbag_context)

        # Skip MCAP loop entirely if no MCAP-level processors need work
        if not active_mcap_processors and not active_hybrid_processors:
            logger.info("No MCAP-level processing needed for this rosbag")
        else:
            logger.info("")  # Add spacing before MCAP loop

            for mcap_idx, mcap_path in enumerate(mcap_files, 1):
                # Create context for this MCAP
                mcap_context = McapProcessingContext(
                    rosbag_path=rosbag_path,
                    mcap_path=mcap_path,
                    config=config
                )
                mcap_name = mcap_context.get_mcap_name()

                # ================================================================
                # MCAP-LEVEL COMPLETION CHECK
                # Check which processors need to run for this specific MCAP
                # ================================================================

                active_processors = []

                for processor in active_mcap_processors + active_hybrid_processors:
                    if not processor.is_mcap_complete(mcap_context):
                        active_processors.append(processor)

                # If no processors need to run, skip this MCAP entirely
                if not active_processors:
                    stats["mcaps_skipped"] += 1
                    # Only log if we're not skipping everything
                    if len(mcap_files) <= 10 or mcap_idx <= 3 or mcap_idx > len(mcap_files) - 2:
                        logger.info(f"[{mcap_idx}/{len(mcap_files)}] MCAP {mcap_path.name}: COMPLETE, skipping")
                    elif mcap_idx == 4:
                        logger.info(f"  ... (skipping completed MCAPs)")
                    continue

                logger.info("")  # Add spacing before each MCAP
                logger.info(f"Processing MCAP ({mcap_idx}/{len(mcap_files)}): {mcap_path.name}")
                logger.info(f"  Active processors: {', '.join(p.name for p in active_processors)}")
                stats["mcaps_processed"] += 1
            
                # Reset only active processors before iteration
                for processor in active_processors:
                    if hasattr(processor, 'reset'):
                        processor.reset()

                # Set MCAP ID for hybrid processors (needed for tracking during collection)
                for processor in active_processors:
                    if hasattr(processor, 'current_mcap_id'):
                        processor.current_mcap_id = mcap_context.get_mcap_id()

                # SINGLE PASS: Iterate through MCAP messages ONCE
                # Only active processors collect in parallel during this iteration
                message_counts: Dict[str, Dict[str, int]] = {}
                for processor in active_processors:
                    message_counts[processor.name] = {}

                # Only open and process MCAP if we have active processors
                with open(mcap_path, "rb") as f:
                    reader = SeekingReader(f, decoder_factories=[DecoderFactory()])

                    # Setup fencepost targets for image preview processor before message iteration
                    if image_topic_previews_processor in active_processors:
                        should_process = image_topic_previews_processor.setup_fencepost_targets_for_mcap(
                            reader, mcap_context, len(active_processors) == 1
                        )
                        if not should_process:
                            continue

                    # Collect all messages once - only for active processors
                    total_messages = 0
                    for schema, channel, message, ros2_msg in reader.iter_decoded_messages(log_time_order=True, reverse=False):
                        total_messages += 1
                        for processor in active_processors:
                            if processor.wants_message(channel.topic, schema.name):
                                processor.collect_message(message, channel, schema, ros2_msg)
                                if channel.topic not in message_counts[processor.name]:
                                    message_counts[processor.name][channel.topic] = 0
                                message_counts[processor.name][channel.topic] += 1

                    # Log collection summary
                    if total_messages > 0 and any(message_counts.values()):
                        logger.info(f"  Processed {total_messages} messages")
                        for processor_name, topic_counts in message_counts.items():
                            if topic_counts:
                                if processor_name == "image_topic_previews_processor":
                                    actual_counts = image_topic_previews_processor.get_collection_counts()
                                    if actual_counts:
                                        topic_summaries = [f"{count} img" for topic, count in actual_counts.items()]
                                        logger.info(f"    {processor_name}: {', '.join(topic_summaries)}")
                                else:
                                    total_collected = sum(topic_counts.values())
                                    logger.info(f"    {processor_name}: {total_collected} msg(s)")

                # Process MCAP-level data after collection
                for processor in active_processors:
                    if processor in active_mcap_processors:
                        processor.process_mcap(mcap_context)
                    elif isinstance(processor, HybridProcessor):
                        if hasattr(processor, 'process_mcap'):
                            processor.process_mcap(mcap_context)

        # Call process_rosbag_after_mcaps only for active hybrid processors
        if active_hybrid_processors:
            logger.info("")  # Add spacing before aggregation
            for processor in active_hybrid_processors:
                processor.process_rosbag_after_mcaps(rosbag_context)

        # Post-loop hook: finalize summaries (timestamps), ensure boundaries, etc.
        # Called after process_rosbag_after_mcaps so derived data (e.g. boundaries)
        # can read freshly written grid data. Each processor is idempotent.
        mcap_processors_all = [
            topics_extraction_processor, timestamp_alignment_processor,
            positional_lookup_processor, image_topic_previews_processor,
            embeddings_processor,
        ]
        for processor in mcap_processors_all:
            processor.on_rosbag_complete(rosbag_context)

        # Compute adjacent similarities for this rosbag (if needed)
        if not similarities_complete:
            logger.info("")  # Add spacing before adjacent similarities
            adjacent_similarities_postprocessor.process_rosbag(rosbag_context)
        else:
            logger.processor_skip("adjacent_similarities_postprocessor", "rosbag already complete")

        logger.success(f"Completed processing for {relative_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    logger.subsection("Pipeline Summary", char="=")
    logger.info(f"Rosbags: {stats['rosbags_processed']} processed, {stats['rosbags_skipped']} skipped (already complete)")
    logger.info(f"MCAPs: {stats['mcaps_processed']} processed, {stats['mcaps_skipped']} skipped (already complete)")

    if stats['rosbags_processed'] == 0 and stats['rosbags_skipped'] > 0:
        logger.success("All rosbags already complete! Nothing to process.")
    else:
        logger.success("Pipeline complete!")

    # ========================================================================
    # WRITE VALID ROSBAGS INDEX
    # ========================================================================

    logger.info("\nWriting valid_rosbags.json index file...")

    # Collect all valid rosbag paths (those with embeddings)
    valid_rosbag_paths = []
    embeddings_dir = config.embeddings_dir

    if embeddings_dir.exists():
        for rosbag_path in rosbags:
            relative_path = rosbag_path.relative_to(config.rosbags_dir)
            rosbag_name = str(relative_path)

            # Check if this rosbag has embeddings in at least one model
            found_in_embeddings = False
            for model_dir in embeddings_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                embeddings_rosbag_path = model_dir / relative_path
                if embeddings_rosbag_path.exists() and embeddings_rosbag_path.is_dir():
                    found_in_embeddings = True
                    break

            if found_in_embeddings:
                valid_rosbag_paths.append(rosbag_name)

    valid_rosbag_paths.sort()

    # Write index file
    index_file = config.base_dir / "valid_rosbags.json"
    index_data = {
        "paths": valid_rosbag_paths,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.success(f"Wrote {len(valid_rosbag_paths)} valid rosbag(s) to {index_file}")

    logger.section("", char="=")


if __name__ == "__main__":
    main()

