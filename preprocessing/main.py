"""
Preprocessing - Main Entry Point

This script orchestrates all 6 processing steps.
Uses granular completion checking to skip already-processed work at multiple levels:
- Processor level: Skip entire processor if all rosbags complete
- Model level: Skip entire model if all rosbags complete for that model
- Rosbag level: Skip entire rosbag if all MCAPs complete
- MCAP level: Skip individual MCAPs if already processed
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
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

# Import processor types for isinstance checks
from preprocessing.abstract import McapProcessor, HybridProcessor


# =============================================================================
# COMPLETION CHECKING HELPERS
# =============================================================================

def is_rosbag_complete_for_mcap_processor(
    processor: McapProcessor,
    rosbag_name: str,
    mcap_names: List[str]
) -> bool:
    """
    Check if all MCAPs in a rosbag are complete for an MCAP-level processor.

    First checks rosbag-level "status": "completed" (fast path).
    Only falls back to MCAP-level checks if rosbag status is not complete.

    Args:
        processor: The MCAP processor to check
        rosbag_name: Name of the rosbag
        mcap_names: List of MCAP names in this rosbag

    Returns:
        True if all MCAPs are complete, False otherwise
    """
    if not mcap_names:
        return True

    tracker = processor.completion_tracker

    # FAST PATH: Check rosbag-level completion status first
    if tracker.is_rosbag_completed(rosbag_name):
        return True

    # SLOW PATH: Check each MCAP individually
    for mcap_name in mcap_names:
        if not tracker.is_mcap_completed(rosbag_name, mcap_name):
            return False
    return True


def extract_mcap_id(mcap_filename: str) -> str:
    """
    Extract MCAP ID from filename.

    Args:
        mcap_filename: MCAP filename like "rosbag2_2025_07_23-07_29_39_0.mcap"

    Returns:
        MCAP ID like "0"
    """
    # Remove .mcap extension and get last underscore-separated part
    stem = mcap_filename.replace('.mcap', '')
    return stem.split('_')[-1]


def construct_embeddings_mcap_name(rosbag_name: str, mcap_filename: str) -> str:
    """
    Construct the MCAP name as stored in embeddings completion.json.

    The EmbeddingsProcessor stores mcap_name as: {rosbag_name}_{mcap_id}.mcap

    Args:
        rosbag_name: Rosbag relative path like "rosbag2_2025_07_25-12_17_25"
        mcap_filename: MCAP filename like "rosbag2_2025_07_25-12_17_25_0.mcap"

    Returns:
        Constructed mcap name for completion tracking
    """
    mcap_id = extract_mcap_id(mcap_filename)
    return f"{rosbag_name}_{mcap_id}.mcap"


def is_rosbag_complete_for_embeddings(
    processor: EmbeddingsProcessor,
    rosbag_name: str,
    mcap_names: List[str]
) -> bool:
    """
    Check if all models and MCAPs are complete for EmbeddingsProcessor.

    First checks rosbag-level "status": "completed" for each model (fast path).
    Only falls back to MCAP-level checks if rosbag status is not complete.

    Args:
        processor: The embeddings processor
        rosbag_name: Name of the rosbag
        mcap_names: List of MCAP names in this rosbag

    Returns:
        True if all models have all MCAPs complete, False otherwise
    """
    if not mcap_names:
        return True

    tracker = processor.completion_tracker

    # Check all models
    for preprocess_id, models in processor.models_by_preprocess.items():
        for model_dir_id, _, _, _ in models:
            # FAST PATH: Check rosbag-level completion status first
            if tracker.is_model_rosbag_completed(model_dir_id, rosbag_name):
                continue  # This model+rosbag is complete

            # SLOW PATH: Check each MCAP individually (only if rosbag not marked complete)
            for mcap_filename in mcap_names:
                mcap_name = construct_embeddings_mcap_name(rosbag_name, mcap_filename)
                if not tracker.is_model_mcap_completed(model_dir_id, rosbag_name, mcap_name):
                    return False
    return True


def get_incomplete_models_for_rosbag(
    processor: EmbeddingsProcessor,
    rosbag_name: str,
    mcap_names: List[str]
) -> List[str]:
    """
    Get list of models that have incomplete MCAPs for a rosbag.

    First checks rosbag-level "status": "completed" for each model (fast path).

    Args:
        processor: The embeddings processor
        rosbag_name: Name of the rosbag
        mcap_names: List of MCAP names in this rosbag

    Returns:
        List of model IDs that need processing
    """
    incomplete_models = []
    tracker = processor.completion_tracker

    for preprocess_id, models in processor.models_by_preprocess.items():
        for model_dir_id, _, _, _ in models:
            # FAST PATH: Check rosbag-level completion status first
            if tracker.is_model_rosbag_completed(model_dir_id, rosbag_name):
                continue  # This model+rosbag is complete

            # SLOW PATH: Check each MCAP individually
            model_complete = True
            for mcap_filename in mcap_names:
                mcap_name = construct_embeddings_mcap_name(rosbag_name, mcap_filename)
                if not tracker.is_model_mcap_completed(model_dir_id, rosbag_name, mcap_name):
                    model_complete = False
                    break
            if not model_complete:
                incomplete_models.append(model_dir_id)

    return incomplete_models


def is_rosbag_complete_for_adjacent_similarities(
    processor: AdjacentSimilaritiesPostprocessor,
    rosbag_name: str,
    embeddings_dir: Path,
    embeddings_need_work: bool = False
) -> bool:
    """
    Check if all models and topics are complete for AdjacentSimilaritiesPostprocessor.

    Args:
        processor: The adjacent similarities processor
        rosbag_name: Name of the rosbag
        embeddings_dir: Directory containing embeddings
        embeddings_need_work: If True, embeddings are being generated so similarities will need work

    Returns:
        True if all models/topics are complete, False otherwise
    """
    # If embeddings need work, similarities will also need work afterward
    if embeddings_need_work:
        return False

    tracker = processor.completion_tracker

    # Check each model directory
    if not embeddings_dir.exists():
        return True  # No embeddings, nothing to process

    found_any_embeddings = False
    for model_path in embeddings_dir.iterdir():
        if not model_path.is_dir():
            continue

        model_name = model_path.name
        rosbag_path = model_path / rosbag_name
        manifest_path = rosbag_path / "manifest.parquet"

        if not manifest_path.exists():
            continue  # No embeddings for this model/rosbag

        found_any_embeddings = True

        # Read manifest to get topics
        try:
            import pandas as pd
            manifest = pd.read_parquet(manifest_path)
            topics = manifest["topic"].unique()

            for topic in topics:
                if not tracker.is_model_topic_completed(model_name, rosbag_name, topic):
                    return False
        except Exception:
            return False

    # If no embeddings found at all, nothing to process
    if not found_any_embeddings:
        return True

    return True


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
        output_dir=config.lookup_tables_dir
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

    #rosbag_processors = []
    #   topics_extraction_processor
    #]
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
        topics_complete = topics_extraction_processor.completion_tracker.is_rosbag_completed(rosbag_name)
        timestamp_complete = is_rosbag_complete_for_mcap_processor(
            timestamp_alignment_processor, rosbag_name, mcap_names
        )
        positional_complete = positional_lookup_processor.completion_tracker.is_rosbag_completed(rosbag_name)
        previews_complete = image_topic_previews_processor.completion_tracker.is_rosbag_completed(rosbag_name)
        embeddings_complete = is_rosbag_complete_for_embeddings(
            embeddings_processor, rosbag_name, mcap_names
        )
        similarities_complete = is_rosbag_complete_for_adjacent_similarities(
            adjacent_similarities_postprocessor, rosbag_name, config.embeddings_dir
        )

        # Check if ALL processors are complete for this rosbag
        all_complete = (
            topics_complete and
            timestamp_complete and
            positional_complete and
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
        if not previews_complete:
            needs_work.append("previews")
        if not embeddings_complete:
            incomplete_models = get_incomplete_models_for_rosbag(
                embeddings_processor, rosbag_name, mcap_names
            )
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
                import json
                with open(topics_file, 'r') as f:
                    topics_data = json.load(f)
                all_topics = list(topics_data.get("topics", {}).keys())
            else:
                all_topics = []

        # Determine which MCAP-level processors need to run for this rosbag
        rosbag_needs_timestamps = not timestamp_complete
        rosbag_needs_positional = not positional_complete
        rosbag_needs_previews = not previews_complete
        rosbag_needs_embeddings = not embeddings_complete

        # Build list of processors that need MCAP iteration for this rosbag
        active_mcap_processors = []
        active_hybrid_processors = []

        if rosbag_needs_timestamps:
            active_mcap_processors.append(timestamp_alignment_processor)
        if rosbag_needs_embeddings:
            active_mcap_processors.append(embeddings_processor)
        if rosbag_needs_positional:
            active_hybrid_processors.append(positional_lookup_processor)
        if rosbag_needs_previews:
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
                    is_done = False

                    if processor.name == "embeddings_processor":
                        # EmbeddingsProcessor checks all models internally
                        is_done = processor.is_mcap_completed(mcap_context)
                    elif processor.name == "positional_lookup_processor":
                        # Check MCAP-level completion
                        is_done = processor.is_mcap_completed(mcap_context)
                    elif processor.name == "timestamp_alignment_processor":
                        is_done = processor.completion_tracker.is_mcap_completed(rosbag_name, mcap_name)
                    elif processor.name == "image_topic_previews_processor":
                        # Image previews are rosbag-level, but we still need to collect from MCAPs
                        # Check if this MCAP is skippable (not in fencepost mapping)
                        is_done = processor.is_mcap_skippable(mcap_context.get_mcap_id())
                    else:
                        is_done = processor.completion_tracker.is_mcap_completed(rosbag_name, mcap_name)

                    if not is_done:
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
                        if processor == timestamp_alignment_processor:
                            processor.process_mcap(mcap_context, all_topics)
                        else:
                            processor.process_mcap(mcap_context)
                    elif isinstance(processor, HybridProcessor):
                        if hasattr(processor, 'process_mcap'):
                            processor.process_mcap(mcap_context)

        # Call process_rosbag_after_mcaps only for active hybrid processors
        if active_hybrid_processors:
            logger.info("")  # Add spacing before aggregation
            for processor in active_hybrid_processors:
                processor.process_rosbag_after_mcaps(rosbag_context)

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

    logger.section("", char="=")


if __name__ == "__main__":
    main()

