"""
Preprocessing - Main Entry Point

This script orchestrates all 6 processing steps.
All step implementations are commented out so they can be implemented one by one.
"""
from pathlib import Path
from typing import Dict, Any
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
from preprocessing.utils import get_logger, get_all_rosbags, get_all_mcaps

# Import all step classes
from preprocessing.steps import (
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
    
    for idx, rosbag_path in enumerate(rosbags, 1):
        # Get all MCAP files for this rosbag
        mcap_files = get_all_mcaps(rosbag_path)
        
        # Create context for this rosbag
        rosbag_context = RosbagProcessingContext(
            rosbag_path=rosbag_path,
            mcap_files=mcap_files,
            config=config
        )
        
        # Use relative path for logging to properly show multi-part rosbags
        relative_path = rosbag_context.get_relative_path()
        logger.subsection(f"Processing rosbag ({idx}/{len(rosbags)}): {relative_path}")
        logger.info(f"Found {len(mcap_files)} mcap file(s) in {relative_path}")
    
        #logger.info("\nStep 1: Extracting topics...")
        topics_data = topics_extraction_processor.process_rosbag(rosbag_context)
        all_topics = list(topics_data.get("topics", {}).keys())
        
        
        # Call process_rosbag_before_mcaps for hybrid processors
        for processor in hybrid_processors:
            processor.process_rosbag_before_mcaps(rosbag_context)
        
        logger.info("")  # Add spacing before MCAP loop
        
        for mcap_idx, mcap_path in enumerate(mcap_files, 1):
            logger.info("")  # Add spacing before each MCAP
            logger.info(f"Processing MCAP ({mcap_idx}/{len(mcap_files)}): {mcap_path.name}")
            
            # Create context for this MCAP
            mcap_context = McapProcessingContext(
                rosbag_path=rosbag_path,
                mcap_path=mcap_path,
                config=config
            )
            
            # Combine mcap and hybrid processors for message collection
            all_mcap_collectors = mcap_processors + hybrid_processors
            
            # Check completion for ALL processors and build filtered list
            active_processors = []
            for processor in all_mcap_collectors:
                # Special handling for processors with custom completion checking
                if processor.name == "embeddings_processor" and hasattr(processor, 'is_mcap_completed'):
                    is_done = processor.is_mcap_completed(mcap_context)
                elif processor.name == "positional_lookup_processor" and hasattr(processor, 'is_mcap_completed'):
                    # PositionalLookupProcessor checks JSON file content to verify MCAP data exists
                    is_done = processor.is_mcap_completed(mcap_context)
                else:
                    # Get output path from processor
                    output_path = processor.get_output_path(mcap_context)
                    
                    # Determine if this processor uses MCAP-level completion
                    # MCAP processors and hybrid processors with process_mcap use MCAP-level completion
                    uses_mcap_completion = isinstance(processor, McapProcessor) or (
                        isinstance(processor, HybridProcessor) and hasattr(processor, 'process_mcap')
                    )
                    
                    # Check completion status
                    is_done = processor.completion_tracker.is_completed(
                        mcap_context,
                        output_path=output_path,
                        processor=processor,
                        mcap_name=mcap_context.get_mcap_name() if uses_mcap_completion else None
                    )
                
                if not is_done:
                    # Processor needs to run, add to active list
                    active_processors.append(processor)
                else:
                    logger.processor_skip(
                        f"{processor.name} for {mcap_context.get_mcap_name()}", 
                        "already completed"
                    )
            
            # If no processors need to run, skip this MCAP entirely
            if not active_processors:
                logger.info(f"All processors completed for {mcap_path.name}, skipping MCAP")
                continue
            
            # Reset only active processors before iteration
            for processor in active_processors:
                if hasattr(processor, 'reset'):
                    processor.reset()
            
            # Set MCAP ID for hybrid processors (needed for tracking during collection)
            for processor in active_processors:
                if hasattr(processor, 'current_mcap_id'):
                    # Set current MCAP ID directly for hybrid processors
                    processor.current_mcap_id = mcap_context.get_mcap_id()
            
            # SINGLE PASS: Iterate through MCAP messages ONCE
            # Only active processors collect in parallel during this iteration
            # Track message counts per processor per topic
            message_counts: Dict[str, Dict[str, int]] = {}  # {processor_name: {topic: count}}
            for processor in active_processors:
                message_counts[processor.name] = {}
            
            # Only open and process MCAP if we have active processors
            with open(mcap_path, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()])
                
                # Setup fencepost targets for image preview processor before message iteration
                # Only if it's in active processors
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
                    # Each active processor filters and collects messages
                    for processor in active_processors:
                        if processor.wants_message(channel.topic, schema.name):
                            processor.collect_message(message, channel, schema, ros2_msg)
                            # Track message counts
                            if channel.topic not in message_counts[processor.name]:
                                message_counts[processor.name][channel.topic] = 0
                            message_counts[processor.name][channel.topic] += 1
                
                # Log collection summary after MCAP iteration - only if we actually collected messages
                if total_messages > 0 and any(message_counts.values()):
                    logger.info(f"Processed {total_messages} total messages across all processors")
                    for processor_name, topic_counts in message_counts.items():
                        if topic_counts:
                            # For ImageTopicPreviewsProcessor, use actual collection counts instead
                            if processor_name == "image_topic_previews_processor":
                                actual_counts = image_topic_previews_processor.get_collection_counts()
                                if actual_counts:
                                    topic_summaries = [f"{count} image(s) from {topic}" for topic, count in actual_counts.items()]
                                    logger.info(f"  {processor_name}: Collected {', '.join(topic_summaries)}")
                            else:
                                # Group topics by processor
                                topic_summaries = [f"{count} messages from {topic}" for topic, count in topic_counts.items()]
                                logger.info(f"  {processor_name}: Collected {', '.join(topic_summaries)}")
            
            # Process MCAP-level data after collection
            for processor in active_processors:
                if processor in mcap_processors:
                    # Pure MCAP processors
                    if processor == timestamp_alignment_processor:
                        processor.process_mcap(mcap_context, all_topics)
                    else:
                        processor.process_mcap(mcap_context)
                elif isinstance(processor, HybridProcessor):
                    # Hybrid processors that implement process_mcap (e.g., PositionalLookupProcessor)
                    if hasattr(processor, 'process_mcap'):
                        processor.process_mcap(mcap_context)
        
        # Call process_rosbag_after_mcaps for hybrid processors
        logger.info("")  # Add spacing before aggregation
        for processor in hybrid_processors:
            processor.process_rosbag_after_mcaps(rosbag_context)
        
        # Compute adjacent similarities for this rosbag (after all MCAPs are processed)
        logger.info("")  # Add spacing before adjacent similarities
        adjacent_similarities_postprocessor.process_rosbag(rosbag_context)
                    
        logger.success(f"Completed processing for {relative_path}")

    logger.subsection("Main pipeline complete!", char="=")
    logger.success("All rosbags processed!")

    # ========================================================================
    # POSTPROCESSORS: RUN AFTER ALL ROSBAGS (if any remain)
    # ========================================================================
    # Note: Adjacent similarities are now computed per-rosbag above.
    # The run() method is kept for backward compatibility but is no longer
    # called by default. Uncomment below if you need to reprocess all rosbags.
    
    # postprocessors = [
    #     adjacent_similarities_postprocessor,
    # ]
    # 
    # for postprocessor in postprocessors:
    #     logger.info(f"\nRunning {type(postprocessor).__name__}...")
    #     postprocessor.run()
    # 
    # logger.success("All postprocessors complete!")

    # ========================================================================
    # DONE
    # ========================================================================
    
    logger.section("PIPELINE COMPLETE!")
    logger.section("", char="=")


if __name__ == "__main__":
    main()

