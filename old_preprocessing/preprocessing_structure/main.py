"""
Rosbag Processing Pipeline - Main Entry Point

This script shows the explicit iteration logic for the rosbag processing pipeline.
You have full control over the iteration flow.
"""
from pathlib import Path
from mcap_ros2.decoder import DecoderFactory
from mcap.reader import SeekingReader

# Import configuration
from preprocessing.config import Config

# Import processors
from preprocessing.processors import (
    TopicsExtractor,
    FencepostCalculator,
    TimestampAlignmentBuilder,
    PositionalLookupBuilder,
    EmbeddingGenerator,
)

# Import collectors
from preprocessing.collectors import (
    TimestampsCollector,
    ImageMessagesCollector,
    PositionMessagesCollector,
    FencepostImageCollector,
)

# Import postprocessors
from preprocessing.postprocessors import (
    AdjacentSimilarityAnalyzer,
    PositionalLookupAggregator,
    RepresentativePreviewsStitcher,
)

# Import core
from preprocessing.core import RosbagProcessingContext, McapProcessingContext


# Import utils
from preprocessing.utils import (
    get_logger,
    get_all_rosbags,
    get_all_mcaps,
)


def main():
    """
    Main pipeline with explicit iteration logic.
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Load configuration from .env file
    config = Config.load_config()
    
    # Setup logger
    logger = get_logger()
    logger.section("ROSBAG PROCESSING PIPELINE")

    # ========================================================================
    # COMPONENT REGISTRATION
    # ========================================================================
    
    logger.info("\nRegistering pipeline components...")
    
    # Level 1: Rosbag-level processors (no collectors needed)
    rosbag_processors = [
        TopicsExtractor(),
        FencepostCalculator(),
    ]
    
    # Create lookup for processors by name
    rosbag_processors_by_name = {p.name: p for p in rosbag_processors}

    # Level 2: MCAP-level processors (with collector dependencies)
    mcap_processors = [
        #TimestampAlignmentBuilder(),
        #PositionalLookupBuilder(grid_size=config.grid_size),
        #EmbeddingGenerator(
        #    models=config.embedding_models,
        #    shard_size=config.shard_size
        #),
    ]
    
    # Auto-discover required collector classes from processors
    required_collector_classes = set()
    for processor in mcap_processors:
        required_collector_classes.update(processor.required_collectors)
    
    # Level 3: Postprocessors (run after all rosbags)
    postprocessors = [
        RepresentativePreviewsStitcher(config.representative_previews_dir),
        #AdjacentSimilarityAnalyzer(
        #    embeddings_dir=config.embeddings_dir,
        #    output_dir=config.adjacent_similarities_dir
        #),
    ]

    for postprocessor in postprocessors:
        required_collector_classes.update(postprocessor.required_collectors)
    
    logger.success(f"Registered {len(rosbag_processors)} rosbag-level processors")
    logger.success(f"Registered {len(mcap_processors)} mcap-level processors")
    logger.success(f"Registered {len(postprocessors)} postprocessors")
    
    if required_collector_classes:
        logger.info(f"Auto-discovered {len(required_collector_classes)} required collector type(s): "
                    f"{', '.join([cls.__name__ for cls in required_collector_classes])}")
    else:
        logger.info("No collectors required by registered processors")

    # ========================================================================
    # MAIN PIPELINE: ITERATE THROUGH ROSBAGS
    # ========================================================================
    
    logger.subsection("Running main pipeline...")
    
    # Get all rosbag directories using helper function
    rosbags = get_all_rosbags(config.rosbags_dir)
    logger.info(f"Found {len(rosbags)} rosbag(s) to process\n")
    
    for idx, rosbag_path in enumerate(rosbags, 1):
        logger.subsection(f"Processing rosbag ({idx}/{len(rosbags)}): {rosbag_path.name}")
        
        mcap_files = get_all_mcaps(rosbag_path)
        logger.info(f"Found {len(mcap_files)} mcap file(s) in {rosbag_path}")

        # Create context for this rosbag
        rosbag_context = RosbagProcessingContext(
            rosbag_path=rosbag_path,
            mcap_files=mcap_files,
            config=config
        )
    
        # ====================================================================
        # RUN ROSBAG-LEVEL PROCESSORS
        # ====================================================================
        
        for rosbag_processor in rosbag_processors:
            if rosbag_processor.should_process(rosbag_context):
                rosbag_processor.process(rosbag_context, data=None)
        
        # ====================================================================
        # ITERATE THROUGH MCAP FILES
        # ====================================================================
        
        # Get all MCAP files using helper function
        
        for mcap_path in mcap_files:
            mcap_id = int(mcap_path.stem.split('_')[-1])
            logger.info(f"\n  Processing mcap ({mcap_id + 1}/{len(mcap_files)}): {mcap_path.name}")
            
            # Create context for this mcap
            mcap_context = McapProcessingContext(
                rosbag_path=rosbag_path, # do we need this?
                mcap_path=mcap_path,
                config=config
            )
  
            # Open MCAP file once for both summary extraction and message iteration
            with open(mcap_path, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()])
                
                # ================================================================
                # EXTRACT TOPIC MESSAGE COUNTS
                # ================================================================
                
                channels = reader.get_summary().channels
                channel_message_counts = reader.get_summary().statistics.channel_message_counts
                
                # Create dict with topic name and message count for each topic
                topic_info = {
                    channel.topic: {
                        "topic_id": channel.id,
                        "message_count": channel_message_counts.get(channel.id, 0)
                    }
                    for channel_id, channel in channels.items()
                }
                
                logger.info(f"    Found {len(topic_info)} topic(s) in MCAP")
                
                # ================================================================
                # INSTANTIATE COLLECTORS (fresh instances for this MCAP)
                # ================================================================
                
                collectors = []
                for collector_class in required_collector_classes:
                    if collector_class == FencepostImageCollector:
                        # Pass mcap_context, fencepost_calculator, and topic_info
                        fencepost_calc = rosbag_processors_by_name.get("fencepost_calculator")
                        collectors.append(FencepostImageCollector(mcap_context, fencepost_calc, topic_info))
                    else:
                        # For collectors that don't need special initialization
                        collectors.append(collector_class())
                
                if not collectors:
                    # No collectors needed, processors can run without data
                    logger.info("    No collectors needed")
                    for processor in mcap_processors:
                        if processor.should_process(mcap_context):
                            processor.process(mcap_context, data={})
                    continue
                
                logger.info(f"    Running {len(collectors)} collector(s)...")
                
                # ================================================================
                # SINGLE PASS: ITERATE MESSAGES AND COLLECT DATA
                # ================================================================
                
                for schema, channel, message, ros2_msg in reader.iter_decoded_messages(log_time_order=True, reverse=False):
                    # Each collector does its own filtering in collect_message
                    for collector in collectors:
                        collector.collect_message(message, channel, schema)
            
            # ================================================================
            # PACKAGE COLLECTED DATA
            # ================================================================
            
            collector_data = {
                type(collector).__name__: collector.get_data() 
                for collector in collectors
            }
            
            logger.info(f"    Collected data from {len(collector_data)} collectors")

            print(collector_data.items())
            
            exit()
            # ================================================================
            # RUN MCAP-LEVEL PROCESSORS WITH COLLECTED DATA
            # ================================================================
            
            for processor in mcap_processors:
                if processor.should_process(mcap_context):
                    processor.process(mcap_context, collector_data)

    logger.subsection("Main pipeline complete!", char="=")
    logger.success("All rosbags processed!")
    
    # ========================================================================
    # RUN POSTPROCESSORS
    # ========================================================================
    
    logger.section("Running postprocessors...")
    
    for postprocessor in postprocessors:
        logger.info(f"\nRunning {type(postprocessor).__name__}...")
        postprocessor.run()
    
    logger.success("All postprocessors complete!")
    
    # ========================================================================
    # DONE
    # ========================================================================
    logger.section("PIPELINE COMPLETE!")
    logger.section("", char="=")

if __name__ == "__main__":
    main()
