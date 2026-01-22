"""
MCAP-level processor for generating embeddings from image messages.
"""
from pathlib import Path
from typing import List, Dict, Any
from ...core import Processor, ProcessingLevel, McapProcessingContext, CompletionTracker
from ...collectors import ImageMessagesCollector


class EmbeddingGenerator(Processor):
    """
    Generate embeddings for image messages.
    
    Uses ImageMessagesCollector to gather image data.
    Generates embeddings using specified models and writes to shards.
    
    Operates at MCAP level - runs once per mcap file.
    """
    
    def __init__(self, models: List[str], shard_size: int = 100000):
        super().__init__("embeddings", ProcessingLevel.MCAP)
        self.models = models
        self.shard_size = shard_size
        self.required_collectors = [ImageMessagesCollector]
        
        # Per-rosbag state (will need to persist across mcaps)
        self.current_shard = {}
        self.shard_counts = {}
    
    def process(self, context: McapProcessingContext, data: Any) -> Dict:
        """
        Extract images, preprocess, and generate embeddings.
        Write to shards when shard_size is reached.
        
        Args:
            context: Processing context with mcap_path set
            data: Dictionary containing collector results
        
        Returns:
            Embedding generation info
        """
        # Check completion
        output_dir = context.config.embeddings_dir / context.get_rosbag_name()
        completion_file = context.config.embeddings_dir / "completion.json"
        completion_tracker = CompletionTracker(completion_file)
        
        if completion_tracker.is_completed(context.get_rosbag_name(), context.get_mcap_name()):
            print(f"    ✓ Embeddings already generated for {context.get_mcap_name()}, skipping")
            return {}
        
        print(f"    Generating embeddings for {context.get_mcap_name()}...")
        
        # Get images from collector
        messages_by_topic = data.get("ImageMessagesCollector", {})
        
        embeddings = []
        
        # TODO: Implement embedding generation
        # 
        # High-level algorithm:
        # 1. For each image topic and its messages:
        #    - For each image message:
        #      a. Preprocess image for each model
        #      b. Generate embeddings using the models
        #      c. Store embedding with metadata (timestamp, topic, etc.)
        # 
        # 2. Manage sharding:
        #    - Accumulate embeddings in memory
        #    - When shard_size is reached, write shard to disk
        #    - Track shard numbers across mcaps in same rosbag
        # 
        # Example structure:
        # for topic, messages in messages_by_topic.items():
        #     for msg in messages:
        #         # Preprocess image
        #         # preprocessed = preprocess_for_models(msg.data, self.models)
        #         
        #         # Generate embeddings for each model
        #         for model_name in self.models:
        #             # embedding = generate_embedding(preprocessed, model_name)
        #             
        #             embeddings.append({
        #                 "topic": topic,
        #                 "timestamp": msg.timestamp,
        #                 "model": model_name,
        #                 "embedding": None,  # Replace with actual embedding vector
        #                 "mcap": context.get_mcap_name()
        #             })
        # 
        # # Write to shard if needed
        # self._write_to_shard(context, embeddings)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mark as completed
        completion_tracker.mark_completed(
            context.get_rosbag_name(),
            output_dir,
            mcap_name=context.get_mcap_name(),
            metadata={"embedding_count": len(embeddings)}
        )
        
        print(f"    ✓ Generated {len(embeddings)} embeddings")
        return {"embedding_count": len(embeddings)}
    
    def _write_to_shard(self, context: McapProcessingContext, embeddings: List):
        """
        Write embeddings to appropriate shard.
        
        Args:
            context: Processing context
            embeddings: List of embeddings to write
        """
        # TODO: Implement shard writing
        # 
        # 1. Load or initialize current shard for this rosbag
        # 2. Append embeddings to current shard
        # 3. If shard size exceeded:
        #    - Write shard to disk (e.g., shard_0000.pkl)
        #    - Increment shard counter
        #    - Start new shard
        # 
        # Example:
        # rosbag_name = context.get_rosbag_name()
        # 
        # if rosbag_name not in self.current_shard:
        #     self.current_shard[rosbag_name] = []
        #     self.shard_counts[rosbag_name] = 0
        # 
        # self.current_shard[rosbag_name].extend(embeddings)
        # 
        # if len(self.current_shard[rosbag_name]) >= self.shard_size:
        #     # Write shard
        #     output_dir = context.output_dir / "embeddings" / rosbag_name
        #     shard_file = output_dir / f"shard_{self.shard_counts[rosbag_name]:04d}.pkl"
        #     
        #     # Save with pickle
        #     # with open(shard_file, 'wb') as f:
        #     #     pickle.dump(self.current_shard[rosbag_name], f)
        #     
        #     # Reset for next shard
        #     self.current_shard[rosbag_name] = []
        #     self.shard_counts[rosbag_name] += 1
        pass

