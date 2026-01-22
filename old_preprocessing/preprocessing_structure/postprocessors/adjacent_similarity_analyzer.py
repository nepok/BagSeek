"""
Postprocessor for computing similarities between adjacent embeddings.
"""
from pathlib import Path
from ..core import CompletionTracker


class AdjacentSimilarityAnalyzer:
    """
    Compute similarities between adjacent embeddings.
    
    Runs after main pipeline - analyzes embedding shards.
    """
    
    def __init__(self, embeddings_dir: Path, output_dir: Path):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
    
    def run(self):
        """
        Read all embeddings per topic per rosbag,
        compute adjacent similarities, create heatmaps.
        """
        print("\nComputing adjacent similarities...")
        
        # Check completion
        completion_tracker = CompletionTracker(self.output_dir / "completion.json")
        
        # Get all rosbag directories
        rosbag_dirs = [d for d in self.embeddings_dir.iterdir() if d.is_dir()]
        
        for rosbag_dir in rosbag_dirs:
            rosbag_name = rosbag_dir.name
            
            if completion_tracker.is_completed(rosbag_name):
                print(f"  ✓ Similarities already computed for {rosbag_name}, skipping")
                continue
            
            print(f"  Computing similarities for {rosbag_name}...")
            
            # TODO: Implement similarity computation
            # 
            # Algorithm:
            # 1. Load all embedding shards for this rosbag
            # 2. Group embeddings by topic
            # 3. For each topic:
            #    - Sort embeddings by timestamp
            #    - Compute similarity between adjacent pairs
            #    - Store similarities
            # 4. Generate visualization (heatmap)
            # 
            # Example structure:
            # import pickle
            # import numpy as np
            # 
            # # Load shards
            # embeddings_by_topic = {}
            # shard_files = sorted(rosbag_dir.glob("shard_*.pkl"))
            # 
            # for shard_file in shard_files:
            #     with open(shard_file, 'rb') as f:
            #         shard_data = pickle.load(f)
            #     
            #     for item in shard_data:
            #         topic = item['topic']
            #         if topic not in embeddings_by_topic:
            #             embeddings_by_topic[topic] = []
            #         embeddings_by_topic[topic].append(item)
            # 
            # # Compute similarities
            # similarities_by_topic = {}
            # for topic, embeddings in embeddings_by_topic.items():
            #     # Sort by timestamp
            #     embeddings.sort(key=lambda x: x['timestamp'])
            #     
            #     # Compute adjacent similarities
            #     similarities = []
            #     for i in range(len(embeddings) - 1):
            #         emb1 = embeddings[i]['embedding']
            #         emb2 = embeddings[i + 1]['embedding']
            #         # similarity = cosine_similarity(emb1, emb2)
            #         # similarities.append(similarity)
            #     
            #     similarities_by_topic[topic] = similarities
            # 
            # # Save results
            # output_file = self.output_dir / f"{rosbag_name}_similarities.json"
            # # Save similarities_by_topic
            # 
            # # Generate heatmap
            # # create_heatmap(similarities_by_topic, self.output_dir / f"{rosbag_name}_heatmap.png")
            
            # Mark as completed
            output_file = self.output_dir / f"{rosbag_name}_similarities.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            completion_tracker.mark_completed(rosbag_name, output_file)
            
            print(f"  ✓ Computed similarities for {rosbag_name}")
        
        print("✓ All similarity computations complete!")

