"""
Postprocessor for stitching representative preview images.
"""
import json
from pathlib import Path
from typing import Dict, List
from ..core import CompletionTracker
from ..collectors import FencepostImageCollector


class RepresentativePreviewsStitcher:
    """
    Stitch representative preview images into final preview grids.
    
    Takes the individual fencepost images extracted by FencepostCalculator
    and combines them into a single preview grid for each rosbag.
    
    Runs after main pipeline as a postprocessor.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize stitcher.
        
        Args:
            output_dir: Base directory for representative previews
        """
        self.output_dir = Path(output_dir)
        self.stitched_dir = self.output_dir / "stitched"
        self.required_collectors = [FencepostImageCollector]
    
    def run(self):
        """
        Stitch preview images for all rosbags.
        """
        print("\n" + "="*60)
        print("Stitching representative preview images...")
        print("="*60)
        
        # Check if output directory exists
        if not self.output_dir.exists():
            print("  No preview data found, skipping")
            return
        
        # Check completion
        completion_tracker = CompletionTracker(
            self.stitched_dir / "completion.json"
        )
        
        # Get all rosbag directories
        rosbag_dirs = [d for d in self.output_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.') and d.name != "stitched"]
        
        if not rosbag_dirs:
            print("  No rosbag preview data found")
            return
        
        print(f"  Found {len(rosbag_dirs)} rosbag(s) with preview data\n")
        
        for rosbag_dir in rosbag_dirs:
            rosbag_name = rosbag_dir.name
            
            if completion_tracker.is_completed(rosbag_name):
                print(f"  ✓ Preview already stitched for {rosbag_name}, skipping")
                continue
            
            print(f"  Processing {rosbag_name}...")
            
            # Load preview info
            info_file = rosbag_dir / "preview_info.json"
            if not info_file.exists():
                print(f"    ⚠ No preview_info.json found, skipping")
                continue
            
            with open(info_file, 'r') as f:
                preview_info = json.load(f)
            
            # Get list of parts
            parts = preview_info.get("previews", [])
            topics = preview_info.get("topics", [])
            
            print(f"    Found {len(parts)} part(s) from {len(topics)} topic(s)")
            
            # TODO: Implement actual stitching logic
            # For now, just log what would be stitched
            self._log_stitching_plan(rosbag_name, parts, topics)
            
            # Create output directory
            output_dir = self.stitched_dir / rosbag_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata about what's ready for stitching
            stitch_metadata = {
                "rosbag": rosbag_name,
                "total_parts": len(parts),
                "topics": topics,
                "parts": parts,
                "status": "ready_for_stitching"
            }
            
            metadata_file = output_dir / "stitch_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(stitch_metadata, f, indent=2)
            
            # Mark as completed
            completion_tracker.mark_completed(
                rosbag_name,
                output_dir,
                metadata={"parts_count": len(parts), "topics_count": len(topics)}
            )
            
            print(f"    ✓ Prepared stitching metadata")
        
        print("\n✓ All preview stitching metadata prepared!")
        print("\nNote: Actual image stitching will be implemented when image saving is added")
    
    def _log_stitching_plan(self, rosbag_name: str, parts: List[Dict], topics: List[str]):
        """
        Log the stitching plan for debugging.
        
        Args:
            rosbag_name: Name of the rosbag
            parts: List of part metadata
            topics: List of topics
        """
        print(f"    Stitching plan:")
        print(f"      Topics: {', '.join(topics)}")
        print(f"      Parts to stitch:")
        
        for part in sorted(parts, key=lambda x: (x['part_index'], x['topic'])):
            print(f"        Part {part['part_index']}: {part['topic']} "
                  f"(MCAP {part['mcap_index']}, index {part['message_index']})")
    
    def _stitch_images(self, image_files: List[Path], output_path: Path):
        """
        Stitch multiple images into a grid.
        
        This is a stub for the actual stitching implementation.
        
        Args:
            image_files: List of image file paths to stitch
            output_path: Output path for stitched image
        """
        # TODO: Implement image stitching
        # 
        # Example using PIL:
        # from PIL import Image
        # 
        # # Load all images
        # images = [Image.open(f) for f in image_files]
        # 
        # # Calculate grid dimensions
        # num_images = len(images)
        # cols = min(num_images, 4)  # Max 4 columns
        # rows = (num_images + cols - 1) // cols
        # 
        # # Get max dimensions
        # max_width = max(img.width for img in images)
        # max_height = max(img.height for img in images)
        # 
        # # Create output image
        # grid_width = cols * max_width
        # grid_height = rows * max_height
        # grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        # 
        # # Paste images into grid
        # for idx, img in enumerate(images):
        #     row = idx // cols
        #     col = idx % cols
        #     x = col * max_width
        #     y = row * max_height
        #     grid.paste(img, (x, y))
        # 
        # # Save result
        # grid.save(output_path)
        pass
