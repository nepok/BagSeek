# completion.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Default completion file path (can be overridden)
COMPLETION_FILE = Path("completion.json")

def get_timestamp() -> str:
    """Get current timestamp in readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_completion() -> dict:
    """Load completion status from JSON file.
    
    Returns:
        Dictionary with completion data, empty dict if file doesn't exist
    """
    if not COMPLETION_FILE.exists():
        return {}
    
    try:
        with open(COMPLETION_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_completion(data: dict):
    """Save completion status to JSON file.
    
    Args:
        data: Dictionary with completion data
    """
    COMPLETION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPLETION_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def completion_done(collector_name: str, mcap_path: Optional[Path] = None, mcap_id: Optional[str] = None) -> bool:
    """Check if a collector has already completed processing.
    
    Args:
        collector_name: Name of the collector (e.g., "timestamp_alignment")
        mcap_path: Optional path to MCAP file (for per-MCAP tracking)
        mcap_id: Optional MCAP identifier (for per-MCAP tracking)
        
    Returns:
        True if completed, False otherwise
    """
    completion = load_completion()
    
    # Check if collector exists in completion data
    if collector_name not in completion:
        return False
    
    collector_data = completion[collector_name]
    
    # If mcap_id is provided, check per-MCAP completion
    if mcap_id is not None:
        if "mcaps" not in collector_data:
            return False
        mcap_entry = collector_data["mcaps"].get(mcap_id)
        if mcap_entry is None:
            return False
        return mcap_entry.get("status") == "success"
    
    # Otherwise check global completion
    if "status" in collector_data:
        return collector_data["status"] == "success"
    
    return False

def update_completion(collectors, mcap_path: Optional[Path] = None, mcap_id: Optional[str] = None):
    """Update completion status for collectors.
    
    Args:
        collectors: List of collector instances
        mcap_path: Optional path to MCAP file (for per-MCAP tracking)
        mcap_id: Optional MCAP identifier (for per-MCAP tracking)
    """
    completion = load_completion()
    
    for collector in collectors:
        # Get collector name from class name (e.g., "TimestampAlignmentCollector" -> "timestamp_alignment")
        collector_name = collector.__class__.__name__
        # Convert CamelCase to snake_case
        collector_name = ''.join(['_' + c.lower() if c.isupper() else c for c in collector_name]).lstrip('_')
        
        # Initialize collector entry if it doesn't exist
        if collector_name not in completion:
            completion[collector_name] = {}
        
        collector_data = completion[collector_name]
        
        # Check if collector has output file (simple success check)
        success = False
        if hasattr(collector, 'csv_path'):
            success = Path(collector.csv_path).exists()
        elif hasattr(collector, 'output_path'):
            success = Path(collector.output_path).exists()
        
        if success:
            # If mcap_id is provided, track per-MCAP completion
            if mcap_id is not None:
                if "mcaps" not in collector_data:
                    collector_data["mcaps"] = {}
                
                collector_data["mcaps"][mcap_id] = {
                    "status": "success",
                    "completed_at": get_timestamp()
                }
                
                # Update total counts
                collector_data["total_mcaps"] = len(collector_data["mcaps"])
                collector_data["completed_mcaps"] = sum(
                    1 for m in collector_data["mcaps"].values() 
                    if m.get("status") == "success"
                )
            else:
                # Global completion
                collector_data["status"] = "success"
                collector_data["completed_at"] = get_timestamp()
        else:
            # Mark as failed
            if mcap_id is not None:
                if "mcaps" not in collector_data:
                    collector_data["mcaps"] = {}
                collector_data["mcaps"][mcap_id] = {
                    "status": "failed",
                    "completed_at": get_timestamp()
                }
            else:
                collector_data["status"] = "failed"
                collector_data["completed_at"] = get_timestamp()
    
    save_completion(completion)
