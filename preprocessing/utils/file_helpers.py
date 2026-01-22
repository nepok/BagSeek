"""
File system helper utilities for rosbag processing.

Provides functions to find and list rosbag directories and MCAP files.
"""
from pathlib import Path
from typing import List


def get_all_rosbags(rosbags_dir: Path) -> List[Path]:
    """
    Discover all rosbags in a directory (including multi-part rosbags).
    
    Handles two types of rosbags:
    1. Regular rosbag: rosbags/my_rosbag/
    2. Multi-part rosbag: rosbags/my_rosbag_multi_parts/Part_1/, Part_2/, etc.
    
    Args:
        rosbags_dir: Parent directory containing rosbag folders
    
    Returns:
        List of Path objects, one for each rosbag, sorted by base name then Part number
    
    Example:
        >>> rosbags = get_all_rosbags(Path("/data/rosbags"))
        >>> for rosbag in rosbags:
        ...     print(rosbag.name)
    """
    # DEBUG: Return only one specific rosbag for debugging
    #debug_rosbag = Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_09_10-09_15_13")
    """debug_rosbags = [Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_09_24-14_13_40"),
                     Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_07_25-12_17_25"),
                     Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_07_23-07_29_39_multi_parts/Part_2"),
                     Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_09_26-12_49_12")]"""
    #if any(rosbag.exists() and rosbag.is_dir() for rosbag in debug_rosbags):
    #    return debug_rosbags
    
    rosbags_dir = Path(rosbags_dir)
    
    if not rosbags_dir.exists():
        raise ValueError(f"Rosbags directory does not exist: {rosbags_dir}")
    
    if not rosbags_dir.is_dir():
        raise ValueError(f"Path is not a directory: {rosbags_dir}")
    
    rosbags = []
    
    # Iterate over all items in rosbags_dir
    for item in rosbags_dir.iterdir():
        # Skip non-directories and EXCLUDED folders
        if not item.is_dir() or "EXCLUDED" in item.name:
            continue
        
        # Check if this is a multi-part rosbag container
        if item.name.endswith("_multi_parts"):
            # Go one level deeper and collect all Part_N folders
            for part_dir in item.iterdir():
                if not part_dir.is_dir():
                    continue
                
                # Check if it's a Part_N folder (Part_1, Part_2, etc.)
                if part_dir.name.startswith("Part_"):
                    part_suffix = part_dir.name[5:]  # Remove "Part_" prefix
                    # Verify it's a valid Part_N format (N is a number)
                    if part_suffix.isdigit():
                        rosbags.append(part_dir)
        else:
            # Regular rosbag - include the folder itself
            rosbags.append(item)
    
    # Sort rosbags: primary by base name (without _multi_parts), secondary by Part number
    def sort_key(path: Path):
        """Extract base name and part number for sorting."""
        if path.name.startswith("Part_"):
            # Multi-part rosbag: extract base name from parent and Part number
            parent_name = path.parent.name
            base_name = parent_name.replace("_multi_parts", "")
            part_num = int(path.name[5:])  # Extract Part number
        else:
            # Regular rosbag: use folder name as base, Part number is 0
            base_name = path.name
            part_num = 0
        
        return (base_name, part_num)
    
    rosbags.sort(key=sort_key)
    return rosbags


def get_all_mcaps(rosbag_dir: Path) -> List[Path]:
    """
    Get all MCAP files from a rosbag, sorted numerically.
    
    Filename format: rosbag2_YYYY_MM_DD-HH_MM_SS_N.mcap
    Files are sorted by the number N at the end.
    
    Args:
        rosbag_dir: Rosbag directory containing .mcap files
    
    Returns:
        List of Path objects to MCAP files, sorted by numeric order
    
    Example:
        >>> mcaps = get_all_mcaps(Path("/data/rosbags/rosbag_2024_01_01"))
        >>> for mcap in mcaps:
        ...     print(mcap.name)
    """
    rosbag_dir = Path(rosbag_dir)
    
    if not rosbag_dir.exists():
        raise ValueError(f"Rosbag directory does not exist: {rosbag_dir}")
    
    if not rosbag_dir.is_dir():
        raise ValueError(f"Path is not a directory: {rosbag_dir}")
    
    # Use glob to find all .mcap files
    mcaps = list(rosbag_dir.glob("*.mcap"))
    
    def extract_number(path: Path) -> float:
        """Extract the number from the mcap filename for sorting."""
        stem = path.stem  # filename without extension
        # Extract number after the last underscore
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        # If no number found, return a large number to put it at the end
        return float('inf')
    
    mcaps.sort(key=extract_number)
    return mcaps


def validate_rosbag(rosbag_dir: Path) -> bool:
    """
    Validate that a directory is a valid rosbag.
    
    Args:
        rosbag_dir: Directory to validate
    
    Returns:
        True if valid rosbag, False otherwise
    """
    # TODO: Implement rosbag validation
    # 
    # Checks could include:
    # - Directory exists
    # - Contains at least one .mcap file
    # - MCAP files are valid (not corrupted)
    # - Has expected metadata files
    # - Minimum size requirements
    # 
    # Example:
    # if not rosbag_dir.exists() or not rosbag_dir.is_dir():
    #     return False
    # 
    # mcaps = list(rosbag_dir.glob("*.mcap"))
    # if not mcaps:
    #     return False
    # 
    # # Optional: Validate MCAP file headers
    # # for mcap in mcaps:
    # #     if not validate_mcap_file(mcap):
    # #         return False
    # 
    # return True
    
    # Placeholder
    return rosbag_dir.exists() and rosbag_dir.is_dir()


def validate_mcap_file(mcap_path: Path) -> bool:
    """
    Validate that a file is a valid MCAP file.
    
    Args:
        mcap_path: Path to MCAP file
    
    Returns:
        True if valid MCAP, False otherwise
    """
    # TODO: Implement MCAP file validation
    # 
    # Checks could include:
    # - File exists and is readable
    # - Has .mcap extension
    # - Valid MCAP header/magic bytes
    # - Not corrupted (can be opened by mcap reader)
    # - Minimum size (not empty)
    # 
    # Example:
    # if not mcap_path.exists() or not mcap_path.is_file():
    #     return False
    # 
    # if mcap_path.suffix != '.mcap':
    #     return False
    # 
    # if mcap_path.stat().st_size == 0:
    #     return False
    # 
    # # Try to open with mcap reader
    # try:
    #     from mcap.reader import make_reader
    #     with open(mcap_path, 'rb') as f:
    #         reader = make_reader(f)
    #         # Check if we can read header
    #         summary = reader.get_summary()
    #     return True
    # except Exception:
    #     return False
    
    # Placeholder
    return mcap_path.exists() and mcap_path.is_file() and mcap_path.suffix == '.mcap'


def get_rosbag_info(rosbag_dir: Path) -> dict:
    """
    Get information about a rosbag.
    
    Args:
        rosbag_dir: Rosbag directory
    
    Returns:
        Dictionary with rosbag information (name, mcap_count, total_size, etc.)
    """
    # TODO: Implement rosbag info extraction
    # 
    # Information to include:
    # - Rosbag name
    # - Number of MCAP files
    # - Total size
    # - Date range (from filenames or metadata)
    # - Topics (if readily available)
    # 
    # Example:
    # mcaps = get_all_mcaps(rosbag_dir)
    # total_size = sum(m.stat().st_size for m in mcaps)
    # 
    # return {
    #     "name": rosbag_dir.name,
    #     "path": str(rosbag_dir),
    #     "mcap_count": len(mcaps),
    #     "total_size_bytes": total_size,
    #     "total_size_gb": total_size / (1024**3),
    # }
    
    # Placeholder
    mcaps = get_all_mcaps(rosbag_dir)
    return {
        "name": rosbag_dir.name,
        "path": str(rosbag_dir),
        "mcap_count": len(mcaps),
    }

