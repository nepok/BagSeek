"""Topic utility functions."""
from typing import Optional


def get_camera_position_order(topic_name: str) -> int:
    """
    Get ordering for camera positions within the same priority level.
    Returns order where lower order = appears first.
    
    Camera position order:
    0: side left
    1: side right
    2: rear left
    3: rear mid
    4: rear right
    5: everything else (will be sorted alphabetically)
    """
    topic_lower = topic_name.lower()
    
    # Check for camera position keywords (order matters - check more specific first)
    if "side" in topic_lower and "left" in topic_lower:
        return 0
    elif "side" in topic_lower and "right" in topic_lower:
        return 1
    elif "rear" in topic_lower and "left" in topic_lower:
        return 2
    elif "rear" in topic_lower and "mid" in topic_lower:
        return 3
    elif "rear" in topic_lower and "right" in topic_lower:
        return 4
    else:
        return 5


def get_topic_sort_priority(topic_name: str, topic_type: Optional[str] = None) -> int:
    """
    Get sorting priority for a topic.
    Returns priority where lower priority = appears first.
    
    Priority order:
    0: Topics containing "zed"
    1: Image topics (sensor_msgs/msg/Image, sensor_msgs/msg/CompressedImage)
    2: PointCloud topics (sensor_msgs/msg/PointCloud2)
    3: Positional topics (NavSatFix, GPS, GNSS, TF, pose, position, odom)
    4: Everything else (alphabetically)
    """
    topic_lower = topic_name.lower()
    
    # Priority 0: Topics containing "zed"
    if "zed" in topic_lower:
        return 0
    
    # Determine topic category
    is_image = False
    is_pointcloud = False
    is_positional = False
    
    if topic_type:
        # Use provided topic type
        type_lower = topic_type.lower()
        is_image = "image" in type_lower and ("sensor_msgs" in type_lower or "compressedimage" in type_lower)
        is_pointcloud = "pointcloud" in type_lower or "point_cloud" in type_lower
        is_positional = any(x in type_lower for x in ["navsatfix", "gps", "gnss", "tf", "odom", "pose"])
    else:
        # Infer from topic name
        is_image = any(x in topic_lower for x in ["image", "camera", "rgb", "color"])
        is_pointcloud = any(x in topic_lower for x in ["pointcloud", "point_cloud", "lidar", "pcl"])
        is_positional = any(x in topic_lower for x in ["gps", "gnss", "navsat", "tf", "odom", "pose", "position"])
    
    if is_image:
        return 1
    elif is_pointcloud:
        return 2
    elif is_positional:
        return 3
    else:
        return 4


def sort_topics(topics: list[str], topic_types: Optional[dict[str, str]] = None) -> list[str]:
    """
    Sort topics according to the default priority order.
    Within image topics, camera positions are ordered: side left, side right, rear left, rear mid, rear right.
    
    Args:
        topics: List of topic names
        topic_types: Optional dict mapping topic names to their types
    
    Returns:
        Sorted list of topics
    """
    if topic_types is None:
        topic_types = {}
    
    def sort_key(topic: str) -> tuple[int, int, str]:
        topic_type = topic_types.get(topic)
        priority = get_topic_sort_priority(topic, topic_type)
        
        # For image topics (priority 1), apply camera position ordering
        camera_order = 5  # Default (alphabetical)
        if priority == 1:  # Image topics
            camera_order = get_camera_position_order(topic)
        
        return (priority, camera_order, topic.lower())
    
    return sorted(topics, key=sort_key)
