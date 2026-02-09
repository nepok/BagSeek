/**
 * Topic sorting utilities for frontend.
 * Matches the backend sorting logic.
 */

function getCameraPositionOrder(topicName: string): number {
  const topicLower = topicName.toLowerCase();
  
  if (topicLower.includes("side") && topicLower.includes("left")) {
    return 0;
  } else if (topicLower.includes("side") && topicLower.includes("right")) {
    return 1;
  } else if (topicLower.includes("rear") && topicLower.includes("left")) {
    return 2;
  } else if (topicLower.includes("rear") && topicLower.includes("mid")) {
    return 3;
  } else if (topicLower.includes("rear") && topicLower.includes("right")) {
    return 4;
  } else {
    return 5;
  }
}

function getTopicSortPriority(topicName: string, topicType?: string): number {
  const topicLower = topicName.toLowerCase();
  const isZed = topicLower.includes("zed");
  
  let isImage = false;
  let isPointcloud = false;
  let isPositional = false;
  
  if (topicType) {
    const typeLower = topicType.toLowerCase();
    isImage = typeLower.includes("sensor_msgs") && (typeLower.includes("image") || typeLower.includes("compressedimage"));
    isPointcloud = typeLower.includes("pointcloud") || typeLower.includes("point_cloud");
    isPositional = ["navsatfix", "gps", "gnss", "tf", "odom", "pose"].some(x => typeLower.includes(x));
  } else {
    isImage = ["image", "camera", "rgb", "color"].some(x => topicLower.includes(x));
    isPointcloud = ["pointcloud", "point_cloud", "lidar", "pcl"].some(x => topicLower.includes(x));
    isPositional = ["gps", "gnss", "navsat", "tf", "odom", "pose", "position"].some(x => topicLower.includes(x));
  }
  
  // Priority 0: Zed image topics
  if (isZed && isImage) {
    return 0;
  }
  
  // Priority 1: Camera/image topics (non-zed)
  if (isImage) {
    return 1;
  }
  
  // Priority 2: PointCloud topics
  if (isPointcloud) {
    return 2;
  }
  
  // Priority 3: Positional topics
  if (isPositional) {
    return 3;
  }
  
  // Priority 4: Everything else (including non-image zed topics)
  return 4;
}

/**
 * Sort topics according to priority order.
 * Priority: zed images -> camera images -> pointclouds -> positional -> rest
 * Within image topics, camera positions are ordered: side left, side right, rear left, rear mid, rear right.
 */
export function sortTopics(
  topics: string[],
  topicTypes?: Record<string, string>
): string[] {
  return [...topics].sort((a, b) => {
    const priorityA = getTopicSortPriority(a, topicTypes?.[a]);
    const priorityB = getTopicSortPriority(b, topicTypes?.[b]);
    
    if (priorityA !== priorityB) {
      return priorityA - priorityB;
    }
    
    // For image topics (priority 0 or 1), apply camera position ordering
    if (priorityA === 0 || priorityA === 1) {
      const cameraOrderA = getCameraPositionOrder(a);
      const cameraOrderB = getCameraPositionOrder(b);
      
      if (cameraOrderA !== cameraOrderB) {
        return cameraOrderA - cameraOrderB;
      }
    }
    
    // Finally, sort alphabetically
    return a.toLowerCase().localeCompare(b.toLowerCase());
  });
}

/**
 * Sort a topics object (Record<string, string>) and return a new object with sorted keys.
 */
export function sortTopicsObject(
  topics: Record<string, string>
): Record<string, string> {
  const sortedKeys = sortTopics(Object.keys(topics), topics);
  const sorted: Record<string, string> = {};
  for (const key of sortedKeys) {
    sorted[key] = topics[key];
  }
  return sorted;
}
