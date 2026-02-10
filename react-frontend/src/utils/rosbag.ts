/**
 * Extract the rosbag display name from a path, preserving parent for multipart rosbags.
 * Mirrors backend's extract_rosbag_name_from_path logic.
 *
 * Examples:
 *   - Regular: 'rosbag2_2025_07_25-12_17_25' -> 'rosbag2_2025_07_25-12_17_25'
 *   - Multipart: 'rosbag2_xxx_multi_parts/Part_1' -> 'rosbag2_xxx_multi_parts/Part_1'
 *   - Deep path: '/data/rosbag2_xxx_multi_parts/Part_1' -> 'rosbag2_xxx_multi_parts/Part_1'
 */
export function extractRosbagName(rosbagPath: string): string {
  const parts = rosbagPath.split('/');
  if (parts.length >= 2) {
    const parent = parts[parts.length - 2];
    const basename = parts[parts.length - 1];
    if (parent.endsWith('_multi_parts') && basename.startsWith('Part_')) {
      return `${parent}/${basename}`;
    }
  }
  return parts[parts.length - 1];
}
