"""MCAP utility functions for message formatting."""
import base64
import struct
import logging
import numpy as np
from flask import jsonify


def normalize_topic(topic: str) -> str:
    """Normalize topic name: replace / with _ and remove leading _"""
    return topic.replace('/', '_').lstrip('_')


def format_message_response(msg, topic_type: str, timestamp: str):
    """
    Format a deserialized ROS2 message into JSON response based on message type.
    
    Args:
        msg: Deserialized ROS2 message object
        topic_type: String type of the message (e.g., 'sensor_msgs/msg/PointCloud2')
        timestamp: Timestamp string to include in response
    
    Returns:
        Flask jsonify response
    """
    match topic_type:
        case 'sensor_msgs/msg/CompressedImage' | 'sensor_msgs/msg/Image':
            # Extract image data
            image_data_bytes = bytes(msg.data) if hasattr(msg, 'data') else None
            if not image_data_bytes:
                return jsonify({'error': 'No image data found'}), 404
            
            # Detect format
            format_str = 'jpeg'  # default
            if hasattr(msg, 'format') and msg.format:
                format_lower = msg.format.lower()
                if 'png' in format_lower:
                    format_str = 'png'
                elif 'jpeg' in format_lower or 'jpg' in format_lower:
                    format_str = 'jpeg'
            
            # Encode to base64
            image_data_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
            
            return jsonify({
                'type': 'image',
                'image': image_data_base64,
                'format': format_str,
                'timestamp': timestamp
            })
        case 'sensor_msgs/msg/PointCloud2':
            # Extract point cloud data, filtering out NaNs, Infs, and zeros
            pointCloud = []
            colors = []
            point_step = msg.point_step
            is_bigendian = msg.is_bigendian
            
            # Check fields to find color data
            has_rgb = False
            has_separate_rgb = False
            rgb_offset = None
            r_offset = None
            g_offset = None
            b_offset = None
            rgb_datatype = None
            r_datatype = None
            g_datatype = None
            b_datatype = None
            
            for field in msg.fields:
                # Debug: log all fields to understand structure
                logging.debug(f"PointCloud2 field: name={field.name}, offset={field.offset}, datatype={field.datatype}, count={field.count}")
                if field.name == 'rgb' or field.name == 'rgba':
                    rgb_offset = field.offset
                    rgb_datatype = field.datatype
                    has_rgb = True
                    logging.debug(f"Found RGB field: offset={rgb_offset}, datatype={rgb_datatype}")
                    break
                elif field.name == 'r':
                    r_offset = field.offset
                    r_datatype = field.datatype
                    has_separate_rgb = True
                elif field.name == 'g':
                    g_offset = field.offset
                    g_datatype = field.datatype
                elif field.name == 'b':
                    b_offset = field.offset
                    b_datatype = field.datatype
            
            logging.debug(f"Color detection: has_rgb={has_rgb}, has_separate_rgb={has_separate_rgb}, rgb_datatype={rgb_datatype}")
            
            # Extract points and colors
            for i in range(0, len(msg.data), point_step):
                # Extract x, y, z coordinates
                if is_bigendian:
                    x, y, z = struct.unpack_from('>fff', msg.data, i)
                else:
                    x, y, z = struct.unpack_from('<fff', msg.data, i)
                
                if all(np.isfinite([x, y, z])) and not (x == 0 and y == 0 and z == 0):
                    pointCloud.extend([x, y, z])
                    
                    # Extract RGB colors if available
                    if has_rgb and rgb_offset is not None:
                        try:
                            # Extract RGB as uint32 (4 bytes) - most common format
                            if is_bigendian:
                                rgb = struct.unpack_from('>I', msg.data, i + rgb_offset)[0]
                                # Big-endian: RRGGBBAA format
                                r = (rgb >> 24) & 0xFF
                                g = (rgb >> 16) & 0xFF
                                b = (rgb >> 8) & 0xFF
                            else:
                                rgb = struct.unpack_from('<I', msg.data, i + rgb_offset)[0]
                                # Little-endian: Try different formats
                                b = rgb & 0xFF
                                g = (rgb >> 8) & 0xFF
                                r = (rgb >> 16) & 0xFF
                            colors.extend([r, g, b])
                        except Exception as e:
                            logging.warning(f"Failed to extract RGB color at offset {i + rgb_offset}: {e}")
                            pass
                    elif has_separate_rgb and r_offset is not None and g_offset is not None and b_offset is not None:
                        # Extract separate r, g, b fields
                        try:
                            if r_datatype == 7:  # FLOAT32
                                if is_bigendian:
                                    r_val = struct.unpack_from('>f', msg.data, i + r_offset)[0]
                                    g_val = struct.unpack_from('>f', msg.data, i + g_offset)[0]
                                    b_val = struct.unpack_from('>f', msg.data, i + b_offset)[0]
                                else:
                                    r_val = struct.unpack_from('<f', msg.data, i + r_offset)[0]
                                    g_val = struct.unpack_from('<f', msg.data, i + g_offset)[0]
                                    b_val = struct.unpack_from('<f', msg.data, i + b_offset)[0]
                                # Convert float (0.0-1.0) to uint8 (0-255)
                                r = int(r_val * 255) if r_val <= 1.0 else int(r_val)
                                g = int(g_val * 255) if g_val <= 1.0 else int(g_val)
                                b = int(b_val * 255) if b_val <= 1.0 else int(b_val)
                            else:  # Assume uint8
                                r = struct.unpack_from('B', msg.data, i + r_offset)[0]
                                g = struct.unpack_from('B', msg.data, i + g_offset)[0]
                                b = struct.unpack_from('B', msg.data, i + b_offset)[0]
                            colors.extend([r, g, b])
                        except Exception as e:
                            logging.warning(f"Failed to extract separate RGB: {e}")
                            pass
            
            # Return with colors if available, otherwise just positions
            if colors:
                return jsonify({
                    'type': 'pointCloud',
                    'pointCloud': {'positions': pointCloud, 'colors': colors},
                    'timestamp': timestamp
                })
            else:
                return jsonify({
                    'type': 'pointCloud',
                    'pointCloud': {'positions': pointCloud, 'colors': []},
                    'timestamp': timestamp
                })
        case 'sensor_msgs/msg/NavSatFix':
            return jsonify({'type': 'position', 'position': {'latitude': msg.latitude, 'longitude': msg.longitude, 'altitude': msg.altitude}, 'timestamp': timestamp})
        case 'novatel_oem7_msgs/msg/BESTPOS':
            return jsonify({'type': 'position', 'position': {'latitude': msg.lat, 'longitude': msg.lon, 'altitude': msg.hgt}, 'timestamp': timestamp})
        case 'tf2_msgs/msg/TFMessage':
            # Assume single transform for simplicity
            if len(msg.transforms) > 0:
                transform = msg.transforms[0]
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                tf_data = {
                    'translation': {
                        'x': translation.x,
                        'y': translation.y,
                        'z': translation.z
                    },
                    'rotation': {
                        'x': rotation.x,
                        'y': rotation.y,
                        'z': rotation.z,
                        'w': rotation.w
                    }
                }
                return jsonify({'type': 'tf', 'tf': tf_data, 'timestamp': timestamp})
        case 'sensor_msgs/msg/Imu':
            imu_data = {
                "orientation": {
                    "x": msg.orientation.x,
                    "y": msg.orientation.y,
                    "z": msg.orientation.z,
                    "w": msg.orientation.w
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x,
                    "y": msg.angular_velocity.y,
                    "z": msg.angular_velocity.z
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x,
                    "y": msg.linear_acceleration.y,
                    "z": msg.linear_acceleration.z
                }
            }
            return jsonify({'type': 'imu', 'imu': imu_data, 'timestamp': timestamp})
        case 'nav_msgs/msg/Odometry':
            # Extract odometry data: pose (position + orientation) and twist (linear + angular velocity)
            odometry_data = {
                "header": {
                    "frame_id": msg.header.frame_id,
                    "stamp": {
                        "sec": msg.header.stamp.sec,
                        "nanosec": msg.header.stamp.nanosec
                    }
                },
                "child_frame_id": msg.child_frame_id,
                "pose": {
                    "position": {
                        "x": msg.pose.pose.position.x,
                        "y": msg.pose.pose.position.y,
                        "z": msg.pose.pose.position.z
                    },
                    "orientation": {
                        "x": msg.pose.pose.orientation.x,
                        "y": msg.pose.pose.orientation.y,
                        "z": msg.pose.pose.orientation.z,
                        "w": msg.pose.pose.orientation.w
                    },
                    "covariance": list(msg.pose.covariance) if hasattr(msg.pose, 'covariance') else []
                },
                "twist": {
                    "linear": {
                        "x": msg.twist.twist.linear.x,
                        "y": msg.twist.twist.linear.y,
                        "z": msg.twist.twist.linear.z
                    },
                    "angular": {
                        "x": msg.twist.twist.angular.x,
                        "y": msg.twist.twist.angular.y,
                        "z": msg.twist.twist.angular.z
                    },
                    "covariance": list(msg.twist.covariance) if hasattr(msg.twist, 'covariance') else []
                }
            }
            return jsonify({'type': 'odometry', 'odometry': odometry_data, 'timestamp': timestamp})
        case _:
            # Fallback for unsupported or unknown message types: return string representation
            return jsonify({'type': 'text', 'text': str(msg), 'timestamp': timestamp})
