"""
3D Landmark Utilities for Movement Analysis

This module provides utilities for converting MediaPipe 2D pose landmarks 
into 3D coordinates using depth information from Intel RealSense cameras.
"""

import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("Warning: pyrealsense2 not available. True 3D coordinates will not work.")


def get_3d_landmark(landmarks, index, color_w, color_h, depth_w, depth_h, depth_frame):
    """
    Convert a MediaPipe landmark to 3D coordinates using depth information.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        index: Index of the landmark to convert (0-32 for pose landmarks)
        color_w, color_h: Color frame dimensions (width, height)
        depth_w, depth_h: Depth frame dimensions (width, height)
        depth_frame: RealSense depth frame object
        
    Returns:
        list: [x_pixel, y_pixel, depth_meters] - Hybrid 3D coordinates
    """
    # Convert normalized coordinates to pixel coordinates
    # Use color frame dimensions for MediaPipe landmark conversion
    cx_color = int(landmarks[index].x * color_w)
    cy_color = int(landmarks[index].y * color_h)
    
    # Convert to depth frame coordinates (in case they're different)
    cx_depth = int(landmarks[index].x * depth_w)
    cy_depth = int(landmarks[index].y * depth_h)
    
    # Clamp coordinates to valid depth frame bounds
    cx_depth = max(0, min(cx_depth, depth_w - 1))
    cy_depth = max(0, min(cy_depth, depth_h - 1))
    
    # Get depth value using depth frame coordinates
    try:
        depth_val = depth_frame.get_distance(cx_depth, cy_depth)
        if depth_val == 0:  # No depth data at this point
            depth_val = 1.0  # Default depth in meters
    except:
        depth_val = 1.0  # Fallback depth
    
    # Return using color frame coordinates for display consistency
    return [cx_color, cy_color, depth_val]


def get_true_3d_landmark(landmarks, index, color_w, color_h, depth_w, depth_h, depth_frame, camera_intrinsics):
    """
    Convert a MediaPipe landmark to TRUE 3D world coordinates using camera intrinsics.
    """
    # Convert normalized coordinates to pixel coordinates
    cx_color = int(landmarks[index].x * color_w)
    cy_color = int(landmarks[index].y * color_h)
    
    # Convert to depth frame coordinates
    cx_depth = int(landmarks[index].x * depth_w)
    cy_depth = int(landmarks[index].y * depth_h)
    
    # Clamp coordinates to valid depth frame bounds
    cx_depth = max(0, min(cx_depth, depth_w - 1))
    cy_depth = max(0, min(cy_depth, depth_h - 1))
    
    # Get depth value
    try:
        depth_val = depth_frame.get_distance(cx_depth, cy_depth)
        if depth_val == 0:
            depth_val = 1.0  # Default depth in meters
    except:
        depth_val = 1.0
    
    # Convert pixel coordinates to 3D world coordinates using camera intrinsics
    if rs is not None:
        point_3d = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [cx_depth, cy_depth], depth_val)
        return list(point_3d)  # [x_meters, y_meters, z_meters]
    else:
        # Fallback if RealSense not available
        return [cx_color, cy_color, depth_val]


def get_body_landmarks_3d(landmarks, color_w, color_h, depth_w, depth_h, depth_frame, use_true_3d=False, camera_intrinsics=None):
    """
    Get common body landmarks for exercise analysis in 3D.
    """
    # Define key body landmarks for exercise analysis
    landmark_indices = {
        'right_shoulder': 12,
        'left_shoulder': 11,
        'right_elbow': 14,
        'left_elbow': 13,
        'right_wrist': 16,
        'left_wrist': 15,
        'right_hip': 24,
        'left_hip': 23,
        'right_knee': 26,
        'left_knee': 25,
        'right_ankle': 28,
        'left_ankle': 27
    }
    
    body_landmarks = {}
    for name, idx in landmark_indices.items():
        if use_true_3d and camera_intrinsics:
            body_landmarks[name] = get_true_3d_landmark(
                landmarks, idx, color_w, color_h, depth_w, depth_h, depth_frame, camera_intrinsics
            )
        else:
            body_landmarks[name] = get_3d_landmark(
                landmarks, idx, color_w, color_h, depth_w, depth_h, depth_frame
            )
    
    return body_landmarks