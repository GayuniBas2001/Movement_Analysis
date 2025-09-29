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
    
    Args:
        landmarks: MediaPipe pose landmarks list
        index: Index of the landmark to convert (0-32 for pose landmarks)
        color_w, color_h: Color frame dimensions (width, height)
        depth_w, depth_h: Depth frame dimensions (width, height)
        depth_frame: RealSense depth frame object
        camera_intrinsics: RealSense camera intrinsics object
        
    Returns:
        list: [x_meters, y_meters, z_meters] - True 3D world coordinates
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
    # This gives true 3D coordinates in meters
    if rs is not None:
        point_3d = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [cx_depth, cy_depth], depth_val)
        return list(point_3d)  # [x_meters, y_meters, z_meters]
    else:
        # Fallback if RealSense not available
        return [cx_color, cy_color, depth_val]


def get_multiple_3d_landmarks(landmarks, indices, color_w, color_h, depth_w, depth_h, depth_frame, camera_intrinsics=None):
    """
    Convert multiple MediaPipe landmarks to 3D coordinates efficiently.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        indices: List of landmark indices to convert
        color_w, color_h: Color frame dimensions (width, height)
        depth_w, depth_h: Depth frame dimensions (width, height)
        depth_frame: RealSense depth frame object
        
    Returns:
        dict: Dictionary mapping index to [x_pixel, y_pixel, depth_meters]
    """
    result = {}
    for idx in indices:
        if camera_intrinsics:
            result[idx] = get_true_3d_landmark(landmarks, idx, color_w, color_h, depth_w, depth_h, depth_frame, camera_intrinsics)
        else:
            result[idx] = get_3d_landmark(landmarks, idx, color_w, color_h, depth_w, depth_h, depth_frame)
    return result


def get_camera_intrinsics(pipeline_profile):
    """
    Get camera intrinsics for true 3D coordinate conversion.
    
    Args:
        pipeline_profile: RealSense pipeline profile
        
    Returns:
        camera intrinsics object for 3D coordinate conversion
    """
    try:
        # Get the depth stream profile
        depth_stream = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        # Get camera intrinsics
        depth_intrinsics = depth_stream.get_intrinsics()
        return depth_intrinsics
    except Exception as e:
        print(f"Error getting camera intrinsics: {e}")
        return None


def get_body_landmarks_3d(landmarks, color_w, color_h, depth_w, depth_h, depth_frame, use_true_3d=False, camera_intrinsics=None):
    """
    Get common body landmarks for exercise analysis in 3D.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        color_w, color_h: Color frame dimensions (width, height)
        depth_w, depth_h: Depth frame dimensions (width, height)
        depth_frame: RealSense depth frame object
        use_true_3d: If True, return true 3D world coordinates; if False, return hybrid coordinates
        camera_intrinsics: Required if use_true_3d=True
        
    Returns:
        dict: Dictionary with body part names mapped to 3D coordinates
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


def validate_landmark_quality(landmarks, depth_frame, color_w, color_h, depth_w, depth_h, 
                            min_depth=0.3, max_depth=5.0):
    """
    Validate the quality of 3D landmarks based on depth consistency.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        depth_frame: RealSense depth frame object
        color_w, color_h: Color frame dimensions
        depth_w, depth_h: Depth frame dimensions
        min_depth, max_depth: Valid depth range in meters
        
    Returns:
        dict: Quality metrics including validity percentage and depth statistics
    """
    valid_landmarks = 0
    total_landmarks = len(landmarks)
    depth_values = []
    
    for i in range(total_landmarks):
        try:
            # Get depth coordinates
            cx_depth = int(landmarks[i].x * depth_w)
            cy_depth = int(landmarks[i].y * depth_h)
            cx_depth = max(0, min(cx_depth, depth_w - 1))
            cy_depth = max(0, min(cy_depth, depth_h - 1))
            
            depth_val = depth_frame.get_distance(cx_depth, cy_depth)
            
            if min_depth <= depth_val <= max_depth:
                valid_landmarks += 1
                depth_values.append(depth_val)
        except:
            continue
    
    validity_percentage = (valid_landmarks / total_landmarks) * 100 if total_landmarks > 0 else 0
    
    quality_metrics = {
        'validity_percentage': validity_percentage,
        'valid_landmarks': valid_landmarks,
        'total_landmarks': total_landmarks,
        'mean_depth': np.mean(depth_values) if depth_values else 0,
        'depth_std': np.std(depth_values) if depth_values else 0,
        'depth_range': (min(depth_values), max(depth_values)) if depth_values else (0, 0)
    }
    
    return quality_metrics