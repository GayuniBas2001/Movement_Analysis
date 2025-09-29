import numpy as np
import pyrealsense2 as rs    # Intel RealSense SDK for depth cameras
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def configure_depth_sensor(DEPTH_PRESETS, depth_sensor, preset_name):
    """Apply preset configuration to depth sensor"""
    preset = DEPTH_PRESETS[preset_name]
    settings_applied = []
    settings_failed = []
    
    # Try each setting individually
    settings = [
        ("laser_power", rs.option.laser_power, preset["laser_power"]),
        ("receiver_gain", rs.option.receiver_gain, preset["receiver_gain"]),
        ("auto_exposure_priority", rs.option.auto_exposure_priority, preset["auto_exposure_priority"]),
        ("holes_fill", rs.option.holes_fill, preset["holes_fill"])
    ]
    
    for setting_name, option, value in settings:
        try:
            if depth_sensor.supports(option):
                depth_sensor.set_option(option, value)
                settings_applied.append(setting_name)
            else:
                settings_failed.append(f"{setting_name} (not supported)")
        except Exception as e:
            settings_failed.append(f"{setting_name} ({str(e)})")
    
    print(f"Applied preset: {preset['name']}")
    if settings_applied:
        print(f"✅ Applied: {', '.join(settings_applied)}")
    if settings_failed:
        print(f"⚠️ Skipped: {', '.join(settings_failed)}")
    
    return len(settings_applied) > 0
    
def calculate_depth_quality(depth_frame):
    """Calculate the percentage of valid depth pixels"""
    depth_image = np.asanyarray(depth_frame.get_data())
    valid_pixels = np.sum(depth_image > 0)
    total_pixels = depth_image.size
    validity_ratio = valid_pixels / total_pixels
    return validity_ratio

def get_center_distance(depth_frame):
    """Get distance in meters at the center of the frame"""
    height, width = depth_frame.get_height(), depth_frame.get_width()
    center_x, center_y = width // 2, height // 2
    return depth_frame.get_distance(center_x, center_y)

def draw_landmarks_on_image(image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(image)
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

def user_interface_for_preset_selection(DEPTH_PRESETS, depth_sensor):
    """Simple command-line interface for selecting depth presets"""
    print("🎯 RealSense Depth Camera Configuration")
    print("=" * 50)
    for i, (key, preset) in enumerate(DEPTH_PRESETS.items()):
        print(f"{i+1}. {preset['name']}")

    selected = None
    while selected not in DEPTH_PRESETS.keys():
        try:
            choice = int(input("\nSelect a configuration preset (1-4): ")) - 1
            selected = list(DEPTH_PRESETS.keys())[choice]
        except (ValueError, IndexError):
            print("Invalid selection. Please choose 1-4.")

    configure_depth_sensor(DEPTH_PRESETS, depth_sensor, selected)
    return selected