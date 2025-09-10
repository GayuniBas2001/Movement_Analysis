import pyrealsense2 as rs
import numpy as np
import cv2

# Define configuration presets for different scenarios
DEPTH_PRESETS = {
    "default": {
        "name": "🏠 Default Indoor (1-3m)",
        "laser_power": 180,
        "receiver_gain": 16,
        "auto_exposure_priority": 0,
        "holes_fill": 5
    },
    "long_range": {
        "name": "🏢 Long Range Gym (3m+)",
        "laser_power": 360,
        "receiver_gain": 20,
        "auto_exposure_priority": 0,
        "holes_fill": 5
    },
    "bright_light": {
        "name": "☀️ Bright Light/Outdoors",
        "laser_power": 360,
        "receiver_gain": 16,
        "auto_exposure_priority": 0,
        "holes_fill": 5
    },
    "close_up": {
        "name": "🔍 Close Range (<1m)",
        "laser_power": 100,
        "receiver_gain": 16,
        "auto_exposure_priority": 0,
        "holes_fill": 5
    }
}

def configure_depth_sensor(depth_sensor, preset_name):
    """Apply preset configuration to depth sensor"""
    preset = DEPTH_PRESETS[preset_name]
    try:
        depth_sensor.set_option(rs.option.laser_power, preset["laser_power"])
        depth_sensor.set_option(rs.option.receiver_gain, preset["receiver_gain"])
        depth_sensor.set_option(rs.option.auto_exposure_priority, preset["auto_exposure_priority"])
        depth_sensor.set_option(rs.option.holes_fill, preset["holes_fill"])
        print(f"Applied preset: {preset['name']}")
        return True
    except Exception as e:
        print(f"Error applying preset: {e}")
        return False

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

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()

# User interface for preset selection
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

# Apply selected preset
configure_depth_sensor(depth_sensor, selected)

# Setup alignment
align_to = rs.stream.color
align = rs.align(align_to)

# Calibration target distance
TARGET_DISTANCE = 3.0  # meters
calibration_mode = input("\nEnable calibration mode? (y/n): ").lower().startswith('y')

print("\nStarting video stream...")
print("Press '1-4' to change presets")
print("Press '+'/'-' to adjust receiver gain")
print("Press 'c' to toggle calibration mode")
print("Press 'q' to quit")

current_gain = DEPTH_PRESETS[selected]["receiver_gain"]
current_preset = selected

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Process images
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Get metrics
        center_distance = get_center_distance(depth_frame)
        data_quality = calculate_depth_quality(depth_frame)
        
        # Draw crosshair
        height, width = color_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.drawMarker(color_image, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(depth_colormap, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Display information
        info_text = [
            f"Preset: {DEPTH_PRESETS[current_preset]['name']}",
            f"Gain: {current_gain}",
            f"Quality: {data_quality:.1%}",
            f"Center: {center_distance:.3f}m"
        ]
        
        if calibration_mode:
            calibration_error = abs(center_distance - TARGET_DISTANCE)
            info_text.append(f"Target: {TARGET_DISTANCE}m")
            info_text.append(f"Error: {calibration_error:.3f}m")
            info_text.append("Adjust with '+'/'-' keys")
        
        # Draw text with background for readability
        for i, text in enumerate(info_text):
            y_position = 30 + i * 25
            cv2.putText(color_image, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(color_image, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(depth_colormap, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(depth_colormap, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Combine images
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Configuration Tool', images)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"Calibration mode {'enabled' if calibration_mode else 'disabled'}")
        elif key in [ord('+'), ord('=')]:  # + key
            current_gain = min(28, current_gain + 1)
            depth_sensor.set_option(rs.option.receiver_gain, current_gain)
            print(f"Gain increased to: {current_gain}")
        elif key == ord('-'):  # - key
            current_gain = max(0, current_gain - 1)
            depth_sensor.set_option(rs.option.receiver_gain, current_gain)
            print(f"Gain decreased to: {current_gain}")
        elif key in [ord(str(i)) for i in range(1, 5)]:  # Number keys 1-4
            preset_index = int(chr(key)) - 1
            new_preset = list(DEPTH_PRESETS.keys())[preset_index]
            if configure_depth_sensor(depth_sensor, new_preset):
                current_preset = new_preset
                current_gain = DEPTH_PRESETS[new_preset]["receiver_gain"]

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nConfiguration complete!")
    print(f"Final settings - Preset: {current_preset}, Gain: {current_gain}")