import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import pyrealsense2 as rs

from angle_utils import calculate_angle
from pose_evaluation import evaluate_bicep_curl
from rep_counter import BicepCurlCounter

#############################################################
# DEPTH CONFIGURATION
#############################################################
DEPTH_PRESETS = {
    "default": {"name": "🏠 Default Indoor (1-3m)", "laser_power": 180, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5},
    "long_range": {"name": "🏢 Long Range Gym (3m+)", "laser_power": 360, "receiver_gain": 20, "auto_exposure_priority": 0, "holes_fill": 5},
    "bright_light": {"name": "☀️ Bright Light/Outdoors", "laser_power": 360, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5},
    "close_up": {"name": "🔍 Close Range (<1m)", "laser_power": 100, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5}
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

#############################################################
# MEDIAPIPE POSE LANDMARKER
#############################################################
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    print("Downloading model...")
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

rep_counter = BicepCurlCounter()

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

#############################################################
# REALSENSE PIPELINE
#############################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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

configure_depth_sensor(depth_sensor, selected)

# Setup frame alignment - align depth to color stream
align_to = rs.stream.color  # ✅ This is the stream type
align = rs.align(align_to)  # ✅ This creates the align processor

# Calibration target distance
TARGET_DISTANCE = 3.0  # meters
calibration_mode = input("\nEnable calibration mode? (y/n): ").lower().startswith('y')

print("\nStarting video stream...")
print("Press '1-4' to change presets")
print("Press 'q' to quit, 'r' to reset rep counter")

current_gain = DEPTH_PRESETS[selected]["receiver_gain"]
current_preset = selected


#############################################################
# MAIN LOOP
#############################################################
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
        
    #### ADDED FROM realsense_test.py ####    
        # Get metrics
        center_distance = get_center_distance(depth_frame)
        data_quality = calculate_depth_quality(depth_frame)
        
        # Draw crosshair
        height, width = color_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.drawMarker(color_image, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(depth_colormap, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    #### ADDED FROM realsense_test.py ####   

        frame = np.asanyarray(color_frame.get_data())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)

        # Extract landmarks
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]

            # Get frame dimensions (use color frame for MediaPipe, depth frame for distance)
            color_h, color_w, _ = frame.shape
            depth_h = depth_frame.get_height()
            depth_w = depth_frame.get_width()

            def get_3d_landmark(index):
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

            right_shoulder = get_3d_landmark(12)
            right_elbow = get_3d_landmark(14)
            right_wrist = get_3d_landmark(16)
            right_hip = get_3d_landmark(24)
            right_knee = get_3d_landmark(26)

            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist, use_2d=False)  # True 3D
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)      # True 3D
            shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow, use_2d=False) # True 3D

            print(f"3D Angles - Elbow: {elbow_angle:.1f}°, Hip: {hip_angle:.1f}°, Shoulder: {shoulder_angle:.1f}°")

            # Evaluate form and get feedback (using 3D calculated angles)
            feedback, form_score, is_good_form = evaluate_bicep_curl({
                "elbow_angle": elbow_angle,
                "shoulder_angle": shoulder_angle,
                "hip_angle": hip_angle
            })
            
            # Count reps based on elbow angle and form
            rep_counted, current_phase, total_reps = rep_counter.evaluate_rep(
                elbow_angle, is_good_form
            )
            
            # Show rep count notification if a rep was just counted
            if rep_counted:
                print(f"🎉 REP COMPLETED! Total reps: {total_reps}")

            # Add angle information to info text when pose is detected
            angle_info = [
                f"Elbow: {elbow_angle:.1f}°",
                f"Hip: {hip_angle:.1f}°", 
                f"Shoulder: {shoulder_angle:.1f}°"
            ]
        else:
            angle_info = ["No pose detected"]
            feedback = "No pose detected"
            form_score = 0.0
            is_good_form = False
            total_reps = rep_counter.rep_count

        # Draw landmarks on both color and depth frames
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Also draw landmarks on depth colormap
        # Convert depth colormap to RGB for landmark drawing, then back to BGR
        depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        annotated_depth = draw_landmarks_on_image(depth_rgb, detection_result)
        depth_with_landmarks = cv2.cvtColor(annotated_depth, cv2.COLOR_RGB2BGR)

        # Add depth information display text to both frames
        info_text = [
            f"Preset: {DEPTH_PRESETS[current_preset]['name']}",
            f"Center Distance: {center_distance:.3f}m",
            f"Data Quality: {data_quality:.1%}"
        ]
        
        # Add basic info to color frame
        for i, text in enumerate(info_text):
            y_position = 30 + i * 25
            cv2.putText(display_frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Add basic info to depth frame (no angles since they're in feedback panel now)
        for i, text in enumerate(info_text):
            y_position = 30 + i * 25
            cv2.putText(depth_with_landmarks, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(depth_with_landmarks, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Combine both videos side by side
        video_combined = np.hstack((display_frame, depth_with_landmarks))
        
        # Create a feedback panel (same width as combined video, height 200px)
        panel_width = video_combined.shape[1]  # Same width as video
        panel_height = 200  # Fixed height at bottom
        feedback_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255  # white background

        # Add feedback content if pose detected
        if detection_result.pose_landmarks:
            # Left side: Rep counter and form score
            cv2.putText(feedback_panel, f"REPS: {total_reps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 3)
            cv2.putText(feedback_panel, f"Phase: {rep_counter.get_phase_description()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(feedback_panel, f"Form Score: {form_score:.1%}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0) if is_good_form else (0, 0, 150), 2)
            
            # Middle section: 3D Analysis
            middle_x = panel_width // 3
            cv2.putText(feedback_panel, "3D Analysis:", (middle_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)
            
            # Add 3D angle information
            for i, angle_text in enumerate(angle_info):
                y_position = 55 + i * 20
                cv2.putText(feedback_panel, angle_text, (middle_x, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Right section: Form Feedback
            right_x = (panel_width * 2) // 3
            cv2.putText(feedback_panel, "Form Feedback:", (right_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 0), 2)
            
            y_start = 55  # Starting Y position after headers
            line_height = 20  # Height between lines
            
            for i, line in enumerate(feedback.split("\n")):
                if line.strip():  # Only display non-empty lines
                    y_position = y_start + i * line_height
                    if y_position < panel_height - 10:  # Make sure text fits in panel
                        cv2.putText(feedback_panel, line.strip(), (right_x, y_position),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # When no pose detected, show "No person detected" message
            center_x = panel_width // 2 - 100
            cv2.putText(feedback_panel, "No Person Detected", (center_x, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(feedback_panel, "Please step into view for 3D analysis", (center_x - 50, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        # Combine videos and feedback panel vertically: videos on top, feedback at bottom
        final_combined = np.vstack((video_combined, feedback_panel))
        
        # Show result with color, depth, and feedback
        cv2.imshow("3D Pose Analysis: Color + Depth + Feedback", final_combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_counter.reset_counter()
            print("Rep counter reset!")
        
        # Check if window was closed
        if cv2.getWindowProperty("3D Pose Analysis: Color + Depth + Feedback", cv2.WND_PROP_VISIBLE) < 1:
            break
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
