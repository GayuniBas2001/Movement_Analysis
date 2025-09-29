import cv2                    # OpenCV - Computer vision library
import numpy as np           # NumPy - Numerical computations
import mediapipe as mp       # Google's ML framework for pose detection
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import pyrealsense2 as rs    # Intel RealSense SDK for depth cameras

from rgbd_set_up import (
    configure_depth_sensor, 
    calculate_depth_quality, 
    get_center_distance,
    draw_landmarks_on_image,
    user_interface_for_preset_selection
)
from angle_utils import calculate_angle
from exercise_selector import select_exercise
from rep_counter import BicepCurlCounter, ArmRaiseCounter
from landmark_utils import get_3d_landmark, get_multiple_3d_landmarks, get_true_3d_landmark, get_camera_intrinsics

#############################################################
# DEPTH CONFIGURATION
#############################################################
DEPTH_PRESETS = {
    "default": {"name": "🏠 Default Indoor (1-3m)", "laser_power": 180, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5},
    "long_range": {"name": "🏢 Long Range Gym (3m+)", "laser_power": 360, "receiver_gain": 20, "auto_exposure_priority": 0, "holes_fill": 5},
    "bright_light": {"name": "☀️ Bright Light/Outdoors", "laser_power": 360, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5},
    "close_up": {"name": "🔍 Close Range (<1m)", "laser_power": 100, "receiver_gain": 16, "auto_exposure_priority": 0, "holes_fill": 5}
}

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

selected_exercise_name, selected_exercise_func = select_exercise()

# Select appropriate rep counter based on exercise
if selected_exercise_name == "Bicep Curl":
    rep_counter = BicepCurlCounter()
elif selected_exercise_name == "Arm Raise":
    rep_counter = ArmRaiseCounter()
else:
    # Default to BicepCurlCounter for other exercises (can be expanded later)
    rep_counter = BicepCurlCounter()
    print(f"⚠️ Using default BicepCurlCounter for {selected_exercise_name} - may not track reps correctly")

#############################################################
# REALSENSE PIPELINE
#############################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()

# Get camera intrinsics for true 3D coordinate conversion
camera_intrinsics = get_camera_intrinsics(profile)
print(f"Camera intrinsics loaded: {camera_intrinsics is not None}")

selected = user_interface_for_preset_selection(DEPTH_PRESETS, depth_sensor)

# Setup frame alignment - align depth to color stream
align_to = rs.stream.color  # This is the stream type
align = rs.align(align_to)  # This creates the align processor

# Calibration target distance
TARGET_DISTANCE = 3.0  # meters

print("\nStarting video stream...")
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
         
        # Get metrics
        center_distance = get_center_distance(depth_frame)
        data_quality = calculate_depth_quality(depth_frame)
        
        # Draw crosshair
        height, width = color_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.drawMarker(color_image, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(depth_colormap, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

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

            # Extract TRUE 3D landmarks using camera intrinsics (all coordinates in meters)
            # Include all landmarks needed by exercise functions: left/right shoulder, elbow, wrist, hip, knee, ankle
            landmark_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            if camera_intrinsics:
                landmarks_3d = get_multiple_3d_landmarks(landmarks, landmark_indices, color_w, color_h, depth_w, depth_h, depth_frame, camera_intrinsics)
                coordinate_system = "TRUE 3D (meters)"
            else:
                landmarks_3d = get_multiple_3d_landmarks(landmarks, landmark_indices, color_w, color_h, depth_w, depth_h, depth_frame)
                coordinate_system = "HYBRID (pixels+depth)"

            feedback, form_score, is_good_form, calculated_angles = selected_exercise_func(landmarks_3d)

            print(" | ".join(f"{key}: {value:.2f}" for key, value in calculated_angles.items()))

            # Count reps based on elbow angle, wrist movement, and form
            rep_counted, current_phase, total_reps, transition_state, is_form_good_global = rep_counter.evaluate_rep(
                landmarks_3d, calculated_angles, is_good_form 
            )
            
            # Show rep count notification if a rep was just counted
            if rep_counted:
                print(f"🎉 REP COMPLETED! Total reps: {total_reps}")

            angle_info = [f"{name}: {value:.1f}°" for name, value in calculated_angles.items()]
        else:
            angle_info = ["No pose detected"]
            feedback = "No pose detected"
            form_score = 0.0
            is_good_form = False
            total_reps = rep_counter.rep_count
            transition_state = rep_counter.transition_state
            is_form_good_global = rep_counter.is_form_good_global

        # Draw landmarks on both color and depth frames
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Also draw landmarks on depth colormap
        # Convert depth colormap to RGB for landmark drawing, then back to BGR
        depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        annotated_depth = draw_landmarks_on_image(depth_rgb, detection_result)
        depth_with_landmarks = cv2.cvtColor(annotated_depth, cv2.COLOR_RGB2BGR)

        # Add depth information display text to both frames
        coord_type = "TRUE 3D" if camera_intrinsics else "HYBRID"
        info_text = [
            f"Preset: {DEPTH_PRESETS[current_preset]['name']}",
            f"Center Distance: {center_distance:.3f}m",
            f"Data Quality: {data_quality:.1%}",
            f"Coordinates: {coord_type}"
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
        
        # Create a feedback panel (same width as combined video, height 300px)
        panel_width = video_combined.shape[1]  # Same width as video
        panel_height = 300  # Increased height for better readability
        feedback_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255  # white background

        # Add feedback content if pose detected
        if detection_result.pose_landmarks:
            # Exercise name at the top
            cv2.putText(feedback_panel, f"Exercise: {selected_exercise_name}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 200), 3)
            
            # Left side: Rep counter and form score
            cv2.putText(feedback_panel, f"REPS: {total_reps}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 100, 0), 4)
            cv2.putText(feedback_panel, f"Phase: {rep_counter.get_phase_description()}", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(feedback_panel, f"Form Score: {form_score:.1%}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0) if is_good_form else (0, 0, 150), 2)
            
            # Add movement form status
            form_status = rep_counter.get_form_status()
            form_color = (0, 150, 0) if form_status == "CORRECT" else (0, 0, 200)
            cv2.putText(feedback_panel, f"Movement: {form_status}", (10, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
            
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
        window_title = f"3D Pose Analysis: {selected_exercise_name}"
        cv2.imshow(window_title, final_combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_counter.reset_counter()
            print("Rep counter reset!")
        
        # Check if window was closed
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
