import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from angle_utils import calculate_angle  # Import the angle calculation function
from pose_evaluation import evaluate_bicep_curl  # Import the evaluation function
from rep_counter import BicepCurlCounter  # Import the rep counter

# Download the pose landmark model if not present
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    print("Downloading model...")
    urllib.request.urlretrieve(url, MODEL_PATH)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

# STEP 1: Initialize MediaPipe Pose Landmarker

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# Initialize rep counter
rep_counter = BicepCurlCounter()

# Real-time video processing with OpenCV######################
# STEP 2: Initialize camera
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit, 'r' to reset rep counter")

# STEP 3: Real-time processing loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run pose detection
        detection_result = detector.detect(mp_image)

        # Extract landmarks if pose detected
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]  # first detected person

            # RIGHT ARM (user's right, appears on left side of screen)
            right_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            right_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
            right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z] 
            right_hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
            right_ankle = [landmarks[28].x, landmarks[28].y, landmarks[28].z]
            right_knee = [landmarks[26].x, landmarks[26].y, landmarks[26].z]

            # Calculate angles once
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist, use_2d=True)
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=True)
            shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow, use_2d=True)

            print(f"Right Elbow: {elbow_angle:.1f}°, Right Hip: {hip_angle:.1f}°, Right Shoulder: {shoulder_angle:.1f}°")

        # Draw landmarks on the frame
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        
        # Convert back to BGR for OpenCV display
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Create a feedback panel (same height as video, width 350px for text area)
        panel_width = 350
        feedback_panel = np.ones((display_frame.shape[0], panel_width, 3), dtype=np.uint8) * 255  # white background

        # Add angle text overlay and feedback if pose detected
        if detection_result.pose_landmarks:
            # Use the already calculated angles (no recalculation needed)
            # Add text to the annotated frame (RGB format)
            cv2.putText(display_frame, f"Elbow: {int(elbow_angle)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Hip: {int(hip_angle)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Shoulder: {int(shoulder_angle)}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Evaluate form and get feedback
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

            # Rep counter section
            cv2.putText(feedback_panel, f"REPS: {total_reps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 3)
            cv2.putText(feedback_panel, f"Phase: {rep_counter.get_phase_description()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(feedback_panel, f"Form Score: {form_score:.1%}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0) if is_good_form else (0, 0, 150), 2)
            
            # Add separator line
            cv2.line(feedback_panel, (10, 95), (panel_width-10, 95), (200, 200, 200), 1)

            # Write feedback text on the panel - each line properly spaced
            y_start = 110  # Starting Y position after rep info
            line_height = 25  # Height between lines for better readability
            
            for i, line in enumerate(feedback.split("\n")):
                if line.strip():  # Only display non-empty lines
                    y_position = y_start + i * line_height
                    cv2.putText(feedback_panel, line.strip(), (10, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # When no pose detected, show "No person detected" message
            cv2.putText(feedback_panel, "No Person Detected", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(feedback_panel, "Please step into view", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        # Always concatenate video + panel for consistent interface
        combined_frame = np.hstack((display_frame, feedback_panel))
        
        # Show final output
        cv2.imshow("Real-time Pose Landmarks + Feedback", combined_frame)
        
        # Check for key press with longer wait time
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
        elif key == ord('r'):  # 'r' key to reset rep counter
            rep_counter.reset_counter()
            print("Rep counter reset!")
        
        # Also check if window was closed
        if cv2.getWindowProperty("Real-time Pose Landmarks + Feedback", cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\nApplication interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Cleaning up and closing application...")
#############################################################

# STEP 4: Cleanup
cap.release()
cv2.destroyAllWindows()

# Force close any remaining windows
cv2.waitKey(1)  # Allow time for cleanup
print("Application closed successfully")
