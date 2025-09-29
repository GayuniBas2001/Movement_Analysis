"""
Data Collection Script for Movement Analysis ML Training

This script captures synchronized RGB and depth video data while collecting 
rule-based evaluation labels for training machine learning models.
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import sys
import time

# Add utils path for importing evaluation functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from pose_evaluation import evaluate_bicep_curl
from rep_counter import BicepCurlCounter
from landmark_utils import get_body_landmarks_3d
from angle_calculator import SelectiveAngleCalculator

try:
    import pyrealsense2 as rs
    import mediapipe as mp
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: RealSense or MediaPipe not available. Mock data collection mode.")


class MLDataCollector:
    def __init__(self, exercise_type="Bicep Curl", output_dir="collected_data"):
        """
        Initialize ML data collector for exercise videos with labels.
        
        Args:
            exercise_type: Type of exercise to collect data for
            output_dir: Directory to save collected data
        """
        self.exercise_type = exercise_type
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.rgb_dir = os.path.join(output_dir, "rgb_videos")
        self.depth_dir = os.path.join(output_dir, "depth_videos")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Initialize MediaPipe
        if REALSENSE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # Use heavy model for better accuracy
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
        # Initialize rep counter for bicep curl
        self.rep_counter = BicepCurlCounter()
            
        # Data storage for current session
        self.frame_labels = []
        self.video_metadata = {
            'exercise_type': exercise_type,
            'session_id': self.session_id,
            'total_frames': 0,
            'fps': 30,
            'resolution': {'width': 640, 'height': 480}
        }
        
    def collect_session(self, duration_seconds=60):
        """
        Collect a training session with synchronized video and labels.
        
        Args:
            duration_seconds: Duration of data collection session
        """
        if not REALSENSE_AVAILABLE:
            print("Cannot collect real data without RealSense/MediaPipe")
            self._create_mock_data(duration_seconds)
            return
            
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            pipeline.start(config)
            profile = pipeline.get_active_profile()
            
            # Setup video writers
            rgb_filename = os.path.join(self.rgb_dir, f"{self.session_id}_rgb.avi")
            depth_filename = os.path.join(self.depth_dir, f"{self.session_id}_depth.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            rgb_writer = cv2.VideoWriter(rgb_filename, fourcc, 30.0, (640, 480))
            depth_writer = cv2.VideoWriter(depth_filename, fourcc, 30.0, (640, 480))
            
            print(f"Starting data collection for {duration_seconds} seconds...")
            print(f"Exercise: {self.exercise_type}")
            print("Press 'q' to stop early")
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration_seconds:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                    
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract frame data and generate labels
                    frame_data = self._process_frame(
                        results.pose_landmarks.landmark,
                        color_frame, depth_frame, frame_count
                    )
                    
                    self.frame_labels.append(frame_data)
                
                # Write video frames
                rgb_writer.write(color_image)
                
                # Convert depth to viewable format and write
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                depth_writer.write(depth_colormap)
                
                # Display feedback
                display_frame = color_image.copy()
                if results.pose_landmarks:
                    self._draw_pose_info(display_frame, frame_data)
                
                cv2.imshow('Data Collection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
            
            # Cleanup
            rgb_writer.release()
            depth_writer.release()
            pipeline.stop()
            cv2.destroyAllWindows()
            
            # Save labels and metadata
            self._save_session_data(frame_count)
            
            print(f"\\nData collection complete!")
            print(f"Collected {frame_count} frames")
            print(f"RGB video: {rgb_filename}")
            print(f"Depth video: {depth_filename}")
            
        except Exception as e:
            print(f"Error during data collection: {e}")
            pipeline.stop()
    
    def _process_frame(self, landmarks, color_frame, depth_frame, frame_idx):
        """Process a single frame to generate training labels."""
        # Frame dimensions
        color_w, color_h = 640, 480
        depth_w, depth_h = 640, 480
        
        # Get 3D landmarks (using hybrid coordinates for now)
        landmarks_3d = []
        for i in range(33):  # MediaPipe has 33 pose landmarks
            x_px = int(landmarks[i].x * color_w)
            y_px = int(landmarks[i].y * color_h)
            
            # Get depth value
            try:
                depth_val = depth_frame.get_distance(
                    min(x_px, depth_w-1), min(y_px, depth_h-1)
                )
                if depth_val == 0:
                    depth_val = 1.0
            except:
                depth_val = 1.0
                
            landmarks_3d.append([x_px, y_px, depth_val])
        
        # Generate rule-based evaluation labels for bicep curl
        feedback, form_score, is_good_form, calculated_angles = evaluate_bicep_curl(landmarks_3d)
        
        # Update rep counter
        rep_counted, phase, rep_count, transition_state, form_good = self.rep_counter.evaluate_rep(
            landmarks_3d, calculated_angles, is_good_form
        )
        
        # Create frame label data
        frame_data = {
            'frame_idx': frame_idx,
            'timestamp': time.time(),
            'landmarks_3d': landmarks_3d,
            'rule_based_labels': {
                'feedback': feedback,
                'form_score': form_score,
                'is_good_form': is_good_form,
                'calculated_angles': calculated_angles
            },
            'rep_counter_data': {
                'rep_counted': rep_counted,
                'current_phase': phase,
                'rep_count': rep_count,
                'transition_state': transition_state,
                'form_status': form_good
            }
        }
        
        return frame_data
    
    def _draw_pose_info(self, frame, frame_data):
        """Draw pose information on the frame for visual feedback."""
        labels = frame_data['rule_based_labels']
        rep_data = frame_data['rep_counter_data']
        
        # Display feedback text
        y_offset = 30
        cv2.putText(frame, f"Form: {labels['form_score']:.1f}%", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Reps: {rep_data['rep_count']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Phase: {rep_data['current_phase']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _save_session_data(self, total_frames):
        """Save the collected labels and metadata."""
        # Update metadata
        self.video_metadata['total_frames'] = total_frames
        
        # Save labels
        labels_filename = os.path.join(self.labels_dir, f"{self.session_id}_labels.json")
        with open(labels_filename, 'w') as f:
            json.dump({
                'metadata': self.video_metadata,
                'frame_labels': self.frame_labels
            }, f, indent=2)
        
        print(f"Labels saved: {labels_filename}")
    
    def _create_mock_data(self, duration_seconds):
        """Create mock data for testing when RealSense is not available."""
        print("Creating mock training data...")
        
        # Create mock video files and labels
        mock_rgb = os.path.join(self.rgb_dir, f"{self.session_id}_rgb_mock.txt")
        mock_depth = os.path.join(self.depth_dir, f"{self.session_id}_depth_mock.txt")
        
        with open(mock_rgb, 'w') as f:
            f.write(f"Mock RGB video data for {self.exercise_type}\\n")
            f.write(f"Duration: {duration_seconds} seconds\\n")
            
        with open(mock_depth, 'w') as f:
            f.write(f"Mock depth video data for {self.exercise_type}\\n")
            f.write(f"Duration: {duration_seconds} seconds\\n")
        
        # Create mock labels
        mock_frames = int(30 * duration_seconds)  # 30 FPS
        mock_labels = []
        
        for i in range(mock_frames):
            mock_labels.append({
                'frame_idx': i,
                'timestamp': time.time() + i/30.0,
                'rule_based_labels': {
                    'feedback': 'Mock feedback',
                    'form_score': 85.0 + np.random.normal(0, 5),
                    'is_good_form': True,
                    'calculated_angles': {'elbow_angle': 90 + np.random.normal(0, 10)}
                }
            })
        
        labels_filename = os.path.join(self.labels_dir, f"{self.session_id}_labels_mock.json")
        with open(labels_filename, 'w') as f:
            json.dump({
                'metadata': self.video_metadata,
                'frame_labels': mock_labels
            }, f, indent=2)
        
        print(f"Mock data created: {labels_filename}")


if __name__ == "__main__":
    # Example usage
    collector = MLDataCollector(
        exercise_type="Bicep Curl",
        output_dir="../collected_data"
    )
    
    # Collect 60 seconds of training data
    collector.collect_session(duration_seconds=60)