"""
Selective Angle Calculator Module for Movement Analysis System

This module calculates only the angles needed for specific exercises to optimize performance.
"""

import numpy as np
from angle_utils import calculate_angle
from landmark_utils import get_true_3d_landmark, get_3d_landmark


class SelectiveAngleCalculator:
    def __init__(self, camera_intrinsics=None):
        """
        Initialize the selective angle calculator.
        
        Args:
            camera_intrinsics: Camera intrinsics for true 3D coordinate conversion
        """
        self.camera_intrinsics = camera_intrinsics
        self.coordinate_system = "TRUE 3D (meters)" if camera_intrinsics else "HYBRID (pixels+depth)"
    
    def extract_landmarks(self, landmarks, required_landmark_indices, 
                         color_w, color_h, depth_w, depth_h, depth_frame):
        """
        Extract only the required 3D landmarks for the exercise.
        """
        extracted_landmarks = {}
        
        for idx in required_landmark_indices:
            if self.camera_intrinsics:
                landmark_3d = get_true_3d_landmark(
                    landmarks, idx, color_w, color_h, depth_w, depth_h, 
                    depth_frame, self.camera_intrinsics
                )
            else:
                landmark_3d = get_3d_landmark(
                    landmarks, idx, color_w, color_h, depth_w, depth_h, depth_frame
                )
            
            extracted_landmarks[idx] = landmark_3d
        
        return extracted_landmarks
    
    def calculate_required_angles(self, exercise_name, landmarks_3d, required_angles):
        """
        Calculate only the angles required for the specific exercise.
        """
        calculated_angles = {}
        
        if exercise_name == 'Bicep Curl':
            calculated_angles = self._calculate_bicep_curl_angles(landmarks_3d, required_angles)
        # Additional exercises can be added here in the future
        
        return calculated_angles
    
    def _calculate_bicep_curl_angles(self, landmarks_3d, required_angles):
        """Calculate angles for bicep curl exercise."""
        angles = {}
        
        # Extract landmarks
        right_shoulder = landmarks_3d.get(12)
        right_elbow = landmarks_3d.get(14)
        right_wrist = landmarks_3d.get(16)
        right_hip = landmarks_3d.get(24)
        
        if 'elbow_angle' in required_angles and all([right_shoulder, right_elbow, right_wrist]):
            angles['elbow_angle'] = calculate_angle(right_shoulder, right_elbow, right_wrist, use_2d=False)
        
        if 'shoulder_angle' in required_angles and all([right_hip, right_shoulder, right_elbow]):
            angles['shoulder_angle'] = calculate_angle(right_hip, right_shoulder, right_elbow, use_2d=False)
        
        if 'hip_angle' in required_angles and right_shoulder and right_hip:
            # For hip angle, we might need knee landmark (26)
            right_knee = landmarks_3d.get(26)
            if right_knee:
                angles['hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)
        
        return angles
    
