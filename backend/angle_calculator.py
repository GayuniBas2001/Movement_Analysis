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
        
        Args:
            landmarks: MediaPipe pose landmarks
            required_landmark_indices: List of landmark indices needed
            color_w, color_h: Color frame dimensions
            depth_w, depth_h: Depth frame dimensions  
            depth_frame: RealSense depth frame
            
        Returns:
            dict: Dictionary mapping landmark index to 3D coordinates
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
        
        Args:
            exercise_name: Name of the exercise
            landmarks_3d: Dictionary of extracted 3D landmarks
            required_angles: List of angle names needed for the exercise
            
        Returns:
            dict: Dictionary of calculated angles
        """
        calculated_angles = {}
        
        if exercise_name == 'Bicep Curl':
            calculated_angles = self._calculate_bicep_curl_angles(landmarks_3d, required_angles)
        elif exercise_name == 'Arm Raise':
            calculated_angles = self._calculate_arm_raise_angles(landmarks_3d, required_angles)
        elif exercise_name == 'Trunk Rotation':
            calculated_angles = self._calculate_trunk_rotation_angles(landmarks_3d, required_angles)
        elif exercise_name == 'Sit to Stand':
            calculated_angles = self._calculate_sit_to_stand_angles(landmarks_3d, required_angles)
        elif exercise_name == 'Squat':
            calculated_angles = self._calculate_squat_angles(landmarks_3d, required_angles)
        elif exercise_name == 'Heel Rise':
            calculated_angles = self._calculate_heel_rise_angles(landmarks_3d, required_angles)
        
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
    
    def _calculate_arm_raise_angles(self, landmarks_3d, required_angles):
        """Calculate angles for arm raise exercise with bilateral analysis."""
        angles = {}
        
        # Extract landmarks
        left_shoulder = landmarks_3d.get(11)
        right_shoulder = landmarks_3d.get(12)
        left_elbow = landmarks_3d.get(13)
        right_elbow = landmarks_3d.get(14)
        left_wrist = landmarks_3d.get(15)
        right_wrist = landmarks_3d.get(16)
        left_hip = landmarks_3d.get(23)
        right_hip = landmarks_3d.get(24)
        
        # Shoulder angles: angle(hip – shoulder – wrist)
        if 'right_shoulder_angle' in required_angles and all([right_hip, right_shoulder, right_wrist]):
            angles['right_shoulder_angle'] = calculate_angle(right_hip, right_shoulder, right_wrist, use_2d=False)
        
        if 'left_shoulder_angle' in required_angles and all([left_hip, left_shoulder, left_wrist]):
            angles['left_shoulder_angle'] = calculate_angle(left_hip, left_shoulder, left_wrist, use_2d=False)
        
        # Elbow extension: angle(shoulder – elbow – wrist)
        if 'right_elbow_angle' in required_angles and all([right_shoulder, right_elbow, right_wrist]):
            angles['right_elbow_angle'] = calculate_angle(right_shoulder, right_elbow, right_wrist, use_2d=False)
        
        if 'left_elbow_angle' in required_angles and all([left_shoulder, left_elbow, left_wrist]):
            angles['left_elbow_angle'] = calculate_angle(left_shoulder, left_elbow, left_wrist, use_2d=False)
        
        # Trunk tilt: angle between vertical axis and hip-shoulder vector
        if 'trunk_tilt' in required_angles and right_hip and right_shoulder:
            hip_shoulder_vector = np.array(right_shoulder) - np.array(right_hip)
            vertical_vector = np.array([0, 1, 0])  # Vertical axis (Y-up)
            
            # Calculate angle between vectors
            cos_angle = np.dot(hip_shoulder_vector, vertical_vector) / (
                np.linalg.norm(hip_shoulder_vector) * np.linalg.norm(vertical_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            trunk_tilt_rad = np.arccos(cos_angle)
            angles['trunk_tilt'] = np.degrees(trunk_tilt_rad) - 90  # Adjust for upright position
        
        # Trunk rotation: angle between shoulder line and hip line
        if 'trunk_rotation' in required_angles and all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
            hip_vector = np.array(right_hip) - np.array(left_hip)
            
            # Calculate angle between shoulder line and hip line
            cos_angle = np.dot(shoulder_vector, hip_vector) / (
                np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            rotation_rad = np.arccos(cos_angle)
            angles['trunk_rotation'] = np.degrees(rotation_rad)
        
        return angles
    
    def _calculate_trunk_rotation_angles(self, landmarks_3d, required_angles):
        """Calculate angles for trunk rotation exercise."""
        angles = {}
        
        # Similar to arm raise trunk rotation calculation
        left_shoulder = landmarks_3d.get(11)
        right_shoulder = landmarks_3d.get(12)
        left_hip = landmarks_3d.get(23)
        right_hip = landmarks_3d.get(24)
        
        if 'shoulder_angle' in required_angles and all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
            hip_vector = np.array(right_hip) - np.array(left_hip)
            
            cos_angle = np.dot(shoulder_vector, hip_vector) / (
                np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            rotation_rad = np.arccos(cos_angle)
            angles['shoulder_angle'] = np.degrees(rotation_rad)
        
        return angles
    
    def _calculate_sit_to_stand_angles(self, landmarks_3d, required_angles):
        """Calculate angles for sit-to-stand exercise."""
        angles = {}
        
        right_hip = landmarks_3d.get(24)
        right_knee = landmarks_3d.get(26)
        right_ankle = landmarks_3d.get(28)
        
        if 'hip_angle' in required_angles and right_hip and right_knee:
            # Need additional landmark for hip angle calculation
            right_shoulder = landmarks_3d.get(12)
            if right_shoulder:
                angles['hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)
        
        if 'knee_angle' in required_angles and all([right_hip, right_knee, right_ankle]):
            angles['knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle, use_2d=False)
        
        return angles
    
    def _calculate_squat_angles(self, landmarks_3d, required_angles):
        """Calculate angles for squat exercise."""
        angles = {}
        
        right_shoulder = landmarks_3d.get(12)
        right_hip = landmarks_3d.get(24)
        right_knee = landmarks_3d.get(26)
        
        if 'hip_angle' in required_angles and all([right_shoulder, right_hip, right_knee]):
            angles['hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)
        
        if 'knee_angle' in required_angles and right_hip and right_knee:
            right_ankle = landmarks_3d.get(28)
            if right_ankle:
                angles['knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle, use_2d=False)
        
        if 'shoulder_angle' in required_angles and right_shoulder and right_hip:
            # Trunk lean angle
            angles['shoulder_angle'] = calculate_angle([right_hip[0], right_hip[1] + 0.1, right_hip[2]], 
                                                     right_hip, right_shoulder, use_2d=False)
        
        return angles
    
    def _calculate_heel_rise_angles(self, landmarks_3d, required_angles):
        """Calculate angles for heel rise exercise."""
        angles = {}
        
        right_hip = landmarks_3d.get(24)
        right_knee = landmarks_3d.get(26)
        right_ankle = landmarks_3d.get(28)
        right_heel = landmarks_3d.get(30)
        right_foot_index = landmarks_3d.get(32)
        
        if 'ankle_angle' in required_angles and all([right_knee, right_ankle, right_foot_index]):
            angles['ankle_angle'] = calculate_angle(right_knee, right_ankle, right_foot_index, use_2d=False)
        
        if 'knee_angle' in required_angles and all([right_hip, right_knee, right_ankle]):
            angles['knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle, use_2d=False)
        
        if 'hip_angle' in required_angles and right_hip and right_knee:
            right_shoulder = landmarks_3d.get(12)
            if right_shoulder:
                angles['hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)
        
        return angles
    
    def get_coordinate_system(self):
        """Return the coordinate system being used."""
        return self.coordinate_system