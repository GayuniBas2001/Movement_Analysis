from angle_calculator import calculate_angle
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def polygon_area(points):
    """Shoelace formula in 2D (x,y) for torso area calculation."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(x[i]*y[(i+1)%len(points)] - x[(i+1)%len(points)]*y[i] for i in range(len(points))))

def evaluate_bicep_curl(landmarks: dict) -> tuple:
    """
    Evaluates if the bicep curl exercise is being done correctly.

    Parameters:
        landmarks (dict): Dictionary of 3D landmarks with keys:
                       'elbow_angle', 'shoulder_angle', 'hip_angle'.
                       Example: {"elbow_angle": 45, "shoulder_angle": 10, "hip_angle": 175}

    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_angles)
               - feedback_string: Feedback message to display
               - form_score: Float between 0-1 indicating form quality
               - is_good_form: Boolean indicating if form is acceptable for rep counting
               - calculated_angles: Dictionary of calculated angles
    """
    feedback = []
    form_score = 0.0
    good_form_count = 0
    total_checks = 3  # Number of form checks we perform

    # Ideal ranges (adjust as needed)
    ideal_elbow_min = 30    # fully curled
    ideal_elbow_max = 160   # extended
    ideal_shoulder_max = 20 # shoulder should stay stable
    ideal_hip_min = 150     # slight lean forward allowed
    ideal_hip_max = 170     # fully upright

    # Extract angles from landmarks dictionary
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]
    right_wrist = landmarks[16]
    right_hip = landmarks[24]
    right_knee = landmarks[26]

    # Calculate angles
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist, use_2d=False)  # True 3D
    hip_angle = calculate_angle(right_shoulder, right_hip, right_knee, use_2d=False)      # True 3D
    shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow, use_2d=False) # True 3D

    calculated_angles = {
        "elbow_angle": elbow_angle,
        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle
    }

    if elbow_angle is None or shoulder_angle is None or hip_angle is None:
        return "Missing angle data for evaluation.", 0.0, False, calculated_angles

    # --- Elbow motion ---
    if ideal_elbow_min <= elbow_angle <= ideal_elbow_max:
        feedback.append("GOOD elbow range of motion.")
        good_form_count += 1
    else:
        if elbow_angle < ideal_elbow_min:
            feedback.append(f"⚠ Over-curling — relax the elbow by ~{ideal_elbow_min - elbow_angle}°.")
        elif elbow_angle > ideal_elbow_max:
            feedback.append(f"⚠ Not curling enough — bend the elbow ~{elbow_angle - ideal_elbow_max}° more.")

    # --- Shoulder stability ---
    if abs(shoulder_angle) <= ideal_shoulder_max:
        feedback.append("GOOD Shoulders are stable.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Too much shoulder movement — reduce by ~{abs(shoulder_angle - ideal_shoulder_max)}°.")

    # --- Hip posture ---
    if ideal_hip_min <= hip_angle <= ideal_hip_max:
        feedback.append("GOOD upright posture.")
        good_form_count += 1
    else:
        if hip_angle < ideal_hip_min:
            feedback.append(f"⚠ Leaning too far forward — straighten up by ~{ideal_hip_min - hip_angle}°.")
        else:
            feedback.append(f"⚠ Leaning backward — adjust forward by ~{hip_angle - ideal_hip_max}°.")

    # Calculate form score
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.65  # At least 2/3 criteria must be good

    # --- Final verdict ---
    if form_score == 1.0:
        return "🎯 Perfect form! Strong curls", form_score, is_good_form, calculated_angles

    return "\n".join(feedback), form_score, is_good_form, calculated_angles

