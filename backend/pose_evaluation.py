from angle_calculator import calculate_angle
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def polygon_area(points):
    """Shoelace formula in 2D (x,y) for torso area calculation."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(x[i]*y[(i+1)%len(points)] - x[(i+1)%len(points)]*y[i] for i in range(len(points))))

# def evaluate_bicep_curl(angles: dict) -> tuple:
def evaluate_bicep_curl(landmarks: dict) -> tuple:
    """
    Evaluates if the bicep curl exercise is being done correctly.

    Parameters:
        landmarks (dict): Dictionary of 3D landmarks with keys:
                       'elbow_angle', 'shoulder_angle', 'hip_angle'.
                       Example: {"elbow_angle": 45, "shoulder_angle": 10, "hip_angle": 175}

    Returns:
        tuple: (feedback_string, form_score, is_good_form)
               - feedback_string: Feedback message to display
               - form_score: Float between 0-1 indicating form quality
               - is_good_form: Boolean indicating if form is acceptable for rep counting
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

def evaluate_arm_raise(landmarks: dict, torso_area_start: float = None) -> tuple:
    """
    Evaluates arm raise exercise according to KIMORE criteria (clinically validated).

    The subject holds a bar with both hands, arms extended along the body, slightly apart, 
    and raises the arms overhead, keeping elbows extended and avoiding pelvic tilt.

    Parameters:
        landmarks (dict): Dictionary of 3D landmarks (keypoints)
        torso_area_start (float): Optional initial torso area to compute torso stability

    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_metrics)
    """
    # Key landmarks
    rs, re, rw = landmarks[12], landmarks[14], landmarks[16]  # Right shoulder, elbow, wrist
    ls, le, lw = landmarks[11], landmarks[13], landmarks[15]  # Left shoulder, elbow, wrist
    rh, rk, ra = landmarks[24], landmarks[26], landmarks[28]  # Right hip, knee, ankle
    lh, lk, la = landmarks[23], landmarks[25], landmarks[27]  # Left hip, knee, ankle

    # Angles (use trunk→shoulder→elbow for shoulder flexion)
    shoulder_angle_r = calculate_angle(rh, rs, re, use_2d=False)
    shoulder_angle_l = calculate_angle(lh, ls, le, use_2d=False)
    elbow_angle_r = calculate_angle(rs, re, rw, use_2d=False)
    elbow_angle_l = calculate_angle(ls, le, lw, use_2d=False)
    hip_angle_r = calculate_angle(rs, rh, rk, use_2d=False)
    hip_angle_l = calculate_angle(ls, lh, lk, use_2d=False)
    knee_angle_r = calculate_angle(rh, rk, ra, use_2d=False)
    knee_angle_l = calculate_angle(lh, lk, la, use_2d=False)

    # Torso area (shoulders + hips)
    torso_pts = [rs, ls, lh, rh]
    torso_area_end = polygon_area(torso_pts)

    # Distances
    hand_distance = euclidean_distance(rw, lw)
    ankle_distance = euclidean_distance(ra, la)

    # Metrics dictionary
    calculated_metrics = {
        "shoulder_angle_r": shoulder_angle_r,
        "shoulder_angle_l": shoulder_angle_l,
        "elbow_angle_r": elbow_angle_r,
        "elbow_angle_l": elbow_angle_l,
        "hip_angle_r": hip_angle_r,
        "hip_angle_l": hip_angle_l,
        "knee_angle_r": knee_angle_r,
        "knee_angle_l": knee_angle_l,
        "torso_area_end": torso_area_end,
        "hand_distance": hand_distance,
        "ankle_distance": ankle_distance
    }

    # Feedback
    feedback, good_form_count = [], 0
    total_checks = 7  # shoulder, elbow, hip, knee, torso, hand, ankle

    # Clinically validated thresholds (KIMORE)
    ideal_shoulder_min, ideal_shoulder_max = 160, 180
    ideal_elbow_min, ideal_elbow_max = 175, 180
    ideal_hip_min, ideal_hip_max = 170, 180
    ideal_knee_min, ideal_knee_max = 170, 180
    ideal_hand_distance = 0.4  # meters
    ideal_ankle_distance = 0.25  # meters
    hand_tol = 0.05
    ankle_tol = 0.05
    torso_tol = 0.05  # 5% area change

    # Shoulder flexion
    if ideal_shoulder_min <= shoulder_angle_r <= ideal_shoulder_max and \
       ideal_shoulder_min <= shoulder_angle_l <= ideal_shoulder_max:
        feedback.append("GOOD : Full shoulder elevation achieved.")
        good_form_count += 1
    else:
        feedback.append("⚠ Raise both arms fully overhead (160–180°).")

    # Elbow extension
    if ideal_elbow_min <= elbow_angle_r <= ideal_elbow_max and \
       ideal_elbow_min <= elbow_angle_l <= ideal_elbow_max:
        feedback.append("GOOD : Elbows fully extended.")
        good_form_count += 1
    else:
        feedback.append("⚠ Keep elbows fully straight (175–180°).")

    # Hip posture
    if ideal_hip_min <= hip_angle_r <= ideal_hip_max and \
       ideal_hip_min <= hip_angle_l <= ideal_hip_max:
        feedback.append("GOOD : Upright trunk posture.")
        good_form_count += 1
    else:
        feedback.append("⚠ Avoid trunk bending or pelvic tilt.")

    # Knee posture
    if ideal_knee_min <= knee_angle_r <= ideal_knee_max and \
       ideal_knee_min <= knee_angle_l <= ideal_knee_max:
        feedback.append("GOOD : Knees stable and nearly extended.")
        good_form_count += 1
    else:
        feedback.append("⚠ Keep knees extended (170–180°).")

    # Torso stability
    if torso_area_start is not None:
        torso_change = abs(torso_area_end - torso_area_start) / torso_area_start
        if torso_change <= torso_tol:
            feedback.append("GOOD : Torso remains stable.")
            good_form_count += 1
        else:
            feedback.append(f"⚠ Torso instability detected (area change {torso_change*100:.1f}%).")
    else:
        feedback.append("INFO : Torso stability not evaluated (no reference start area).")

    # Hand distance
    if abs(hand_distance - ideal_hand_distance) <= hand_tol:
        feedback.append("GOOD : Hands placed correctly.")
        good_form_count += 1
    else:
        diff = hand_distance - ideal_hand_distance
        if diff > 0:
            feedback.append(f"⚠ Move hands closer by ~{diff:.2f} m.")
        else:
            feedback.append(f"⚠ Move hands farther by ~{abs(diff):.2f} m.")

    # Ankle distance
    if abs(ankle_distance - ideal_ankle_distance) <= ankle_tol:
        feedback.append("GOOD : Feet placed correctly apart.")
        good_form_count += 1
    else:
        diff = ankle_distance - ideal_ankle_distance
        if diff > 0:
            feedback.append(f"⚠ Move feet closer by ~{diff:.2f} m.")
        else:
            feedback.append(f"⚠ Move feet farther by ~{abs(diff):.2f} m.")

    # Final scoring
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.65

    if form_score == 1.0:
        return "🎯 Perfect arm raise form!", form_score, is_good_form, calculated_metrics

    return "\n".join(feedback), form_score, is_good_form, calculated_metrics

def evaluate_trunk_rotation(landmarks: dict) -> tuple:
    """
    Evaluates trunk rotation exercise.
    
    Parameters:
        landmarks (dict): Dictionary of 3D landmarks
        
    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_angles)
    """
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
    
    feedback = []
    form_score = 0.0
    good_form_count = 0
    total_checks = 2
    
    # Ideal ranges for trunk rotation
    ideal_rotation_min = 30    # minimum rotation angle
    ideal_rotation_max = 60    # maximum safe rotation
    ideal_hip_min = 160        # maintain stable hips
    
    shoulder_angle_abs = abs(shoulder_angle)
    hip_angle_val = hip_angle
    
    # Check rotation range
    if ideal_rotation_min <= shoulder_angle_abs <= ideal_rotation_max:
        feedback.append("✓ Good rotation range.")
        good_form_count += 1
    else:
        if shoulder_angle_abs < ideal_rotation_min:
            feedback.append(f"⚠ Rotate more — increase by ~{ideal_rotation_min - shoulder_angle_abs}°.")
        else:
            feedback.append(f"⚠ Don't over-rotate — reduce by ~{shoulder_angle_abs - ideal_rotation_max}°.")
    
    # Check hip stability
    if hip_angle_val >= ideal_hip_min:
        feedback.append("✓ Good, hips stay stable.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Keep hips stable — straighten by ~{ideal_hip_min - hip_angle_val}°.")
    
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.5
    
    if form_score == 1.0:
        return "🎯 Perfect trunk rotation!", form_score, is_good_form, calculated_angles
    
    return "\n".join(feedback), form_score, is_good_form, calculated_angles


def evaluate_sit_to_stand(landmarks: dict) -> tuple:
    """
    Evaluates sit-to-stand exercise.
    
    Parameters:
        landmarks (dict): Dictionary of 3D landmarks
        
    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_angles)
    """
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
    
    feedback = []
    form_score = 0.0
    good_form_count = 0
    total_checks = 2
    
    # Ideal ranges for sit-to-stand
    ideal_hip_min = 160        # standing position
    ideal_knee_min = 160       # extended knees when standing
    
    # Check hip extension
    if hip_angle >= ideal_hip_min:
        feedback.append("✓ Good hip extension to standing.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Stand up straighter — extend hips by ~{ideal_hip_min - hip_angle}°.")
    
    # Check elbow position (arms should be relaxed)
    if elbow_angle >= ideal_knee_min:
        feedback.append("✓ Good arm position.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Keep arms relaxed — extend by ~{ideal_knee_min - elbow_angle}°.")
    
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.5
    
    if form_score == 1.0:
        return "🎯 Perfect sit-to-stand form!", form_score, is_good_form, calculated_angles
    
    return "\n".join(feedback), form_score, is_good_form, calculated_angles


def evaluate_squat(landmarks: dict) -> tuple:
    """
    Evaluates squat exercise.
    
    Parameters:
        landmarks (dict): Dictionary of 3D landmarks
        
    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_angles)
    """
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
    
    feedback = []
    form_score = 0.0
    good_form_count = 0
    total_checks = 3
    
    # Ideal ranges for squat
    ideal_hip_min = 90         # deep squat position
    ideal_hip_max = 170        # standing position
    ideal_elbow_min = 90       # arms can be relaxed or out
    ideal_elbow_max = 170      # standing position
    ideal_shoulder_max = 15    # keep torso upright
    
    hip_angle_val = hip_angle
    elbow_angle_val = elbow_angle
    shoulder_angle_abs = abs(shoulder_angle)
    
    # Check hip range
    if ideal_hip_min <= hip_angle <= ideal_hip_max:
        feedback.append("✓ Good hip movement range.")
        good_form_count += 1
    else:
        if hip_angle < ideal_hip_min:
            feedback.append(f"⚠ Don't squat too deep — raise by ~{ideal_hip_min - hip_angle}°.")
        else:
            feedback.append(f"⚠ Squat deeper — lower by ~{hip_angle - ideal_hip_max}°.")
    
    # Check elbow position (arms should be controlled)
    if ideal_elbow_min <= elbow_angle_val <= ideal_elbow_max:
        feedback.append("✓ Good arm position.")
        good_form_count += 1
    else:
        feedback.append("⚠ Keep arms controlled during movement.")
    
    # Check torso position
    if shoulder_angle_abs <= ideal_shoulder_max:
        feedback.append("✓ Good upright torso.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Keep torso more upright — reduce lean by ~{shoulder_angle_abs - ideal_shoulder_max}°.")
    
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.65
    
    if form_score == 1.0:
        return "🎯 Perfect squat form!", form_score, is_good_form, calculated_angles
    
    return "\n".join(feedback), form_score, is_good_form, calculated_angles


def evaluate_heel_rise(landmarks: dict) -> tuple:
    """
    Evaluates heel rise (tiptoe standing) exercise.
    
    Parameters:
        landmarks (dict): Dictionary of 3D landmarks
        
    Returns:
        tuple: (feedback_string, form_score, is_good_form, calculated_angles)
    """
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
    
    feedback = []
    form_score = 0.0
    good_form_count = 0
    total_checks = 3
    
    # Ideal ranges for heel rise
    ideal_elbow_min = 160      # keep arms relaxed and straight
    ideal_shoulder_max = 15    # minimal shoulder movement
    ideal_hip_min = 160        # maintain upright posture
    
    elbow_angle_val = elbow_angle
    shoulder_angle_abs = abs(shoulder_angle)
    hip_angle_val = hip_angle
    
    # Check elbow position (arms should be relaxed)
    if elbow_angle_val >= ideal_elbow_min:
        feedback.append("✓ Good arm position.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Keep arms relaxed — extend by ~{ideal_elbow_min - elbow_angle_val}°.")
    
    # Check shoulder stability
    if shoulder_angle_abs <= ideal_shoulder_max:
        feedback.append("✓ Good, shoulders stay stable.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Keep shoulders stable — reduce movement by ~{shoulder_angle_abs - ideal_shoulder_max}°.")
    
    # Check posture
    if hip_angle_val >= ideal_hip_min:
        feedback.append("✓ Good upright posture.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Stand more upright — straighten by ~{ideal_hip_min - hip_angle_val}°.")
    
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.65
    
    if form_score == 1.0:
        return "🎯 Perfect heel rise form!", form_score, is_good_form, calculated_angles
    
    return "\n".join(feedback), form_score, is_good_form, calculated_angles
