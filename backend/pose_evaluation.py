def evaluate_bicep_curl(angles: dict) -> tuple:
    """
    Evaluates if the bicep curl exercise is being done correctly.

    Parameters:
        angles (dict): Dictionary of angles with keys:
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

    elbow_angle = angles.get("elbow_angle")
    shoulder_angle = angles.get("shoulder_angle")
    hip_angle = angles.get("hip_angle")

    if elbow_angle is None or shoulder_angle is None or hip_angle is None:
        return "Missing angle data for evaluation.", 0.0, False

    # --- Elbow motion ---
    if ideal_elbow_min <= elbow_angle <= ideal_elbow_max:
        feedback.append("✓ Good elbow range of motion.")
        good_form_count += 1
    else:
        if elbow_angle < ideal_elbow_min:
            feedback.append(f"⚠ Over-curling — relax the elbow by ~{ideal_elbow_min - elbow_angle}°.")
        elif elbow_angle > ideal_elbow_max:
            feedback.append(f"⚠ Not curling enough — bend the elbow ~{elbow_angle - ideal_elbow_max}° more.")

    # --- Shoulder stability ---
    if abs(shoulder_angle) <= ideal_shoulder_max:
        feedback.append("✓ Good, Shoulders are stable.")
        good_form_count += 1
    else:
        feedback.append(f"⚠ Too much shoulder movement — reduce by ~{abs(shoulder_angle - ideal_shoulder_max)}°.")

    # --- Hip posture ---
    if ideal_hip_min <= hip_angle <= ideal_hip_max:
        feedback.append("✓ Good upright posture.")
        good_form_count += 1
    else:
        if hip_angle < ideal_hip_min:
            feedback.append(f"⚠ Leaning too far forward — straighten up by ~{ideal_hip_min - hip_angle}°.")
        else:
            feedback.append(f"⚠ Leaning backward — adjust forward by ~{hip_angle - ideal_hip_max}°.")

    # Calculate form score
    form_score = good_form_count / total_checks
    is_good_form = form_score >= 0.67  # At least 2/3 criteria must be good

    # --- Final verdict ---
    if form_score == 1.0:
        return "🎯 Perfect form! Strong curls", form_score, is_good_form

    return "\n".join(feedback), form_score, is_good_form
