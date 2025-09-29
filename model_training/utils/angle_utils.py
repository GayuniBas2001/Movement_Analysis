import numpy as np

def calculate_angle(a, b, c, use_2d=True):
    """
    Calculate the angle at point b given three points (a, b, c).
    Each point is an (x,y,z) coordinate.
    
    Args:
        a, b, c: Points as [x, y, z] coordinates
        use_2d: If True, only use x,y coordinates (ignore z)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Use only 2D coordinates for most joint angles
    if use_2d:
        a, b, c = a[:2], b[:2], c[:2]
    
    v1, v2 = a - b, c - b
    
    # Handle edge case where vectors have zero length
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def knee_angle(hip, knee, ankle):
    return calculate_angle(hip, knee, ankle)

def shoulder_angle(elbow, shoulder, hip):
    return calculate_angle(elbow, shoulder, hip)

def elbow_angle(wrist, elbow, shoulder):
    return calculate_angle(wrist, elbow, shoulder)

if __name__ == "__main__":
    # Example usage
    hip = (0, 0, 0)
    knee = (1, 1, 0)
    ankle = (2, 0, 0)
    angle = knee_angle(hip, knee, ankle)
    print(f"Knee Angle: {angle}")

    elbow = (3, 1, 0)
    shoulder = (4, 2, 0)
    angle = elbow_angle(elbow, shoulder, hip)
    print(f"Elbow Angle: {angle}")

    wrist = (5, 1, 0)
    angle = shoulder_angle(wrist, elbow, shoulder)
    print(f"Shoulder Angle: {angle}")