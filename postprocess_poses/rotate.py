import numpy as np

# H36M joint indices
ROOT, RHIP, RKNE, RANK, LHIP, LKNE, LANK, BELLY, NECK, NOSE, HEAD, LSHO, LELB, LWRI, RSHO, RELB, RWRI = range(17)

def calculate_hip_orientation_angle(pose_3d):
    """Calculate the hip orientation angle in the XZ plane."""
    left_hip = pose_3d[LHIP]
    right_hip = pose_3d[RHIP]
    
    # Calculate hip vector (from left to right hip)
    hip_vector = right_hip - left_hip
    
    # Project to XZ plane (ignore Y coordinate)
    hip_vector_xz = np.array([hip_vector[0], hip_vector[2]])
    
    # Calculate angle with respect to positive X-axis
    if np.linalg.norm(hip_vector_xz) > 0:
        angle_rad = np.arctan2(hip_vector_xz[1], hip_vector_xz[0])
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = 0.0
    
    return angle_deg

def calculate_rotation_to_front_facing(pose_3d, target_angle_deg=-178.55):
    """
    Calculate the rotation needed to make pose front-facing.
    
    Args:
        pose_3d: 3D pose array
        target_angle_deg: Target hip angle in degrees (default: test video angle)
    
    Returns:
        rotation_angle_rad: Rotation angle in radians around Y-axis
    """
    # Calculate current hip orientation angle
    current_hip_angle = calculate_hip_orientation_angle(pose_3d)
    
    # Calculate the difference in angles
    angle_diff = target_angle_deg - current_hip_angle
    
    # Normalize angle to [-180, 180] range
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    

    
    # Convert to radians
    rotation_angle_rad = np.radians(angle_diff)
    
    return rotation_angle_rad

# Rotate point around specified axis
def rotate_point(point, angle, axis='y', center=np.zeros(3)):
    c, s = np.cos(angle), np.sin(angle)
    rotated = point - center
    if axis == 'x':
        mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # 'z'
        mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return mat @ rotated + center

# Rotate entire skeleton
def rotate_skeleton(pose, angle_x, angle_y, angle_z):
    root = pose[ROOT].copy()
    rotated_pose = pose.copy()

    # Rotate around X-axis
    for i in range(len(rotated_pose)):
        rotated_pose[i] = rotate_point(rotated_pose[i], angle_x, axis='x', center=root)

    # Rotate around Y-axis
    for i in range(len(rotated_pose)):
        rotated_pose[i] = rotate_point(rotated_pose[i], angle_y, axis='y', center=root)

    # Rotate around Z-axis
    for i in range(len(rotated_pose)):
        rotated_pose[i] = rotate_point(rotated_pose[i], angle_z, axis='z', center=root)

    return rotated_pose

def rotate_skeleton_to_front_facing(pose, target_angle_deg=-178.55):
    """
    Rotate skeleton to front-facing orientation using dynamic calculation.
    Automatically applies 180-degree horizontal flip to ensure same facing direction.
    
    Args:
        pose: 3D pose array
        target_angle_deg: Target hip angle in degrees (default: test video angle)
    
    Returns:
        rotated_pose: Rotated pose array
    """
    # Calculate rotation needed
    rotation_angle = calculate_rotation_to_front_facing(pose, target_angle_deg)
    
    # Apply rotation around Y-axis only (for front-facing alignment)
    root = pose[ROOT].copy()
    rotated_pose = pose.copy()
    
    for i in range(len(rotated_pose)):
        rotated_pose[i] = rotate_point(rotated_pose[i], rotation_angle, axis='y', center=root)
    
    # Apply 180-degree horizontal flip to ensure same facing direction
    flip_angle = np.pi  # 180 degrees
    for i in range(len(rotated_pose)):
        rotated_pose[i] = rotate_point(rotated_pose[i], flip_angle, axis='y', center=root)
    
    return rotated_pose

# Main process for list of pose segments
def process_pose_segments(pose_segments, target_angle_deg=-178.55):
    """
    Process multiple pose segments (list of arrays) with dynamic rotation calculation.
    
    Args:
        pose_segments: List of pose segments
        target_angle_deg: Target hip angle in degrees (default: test video angle)
    
    Returns:
        rotated_segments: List of rotated pose segments
    """
    if not pose_segments:
        return pose_segments
    
    print(f"🔄 Rotating poses to front-facing orientation (target angle: {target_angle_deg:.2f}°)")
    
    rotated_segments = []
    for segment_idx, segment in enumerate(pose_segments):
        if segment.shape[0] == 0:
            rotated_segments.append(segment)
            continue
        
        print(f"   Processing segment {segment_idx + 1}/{len(pose_segments)} (shape: {segment.shape})")
        
        # Use dynamic rotation calculation
        rotated_segment = np.array([
            rotate_skeleton_to_front_facing(pose, target_angle_deg) 
            for pose in segment
        ])
        
        rotated_segments.append(rotated_segment)
    
    print("✅ Dynamic rotation completed!")
    return rotated_segments

# Legacy function for backward compatibility
def process_pose_segments_legacy(pose_segments):
    """Legacy function using fixed Euler angles for backward compatibility."""
    if not pose_segments:
        return pose_segments
        
    # Euler angles (in radians) calculated from sample frame analysis
    angle_x = -1.687  # -96.6° rotation around X-axis
    angle_y = -0.157  # -9.0° rotation around Y-axis
    angle_z = -2.4    # -120.8° rotation around Z-axis

    rotated_segments = []
    for segment in pose_segments:
        if segment.shape[0] == 0:
            rotated_segments.append(segment)
            continue
        
        rotated_segment = np.array([rotate_skeleton(pose, angle_x, angle_y, angle_z) for pose in segment])
        rotated_segments.append(rotated_segment)
    
    return rotated_segments