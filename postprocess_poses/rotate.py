import numpy as np

# H36M joint indices
ROOT, RHIP, RKNE, RANK, LHIP, LKNE, LANK, BELLY, NECK, NOSE, HEAD, LSHO, LELB, LWRI, RSHO, RELB, RWRI = range(17)

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

# Main process for list of pose segments
def process_pose_segments(pose_segments):
    """Process multiple pose segments (list of arrays)."""
    if not pose_segments:
        return pose_segments
        
    # Euler angles (in radians) calculated from sample frame analysis
    angle_x = -1.687  # -96.6° rotation around X-axis
    angle_y = -0.157  # -9.0° rotation around Y-axis
    angle_z = -2.4  # -120.8° rotation around Z-axis

    rotated_segments = []
    for segment in pose_segments:
        if segment.shape[0] == 0:
            rotated_segments.append(segment)
            continue
        
        rotated_segment = np.array([rotate_skeleton(pose, angle_x, angle_y, angle_z) for pose in segment])
        rotated_segments.append(rotated_segment)
    
    return rotated_segments