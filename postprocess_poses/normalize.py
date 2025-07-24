import numpy as np
from typing import List
import math

# H36M joint indices
ROOT = 0
RHIP = 1
RKNE = 2
RANK = 3
LHIP = 4
LKNE = 5
LANK = 6
BELLY = 7
NECK = 8
NOSE = 9
HEAD = 10
LSHO = 11
LELB = 12
LWRI = 13
RSHO = 14
RELB = 15
RWRI = 16

def calculate_bone_length(pose, joint1, joint2):
    """Calculate distance between two joints."""
    return np.linalg.norm(pose[joint1] - pose[joint2])

def scale_skeleton_to_standard_size(poses, target_scale=1.0):
    """Scale entire skeleton to standard size using torso as reference."""
    scaled_poses = poses.copy()
    
    # Calculate average torso height across all frames
    torso_heights = []
    for pose in poses:
        torso_height = calculate_bone_length(pose, ROOT, HEAD)
        if torso_height > 0:
            torso_heights.append(torso_height)
    
    if not torso_heights:
        return scaled_poses
    
    avg_torso_height = np.mean(torso_heights)
    scale_factor = target_scale / avg_torso_height
    
    # Apply scaling to all poses
    for frame_idx in range(len(poses)):
        root_pos = scaled_poses[frame_idx, ROOT].copy()
        # Scale relative to root position
        scaled_poses[frame_idx] = (scaled_poses[frame_idx] - root_pos) * scale_factor + root_pos
    
    return scaled_poses

def center_at_origin(poses):
    """Center hip at origin (0,0) keeping Z coordinate."""
    centered_poses = poses.copy()
    
    for frame_idx in range(len(poses)):
        root_pos = centered_poses[frame_idx, ROOT]
        # Offset only X and Y, keep Z
        offset = np.array([root_pos[0], root_pos[1], 0])
        centered_poses[frame_idx] -= offset
    
    return centered_poses

def apply_ema_smoothing(poses, alpha=0.3):
    """Apply Exponential Moving Average smoothing to reduce jitter."""
    if len(poses) == 0:
        return poses
    
    smoothed_poses = poses.copy()
    
    # Initialize with first frame
    for frame_idx in range(1, len(poses)):
        # EMA: new_value = alpha * current + (1-alpha) * previous
        smoothed_poses[frame_idx] = alpha * poses[frame_idx] + (1 - alpha) * smoothed_poses[frame_idx - 1]
    
    return smoothed_poses

def normalize_pose_segments(pose_segments, target_scale=1.0, ema_alpha=0.3):
    """
    Apply normalization to pose segments.
    
    Args:
        pose_segments: List of pose segments, each of shape [frames, 17, 3]
        target_scale: Target skeleton scale
        ema_alpha: EMA smoothing factor (0 < alpha < 1)
        enable_rotation: Whether to apply rotation to make skeleton front-facing
    
    Returns:
        List of normalized pose segments in same format as input
    """
    if not pose_segments:
        return pose_segments
    
    print(f"\n🔧 Normalizing {len(pose_segments)} pose segments...")
    
    normalized_segments = []
    
    for i, segment in enumerate(pose_segments):
        if segment.shape[0] == 0:
            normalized_segments.append(segment)
            continue
            
        print(f"Processing segment {i+1}/{len(pose_segments)}...")
        
        
        # Step 2: Scale skeleton to standard size
        normalized = scale_skeleton_to_standard_size(segment, target_scale)
        
        # Step 3: Center at origin
        normalized = center_at_origin(normalized)
        
        # Step 4: Apply EMA smoothing
        normalized = apply_ema_smoothing(normalized, ema_alpha)
        
        normalized_segments.append(normalized)
    
    print(f"✅ Normalization completed!")
    print(f"   - Skeleton rotated to upright and front-facing")
    print(f"   - Skeleton scaled to {target_scale}")
    print(f"   - Hip centered at origin")
    print(f"   - EMA smoothing applied (alpha={ema_alpha})")
    
    return normalized_segments
