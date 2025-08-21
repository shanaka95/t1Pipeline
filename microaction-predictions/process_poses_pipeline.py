#!/usr/bin/env python3
"""
Complete Pose Processing Pipeline for Micro-Action Prediction

This script processes pose data through the complete pipeline:
1. Load poses from .pkl file
2. Concatenate all segments into a single long sequence
3. Apply filtering using postprocess_poses/filter.py
4. Apply normalization using postprocess_poses/normalize.py
5. Apply rotation using postprocess_poses/rotate.py
6. Convert to COCO body style format
7. Segment using postprocess_poses/segment.py with 0.05 threshold
8. Predict micro-actions for all segments using MMN model

Usage:
    python process_poses_pipeline.py --input poses/005_t1_20230519/poses_3D.pkl --output results/005_t1_20230519
    python process_poses_pipeline.py --input poses/005_t1_20230519/poses_3D.pkl --output results/005_t1_20230519 --debug-num-clips 5 --debug-clip-length 50
"""

import os
import sys
import numpy as np
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
warnings.filterwarnings('ignore')

# Add action recognition paths for MMN
sys.path.append('../action_recognition/MMN')

# Add postprocess_poses to path
sys.path.append('../postprocess_poses')

# H36M skeleton connections (joint_start -> joint_end)
H36M_CONNECTIONS = [
    (0, 1),   # Hip -> Right Hip
    (1, 2),   # Right Hip -> Right Knee  
    (2, 3),   # Right Knee -> Right Ankle
    (0, 4),   # Hip -> Left Hip
    (4, 5),   # Left Hip -> Left Knee
    (5, 6),   # Left Knee -> Left Ankle
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head
    (8, 11),  # Thorax -> Left Shoulder
    (11, 12), # Left Shoulder -> Left Elbow
    (12, 13), # Left Elbow -> Left Hand
    (8, 14),  # Thorax -> Right Shoulder
    (14, 15), # Right Shoulder -> Right Elbow
    (15, 16)  # Right Elbow -> Right Hand
]

# COCO skeleton connections (joint_start -> joint_end)
COCO_CONNECTIONS = [
    # Head connections
    (0, 1),   # Nose -> Left Eye
    (0, 2),   # Nose -> Right Eye
    (0, 3),   # Nose -> Left Ear
    (0, 4),   # Nose -> Right Ear
    
    # Head to body connections (connect head to shoulders)
    (1, 5),   # Left Eye -> Left Shoulder
    (2, 6),   # Right Eye -> Right Shoulder
    (3, 5),   # Left Ear -> Left Shoulder
    (4, 6),   # Right Ear -> Right Shoulder
    
    # Upper body
    (5, 6),   # Left Shoulder -> Right Shoulder
    (5, 7),   # Left Shoulder -> Left Elbow
    (7, 9),   # Left Elbow -> Left Wrist
    (6, 8),   # Right Shoulder -> Right Elbow
    (8, 10),  # Right Elbow -> Right Wrist
    
    # Torso
    (5, 11),  # Left Shoulder -> Left Hip
    (6, 12),  # Right Shoulder -> Right Hip
    (11, 12), # Left Hip -> Right Hip
    
    # Lower body
    (11, 13), # Left Hip -> Left Knee
    (13, 15), # Left Knee -> Left Ankle
    (12, 14), # Right Hip -> Right Knee
    (14, 16), # Right Knee -> Right Ankle
]

def load_poses_from_pkl(poses_path: str) -> np.ndarray:
    """
    Load and concatenate all pose segments from a PKL file.
    
    Args:
        poses_path (str): Path to the PKL file containing pose segments
        
    Returns:
        np.ndarray: Concatenated pose sequence with shape (total_frames, 17, 3)
    """
    print(f"Loading poses from: {poses_path}")
    
    import pickle
    
    # Load the PKL file
    with open(poses_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict):
        # If it's a dictionary with segment keys
        segment_keys = [key for key in data.keys() if key.startswith('segment_')]
        segment_keys.sort()  # Ensure proper ordering
        
        print(f"Found {len(segment_keys)} pose segments")
        
        # Concatenate all segments
        all_segments = []
        total_frames = 0
        
        for key in segment_keys:
            segment = data[key]
            
            # Remove batch dimension if present
            if len(segment.shape) == 4 and segment.shape[0] == 1:
                segment = segment.squeeze(0)
            
            # Validate segment format
            if len(segment.shape) != 3 or segment.shape[1] != 17 or segment.shape[2] != 3:
                print(f"Warning: Segment {key} has unexpected shape {segment.shape}, skipping...")
                continue
            
            all_segments.append(segment)
            total_frames += segment.shape[0]
            print(f"  {key}: {segment.shape[0]} frames")
        
        # Concatenate all segments
        if all_segments:
            concatenated_poses = np.concatenate(all_segments, axis=0)
            print(f"Concatenated {len(all_segments)} segments into {concatenated_poses.shape[0]} frames")
            return concatenated_poses
        else:
            raise ValueError("No valid pose segments found in the PKL file")
    
    elif isinstance(data, list):
        # If it's a list of segments
        print(f"Found {len(data)} pose segments in list format")
        
        all_segments = []
        total_frames = 0
        
        for i, segment in enumerate(data):
            # Remove batch dimension if present
            if len(segment.shape) == 4 and segment.shape[0] == 1:
                segment = segment.squeeze(0)
            
            # Validate segment format
            if len(segment.shape) != 3 or segment.shape[1] != 17 or segment.shape[2] != 3:
                print(f"Warning: Segment {i} has unexpected shape {segment.shape}, skipping...")
                continue
            
            all_segments.append(segment)
            total_frames += segment.shape[0]
            print(f"  Segment {i}: {segment.shape[0]} frames")
        
        # Concatenate all segments
        if all_segments:
            concatenated_poses = np.concatenate(all_segments, axis=0)
            print(f"Concatenated {len(all_segments)} segments into {concatenated_poses.shape[0]} frames")
            return concatenated_poses
        else:
            raise ValueError("No valid pose segments found in the PKL file")
    
    elif isinstance(data, np.ndarray):
        # If it's already a numpy array
        print(f"Found numpy array with shape: {data.shape}")
        
        # Handle extra batch dimension if present
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        
        # Validate format
        if len(data.shape) != 3 or data.shape[1] != 17 or data.shape[2] != 3:
            raise ValueError(f"Invalid pose array shape: {data.shape}, expected (frames, 17, 3)")
        
        print(f"Using pose array with {data.shape[0]} frames")
        return data
    
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

def filter_poses(poses: np.ndarray, velocity_threshold: float = 0.3, 
                acceleration_threshold: float = 0.5) -> np.ndarray:
    """
    Filter poses using the glitch detection logic from postprocess_poses/filter.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        velocity_threshold: Threshold for velocity-based glitch detection
        acceleration_threshold: Threshold for acceleration-based glitch detection
        
    Returns:
        Filtered pose sequence
    """
    print(f"🔍 Filtering poses with velocity_threshold={velocity_threshold}, acceleration_threshold={acceleration_threshold}")
    
    # Import filter functions
    from filter import PoseGlitchDetector
    
    # Initialize detector
    detector = PoseGlitchDetector(velocity_threshold, acceleration_threshold)
    
    # Detect glitches
    glitch_info = detector.detect_glitches(poses)
    
    if glitch_info["has_glitches"]:
        print(f"⚠️  Found {glitch_info['num_glitch_frames']} glitch frames ({glitch_info['glitch_percentage']:.1f}%)")
        
        # Create mask for non-glitch frames
        glitch_mask = np.ones(poses.shape[0], dtype=bool)
        glitch_mask[glitch_info['glitch_frames']] = False
        
        # Filter out glitch frames
        filtered_poses = poses[glitch_mask]
        print(f"✅ Filtered poses: {poses.shape[0]} -> {filtered_poses.shape[0]} frames")
        
        return filtered_poses
    else:
        print(f"✅ No glitches detected, keeping all {poses.shape[0]} frames")
        return poses

def normalize_poses(poses: np.ndarray, target_scale: float = 1.0, 
                   ema_alpha: float = 0.3) -> np.ndarray:
    """
    Normalize poses using postprocess_poses/normalize.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        target_scale: Target skeleton scale
        ema_alpha: EMA smoothing factor
        
    Returns:
        Normalized pose sequence
    """
    print(f"🔧 Normalizing poses with target_scale={target_scale}, ema_alpha={ema_alpha}")
    
    # Import normalize functions
    from normalize import scale_skeleton_to_standard_size, center_at_origin, apply_ema_smoothing
    
    # Apply normalization steps
    normalized = scale_skeleton_to_standard_size(poses, target_scale)
    normalized = center_at_origin(normalized)
    normalized = apply_ema_smoothing(normalized, ema_alpha)
    
    print(f"✅ Normalization completed!")
    return normalized

def rotate_poses(poses: np.ndarray, target_angle_deg: float = -178.55) -> np.ndarray:
    """
    Rotate poses using postprocess_poses/rotate.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        target_angle_deg: Target hip angle in degrees
        
    Returns:
        Rotated pose sequence
    """
    print(f"🔄 Rotating poses to front-facing orientation (target angle: {target_angle_deg:.2f}°)")
    
    # Import rotate functions
    from rotate import rotate_skeleton_to_front_facing
    
    # Apply rotation to each frame
    rotated_poses = np.array([
        rotate_skeleton_to_front_facing(pose, target_angle_deg) 
        for pose in poses
    ])
    
    print(f"✅ Rotation completed!")
    return rotated_poses

def convert_h36m_to_coco_format(h36m_pose: np.ndarray) -> np.ndarray:
    """
    Convert H36M pose format (17 joints, 3D) to COCO format (44 joints, 2D)
    
    Args:
        h36m_pose: Shape (T, 17, 3) - H36M pose sequence
        
    Returns:
        Shape (T, 44, 2) - COCO pose sequence
    """
    T, V, C = h36m_pose.shape
    
    # Initialize COCO pose with zeros
    coco_pose = np.zeros((T, 44, 2))
    
    # H36M joint names for reference:
    # 0: root, 1: rhip, 2: rkne, 3: rank, 4: lhip, 5: lkne, 6: lank
    # 7: belly, 8: neck, 9: nose, 10: head, 11: lsho, 12: lelb, 13: lwri
    # 14: rsho, 15: relb, 16: rwri
    
    # COCO joint names for reference:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    # 17-43: additional keypoints (mostly zeros in our case)
    
    # Proper mapping from H36M to COCO format
    h36m_to_coco = {
        # Head and face
        9: 0,    # nose -> nose
        10: 0,   # head -> nose (approximation)
        
        # Shoulders
        11: 5,   # lsho -> left_shoulder
        14: 6,   # rsho -> right_shoulder
        
        # Elbows
        12: 7,   # lelb -> left_elbow
        15: 8,   # relb -> right_elbow
        
        # Wrists
        13: 9,   # lwri -> left_wrist
        16: 10,  # rwri -> right_wrist
        
        # Hips
        4: 11,   # lhip -> left_hip
        1: 12,   # rhip -> right_hip
        
        # Knees
        5: 13,   # lkne -> left_knee
        2: 14,   # rkne -> right_knee
        
        # Ankles
        6: 15,   # lank -> left_ankle
        3: 16,   # rank -> right_ankle
    }
    
    # Project 3D to 2D by taking X and Y coordinates
    # Note: H36M uses different coordinate system, so we need to adjust
    h36m_2d = h36m_pose[:, :, :2].copy()
    
    # Apply coordinate system adjustment for better visualization
    # Flip Y-axis to match COCO convention (Y increases downward in COCO)
    h36m_2d[:, :, 1] = -h36m_2d[:, :, 1]
    
    # Copy joints from H36M to COCO
    for h36m_idx, coco_idx in h36m_to_coco.items():
        coco_pose[:, coco_idx, :] = h36m_2d[:, h36m_idx, :]
    
    # Fill in missing face keypoints with better approximations
    if 0 in h36m_to_coco.values():  # If nose is mapped
        nose_pos = coco_pose[:, 0, :]  # nose position
        
        # Get shoulder positions for better head positioning
        left_shoulder_pos = coco_pose[:, 5, :]  # left_shoulder
        right_shoulder_pos = coco_pose[:, 6, :]  # right_shoulder
        
        # Calculate head center position (above shoulders)
        shoulder_center = (left_shoulder_pos + right_shoulder_pos) / 2
        
        # Position head above shoulders (adjust nose position)
        head_height_offset = 0.15  # Distance above shoulders
        coco_pose[:, 0, :] = shoulder_center + np.array([0, head_height_offset])  # nose
        
        # Update nose position for face keypoints
        nose_pos = coco_pose[:, 0, :]
        
        # Approximate eye positions (slightly above and to sides of nose)
        eye_offset_x = 0.03
        eye_offset_y = -0.02
        coco_pose[:, 1, :] = nose_pos + np.array([-eye_offset_x, eye_offset_y])  # left_eye
        coco_pose[:, 2, :] = nose_pos + np.array([eye_offset_x, eye_offset_y])   # right_eye
        
        # Approximate ear positions (slightly to sides of nose)
        ear_offset_x = 0.05
        ear_offset_y = 0.01
        coco_pose[:, 3, :] = nose_pos + np.array([-ear_offset_x, ear_offset_y])  # left_ear
        coco_pose[:, 4, :] = nose_pos + np.array([ear_offset_x, ear_offset_y])   # right_ear
    
    # Normalize the pose to reasonable range for visualization
    # Find the range of non-zero coordinates
    non_zero_mask = np.any(coco_pose != 0, axis=2)
    if np.any(non_zero_mask):
        valid_coords = coco_pose[non_zero_mask]
        if len(valid_coords) > 0:
            coord_range = np.ptp(valid_coords, axis=0)
            max_range = np.max(coord_range)
            if max_range > 0:
                # Scale to reasonable range (e.g., [-1, 1])
                scale_factor = 2.0 / max_range
                coco_pose = coco_pose * scale_factor
    
    return coco_pose

def create_debug_animation(poses: np.ndarray, 
                         output_path: str, 
                         title: str = "Pose Animation", 
                         fps: int = 15) -> None:
    """
    Create a skeleton animation for debugging purposes.
    
    Args:
        poses: 3D pose data with shape (T, 17, 3) or (T, 44, 2)
        output_path: Output path for the GIF
        title: Title for the animation
        fps: Frames per second for the animation
    """
    print(f"   🎬 Creating debug animation with {poses.shape[0]} frames...")
    
    # Determine if it's 3D or 2D poses and format
    is_3d = poses.shape[2] == 3
    is_coco = poses.shape[1] == 44  # COCO has 44 joints
    
    # Choose appropriate connections
    if is_coco:
        connections = COCO_CONNECTIONS
    else:
        connections = H36M_CONNECTIONS
    
    # Set up the figure
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate axis limits from pose data
        all_coords = poses.reshape(-1, 3)
        valid_coords = all_coords[np.any(np.abs(all_coords) > 1e-6, axis=1)]
        
        if len(valid_coords) > 0:
            ranges = np.ptp(valid_coords, axis=0)
            centers = np.mean(valid_coords, axis=0)
            max_range = max(np.max(ranges), 1.0)
            padding = max_range * 0.1 + 0.1
            
            x_lim = [centers[0] - max_range/2 - padding, centers[0] + max_range/2 + padding]
            y_lim = [centers[1] - max_range/2 - padding, centers[1] + max_range/2 + padding]
            z_lim = [centers[2] - max_range/2 - padding, centers[2] + max_range/2 + padding]
        else:
            x_lim = y_lim = z_lim = [-1, 1]
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel('X (Left-Right)')
        ax.set_ylabel('Y (Height)')
        ax.set_zlabel('Z (Forward-Back)')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=15, azim=45)
        
        # Create line objects for 3D view
        lines_3d = []
        for _ in connections:
            line, = ax.plot([], [], [], 'royalblue', linewidth=3, alpha=0.8)
            lines_3d.append(line)
        
        # Create scatter plot for joints in 3D
        joint_scatter = ax.scatter([], [], [], c='red', s=60, alpha=0.9)
        
    else:
        # 2D visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Left-Right (X)')
        ax.set_ylabel('Height (-Y)')
        
        # Create line objects for 2D view
        lines_2d = []
        for _ in connections:
            line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
            lines_2d.append(line)
    
    # Add frame counter
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def init():
        """Initialize animation."""
        if is_3d:
            for line in lines_3d:
                line.set_data([], [])
                line.set_3d_properties([])
            joint_scatter._offsets3d = ([], [], [])
        else:
            for line in lines_2d:
                line.set_data([], [])
        frame_text.set_text('')
        
        if is_3d:
            return lines_3d + [frame_text]
        else:
            return lines_2d + [frame_text]
    
    def animate_frame(frame_idx):
        """Update function for each frame."""
        if frame_idx >= poses.shape[0]:
            if is_3d:
                return lines_3d + [frame_text]
            else:
                return lines_2d + [frame_text]
        
        # Get current frame poses
        current_frame = poses[frame_idx, :, :]
        
        if is_3d:
            x_coords = current_frame[:, 0]
            y_coords = current_frame[:, 1]
            z_coords = current_frame[:, 2]
            
            # Update 3D view
            for i, (start_joint, end_joint) in enumerate(connections):
                lines_3d[i].set_data([x_coords[start_joint], x_coords[end_joint]], 
                                    [y_coords[start_joint], y_coords[end_joint]])
                lines_3d[i].set_3d_properties([z_coords[start_joint], z_coords[end_joint]])
            
            # Update joint positions in 3D
            joint_scatter._offsets3d = (x_coords, y_coords, z_coords)
        else:
            x_coords = current_frame[:, 0]
            y_coords = current_frame[:, 1]
            
            # Update 2D view (flip Y for proper orientation)
            y_coords_2d = -y_coords
            for i, (start_joint, end_joint) in enumerate(connections):
                x_data = [x_coords[start_joint], x_coords[end_joint]]
                y_data = [y_coords_2d[start_joint], y_coords_2d[end_joint]]
                lines_2d[i].set_data(x_data, y_data)
        
        # Update frame counter
        progress = (frame_idx + 1) / poses.shape[0] * 100
        frame_text.set_text(f'Frame: {frame_idx + 1}/{poses.shape[0]} ({progress:.1f}%)')
        
        if is_3d:
            return lines_3d + [frame_text]
        else:
            return lines_2d + [frame_text]
    
    # Create animation
    interval = 1000 // fps
    anim = animation.FuncAnimation(fig, animate_frame, init_func=init, frames=poses.shape[0],
                                 interval=interval, blit=False, repeat=True)
    
    # Save as GIF
    print(f"   💾 Saving debug animation to: {output_path}")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"   ✅ Debug animation saved successfully!")
    except Exception as e:
        print(f"   ❌ Error saving debug animation: {e}")
    finally:
        plt.close(fig)

def save_debug_clips(poses: np.ndarray, output_dir: str, stage_name: str, 
                    num_clips: int = 3, clip_length: int = 30, fps: int = 15) -> None:
    """
    Save small debug clips for a given pose stage.
    
    Args:
        poses: Pose sequence with shape (frames, joints, coords)
        output_dir: Output directory for debug clips
        stage_name: Name of the processing stage (e.g., 'before_coco', 'after_coco')
        num_clips: Number of debug clips to save
        clip_length: Length of each clip in frames
        fps: Frames per second for animations
    """
    print(f"🔍 Saving {num_clips} debug clips for {stage_name} stage...")
    
    # Create debug directory
    debug_dir = os.path.join(output_dir, 'debug_clips')
    os.makedirs(debug_dir, exist_ok=True)
    
    total_frames = poses.shape[0]
    
    if total_frames < clip_length:
        print(f"   ⚠️  Total frames ({total_frames}) is less than clip length ({clip_length}), using all frames")
        clip_length = total_frames
    
    # Select random start positions for clips
    max_start = total_frames - clip_length
    if max_start <= 0:
        start_positions = [0]
    else:
        start_positions = random.sample(range(max_start + 1), min(num_clips, max_start + 1))
    
    for i, start_pos in enumerate(start_positions):
        end_pos = min(start_pos + clip_length, total_frames)
        clip_poses = poses[start_pos:end_pos]
        
        # Create output filename
        output_filename = f"debug_{stage_name}_clip_{i+1:02d}_frames_{start_pos:04d}-{end_pos:04d}.gif"
        output_path = os.path.join(debug_dir, output_filename)
        
        # Create title
        title = f"Debug {stage_name.title()} - Clip {i+1}/{len(start_positions)} (Frames {start_pos}-{end_pos})"
        
        # Create visualization
        try:
            create_debug_animation(clip_poses, output_path, title, fps)
        except Exception as e:
            print(f"   ❌ Failed to create debug clip {i+1}: {e}")
            continue
    
    print(f"   ✅ Debug clips saved to: {debug_dir}")

def segment_poses(poses: np.ndarray, velocity_threshold: float = 0.05, 
                 acceleration_threshold: float = 0.05) -> List[np.ndarray]:
    """
    Segment poses using postprocess_poses/segment.py with specified thresholds
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        velocity_threshold: Threshold for movement detection
        acceleration_threshold: Threshold for movement detection
        
    Returns:
        List of pose segments
    """
    print(f"📊 Segmenting poses with velocity_threshold={velocity_threshold}, acceleration_threshold={acceleration_threshold}")
    
    # Import segment functions
    from segment import MovementSegmenter
    
    # Initialize segmenter
    segmenter = MovementSegmenter(
        max_segment_length=243,
        velocity_threshold=velocity_threshold,
        acceleration_threshold=acceleration_threshold,
        min_segment_length=30
    )
    
    # Segment poses
    segments = segmenter.segment_by_movement(poses)
    
    print(f"✅ Created {len(segments)} segments")
    for i, segment in enumerate(segments):
        print(f"  Segment {i+1}: {segment.shape[0]} frames")
    
    return segments

def create_mmn_data_files(segments: List[np.ndarray], output_dir: str, predictions: Dict[str, int] = None) -> str:
    """
    Create MMN data files in the exact format expected by the feeder
    
    Args:
        segments: List of pose segments
        output_dir: Output directory
        
    Returns:
        Path to the MMN data directory
    """
    print(f"💾 Creating MMN data files...")
    
    # Create MMN data directory structure exactly as expected
    mmn_data_dir = os.path.join(output_dir, 'data', 'MA52')
    os.makedirs(mmn_data_dir, exist_ok=True)
    
    # Prepare data for MMN format
    data_annotations = []
    label_data = []
    segment_labels = {}  # Dictionary to store segment labels
    
    for i, segment in enumerate(segments):
        # MMN expects data in format (T, V, C) where T=64, V=44, C=2
        # Our segment is (T, V, C) where T=variable, V=44, C=2
        
        # Sample to 64 frames if longer, pad if shorter
        if segment.shape[0] > 64:
            indices = np.linspace(0, segment.shape[0]-1, 64, dtype=int)
            segment_resized = segment[indices]
        elif segment.shape[0] < 64:
            # Pad with last frame
            padding = np.repeat(segment[-1:], 64 - segment.shape[0], axis=0)
            segment_resized = np.concatenate([segment, padding], axis=0)
        else:
            segment_resized = segment
        
        # Normalize to reasonable range
        segment_normalized = segment_resized / np.max(np.abs(segment_resized)) if np.max(np.abs(segment_resized)) > 0 else segment_resized
        
        # Create data annotation in the format expected by MMN
        frame_dir = f'segment_{i:03d}'
        data_annotation = {
            'frame_dir': frame_dir,
            'keypoint': [segment_normalized],  # MMN expects list with one element
            'label': 0  # Default label
        }
        data_annotations.append(data_annotation)
        
        # Create label entry
        label_entry = {
            'file_name': frame_dir,
            'label': 0  # Default label
        }
        label_data.append(label_entry)
        
        # Store segment label for JSON output
        segment_labels[f'segment_{i:03d}'] = 0  # Default label as integer
    
    # Create the data structure exactly as expected by MMN
    data_dict = {
        'annotations': data_annotations
    }
    
    # Save data files in the exact format MMN expects
    val_data_file = os.path.join(mmn_data_dir, 'val_data.pkl')
    val_label_file = os.path.join(mmn_data_dir, 'val_label.pkl')
    
    import pickle
    with open(val_data_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    with open(val_label_file, 'wb') as f:
        pickle.dump(label_data, f)
    
    # Save segment labels as JSON
    segment_labels_file = os.path.join(output_dir, 'segment_labels.json')
    import json
    with open(segment_labels_file, 'w') as f:
        json.dump(segment_labels, f, indent=2)
    
    print(f"✅ Created MMN data files:")
    print(f"   - Data file: {val_data_file}")
    print(f"   - Label file: {val_label_file}")
    print(f"   - Segment labels: {segment_labels_file}")
    print(f"   - {len(segments)} segments processed")
    
    return os.path.join(output_dir, 'data')



def run_mmn_inference(config_path: str, weights_path: str, data_dir: str, output_dir: str, num_segments: int) -> Dict[str, int]:
    """
    Run MMN inference using main.py
    
    Args:
        config_path: Path to MMN config file
        weights_path: Path to MMN model weights
        data_dir: Directory containing data files (should contain MA52 subdirectory)
        output_dir: Output directory for results
        
    Returns:
        Dictionary mapping segment names to predicted labels
    """
    print(f"🤖 Running MMN inference...")
    
    # Change to MMN directory
    mmn_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), '..', '..'))
    original_dir = os.getcwd()
    
    try:
        os.chdir(mmn_dir)
        
        # Create symbolic link to our data directory in MMN's expected location
        mmn_data_link = os.path.join(mmn_dir, 'data')
        # The data_dir is relative to the original working directory, not the MMN directory
        abs_data_dir = os.path.abspath(os.path.join(original_dir, data_dir))
        
        # Remove existing data link/directory if it exists
        if os.path.exists(mmn_data_link) or os.path.islink(mmn_data_link):
            if os.path.islink(mmn_data_link):
                os.unlink(mmn_data_link)
            elif os.path.isdir(mmn_data_link):
                import shutil
                shutil.rmtree(mmn_data_link)
            elif os.path.isfile(mmn_data_link):
                os.remove(mmn_data_link)
        
        # Create symbolic link to our data directory
        try:
            os.symlink(abs_data_dir, mmn_data_link)
        except FileExistsError:
            # If it still exists, force remove and try again
            if os.path.exists(mmn_data_link) or os.path.islink(mmn_data_link):
                try:
                    os.unlink(mmn_data_link)
                except:
                    import shutil
                    shutil.rmtree(mmn_data_link)
            os.symlink(abs_data_dir, mmn_data_link)
        print(f"Created data symlink: {mmn_data_link} -> {abs_data_dir}")
        
        # Create work directory for MMN
        work_dir = os.path.join(mmn_dir, 'work_dir', 'test', 'MA52_J')
        os.makedirs(work_dir, exist_ok=True)
        print(f"Created MMN work directory: {work_dir}")
        
        # Add torchlight and torchpack to Python path and run MMN main.py
        torchlight_path = os.path.join(mmn_dir, 'torchlight')
        torchpack_path = os.path.join(mmn_dir, 'torchpack')
        # Use CPU version of MMN main.py
        cmd = f"PYTHONPATH={torchlight_path}:{torchpack_path}:$PYTHONPATH python main_cpu.py --config ./config/test/MA52_J.yaml --weights ./checkpoints/MMN_MA52_J.pt"
        print(f"Running command: {cmd}")
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Clean up symlink
        if os.path.exists(mmn_data_link):
            os.unlink(mmn_data_link)
        
        if result.returncode != 0:
            print(f"MMN inference failed: {result.stderr}")
            raise RuntimeError(f"MMN inference failed with return code {result.returncode}")
        
        print(f"MMN inference completed successfully")
        print(f"Output: {result.stdout}")
        print(f"Error output: {result.stderr}")
        
        # Parse predictions from MMN output
        predictions = parse_mmn_predictions(result.stdout, num_segments)
        
        # If no predictions found, try to debug the issue
        if not predictions or all(pred == 0 for pred in predictions.values()):
            print(f"⚠️  No valid predictions found. Debugging MMN output...")
            print(f"   Output length: {len(result.stdout)}")
            print(f"   Error length: {len(result.stderr)}")
            print(f"   Return code: {result.returncode}")
            
            # Try to find MMN prediction files
            mmn_output_files = []
            
            # First look in the work directory
            work_dir = os.path.join(mmn_dir, 'work_dir', 'test', 'MA52_J')
            if os.path.exists(work_dir):
                for file in os.listdir(work_dir):
                    if file.endswith('.pkl') and 'score' in file:
                        mmn_output_files.append(os.path.join(work_dir, file))
            
            # If not found, search the entire MMN directory
            if not mmn_output_files:
                for root, dirs, files in os.walk(mmn_dir):
                    for file in files:
                        if file.endswith('.pkl') and 'score' in file:
                            mmn_output_files.append(os.path.join(root, file))
            
            if mmn_output_files:
                print(f"   Found MMN score files: {mmn_output_files}")
                
                # Try to load predictions from the score file
                try:
                    import pickle
                    score_file = mmn_output_files[0]  # Use the first score file
                    print(f"   Loading predictions from: {score_file}")
                    
                    with open(score_file, 'rb') as f:
                        score_dict = pickle.load(f)
                    
                    print(f"   Score dict keys: {list(score_dict.keys())[:10]}...")
                    print(f"   Score dict values: {list(score_dict.values())[:5]}...")
                    
                    # Convert scores to predictions
                    for i in range(num_segments):
                        segment_name = f"segment_{i:03d}"
                        if i in score_dict:
                            # Get the predicted class (argmax of scores)
                            scores = score_dict[i]
                            prediction = np.argmax(scores)
                            predictions[segment_name] = int(prediction)
                            print(f"   Loaded prediction: {segment_name} -> {prediction}")
                        else:
                            predictions[segment_name] = 0
                            print(f"   No score for {segment_name}, using 0")
                    
                except Exception as e:
                    print(f"   Error loading score file: {e}")
            else:
                print(f"   No MMN score files found")
        
        return predictions
        
    finally:
        os.chdir(original_dir)

def parse_mmn_predictions(stdout: str, num_segments: int) -> Dict[str, int]:
    """
    Parse predictions from MMN inference output.
    
    Args:
        stdout: Standard output from MMN inference
        num_segments: Number of segments that were processed
        
    Returns:
        Dictionary mapping segment names to predicted labels
    """
    print(f"🔍 Parsing MMN predictions from output...")
    
    predictions = {}
    
    # Try to parse predictions from the output
    # MMN typically outputs predictions in a specific format
    lines = stdout.strip().split('\n')
    
    # Look for prediction lines in the output
    for line in lines:
        line = line.strip()
        
        # Common MMN output patterns
        if 'segment_' in line and ('prediction' in line.lower() or 'label' in line.lower()):
            # Try to extract segment name and prediction
            try:
                # Pattern: segment_XXX: prediction Y
                if ':' in line:
                    parts = line.split(':')
                    segment_part = parts[0].strip()
                    prediction_part = parts[1].strip()
                    
                    # Extract segment name
                    if 'segment_' in segment_part:
                        segment_name = segment_part.split('segment_')[1].split()[0]
                        segment_name = f"segment_{segment_name:03d}"
                        
                        # Extract prediction
                        if 'prediction' in prediction_part.lower():
                            pred_str = prediction_part.split('prediction')[-1].strip()
                        elif 'label' in prediction_part.lower():
                            pred_str = prediction_part.split('label')[-1].strip()
                        else:
                            pred_str = prediction_part
                        
                        # Convert to integer
                        try:
                            prediction = int(pred_str)
                            predictions[segment_name] = prediction
                            print(f"   Found prediction: {segment_name} -> {prediction}")
                        except ValueError:
                            print(f"   Could not parse prediction value: {pred_str}")
                            
            except Exception as e:
                print(f"   Error parsing line: {line} - {e}")
                continue
    
    # If no predictions found, try alternative parsing methods
    if not predictions:
        print(f"   ⚠️  No predictions found in output, trying alternative parsing...")
        
        # Look for any numbers that might be predictions
        import re
        numbers = re.findall(r'\d+', stdout)
        
        print(f"   Found {len(numbers)} numbers in output")
        if len(numbers) > 0:
            print(f"   First 10 numbers: {numbers[:10]}")
        
        if len(numbers) >= num_segments:
            # Assume the first num_segments numbers are predictions
            for i in range(num_segments):
                segment_name = f"segment_{i:03d}"
                try:
                    prediction = int(numbers[i])
                    predictions[segment_name] = prediction
                    print(f"   Assigned prediction: {segment_name} -> {prediction}")
                except (ValueError, IndexError):
                    predictions[segment_name] = 0
                    print(f"   Default prediction: {segment_name} -> 0")
        else:
            # Fallback: assign default predictions
            for i in range(num_segments):
                segment_name = f"segment_{i:03d}"
                predictions[segment_name] = 0
                print(f"   Default prediction: {segment_name} -> 0")
    
    # Additional debugging: check if we have any non-zero predictions
    non_zero_predictions = [pred for pred in predictions.values() if pred != 0]
    print(f"   Non-zero predictions: {len(non_zero_predictions)} out of {len(predictions)}")
    if non_zero_predictions:
        print(f"   Non-zero prediction values: {set(non_zero_predictions)}")
    
    print(f"   ✅ Parsed {len(predictions)} predictions")
    return predictions

def update_segment_labels_with_predictions(output_dir: str, predictions: Dict[str, int]) -> None:
    """
    Update segment labels JSON file with actual predictions from MMN model.
    
    Args:
        output_dir: Output directory containing segment_labels.json
        predictions: Dictionary mapping segment names to predicted labels
    """
    print(f"📝 Updating segment labels with predictions...")
    
    segment_labels_file = os.path.join(output_dir, 'segment_labels.json')
    
    if not os.path.exists(segment_labels_file):
        print(f"   ⚠️  Segment labels file not found: {segment_labels_file}")
        return
    
    # Load existing segment labels
    with open(segment_labels_file, 'r') as f:
        segment_labels = json.load(f)
    
    # Update with predictions
    updated_count = 0
    for segment_name, prediction in predictions.items():
        if segment_name in segment_labels:
            old_label = segment_labels[segment_name]
            segment_labels[segment_name] = prediction
            if old_label != prediction:
                print(f"   Updated {segment_name}: {old_label} -> {prediction}")
                updated_count += 1
        else:
            segment_labels[segment_name] = prediction
            print(f"   Added {segment_name}: {prediction}")
            updated_count += 1
    
    # Save updated segment labels
    with open(segment_labels_file, 'w') as f:
        json.dump(segment_labels, f, indent=2)
    
    print(f"   ✅ Updated {updated_count} segment labels")
    print(f"   📊 Prediction distribution:")
    
    # Count predictions
    pred_counts = {}
    for pred in predictions.values():
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    for pred, count in sorted(pred_counts.items()):
        print(f"      Label {pred}: {count} segments")

def process_poses_pipeline(input_path: str, output_dir: str, 
                          config_path: str = '../action_recognition/MMN/config/test/MA52_J.yaml',
                          weights_path: str = '../action_recognition/MMN/checkpoints/MMN_MA52_J.pt',
                          device: str = 'cuda',
                          enable_debug_clips: bool = True,
                          debug_num_clips: int = 3,
                          debug_clip_length: int = 30) -> Dict:
    """
    Complete pipeline for processing poses and predicting micro-actions
    
    Args:
        input_path: Path to input PKL file
        output_dir: Output directory for results
        config_path: Path to MMN config file
        weights_path: Path to MMN model weights
        device: Device to run inference on
        enable_debug_clips: Whether to save debug clips before and after COCO conversion
        debug_num_clips: Number of debug clips to save
        debug_clip_length: Length of each debug clip in frames
        
    Returns:
        Dictionary with pipeline results
    """
    print("=" * 80)
    print("COMPLETE POSE PROCESSING PIPELINE")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load poses from PKL file
    print(f"\n📁 Step 1: Loading poses from {input_path}")
    poses = load_poses_from_pkl(input_path)
    print(f"✅ Loaded {poses.shape[0]} frames")
    
    # Step 2: Filter poses
    print(f"\n🔍 Step 2: Filtering poses")
    filtered_poses = filter_poses(poses, velocity_threshold=0.3, acceleration_threshold=0.5)
    print(f"✅ Filtered to {filtered_poses.shape[0]} frames")
    
    # Step 3: Normalize poses
    print(f"\n🔧 Step 3: Normalizing poses")
    normalized_poses = normalize_poses(filtered_poses, target_scale=1.0, ema_alpha=0.3)
    print(f"✅ Normalized {normalized_poses.shape[0]} frames")
    
    # Step 4: Rotate poses
    print(f"\n🔄 Step 4: Rotating poses")
    rotated_poses = rotate_poses(normalized_poses, target_angle_deg=-178.55)
    print(f"✅ Rotated {rotated_poses.shape[0]} frames")
    
    # Step 5: Convert to COCO format
    print(f"\n🔄 Step 5: Converting to COCO format")
    
    # Save debug clips before COCO conversion
    if enable_debug_clips:
        print(f"\n🔍 Saving debug clips before COCO conversion...")
        save_debug_clips(rotated_poses, output_dir, 'before_coco', debug_num_clips, debug_clip_length)
    
    coco_poses = convert_h36m_to_coco_format(rotated_poses)
    print(f"✅ Converted to COCO format: {coco_poses.shape}")
    
    # Save debug clips after COCO conversion
    if enable_debug_clips:
        print(f"\n🔍 Saving debug clips after COCO conversion...")
        save_debug_clips(coco_poses, output_dir, 'after_coco', debug_num_clips, debug_clip_length)
    
    # Step 6: Segment poses
    print(f"\n📊 Step 6: Segmenting poses")
    segments = segment_poses(coco_poses, velocity_threshold=0.05, acceleration_threshold=0.05)
    print(f"✅ Created {len(segments)} segments")
    
    # Step 7: Create MMN data files
    print(f"\n💾 Step 7: Creating MMN data files")
    mmn_data_dir = create_mmn_data_files(segments, output_dir)
    
    # Step 8: Run MMN inference
    print(f"\n🤖 Step 8: Running MMN inference")
    predictions = run_mmn_inference(config_path, weights_path, mmn_data_dir, output_dir, len(segments))
    
    # Step 9: Update segment labels with predictions
    print(f"\n📊 Step 9: Updating segment labels with predictions")
    update_segment_labels_with_predictions(output_dir, predictions)
    
    # Step 10: Save results
    print(f"\n💾 Step 10: Saving results")
    
    # Save processed segments
    segments_file = os.path.join(output_dir, 'processed_segments.npz')
    segment_data = {}
    for i, segment in enumerate(segments):
        segment_data[f'segment_{i:03d}'] = segment
    segment_data['num_segments'] = len(segments)
    np.savez_compressed(segments_file, **segment_data)
    
    # Create summary
    summary = {
        'input_path': input_path,
        'output_dir': output_dir,
        'mmn_predictions': predictions,
        'pipeline_steps': {
            'original_frames': poses.shape[0],
            'filtered_frames': filtered_poses.shape[0],
            'normalized_frames': normalized_poses.shape[0],
            'rotated_frames': rotated_poses.shape[0],
            'coco_frames': coco_poses.shape[0],
            'num_segments': len(segments)
        }
    }
    
    summary_file = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}")
    print(f"  - Segments: {segments_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Segment labels: {os.path.join(output_dir, 'segment_labels.json')}")
    print(f"  - MMN data: {mmn_data_dir}")
    print(f"  - MMN predictions: {len(predictions)} segments")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"📊 Pipeline Summary:")
    print(f"   Original frames: {summary['pipeline_steps']['original_frames']:,}")
    print(f"   Filtered frames: {summary['pipeline_steps']['filtered_frames']:,}")
    print(f"   Final segments: {summary['pipeline_steps']['num_segments']}")
    print(f"   MMN inference completed")
    
    print(f"\n📁 Output Files:")
    print(f"   - Processed segments: {segments_file}")
    print(f"   - MMN data directory: {mmn_data_dir}")
    print(f"   - MMN predictions: {len(predictions)} segments")
    if enable_debug_clips:
        debug_dir = os.path.join(output_dir, 'debug_clips')
        print(f"   - Debug clips: {debug_dir}")
    
    return summary

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Complete pose processing pipeline for micro-action prediction')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input PKL file (e.g., poses/005_t1_20230519/poses_3D.pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory (e.g., results/005_t1_20230519)')
    parser.add_argument('--config', type=str, 
                       default='../action_recognition/MMN/config/test/MA52_J.yaml',
                       help='Path to MMN config file')
    parser.add_argument('--weights', type=str,
                       default='../action_recognition/MMN/checkpoints/MMN_MA52_J.pt',
                       help='Path to MMN model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--save-debug-clips', action='store_true', default=True,
                       help='Save debug clips before and after COCO conversion (default: enabled)')
    parser.add_argument('--debug-num-clips', type=int, default=3,
                       help='Number of debug clips to save (default: 3)')
    parser.add_argument('--debug-clip-length', type=int, default=30,
                       help='Length of each debug clip in frames (default: 30)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Validate config and weights files exist
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights file {args.weights} does not exist")
        sys.exit(1)
    
    # Run pipeline
    try:
        summary = process_poses_pipeline(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config,
            weights_path=args.weights,
            device=args.device,
            enable_debug_clips=args.save_debug_clips,
            debug_num_clips=args.debug_num_clips,
            debug_clip_length=args.debug_clip_length
        )
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
