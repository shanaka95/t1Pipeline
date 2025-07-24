import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# =======================
# CONFIGURATION VARIABLE
# =======================
# Change this path to your 3D pose file
POSE_FILE_PATH = "../test/3d_poses/example_with_cropping.npy"

# Output settings
OUTPUT_DIR = "visualizations/poses"
FPS = 30

# Frame processing settings (to handle large files)
MAX_FRAMES = 1000        # Maximum number of frames to animate (set to None for all frames)
FRAME_SKIP = 1          # Take every Nth frame (1 = all frames, 10 = every 10th frame)
START_FRAME = 0          # Which frame to start from (0 = beginning)

def animate_skeleton_3d(poses, output_path, title="3D Skeleton Animation", fps=30):
    """
    Create a 2D skeleton animation from 3D pose data (side view).
    
    Parameters:
    poses (numpy array): Shape (T, 17, 3) - skeleton poses over time
    output_path (str): Output path for the GIF
    title (str): Title for the animation
    fps (int): Frames per second for the animation
    """
    print(f"🎬 Creating animation with {poses.shape[0]} frames...")
    
    # Human3.6M skeleton connections (joint_start -> joint_end)
    connections = [
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
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.5, 1.5)  # X axis (left-right)
    ax.set_ylim(-1.5, 1.5)  # Y axis (up-down, flipped for proper orientation)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Left-Right (X)')
    ax.set_ylabel('Height (-Y)')
    
    # Create line objects for each connection
    lines = []
    for _ in connections:
        line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
        lines.append(line)
    
    # Add frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    def init():
        """Initialize animation."""
        for line in lines:
            line.set_data([], [])
        frame_text.set_text('')
        return lines + [frame_text]
    
    def animate(frame_idx):
        """Update function for each frame."""
        if frame_idx >= poses.shape[0]:
            return lines + [frame_text]
        
        # Get current frame poses (17 joints, 3 coordinates each)
        current_frame = poses[frame_idx, :, :]
        x_coords = current_frame[:, 0]  # X coordinates (left-right)
        y_coords = -current_frame[:, 1]  # -Y coordinates (flip Y for proper orientation)
        
        # Update skeleton connections
        for i, (start_joint, end_joint) in enumerate(connections):
            x_data = [x_coords[start_joint], x_coords[end_joint]]
            y_data = [y_coords[start_joint], y_coords[end_joint]]
            lines[i].set_data(x_data, y_data)
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame_idx + 1}/{poses.shape[0]}')
        
        return lines + [frame_text]
    
    # Create animation
    interval = 1000 // fps
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=poses.shape[0],
                                 interval=interval, blit=True, repeat=True)
    
    # Save as GIF
    print(f"💾 Saving animation to: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"✅ Animation saved successfully!")

def process_poses(poses, start_frame=0, max_frames=None, frame_skip=1):
    """
    Process poses by applying frame selection, limiting, and downsampling.
    
    Parameters:
    poses (numpy array): Original poses (T, 17, 3)
    start_frame (int): Starting frame index
    max_frames (int): Maximum number of frames to keep (after skipping)
    frame_skip (int): Take every Nth frame
    
    Returns:
    numpy array: Processed poses
    """
    # Start from specified frame
    poses = poses[start_frame:]
    
    # Apply frame skipping (downsampling)
    poses = poses[::frame_skip]
    
    # Limit maximum frames
    if max_frames is not None and poses.shape[0] > max_frames:
        poses = poses[:max_frames]
    
    return poses

def load_pose_data(file_path):
    """Load pose data from pickle or npy file."""
    if file_path.endswith('.pkl'):
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # If it's a list of segments, take the first one
        if isinstance(data, list) and len(data) > 0:
            poses = data[0]
        else:
            poses = data
    elif file_path.endswith('.npy'):
        poses = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return poses

def create_pose_segment_visualizations(
    pose_segments,
    output_dir="visualizations/poses",
    segment_indices=None,
    fps=30,
    max_frames_per_segment=500,
    frame_skip=1,
    prefix="segment"
):
    """
    Create animations for multiple pose segments.
    
    Parameters:
    pose_segments (list): List of pose segments (each is numpy array of shape (T, 17, 3))
    output_dir (str): Output directory for GIF files
    segment_indices (list): List of segment indices to visualize (None for all)
    fps (int): Frames per second for animations
    max_frames_per_segment (int): Maximum frames per segment
    frame_skip (int): Take every Nth frame
    prefix (str): Prefix for output filenames
    """
    if not pose_segments:
        print("⚠️ No pose segments provided for visualization")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which segments to visualize
    if segment_indices is None:
        segments_to_visualize = list(range(len(pose_segments)))
    else:
        segments_to_visualize = [i for i in segment_indices if 0 <= i < len(pose_segments)]
    
    print(f"\n🎨 Creating visualizations for {len(segments_to_visualize)} pose segments...")
    print(f"Output directory: {output_dir}")
    print(f"Settings: fps={fps}, max_frames={max_frames_per_segment}, frame_skip={frame_skip}")
    
    successful_animations = 0
    
    for i, segment_idx in enumerate(segments_to_visualize):
        segment = pose_segments[segment_idx]
        
        # Validate segment
        if not isinstance(segment, np.ndarray) or len(segment.shape) != 3:
            print(f"⚠️ Skipping invalid segment {segment_idx}")
            continue
            
        if segment.shape[1] != 17 or segment.shape[2] != 3:
            print(f"⚠️ Skipping segment {segment_idx} with wrong shape: {segment.shape}")
            continue
        
        # Create filename
        output_filename = f"{prefix}_{segment_idx:03d}_frames{segment.shape[0]}.gif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create title
        title = f"Pose Segment {segment_idx} ({segment.shape[0]} frames)"
        
        print(f"\n📹 Processing segment {i+1}/{len(segments_to_visualize)}: {segment_idx}")
        print(f"   Original frames: {segment.shape[0]}")
        
        # Process poses for this segment
        processed_segment = process_poses(segment, 0, max_frames_per_segment, frame_skip)
        processed_frames = processed_segment.shape[0]
        print(f"   Processed frames: {processed_frames}")
        
        if processed_frames == 0:
            print(f"   ⚠️ No frames to process, skipping...")
            continue
        
        # Create animation
        try:
            animate_skeleton_3d(processed_segment, output_path, title, fps)
            successful_animations += 1
        except Exception as e:
            print(f"   ❌ Error creating animation for segment {segment_idx}: {e}")
    
    print(f"\n✅ Visualization completed!")
    print(f"   Successfully created: {successful_animations}/{len(segments_to_visualize)} animations")
    print(f"   Saved in: {output_dir}")

def main():
    """Main function to create skeleton animation."""
    
    # Check if pose file exists
    if not os.path.exists(POSE_FILE_PATH):
        print(f"❌ Error: Pose file not found at {POSE_FILE_PATH}")
        print("Please check the POSE_FILE_PATH variable at the top of this script.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load pose data
    print(f"📁 Loading poses from: {POSE_FILE_PATH}")
    try:
        poses = load_pose_data(POSE_FILE_PATH)
        print(f"✅ Loaded poses with shape: {poses.shape}")
    except Exception as e:
        print(f"❌ Error loading pose file: {e}")
        return
    
    # Validate shape
    if len(poses.shape) != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
        print(f"❌ Error: Expected pose shape (T, 17, 3), got {poses.shape}")
        return
    
    # Process poses (downsample and limit frames)
    original_frames = poses.shape[0]
    poses = process_poses(poses, START_FRAME, MAX_FRAMES, FRAME_SKIP)
    processed_frames = poses.shape[0]
    
    print(f"📊 Frame processing:")
    print(f"   Original frames: {original_frames}")
    print(f"   Start frame: {START_FRAME}")
    print(f"   Frame skip: every {FRAME_SKIP} frame(s)")
    print(f"   Max frames limit: {MAX_FRAMES}")
    print(f"   Final frames: {processed_frames}")
    
    # Estimate memory usage and warn if too large
    estimated_mb = (processed_frames * 17 * 3 * 8) / (1024 * 1024)  # rough estimate
    if processed_frames > 2000:
        print(f"⚠️  Warning: {processed_frames} frames might be too many for GIF creation")
        print(f"   Consider reducing MAX_FRAMES or increasing FRAME_SKIP")
        print(f"   Estimated memory usage: ~{estimated_mb:.1f} MB")
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(POSE_FILE_PATH))[0]
    suffix = f"_skip{FRAME_SKIP}_frames{processed_frames}"
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}_animation.gif")
    
    # Create animation
    animate_skeleton_3d(poses, output_path, f"3D Skeleton Animation - {base_name}", FPS)

if __name__ == "__main__":
    main()
