#!/usr/bin/env python3
"""
Script to visualize random segments from a specified cluster using clustering data and 3D poses.

This script:
1. Takes a clustering data file and poses root directory as arguments
2. Identifies 5 random segments from a specified cluster
3. Loads the corresponding 3D pose data for those segments
4. Creates dual-view GIF visualizations for each segment

Usage:
    python visualize_cluster_segments.py --clustering_data <path> --poses_dir <path> --cluster_id <id> [--output_dir <path>] [--num_segments <n>] [--fps <fps>]

Example:
    python visualize_cluster_segments.py --clustering_data kmeans_top5_labels_results.pkl --poses_dir poses --cluster_id 15
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import random
from typing import List, Tuple, Dict, Any

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

def load_clustering_data(clustering_file: str) -> Dict[str, Any]:
    """
    Load clustering data from pickle file.
    
    Args:
        clustering_file: Path to clustering data pickle file
        
    Returns:
        Dictionary containing clustering results
    """
    print(f"📊 Loading clustering data from: {clustering_file}")
    
    if not os.path.exists(clustering_file):
        raise FileNotFoundError(f"Clustering file not found: {clustering_file}")
    
    with open(clustering_file, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Handle different data structures
    if 'cluster_summary' in raw_data:
        clustering_data = raw_data['cluster_summary']
        print(f"✅ Loaded clustering data with {len(clustering_data)} clusters from 'cluster_summary'")
    elif isinstance(raw_data, dict) and all(isinstance(k, int) for k in raw_data.keys()):
        clustering_data = raw_data
        print(f"✅ Loaded clustering data with {len(clustering_data)} clusters (direct format)")
    else:
        # Check if it's the format we initially saw in the output
        clustering_data = raw_data
        print(f"✅ Loaded clustering data (attempting with original format)")
    
    return clustering_data

def get_random_segments_from_cluster(clustering_data: Dict[str, Any], 
                                   cluster_id: int, 
                                   num_segments: int = 5,
                                   poses_dir: str = None) -> List[Tuple[str, int]]:
    """
    Get random segments from a specified cluster.
    
    Args:
        clustering_data: Dictionary containing clustering results
        cluster_id: ID of the cluster to sample from
        num_segments: Number of segments to randomly select
        poses_dir: Optional poses directory to prioritize available videos
        
    Returns:
        List of tuples (video_name, segment_id)
    """
    print(f"🎯 Getting {num_segments} random segments from cluster {cluster_id}")
    
    if cluster_id not in clustering_data:
        raise ValueError(f"Cluster {cluster_id} not found in clustering data")
    
    cluster_info = clustering_data[cluster_id]
    videos = cluster_info['videos']
    
    # Collect all (video_name, segment_id) pairs
    all_segments = []
    available_segments = []
    
    for video_name, segment_ids in videos.items():
        for segment_id in segment_ids:
            segment_tuple = (video_name, segment_id)
            all_segments.append(segment_tuple)
            
            # Check if poses are available for this video
            if poses_dir:
                pose_file = os.path.join(poses_dir, video_name, 'poses_3D.pkl')
                if os.path.exists(pose_file):
                    available_segments.append(segment_tuple)
    
    print(f"   📈 Cluster {cluster_id} contains {len(all_segments)} total segments from {len(videos)} videos")
    print(f"   📊 Cluster info: {cluster_info['total_segments']} segments, {cluster_info['num_videos']} videos")
    
    if poses_dir and available_segments:
        print(f"   📁 Found {len(available_segments)} segments with available pose data")
        # Prioritize available segments
        if len(available_segments) >= num_segments:
            print(f"   ✨ Using only segments with available pose data")
            selected_segments = random.sample(available_segments, num_segments)
        else:
            print(f"   ⚠️  Only {len(available_segments)} segments with poses available, using all + random others")
            remaining_needed = num_segments - len(available_segments)
            unavailable_segments = [seg for seg in all_segments if seg not in available_segments]
            additional_segments = random.sample(unavailable_segments, min(remaining_needed, len(unavailable_segments)))
            selected_segments = available_segments + additional_segments
    else:
        # Original behavior - purely random selection
        if len(all_segments) < num_segments:
            print(f"   ⚠️  Only {len(all_segments)} segments available, using all of them")
            selected_segments = all_segments
        else:
            selected_segments = random.sample(all_segments, num_segments)
    
    print(f"   ✅ Selected {len(selected_segments)} segments:")
    for i, (video_name, segment_id) in enumerate(selected_segments):
        pose_available = "📁" if poses_dir and os.path.exists(os.path.join(poses_dir, video_name, 'poses_3D.pkl')) else "❓"
        print(f"      {i+1}. {pose_available} Video: {video_name}, Segment: {segment_id}")
    
    return selected_segments

def load_pose_segment(poses_dir: str, video_name: str, segment_id: int) -> np.ndarray:
    """
    Load 3D pose data for a specific segment.
    
    Args:
        poses_dir: Root directory containing pose data
        video_name: Name of the video
        segment_id: ID of the segment within the video
        
    Returns:
        3D pose data with shape (243, 17, 3)
    """
    pose_file = os.path.join(poses_dir, video_name, 'poses_3D.pkl')
    
    print(f"   📁 Loading pose data from: {pose_file}")
    
    if not os.path.exists(pose_file):
        print(f"   ❌ Pose file not found: {pose_file}")
        print(f"   ⚠️  This is expected if not all video poses are available locally")
        return None
    
    try:
        with open(pose_file, 'rb') as f:
            poses_list = pickle.load(f)
        
        if segment_id >= len(poses_list):
            print(f"   ❌ Segment {segment_id} not found in pose data (only {len(poses_list)} segments available)")
            return None
        
        # Get the segment data and remove batch dimension
        segment_data = poses_list[segment_id]
        if len(segment_data.shape) == 4 and segment_data.shape[0] == 1:
            segment_data = segment_data.squeeze(0)  # Remove batch dimension: (1, 243, 17, 3) -> (243, 17, 3)
        
        print(f"   ✅ Loaded pose segment with shape: {segment_data.shape}")
        return segment_data
        
    except Exception as e:
        print(f"   ❌ Error loading pose data: {e}")
        return None

def create_dual_view_animation(poses: np.ndarray, 
                             output_path: str, 
                             title: str = "Dual View Skeleton Animation", 
                             fps: int = 15) -> None:
    """
    Create a dual-view skeleton animation with side view (2D) and isometric view (3D).
    
    Args:
        poses: 3D pose data with shape (T, 17, 3)
        output_path: Output path for the GIF
        title: Title for the animation
        fps: Frames per second for the animation
    """
    print(f"   🎬 Creating dual-view animation with {poses.shape[0]} frames...")
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # --- Setup Side View (2D) ---
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Side View (2D)', fontsize=14)
    ax1.set_xlabel('Left-Right (X)')
    ax1.set_ylabel('Height (-Y)')
    
    # Create line objects for side view
    lines_2d = []
    for _ in H36M_CONNECTIONS:
        line, = ax1.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
        lines_2d.append(line)
    
    # --- Setup Isometric View (3D) ---
    ax2.remove()  # Remove the 2D axis
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
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
    
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)
    ax2.set_xlabel('X (Left-Right)')
    ax2.set_ylabel('Y (Height)')
    ax2.set_zlabel('Z (Forward-Back)')
    ax2.set_title('Isometric View (3D)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Set isometric viewing angle
    ax2.view_init(elev=15, azim=45)
    
    # Create line objects for 3D view
    lines_3d = []
    for _ in H36M_CONNECTIONS:
        line, = ax2.plot([], [], [], 'royalblue', linewidth=3, alpha=0.8)
        lines_3d.append(line)
    
    # Create scatter plot for joints in 3D
    joint_scatter = ax2.scatter([], [], [], c='red', s=60, alpha=0.9)
    
    # Add frame counter
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def init():
        """Initialize animation."""
        # Initialize 2D lines
        for line in lines_2d:
            line.set_data([], [])
        
        # Initialize 3D lines
        for line in lines_3d:
            line.set_data([], [])
            line.set_3d_properties([])
        
        # Initialize scatter plot
        joint_scatter._offsets3d = ([], [], [])
        frame_text.set_text('')
        
        return lines_2d + lines_3d + [frame_text]
    
    def animate_frame(frame_idx):
        """Update function for each frame."""
        if frame_idx >= poses.shape[0]:
            return lines_2d + lines_3d + [frame_text]
        
        # Get current frame poses
        current_frame = poses[frame_idx, :, :]
        x_coords = current_frame[:, 0]
        y_coords = current_frame[:, 1]
        z_coords = current_frame[:, 2]
        
        # Update 2D side view (flip Y for proper orientation)
        y_coords_2d = -y_coords
        for i, (start_joint, end_joint) in enumerate(H36M_CONNECTIONS):
            x_data = [x_coords[start_joint], x_coords[end_joint]]
            y_data = [y_coords_2d[start_joint], y_coords_2d[end_joint]]
            lines_2d[i].set_data(x_data, y_data)
        
        # Update 3D isometric view
        for i, (start_joint, end_joint) in enumerate(H36M_CONNECTIONS):
            lines_3d[i].set_data([x_coords[start_joint], x_coords[end_joint]], 
                                [y_coords[start_joint], y_coords[end_joint]])
            lines_3d[i].set_3d_properties([z_coords[start_joint], z_coords[end_joint]])
        
        # Update joint positions in 3D
        joint_scatter._offsets3d = (x_coords, y_coords, z_coords)
        
        # Update frame counter
        progress = (frame_idx + 1) / poses.shape[0] * 100
        frame_text.set_text(f'Frame: {frame_idx + 1}/{poses.shape[0]} ({progress:.1f}%)')
        
        return lines_2d + lines_3d + [frame_text]
    
    # Create animation
    interval = 1000 // fps
    anim = animation.FuncAnimation(fig, animate_frame, init_func=init, frames=poses.shape[0],
                                 interval=interval, blit=False, repeat=True)
    
    # Save as GIF
    print(f"   💾 Saving dual-view animation to: {output_path}")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"   ✅ Animation saved successfully!")
    except Exception as e:
        print(f"   ❌ Error saving animation: {e}")
    finally:
        plt.close(fig)

def visualize_cluster_segments(clustering_file: str,
                             poses_dir: str,
                             cluster_id: int,
                             output_dir: str = None,
                             num_segments: int = 5,
                             fps: int = 15) -> None:
    """
    Main function to visualize random segments from a specified cluster.
    
    Args:
        clustering_file: Path to clustering data pickle file
        poses_dir: Root directory containing pose data
        cluster_id: ID of the cluster to visualize
        output_dir: Output directory for visualizations (default: ./cluster_visualizations)
        num_segments: Number of segments to visualize (default: 5)
        fps: Frames per second for animations (default: 15)
    """
    print(f"🚀 Starting cluster segment visualization...")
    print(f"   📊 Clustering file: {clustering_file}")
    print(f"   📁 Poses directory: {poses_dir}")
    print(f"   🎯 Target cluster: {cluster_id}")
    print(f"   📈 Number of segments: {num_segments}")
    print(f"   🎬 FPS: {fps}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = "cluster_visualizations"
    
    # Create output directory
    cluster_output_dir = os.path.join(output_dir, f"cluster_{cluster_id:03d}")
    os.makedirs(cluster_output_dir, exist_ok=True)
    print(f"   📁 Output directory: {cluster_output_dir}")
    
    # Load clustering data
    try:
        clustering_data = load_clustering_data(clustering_file)
    except Exception as e:
        print(f"❌ Failed to load clustering data: {e}")
        return
    
    # Get random segments from cluster
    try:
        selected_segments = get_random_segments_from_cluster(clustering_data, cluster_id, num_segments, poses_dir)
    except Exception as e:
        print(f"❌ Failed to get segments from cluster: {e}")
        return
    
    if not selected_segments:
        print(f"❌ No segments found in cluster {cluster_id}")
        return
    
    # Process each selected segment
    successful_visualizations = 0
    
    print(f"\n🎬 Processing {len(selected_segments)} segments...")
    
    for i, (video_name, segment_id) in enumerate(selected_segments):
        print(f"\n📹 Processing segment {i+1}/{len(selected_segments)}: {video_name} - Segment {segment_id}")
        
        # Load pose data for this segment
        pose_data = load_pose_segment(poses_dir, video_name, segment_id)
        
        if pose_data is None:
            print(f"   ⚠️  Skipping segment due to missing or invalid pose data")
            continue
        
        # Create output filename
        output_filename = f"cluster_{cluster_id:03d}_seg_{i+1:02d}_{video_name}_segment_{segment_id}.gif"
        output_path = os.path.join(cluster_output_dir, output_filename)
        
        # Create title
        title = f"Cluster {cluster_id} - Segment {i+1}/{len(selected_segments)} ({video_name})"
        
        # Create visualization
        try:
            create_dual_view_animation(pose_data, output_path, title, fps)
            successful_visualizations += 1
        except Exception as e:
            print(f"   ❌ Failed to create visualization: {e}")
            continue
    
    # Summary
    print(f"\n🎉 Visualization completed!")
    print(f"   ✅ Successfully created {successful_visualizations}/{len(selected_segments)} visualizations")
    print(f"   📁 All visualizations saved in: {cluster_output_dir}")
    
    if successful_visualizations == 0:
        print(f"   ⚠️  No visualizations were created. This is expected if pose data is not available locally.")
        print(f"   💡 The script is ready to run when full pose data is available.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize random segments from a specified cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_cluster_segments.py --clustering_data kmeans_top5_labels_results.pkl --poses_dir poses --cluster_id 15
  python visualize_cluster_segments.py --clustering_data results.pkl --poses_dir /path/to/poses --cluster_id 42 --num_segments 3 --fps 20
        """
    )
    
    parser.add_argument('--clustering_data', '-c', type=str, required=True,
                       help='Path to clustering data pickle file (e.g., kmeans_top5_labels_results.pkl)')
    parser.add_argument('--poses_dir', '-p', type=str, required=True,
                       help='Root directory containing pose data (e.g., poses)')
    parser.add_argument('--cluster_id', '-i', type=int, required=True,
                       help='ID of the cluster to visualize (0-99)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for visualizations (default: ./cluster_visualizations)')
    parser.add_argument('--num_segments', '-n', type=int, default=5,
                       help='Number of segments to visualize (default: 5)')
    parser.add_argument('--fps', '-f', type=int, default=15,
                       help='Frames per second for animations (default: 15)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible segment selection')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Validate inputs
    if not os.path.exists(args.clustering_data):
        print(f"❌ Clustering data file not found: {args.clustering_data}")
        sys.exit(1)
    
    if not os.path.exists(args.poses_dir):
        print(f"❌ Poses directory not found: {args.poses_dir}")
        sys.exit(1)
    
    # Run visualization
    visualize_cluster_segments(
        clustering_file=args.clustering_data,
        poses_dir=args.poses_dir,
        cluster_id=args.cluster_id,
        output_dir=args.output_dir,
        num_segments=args.num_segments,
        fps=args.fps
    )

if __name__ == "__main__":
    main() 