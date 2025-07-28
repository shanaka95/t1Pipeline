#!/usr/bin/env python3
"""
Script to visualize random segments from each cluster using the master poses directory
and clustering results from top5_labels analysis.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from pathlib import Path

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

def create_dual_view_animation(poses, output_path, title="Dual View Skeleton Animation", fps=15):
    """
    Create a dual-view skeleton animation with side view (2D) and isometric view (3D).
    
    Parameters:
    poses (numpy array): Shape (T, 17, 3) - skeleton poses over time
    output_path (str): Output path for the GIF
    title (str): Title for the animation
    fps (int): Frames per second for the animation
    """
    print(f"🎬 Creating dual-view animation with {poses.shape[0]} frames...")
    
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
    print(f"💾 Saving dual-view animation to: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"✅ Dual-view animation saved successfully!")

def load_pose_data(video_dir):
    """
    Load 3D pose data from a video directory.
    
    Parameters:
    video_dir (str): Path to video directory containing poses_3D.pkl
    
    Returns:
    numpy array: 3D poses or None if loading fails
    """
    poses_file = os.path.join(video_dir, "poses_3D.pkl")
    if not os.path.exists(poses_file):
        return None
    
    try:
        with open(poses_file, 'rb') as f:
            poses_3d = pickle.load(f)
        return poses_3d
    except Exception as e:
        print(f"❌ Error loading poses from {poses_file}: {e}")
        return None

def find_optimal_cluster_number(clustering_base_dir):
    """
    Find the optimal cluster number from the previous analysis.
    
    Parameters:
    clustering_base_dir (str): Base directory containing clustering results
    
    Returns:
    int: Optimal cluster number
    """
    # Try to find the analysis results
    analysis_files = [
        "top5_clustering_analysis.json",
        "top5_clustering_executive_summary.txt"
    ]
    
    optimal_clusters = 99  # Default from our previous analysis
    
    for analysis_file in analysis_files:
        file_path = os.path.join(clustering_base_dir, "..", analysis_file)
        if os.path.exists(file_path):
            if analysis_file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Find the best scoring configuration
                    best_config = max(data, key=lambda x: x['silhouette_score'])
                    optimal_clusters = best_config['cluster_number']
                    print(f"✅ Found optimal cluster number: {optimal_clusters} (score: {best_config['silhouette_score']:.6f})")
                    break
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
    
    return optimal_clusters

def visualize_clusters_with_poses_dir(
    poses_master_dir="/home/janus/iwso-datasets/t1-body-poses-final/",
    clustering_base_dir="/home/shanaka/Desktop/thesis/pipeline-final/test/clustering_info_with_top5_labels",
    output_dir="cluster_visualizations_output",
    num_segments_per_cluster=5,
    fps=15
):
    """
    Create visualizations for random segments from each cluster using master poses directory.
    
    Parameters:
    poses_master_dir (str): Master directory containing all pose directories
    clustering_base_dir (str): Base directory containing clustering results for different cluster numbers
    output_dir (str): Output directory for visualizations
    num_segments_per_cluster (int): Number of random segments to visualize per cluster
    fps (int): Frames per second for animations
    """
    print(f"🎯 Starting cluster visualization with poses directory...")
    print(f"📂 Poses master directory: {poses_master_dir}")
    print(f"📊 Clustering base directory: {clustering_base_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎬 Segments per cluster: {num_segments_per_cluster}")
    
    # Find optimal cluster number
    optimal_clusters = find_optimal_cluster_number(clustering_base_dir)
    clustering_dir = os.path.join(clustering_base_dir, str(optimal_clusters))
    
    if not os.path.exists(clustering_dir):
        raise FileNotFoundError(f"Clustering directory not found: {clustering_dir}")
    
    # Load clustering summary
    clustering_summary_path = os.path.join(clustering_dir, "clustering_summary.json")
    if not os.path.exists(clustering_summary_path):
        raise FileNotFoundError(f"Clustering summary not found: {clustering_summary_path}")
    
    print(f"📊 Loading clustering summary from: {clustering_summary_path}")
    with open(clustering_summary_path, 'r') as f:
        clustering_summary = json.load(f)
    
    n_clusters = clustering_summary['n_clusters']
    cluster_summary = clustering_summary['cluster_summary']
    
    print(f"🎯 Found {n_clusters} clusters")
    print(f"📊 Total segments: {clustering_summary['total_segments']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_visualizations = 0
    successful_clusters = 0
    
    # Process each cluster
    for cluster_id in range(n_clusters):
        cluster_key = str(cluster_id)
        if cluster_key not in cluster_summary:
            print(f"⚠️ Cluster {cluster_id} not found in summary")
            continue
        
        cluster_data = cluster_summary[cluster_key]
        cluster_videos = cluster_data['videos']
        
        print(f"\n🎯 Processing Cluster {cluster_id}")
        print(f"   📊 Videos in cluster: {len(cluster_videos)}")
        print(f"   📊 Total segments: {cluster_data['total_segments']}")
        
        # Collect all segments from this cluster
        all_segments = []
        for video_name, segments in cluster_videos.items():
            for segment_id in segments:
                all_segments.append({
                    'video_name': video_name,
                    'segment_id': segment_id
                })
        
        if not all_segments:
            print(f"   ⚠️ No segments found in cluster {cluster_id}")
            continue
        
        # Randomly select segments for visualization
        num_to_visualize = min(num_segments_per_cluster, len(all_segments))
        selected_segments = random.sample(all_segments, num_to_visualize)
        
        print(f"   🎲 Randomly selected {num_to_visualize} segments from {len(all_segments)} total")
        
        # Create cluster visualization directory
        cluster_vis_dir = os.path.join(output_dir, f"cluster_{cluster_id:03d}")
        os.makedirs(cluster_vis_dir, exist_ok=True)
        
        cluster_success = 0
        
        # Create visualizations for selected segments
        for i, segment_info in enumerate(selected_segments):
            video_name = segment_info['video_name']
            segment_id = segment_info['segment_id']
            
            # Find pose directory
            video_dir = os.path.join(poses_master_dir, video_name)
            if not os.path.exists(video_dir):
                print(f"   ⚠️ Video directory not found: {video_dir}")
                continue
            
            # Load pose data
            poses_3d = load_pose_data(video_dir)
            if poses_3d is None:
                print(f"   ⚠️ Failed to load poses from {video_dir}")
                continue
            
            # Extract specific segment (assuming segment_id is frame index or similar)
            # Note: You may need to adjust this based on how segment_id is defined
            try:
                if isinstance(segment_id, int) and segment_id < len(poses_3d):
                    # For single frame, create a short sequence around it
                    start_frame = max(0, segment_id - 15)
                    end_frame = min(len(poses_3d), segment_id + 16)
                    segment_poses = poses_3d[start_frame:end_frame]
                else:
                    # Use entire pose sequence or first 30 frames
                    segment_poses = poses_3d[:30] if len(poses_3d) > 30 else poses_3d
                
                if len(segment_poses) == 0:
                    print(f"   ⚠️ Empty segment for {video_name}:{segment_id}")
                    continue
                
                # Create output filename
                output_filename = f"cluster_{cluster_id:03d}_seg_{i+1:02d}_{video_name}_frame_{segment_id}.gif"
                output_path = os.path.join(cluster_vis_dir, output_filename)
                
                # Create title
                title = f"Cluster {cluster_id} - Segment {i+1}/{num_to_visualize}\n{video_name} (Frame {segment_id})"
                
                print(f"   🎬 Creating visualization {i+1}/{num_to_visualize}: {video_name}:{segment_id}")
                
                # Create dual-view animation
                create_dual_view_animation(segment_poses, output_path, title, fps)
                total_visualizations += 1
                cluster_success += 1
                
            except Exception as e:
                print(f"   ❌ Failed to create visualization for {video_name}:{segment_id}: {e}")
        
        if cluster_success > 0:
            successful_clusters += 1
            print(f"   ✅ Completed cluster {cluster_id}: {cluster_success}/{num_to_visualize} visualizations created")
        else:
            print(f"   ❌ No successful visualizations for cluster {cluster_id}")
    
    print(f"\n🎉 Cluster visualization completed!")
    print(f"🎯 Successful clusters: {successful_clusters}/{n_clusters}")
    print(f"📊 Total visualizations created: {total_visualizations}")
    print(f"📁 All visualizations saved in: {output_dir}")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "visualization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Cluster Visualization Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Generated on: {np.datetime64('now')}\n")
        f.write(f"Poses master directory: {poses_master_dir}\n")
        f.write(f"Clustering directory: {clustering_dir}\n")
        f.write(f"Optimal cluster number: {optimal_clusters}\n")
        f.write(f"Total clusters: {n_clusters}\n")
        f.write(f"Successful clusters: {successful_clusters}\n")
        f.write(f"Segments per cluster: {num_segments_per_cluster}\n")
        f.write(f"Total visualizations created: {total_visualizations}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    print(f"📝 Summary saved to: {summary_file}")
    
    return output_dir

def main():
    """Main function to run cluster visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize random segments from each cluster")
    parser.add_argument("--poses_dir", default="/home/janus/iwso-datasets/t1-body-poses-final/",
                       help="Master directory containing all pose directories")
    parser.add_argument("--clustering_dir", default="/home/shanaka/Desktop/thesis/pipeline-final/test/clustering_info_with_top5_labels",
                       help="Base directory containing clustering results")
    parser.add_argument("--output_dir", default="cluster_visualizations_output",
                       help="Output directory for visualizations")
    parser.add_argument("--num_segments", type=int, default=5,
                       help="Number of random segments to visualize per cluster")
    parser.add_argument("--fps", type=int, default=15,
                       help="Frames per second for animations")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        output_dir = visualize_clusters_with_poses_dir(
            poses_master_dir=args.poses_dir,
            clustering_base_dir=args.clustering_dir,
            output_dir=args.output_dir,
            num_segments_per_cluster=args.num_segments,
            fps=args.fps
        )
        print(f"\n🎉 Visualization completed successfully!")
        print(f"📁 Check the results in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 