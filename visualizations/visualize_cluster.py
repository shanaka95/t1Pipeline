import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

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

def visualize_cluster_poses(clustering_poses_dir, num_vis_per_cluster=5, fps=15):
    """
    Create dual-view visualizations for poses in each cluster.
    
    Parameters:
    clustering_poses_dir: Directory containing clustering results and cluster subdirectories
    num_vis_per_cluster: Number of visualizations to create per cluster (default: 5)
    fps: Frames per second for animations (default: 15)
    """
    print(f"🎯 Starting cluster pose visualization...")
    print(f"📂 Clustering directory: {clustering_poses_dir}")
    print(f"📊 Visualizations per cluster: {num_vis_per_cluster}")
    
    # Load clustering results
    clustering_results_path = os.path.join(clustering_poses_dir, 'kmeans_clustering_results.pkl')
    if not os.path.exists(clustering_results_path):
        # Try parent directory
        parent_dir = os.path.dirname(clustering_poses_dir)
        clustering_results_path = os.path.join(parent_dir, 'clustering_results', 'kmeans_clustering_results.pkl')
    
    if not os.path.exists(clustering_results_path):
        raise FileNotFoundError(f"Cannot find kmeans_clustering_results.pkl in {clustering_poses_dir} or parent directories")
    
    print(f"📊 Loading clustering results from: {clustering_results_path}")
    with open(clustering_results_path, 'rb') as f:
        clustering_results = pickle.load(f)
    
    n_clusters = clustering_results['n_clusters']
    print(f"🎯 Found {n_clusters} clusters")
    
    # Create visualizations directory
    vis_output_dir = os.path.join(clustering_poses_dir, 'visualizations')
    os.makedirs(vis_output_dir, exist_ok=True)
    print(f"📁 Saving visualizations to: {vis_output_dir}")
    
    total_visualizations = 0
    
    # Process each cluster
    for cluster_id in range(n_clusters):
        cluster_dir = os.path.join(clustering_poses_dir, f"cluster_{cluster_id:03d}")
        
        if not os.path.exists(cluster_dir):
            print(f"⚠️ Cluster directory not found: {cluster_dir}")
            continue
        
        poses_file = os.path.join(cluster_dir, "poses.pkl")
        if not os.path.exists(poses_file):
            print(f"⚠️ Poses file not found: {poses_file}")
            continue
        
        print(f"\n🎯 Processing Cluster {cluster_id}")
        
        # Load cluster poses
        with open(poses_file, 'rb') as f:
            cluster_data = pickle.load(f)
        
        poses_list = cluster_data['poses']
        metadata = cluster_data['metadata']
        
        print(f"   📊 Found {len(poses_list)} poses in cluster")
        
        # Create cluster visualization directory
        cluster_vis_dir = os.path.join(vis_output_dir, f"cluster_{cluster_id:03d}")
        os.makedirs(cluster_vis_dir, exist_ok=True)
        
        # Create visualizations for this cluster (up to num_vis_per_cluster)
        num_to_visualize = min(num_vis_per_cluster, len(poses_list))
        
        for i in range(num_to_visualize):
            pose_data = poses_list[i]
            if len(pose_data.shape) == 4:  # Remove batch dimension if present
                pose_data = pose_data.squeeze(0)
            
            # Get metadata for this pose
            video_name = metadata[i]['video_name'] if i < len(metadata) else f"unknown_{i}"
            sequence_id = metadata[i]['sequence_id'] if i < len(metadata) else i
            
            # Create output filename
            output_filename = f"cluster_{cluster_id:03d}_pose_{i:02d}_{video_name}_seq_{sequence_id}.gif"
            output_path = os.path.join(cluster_vis_dir, output_filename)
            
            # Create title
            title = f"Cluster {cluster_id} - Pose {i+1}/{num_to_visualize} ({video_name})"
            
            print(f"   🎬 Creating visualization {i+1}/{num_to_visualize}: {output_filename}")
            
            # Create dual-view animation
            try:
                create_dual_view_animation(pose_data, output_path, title, fps)
                total_visualizations += 1
            except Exception as e:
                print(f"   ❌ Failed to create visualization: {e}")
        
        print(f"   ✅ Completed cluster {cluster_id}: {num_to_visualize} visualizations created")
    
    print(f"\n🎉 Cluster visualization completed!")
    print(f"📊 Total visualizations created: {total_visualizations}")
    print(f"📁 All visualizations saved in: {vis_output_dir}")
    
    return vis_output_dir
