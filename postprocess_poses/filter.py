import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PoseGlitchDetector:
    """
    Detects and visualizes glitches in pose sequences.
    
    A glitch is defined as abnormal fast movement that could be caused by:
    - Tracking errors
    - Noise in pose estimation
    - Sudden jumps in joint positions
    """
    
    def __init__(self, velocity_threshold=0.3, acceleration_threshold=0.5):
        """
        Initialize the glitch detector.
        
        Args:
            velocity_threshold (float): Threshold for detecting high velocity movements
            acceleration_threshold (float): Threshold for detecting high acceleration movements
        """
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
        # H36M joint names for reference
        self.joint_names = [
            'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank',
            'belly', 'neck', 'nose', 'head', 'lsho', 'lelb', 'lwri',
            'rsho', 'relb', 'rwri'
        ]
        
        # H36M skeleton connections for visualization
        self.skeleton_connections = [
            (0, 1),   # root -> right hip
            (1, 2),   # right hip -> right knee  
            (2, 3),   # right knee -> right ankle
            (0, 4),   # root -> left hip
            (4, 5),   # left hip -> left knee
            (5, 6),   # left knee -> left ankle
            (0, 7),   # root -> belly
            (7, 8),   # belly -> neck
            (8, 9),   # neck -> nose
            (9, 10),  # nose -> head
            (8, 11),  # neck -> left shoulder
            (11, 12), # left shoulder -> left elbow
            (12, 13), # left elbow -> left wrist
            (8, 14),  # neck -> right shoulder
            (14, 15), # right shoulder -> right elbow
            (15, 16)  # right elbow -> right wrist
        ]
    
    def calculate_velocities(self, poses: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame velocities for each joint.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - pose sequence
            
        Returns:
            np.ndarray: Shape (frames-1, 17) - velocity magnitudes
        """
        # Calculate displacement between consecutive frames
        displacements = np.diff(poses, axis=0)  # (frames-1, 17, 3)
        
        # Calculate velocity magnitude for each joint
        velocities = np.linalg.norm(displacements, axis=2)  # (frames-1, 17)
        
        return velocities
    
    def calculate_accelerations(self, poses: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame accelerations for each joint.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - pose sequence
            
        Returns:
            np.ndarray: Shape (frames-2, 17) - acceleration magnitudes
        """
        velocities = self.calculate_velocities(poses)
        
        # Calculate acceleration as change in velocity
        accelerations = np.diff(velocities, axis=0)  # (frames-2, 17)
        acceleration_magnitudes = np.abs(accelerations)
        
        return acceleration_magnitudes
    
    def detect_glitches(self, poses: np.ndarray) -> Dict:
        """
        Detect glitches in a pose sequence.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - pose sequence
            
        Returns:
            Dict: Contains glitch information including frames, joints, and metrics
        """
        if poses.shape[0] < 3:
            return {"has_glitches": False, "glitch_frames": [], "glitch_joints": []}
        
        # Calculate movement metrics
        velocities = self.calculate_velocities(poses)
        accelerations = self.calculate_accelerations(poses)
        
        # Detect high velocity frames
        high_velocity_mask = velocities > self.velocity_threshold
        velocity_glitch_frames = np.where(np.any(high_velocity_mask, axis=1))[0]
        
        # Detect high acceleration frames  
        high_accel_mask = accelerations > self.acceleration_threshold
        accel_glitch_frames = np.where(np.any(high_accel_mask, axis=1))[0] + 1  # +1 to align with original frame indices
        
        # Combine both types of glitches
        all_glitch_frames = np.unique(np.concatenate([velocity_glitch_frames, accel_glitch_frames]))
        
        # Get affected joints for each glitch frame
        glitch_joints = {}
        for frame_idx in all_glitch_frames:
            affected_joints = []
            
            # Check velocity-based glitches
            if frame_idx < len(velocities):
                vel_joints = np.where(velocities[frame_idx] > self.velocity_threshold)[0]
                affected_joints.extend(vel_joints.tolist())
            
            # Check acceleration-based glitches
            if frame_idx > 0 and frame_idx-1 < len(accelerations):
                accel_joints = np.where(accelerations[frame_idx-1] > self.acceleration_threshold)[0]
                affected_joints.extend(accel_joints.tolist())
            
            glitch_joints[frame_idx] = list(set(affected_joints))
        
        # Calculate severity metrics
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0
        max_acceleration = np.max(accelerations) if len(accelerations) > 0 else 0
        mean_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        
        return {
            "has_glitches": len(all_glitch_frames) > 0,
            "glitch_frames": all_glitch_frames.tolist(),
            "glitch_joints": glitch_joints,
            "num_glitch_frames": len(all_glitch_frames),
            "glitch_percentage": len(all_glitch_frames) / poses.shape[0] * 100,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "mean_velocity": mean_velocity,
            "velocities": velocities,
            "accelerations": accelerations
        }
    
    def create_glitch_visualization(self, poses: np.ndarray, glitch_info: Dict, 
                                  output_path: str, segment_id: int, fps: int = 30):
        """
        Create an animation visualization of a glitched pose segment.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - pose sequence
            glitch_info (Dict): Glitch detection results
            output_path (str): Output path for the animation
            segment_id (int): Segment identifier
            fps (int): Frames per second for animation
        """
        print(f"Creating visualization for segment {segment_id} with {glitch_info['num_glitch_frames']} glitch frames...")
        
        # Set up the figure and axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left subplot: 3D skeleton animation (side view)
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Segment {segment_id} - 3D Skeleton (Side View)', fontsize=14)
        ax1.set_xlabel('Left-Right (X)')
        ax1.set_ylabel('Height (-Y)')
        
        # Right subplot: Velocity plot
        if len(glitch_info['velocities']) > 0:
            ax2.plot(np.mean(glitch_info['velocities'], axis=1), 'b-', label='Mean Velocity')
            ax2.axhline(y=self.velocity_threshold, color='r', linestyle='--', label='Velocity Threshold')
            ax2.set_ylabel('Velocity')
            ax2.set_xlabel('Frame')
            ax2.set_title('Movement Velocity Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Create line objects for skeleton connections
        lines = []
        for _ in self.skeleton_connections:
            line, = ax1.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
            lines.append(line)
        
        # Add frame counter and glitch indicator
        frame_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        glitch_text = ax1.text(0.02, 0.88, '', transform=ax1.transAxes, fontsize=12,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Velocity indicator on right plot
        velocity_marker = ax2.axvline(x=0, color='g', linestyle='-', alpha=0.7) if len(glitch_info['velocities']) > 0 else None
        
        def init():
            """Initialize animation."""
            for line in lines:
                line.set_data([], [])
            frame_text.set_text('')
            glitch_text.set_text('')
            return lines + [frame_text, glitch_text]
        
        def animate(frame_idx):
            """Update function for each frame."""
            if frame_idx >= poses.shape[0]:
                return lines + [frame_text, glitch_text]
            
            # Get current frame poses
            current_frame = poses[frame_idx, :, :]
            x_coords = current_frame[:, 0]
            y_coords = -current_frame[:, 1]  # Flip Y for proper orientation
            
            # Determine if current frame is a glitch
            is_glitch_frame = frame_idx in glitch_info['glitch_frames']
            
            # Update skeleton connections with color coding
            for i, (start_joint, end_joint) in enumerate(self.skeleton_connections):
                x_data = [x_coords[start_joint], x_coords[end_joint]]
                y_data = [y_coords[start_joint], y_coords[end_joint]]
                
                # Color code based on whether joints are affected
                if is_glitch_frame and frame_idx in glitch_info['glitch_joints']:
                    affected_joints = glitch_info['glitch_joints'][frame_idx]
                    if start_joint in affected_joints or end_joint in affected_joints:
                        lines[i].set_color('red')
                        lines[i].set_linewidth(3)
                    else:
                        lines[i].set_color('orange')
                        lines[i].set_linewidth(2)
                else:
                    lines[i].set_color('blue')
                    lines[i].set_linewidth(2)
                
                lines[i].set_data(x_data, y_data)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{poses.shape[0]}')
            
            # Update glitch indicator
            if is_glitch_frame:
                affected_joint_names = []
                if frame_idx in glitch_info['glitch_joints']:
                    affected_joint_names = [self.joint_names[j] for j in glitch_info['glitch_joints'][frame_idx]]
                glitch_text.set_text(f'GLITCH!\nJoints: {", ".join(affected_joint_names[:3])}')
            else:
                glitch_text.set_text('')
            
            # Update velocity indicator
            if velocity_marker is not None and frame_idx > 0:
                velocity_marker.set_xdata([frame_idx-1])
            
            return lines + [frame_text, glitch_text]
        
        # Create animation
        interval = 1000 // fps
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=poses.shape[0],
                                     interval=interval, blit=False, repeat=True)
        
        # Save as GIF
        print(f"Saving animation to: {output_path}")
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved successfully!")

def detect_and_visualize_glitches(pose_segments: List[np.ndarray], 
                                output_dir: str = "outputs/glitch_analysis",
                                velocity_threshold: float = 0.3,
                                acceleration_threshold: float = 0.5,
                                fps: int = 30) -> Dict:
    """
    Main function to detect glitches in pose segments and generate visualizations.
    
    Args:
        pose_segments (List[np.ndarray]): List of pose segments, each with shape (frames, 17, 3)
                                        Frames can be variable length (minimum 10 frames recommended)
        output_dir (str): Directory to save visualizations
        velocity_threshold (float): Threshold for velocity-based glitch detection  
        acceleration_threshold (float): Threshold for acceleration-based glitch detection
        fps (int): Frames per second for animations
        
    Returns:
        Dict: Summary of glitch detection results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = PoseGlitchDetector(velocity_threshold, acceleration_threshold)
    
    # Results storage
    results = {
        "total_segments": len(pose_segments),
        "glitched_segments": [],
        "glitch_summary": {},
        "visualizations_created": []
    }
    
    print(f"Analyzing {len(pose_segments)} pose segments for glitches...")
    
    # Process each segment
    for segment_id, poses in enumerate(pose_segments):
        print(f"\nProcessing segment {segment_id + 1}/{len(pose_segments)}")
        
        # Check segment format
        if len(poses.shape) != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            print(f"Warning: Segment {segment_id} has unexpected shape {poses.shape}, skipping...")
            continue
        
        # Check minimum frames for meaningful analysis
        if poses.shape[0] < 3:
            print(f"Warning: Segment {segment_id} has too few frames ({poses.shape[0]}), skipping...")
            continue
        
        print(f"  Segment shape: {poses.shape} ({poses.shape[0]} frames)")
        
        # Detect glitches
        glitch_info = detector.detect_glitches(poses)
        
        # Store results
        results["glitch_summary"][segment_id] = {
            "has_glitches": glitch_info["has_glitches"],
            "num_glitch_frames": glitch_info["num_glitch_frames"],
            "glitch_percentage": glitch_info["glitch_percentage"],
            "max_velocity": glitch_info["max_velocity"],
            "max_acceleration": glitch_info["max_acceleration"],
            "total_frames": poses.shape[0]
        }
        
        # If glitches detected, create visualization
        if glitch_info["has_glitches"]:
            results["glitched_segments"].append(segment_id)
            
            print(f"  ⚠️  Glitches detected:")
            print(f"      - {glitch_info['num_glitch_frames']} glitch frames ({glitch_info['glitch_percentage']:.1f}%)")
            print(f"      - Max velocity: {glitch_info['max_velocity']:.3f}")
            print(f"      - Max acceleration: {glitch_info['max_acceleration']:.3f}")
            
            # Create visualization
            output_path = os.path.join(output_dir, f"segment_{segment_id:03d}_glitches_{poses.shape[0]}frames.gif")
            detector.create_glitch_visualization(poses, glitch_info, output_path, segment_id, fps)
            results["visualizations_created"].append(output_path)
        else:
            print(f"  ✅ No glitches detected in {poses.shape[0]} frames")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"GLITCH DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments analyzed: {results['total_segments']}")
    print(f"Segments with glitches: {len(results['glitched_segments'])}")
    print(f"Glitch rate: {len(results['glitched_segments'])/results['total_segments']*100:.1f}%")
    
    if results["glitched_segments"]:
        print(f"\nGlitched segments:")
        for seg_id in results["glitched_segments"]:
            frames = results["glitch_summary"][seg_id]["total_frames"]
            glitch_pct = results["glitch_summary"][seg_id]["glitch_percentage"]
            print(f"  - Segment {seg_id}: {frames} frames, {glitch_pct:.1f}% glitched")
        print(f"\nVisualizations saved to: {output_dir}")
    
    return results

def filter_glitched_segments(pose_segments: List[np.ndarray], 
                           velocity_threshold: float = 0.3,
                           acceleration_threshold: float = 0.5,
                           create_visualizations: bool = False,
                           output_dir: str = "outputs/glitch_analysis") -> List[np.ndarray]:
    """
    Filter out glitched segments from pose data and return only clean segments.
    
    This function is designed to be called from the main processing pipeline to
    automatically remove problematic pose segments before saving.
    
    Args:
        pose_segments (List[np.ndarray]): List of pose segments with variable lengths
        velocity_threshold (float): Threshold for velocity-based glitch detection  
        acceleration_threshold (float): Threshold for acceleration-based glitch detection
        create_visualizations (bool): Whether to create visualizations of glitched segments
        output_dir (str): Directory to save visualizations (if enabled)
        
    Returns:
        List[np.ndarray]: List of clean pose segments (no glitches detected)
    """
    print(f"\n{'='*60}")
    print(f"FILTERING GLITCHED SEGMENTS")
    print(f"{'='*60}")
    
    if create_visualizations:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = PoseGlitchDetector(velocity_threshold, acceleration_threshold)
    
    # Process segments and track results
    clean_segments = []
    glitched_segments = []
    total_segments = len(pose_segments)
    
    print(f"Analyzing {total_segments} pose segments for glitches...")
    
    for segment_id, poses in enumerate(pose_segments):
        # Handle batch dimension if present
        if len(poses.shape) == 4 and poses.shape[0] == 1:
            poses = poses.squeeze(0)
        
        # Check segment format
        if len(poses.shape) != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            print(f"  Segment {segment_id}: Invalid format {poses.shape} - SKIPPED")
            continue
        
        # Check minimum frames for meaningful analysis
        if poses.shape[0] < 3:
            print(f"  Segment {segment_id}: Too few frames ({poses.shape[0]}) - SKIPPED")
            continue
        
        # Detect glitches
        glitch_info = detector.detect_glitches(poses)
        
        if glitch_info["has_glitches"]:
            glitched_segments.append(segment_id)
            print(f"  Segment {segment_id}: ❌ GLITCHED ({glitch_info['num_glitch_frames']} frames, {glitch_info['glitch_percentage']:.1f}%) - REMOVED")
            
            # Create visualization if requested
            if create_visualizations:
                output_path = os.path.join(output_dir, f"removed_segment_{segment_id:03d}_glitches_{poses.shape[0]}frames.gif")
                detector.create_glitch_visualization(poses, glitch_info, output_path, segment_id, fps=30)
                print(f"    Visualization saved: {output_path}")
                
        else:
            clean_segments.append(poses)
            print(f"  Segment {segment_id}: ✅ CLEAN ({poses.shape[0]} frames) - KEPT")
    
    # Print summary
    removed_count = len(glitched_segments)
    kept_count = len(clean_segments)
    removal_rate = (removed_count / total_segments * 100) if total_segments > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments processed: {total_segments}")
    print(f"Clean segments kept: {kept_count}")
    print(f"Glitched segments removed: {removed_count}")
    print(f"Removal rate: {removal_rate:.1f}%")
    
    if glitched_segments:
        print(f"Removed segment IDs: {glitched_segments}")
        if create_visualizations:
            print(f"Visualizations saved to: {output_dir}")
    
    if kept_count == 0:
        print("⚠️  WARNING: No clean segments remain after filtering!")
    else:
        print(f"✅ {kept_count} clean segments ready for further processing")
    
    return clean_segments

def filter_empty_skeleton_segments(pose_segments: List[np.ndarray], 
                                 summary_dir: Optional[str] = None) -> Tuple[List[np.ndarray], Dict]:
    """
    Filter out segments that don't have valid skeleton data.
    
    Args:
        pose_segments: List of pose segments
        summary_dir: Optional directory to save summary JSON
        
    Returns:
        Tuple of (clean_segments, summary_info)
    """
    print(f"🔍 Filtering empty skeleton segments...")
    
    if not pose_segments:
        summary_info = {
            'input_segments': 0,
            'output_segments': 0,
            'removed_segments': 0,
            'removal_percentage': 0.0,
            'total_input_frames': 0,
            'total_output_frames': 0,
            'removed_frames': 0,
            'time_removed_percentage': 0.0
        }
        return [], summary_info
    
    clean_segments = []
    removed_count = 0
    total_input_frames = 0
    total_output_frames = 0
    
    for i, segment in enumerate(pose_segments):
        if len(segment.shape) != 3 or segment.shape[1] != 17 or segment.shape[2] != 3:
            print(f"  Segment {i}: Invalid shape {segment.shape} - removed")
            removed_count += 1
            continue
            
        total_input_frames += segment.shape[0]
        
        # Check if segment has any valid skeleton data
        # A segment is considered empty if all coordinates are zero or very close to zero
        max_coord = np.max(np.abs(segment))
        if max_coord < 1e-6:  # All coordinates are essentially zero
            print(f"  Segment {i}: No skeleton data (max coord: {max_coord}) - removed")
            removed_count += 1
            continue
            
        # Check if segment has sufficient non-zero joints
        non_zero_frames = np.sum(np.any(np.abs(segment) > 1e-6, axis=(1, 2)))
        if non_zero_frames < segment.shape[0] * 0.5:  # Less than 50% frames have data
            print(f"  Segment {i}: Insufficient skeleton data ({non_zero_frames}/{segment.shape[0]} frames) - removed")
            removed_count += 1
            continue
            
        clean_segments.append(segment)
        total_output_frames += segment.shape[0]
    
    removal_percentage = (removed_count / len(pose_segments)) * 100 if pose_segments else 0
    time_removed_percentage = ((total_input_frames - total_output_frames) / total_input_frames) * 100 if total_input_frames > 0 else 0
    
    summary_info = {
        'input_segments': len(pose_segments),
        'output_segments': len(clean_segments),
        'removed_segments': removed_count,
        'removal_percentage': removal_percentage,
        'total_input_frames': total_input_frames,
        'total_output_frames': total_output_frames,
        'removed_frames': total_input_frames - total_output_frames,
        'time_removed_percentage': time_removed_percentage
    }
    
    print(f"📊 Empty skeleton filter summary:")
    print(f"   Input segments: {summary_info['input_segments']}")
    print(f"   Output segments: {summary_info['output_segments']}")
    print(f"   Removed segments: {summary_info['removed_segments']} ({removal_percentage:.1f}%)")
    print(f"   Removed frames: {summary_info['removed_frames']} ({time_removed_percentage:.1f}%)")
    
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, 'empty_skeleton_filter_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_info, f, indent=2)
        print(f"💾 Empty skeleton filter summary saved to: {summary_file}")
    
    return clean_segments, summary_info

def process_poses_with_glitch_filtering(raw_poses, 
                                      velocity_threshold: float = 0.3,
                                      acceleration_threshold: float = 0.5,
                                      create_visualizations: bool = True,
                                      output_dir: str = "outputs/removed_glitches",
                                      summary_dir: Optional[str] = None) -> List[np.ndarray]:
    """
    Complete pipeline function for processing raw poses with glitch filtering.
    
    This function handles the complete workflow:
    1. Parse raw pose data into segments
    2. Remove batch dimensions if present  
    3. Filter out glitched segments
    4. Return clean segments ready for saving
    5. Optionally create summary statistics
    
    Args:
        raw_poses: Raw pose data from extract_pose (list or array)
        velocity_threshold (float): Threshold for velocity-based glitch detection
        acceleration_threshold (float): Threshold for acceleration-based glitch detection  
        create_visualizations (bool): Whether to create visualizations of removed segments
        output_dir (str): Directory to save visualizations
        summary_dir (Optional[str]): Directory to save summary JSON files
        
    Returns:
        List[np.ndarray]: List of clean pose segments
    """
    print(f"🔍 Processing poses with glitch filtering...")
    
    original_segment_count = 0
    original_frame_count = 0
    
    # Parse raw poses into segments (reuse logic from test script)
    if isinstance(raw_poses, list):
        print(f"Found {len(raw_poses)} pose segments in list format")
        segments = []
        original_segment_count = len(raw_poses)
        
        for i, segment in enumerate(raw_poses):
            # Handle extra batch dimension if present
            if len(segment.shape) == 4 and segment.shape[0] == 1:
                segment = segment.squeeze(0)
            
            original_frame_count += segment.shape[0] if len(segment.shape) >= 1 else 0
            
            # Validate segment format
            if len(segment.shape) == 3 and segment.shape[1] == 17 and segment.shape[2] == 3:
                if segment.shape[0] >= 10:  # Minimum frames
                    segments.append(segment)
                else:
                    print(f"  Segment {i}: Too few frames ({segment.shape[0]}) - skipped")
            else:
                print(f"  Segment {i}: Invalid format {segment.shape} - skipped")
        
    elif isinstance(raw_poses, np.ndarray):
        print(f"Found numpy array with shape: {raw_poses.shape}")
        original_frame_count = raw_poses.shape[0] if len(raw_poses.shape) >= 1 else 0
        
        # Handle extra batch dimension if present
        if len(raw_poses.shape) == 4 and raw_poses.shape[0] == 1:
            raw_poses = raw_poses.squeeze(0)
        
        # Split into segments (243 frames each, allow partial last segment)
        segments = []
        total_frames = raw_poses.shape[0]
        segment_length = 243
        original_segment_count = (total_frames + segment_length - 1) // segment_length
        
        for start_idx in range(0, total_frames, segment_length):
            end_idx = min(start_idx + segment_length, total_frames)
            segment = raw_poses[start_idx:end_idx]
            
            if segment.shape[0] >= 10:  # Minimum frames for analysis
                segments.append(segment)
            else:
                print(f"  Skipping segment: frames {start_idx}-{end_idx-1} ({segment.shape[0]} frames - too short)")
    
    else:
        raise ValueError(f"Unexpected raw_poses format: {type(raw_poses)}")
    
    print(f"Prepared {len(segments)} segments for glitch filtering")
    
    # Filter out glitched segments
    clean_segments = filter_glitched_segments(
        pose_segments=segments,
        velocity_threshold=velocity_threshold,
        acceleration_threshold=acceleration_threshold,
        create_visualizations=create_visualizations,
        output_dir=output_dir
    )
    
    # Create summary if requested
    if summary_dir:
        final_frame_count = sum(seg.shape[0] for seg in clean_segments)
        removed_segments = len(segments) - len(clean_segments)
        removed_frames = sum(seg.shape[0] for seg in segments) - final_frame_count
        
        glitch_summary = {
            'original_segments': original_segment_count,
            'parsed_segments': len(segments),
            'final_segments': len(clean_segments),
            'removed_segments': removed_segments,
            'removal_percentage': (removed_segments / len(segments)) * 100 if segments else 0,
            'original_frames': original_frame_count,
            'parsed_frames': sum(seg.shape[0] for seg in segments),
            'final_frames': final_frame_count,
            'removed_frames': removed_frames,
            'time_removed_percentage': (removed_frames / sum(seg.shape[0] for seg in segments)) * 100 if segments else 0,
            'velocity_threshold': velocity_threshold,
            'acceleration_threshold': acceleration_threshold
        }
        
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, 'glitch_filter_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(glitch_summary, f, indent=2)
        print(f"💾 Glitch filter summary saved to: {summary_file}")
    
    return clean_segments

# Test function for the current pipeline
def test_glitch_detection():
    """Test function to verify glitch detection with sample data."""
    print("Testing glitch detection with sample data...")
    
    # Create sample pose data with artificial glitches
    frames, joints, coords = 243, 17, 3
    
    # Normal motion
    normal_poses = np.random.randn(frames, joints, coords) * 0.1
    
    # Add artificial glitches at specific frames
    glitch_frames = [50, 100, 150, 200]
    for frame in glitch_frames:
        # Add large jumps to simulate glitches
        normal_poses[frame] += np.random.randn(joints, coords) * 2.0
    
    # Test detection
    detector = PoseGlitchDetector()
    glitch_info = detector.detect_glitches(normal_poses)
    
    print(f"Detected {glitch_info['num_glitch_frames']} glitch frames")
    print(f"Expected around {len(glitch_frames)} glitch frames")
    print(f"Detection successful: {len(glitch_info['glitch_frames']) > 0}")
    
    return glitch_info

if __name__ == "__main__":
    # Run test when script is executed directly
    test_glitch_detection()
