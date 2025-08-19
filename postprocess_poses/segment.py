import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

class MovementSegmenter:
    """
    Segments pose sequences based on movement detection using velocity and acceleration thresholds.
    
    This class identifies segments of video that have considerable movement within a maximum frame range.
    It uses the same movement detection logic as the PoseGlitchDetector but for segmentation purposes.
    """
    
    def __init__(self, max_segment_length: int = 243, 
                 velocity_threshold: float = 0.3, 
                 acceleration_threshold: float = 0.5,
                 min_segment_length: int = 30):
        """
        Initialize the movement segmenter.
        
        Args:
            max_segment_length (int): Maximum length of each segment in frames
            velocity_threshold (float): Threshold for detecting significant movement velocity
            acceleration_threshold (float): Threshold for detecting significant movement acceleration
            min_segment_length (int): Minimum length of each segment in frames
        """
        self.max_segment_length = max_segment_length
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_segment_length = min_segment_length
        
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
    
    def detect_movement_frames(self, poses: np.ndarray) -> np.ndarray:
        """
        Detect frames with significant movement based on velocity and acceleration thresholds.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - pose sequence
            
        Returns:
            np.ndarray: Boolean array indicating frames with significant movement
        """
        if poses.shape[0] < 3:
            return np.zeros(poses.shape[0], dtype=bool)
        
        # Calculate movement metrics
        velocities = self.calculate_velocities(poses)
        accelerations = self.calculate_accelerations(poses)
        
        # Detect high velocity frames
        high_velocity_mask = velocities > self.velocity_threshold
        velocity_movement_frames = np.any(high_velocity_mask, axis=1)
        
        # Detect high acceleration frames  
        high_accel_mask = accelerations > self.acceleration_threshold
        accel_movement_frames = np.any(high_accel_mask, axis=1)
        
        # Combine both types of movement
        # Initialize movement frames array
        movement_frames = np.zeros(poses.shape[0], dtype=bool)
        
        # Velocity affects frames 1 to n-1 (since it's frame-to-frame)
        movement_frames[1:len(velocity_movement_frames)+1] |= velocity_movement_frames
        
        # Acceleration affects frames 2 to n-1 (since it's change in velocity)
        movement_frames[2:len(accel_movement_frames)+2] |= accel_movement_frames
        
        return movement_frames
    
    def segment_by_movement(self, poses: np.ndarray) -> List[np.ndarray]:
        """
        Segment pose sequence based on movement detection.
        
        Args:
            poses (np.ndarray): Shape (frames, 17, 3) - long pose sequence
            
        Returns:
            List[np.ndarray]: List of pose segments
        """
        print(f"Segmenting {poses.shape[0]} frames based on movement...")
        
        # Detect movement frames
        movement_frames = self.detect_movement_frames(poses)
        
        # Find segments with movement
        segments = []
        current_start = 0
        
        for i in range(poses.shape[0]):
            # Check if we've reached max segment length
            if i - current_start >= self.max_segment_length:
                # Force segment end at max length
                if i - current_start >= self.min_segment_length:
                    segment = poses[current_start:i]
                    segments.append(segment)
                    print(f"  Created segment {len(segments)}: frames {current_start}-{i-1} ({segment.shape[0]} frames)")
                current_start = i
                continue
            
            # Check if current frame has movement
            if movement_frames[i]:
                # Look ahead to see if we have enough movement frames
                look_ahead = min(30, poses.shape[0] - i)  # Look ahead 30 frames
                future_movement = np.sum(movement_frames[i:i+look_ahead])
                
                # If we have significant movement ahead, continue current segment
                if future_movement >= 5:  # At least 5 movement frames in next 30
                    continue
                else:
                    # End segment here if it's long enough
                    if i - current_start >= self.min_segment_length:
                        segment = poses[current_start:i+1]
                        segments.append(segment)
                        print(f"  Created segment {len(segments)}: frames {current_start}-{i} ({segment.shape[0]} frames) - movement detected")
                    current_start = i + 1
        
        # Handle final segment
        if poses.shape[0] - current_start >= self.min_segment_length:
            final_segment = poses[current_start:]
            segments.append(final_segment)
            print(f"  Created final segment {len(segments)}: frames {current_start}-{poses.shape[0]-1} ({final_segment.shape[0]} frames)")
        
        print(f"Created {len(segments)} segments from {poses.shape[0]} frames")
        return segments
    
    def create_segmentation_visualization(self, poses: np.ndarray, segments: List[np.ndarray], 
                                        output_path: str, fps: int = 30):
        """
        Create a visualization of the segmentation results.
        
        Args:
            poses (np.ndarray): Original long pose sequence
            segments (List[np.ndarray]): List of segmented poses
            output_path (str): Output path for the visualization
            fps (int): Frames per second for animation
        """
        print(f"Creating segmentation visualization...")
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Top subplot: Movement detection over time
        movement_frames = self.detect_movement_frames(poses)
        time_axis = np.arange(len(movement_frames))
        
        ax1.plot(time_axis, movement_frames, 'b-', linewidth=2, label='Movement Detection')
        ax1.set_ylabel('Movement Detected')
        ax1.set_xlabel('Frame')
        ax1.set_title('Movement Detection Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Color segments on the movement plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        current_frame = 0
        
        for i, segment in enumerate(segments):
            segment_start = current_frame
            segment_end = current_frame + segment.shape[0]
            ax1.axvspan(segment_start, segment_end, alpha=0.3, color=colors[i], 
                       label=f'Segment {i+1} ({segment.shape[0]} frames)')
            current_frame = segment_end
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Bottom subplot: 3D skeleton animation (side view)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('3D Skeleton Animation (Side View)', fontsize=14)
        ax2.set_xlabel('Left-Right (X)')
        ax2.set_ylabel('Height (-Y)')
        
        # Create line objects for skeleton connections
        lines = []
        for _ in self.skeleton_connections:
            line, = ax2.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
            lines.append(line)
        
        # Add frame counter and segment indicator
        frame_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=12,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        segment_text = ax2.text(0.02, 0.88, '', transform=ax2.transAxes, fontsize=12,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Movement indicator on top plot
        movement_marker = ax1.axvline(x=0, color='r', linestyle='-', alpha=0.7)
        
        def init():
            """Initialize animation."""
            for line in lines:
                line.set_data([], [])
            frame_text.set_text('')
            segment_text.set_text('')
            return lines + [frame_text, segment_text]
        
        def animate(frame_idx):
            """Update function for each frame."""
            if frame_idx >= poses.shape[0]:
                return lines + [frame_text, segment_text]
            
            # Get current frame poses
            current_frame = poses[frame_idx, :, :]
            x_coords = current_frame[:, 0]
            y_coords = -current_frame[:, 1]  # Flip Y for proper orientation
            
            # Update skeleton connections
            for i, (start_joint, end_joint) in enumerate(self.skeleton_connections):
                x_data = [x_coords[start_joint], x_coords[end_joint]]
                y_data = [y_coords[start_joint], y_coords[end_joint]]
                lines[i].set_data(x_data, y_data)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{poses.shape[0]}')
            
            # Update segment indicator
            current_segment = None
            current_frame_in_segment = 0
            frame_count = 0
            
            for seg_idx, segment in enumerate(segments):
                if frame_idx < frame_count + segment.shape[0]:
                    current_segment = seg_idx + 1
                    current_frame_in_segment = frame_idx - frame_count + 1
                    break
                frame_count += segment.shape[0]
            
            if current_segment:
                segment_text.set_text(f'Segment: {current_segment}/{len(segments)}\nFrame in segment: {current_frame_in_segment}')
            else:
                segment_text.set_text('')
            
            # Update movement indicator
            movement_marker.set_xdata([frame_idx])
            
            return lines + [frame_text, segment_text]
        
        # Create animation
        interval = 1000 // fps
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=poses.shape[0],
                                     interval=interval, blit=False, repeat=True)
        
        # Save as GIF
        print(f"Saving animation to: {output_path}")
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved successfully!")

def load_poses_from_npz(poses_path: str) -> np.ndarray:
    """
    Load and concatenate all pose segments from an NPZ file.
    
    Args:
        poses_path (str): Path to the NPZ file containing pose segments
        
    Returns:
        np.ndarray: Concatenated pose sequence with shape (total_frames, 17, 3)
    """
    print(f"Loading poses from: {poses_path}")
    
    # Load the NPZ file
    data = np.load(poses_path)
    
    # Get all segment keys
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
        raise ValueError("No valid pose segments found in the NPZ file")

def segment_poses(poses_path: str, 
                 output_path: str,
                 max_segment_length: int = 243,
                 velocity_threshold: float = 0.1,
                 acceleration_threshold: float = 0.1,
                 min_segment_length: int = 30,
                 create_visualization: bool = True,
                 fps: int = 30) -> Dict:
    """
    Main function to segment poses based on movement detection.
    
    Args:
        poses_path (str): Path to the NPZ file containing pose segments
        output_dir (str): Directory to save segmented poses and visualizations
        max_segment_length (int): Maximum length of each segment in frames
        velocity_threshold (float): Threshold for detecting significant movement velocity
        acceleration_threshold (float): Threshold for detecting significant movement acceleration
        min_segment_length (int): Minimum length of each segment in frames
        create_visualization (bool): Whether to create visualization of segmentation
        fps (int): Frames per second for animations
        
    Returns:
        Dict: Summary of segmentation results
    """
    # Load and concatenate poses
    poses = load_poses_from_npz(poses_path)
    
    # Initialize segmenter
    segmenter = MovementSegmenter(
        max_segment_length=max_segment_length,
        velocity_threshold=velocity_threshold,
        acceleration_threshold=acceleration_threshold,
        min_segment_length=min_segment_length
    )
    
    # Segment poses
    segments = segmenter.segment_by_movement(poses)
    
    # Save segmented poses in the same format as original
    segment_data = {}
    
    for i, segment in enumerate(segments):
        # Add batch dimension to match original format (1, frames, 17, 3)
        segment_with_batch = segment[np.newaxis, :, :, :]
        segment_data[f'segment_{i:03d}'] = segment_with_batch
    
    # Add metadata
    segment_data['num_segments'] = len(segments)
    segment_data['segment_shapes'] = np.array([seg.shape for seg in segments])
    segment_data['original_frames'] = poses.shape[0]
    segment_data['max_segment_length'] = max_segment_length
    segment_data['velocity_threshold'] = velocity_threshold
    segment_data['acceleration_threshold'] = acceleration_threshold
    segment_data['min_segment_length'] = min_segment_length
    
    np.savez_compressed(output_path, **segment_data)
    print(f"Saved {len(segments)} segments to: {output_path}")
    
    # Create visualization if requested
    if create_visualization:
        output_dir = os.path.dirname(output_path)
        viz_path = os.path.join(output_dir, 'segmentation_visualization.gif')
        segmenter.create_segmentation_visualization(poses, segments, viz_path, fps)
    
    # Calculate comprehensive summary statistics
    segment_lengths = [seg.shape[0] for seg in segments]
    original_segment_count = poses.shape[0] // 243  # Assuming original segments were 243 frames
    
    # Calculate duration statistics (assuming 30 fps)
    fps = 30
    segment_durations = [length / fps for length in segment_lengths]
    
    # Calculate movement statistics
    movement_frames = segmenter.detect_movement_frames(poses)
    total_movement_frames = np.sum(movement_frames)
    movement_percentage = (total_movement_frames / poses.shape[0]) * 100
    
    summary = {
        'input_frames': poses.shape[0],
        'input_duration_seconds': poses.shape[0] / fps,
        'num_segments': len(segments),
        'segment_lengths': segment_lengths,
        'segment_durations_seconds': segment_durations,
        'min_segment_length': min(segment_lengths) if segment_lengths else 0,
        'max_segment_length': max(segment_lengths) if segment_lengths else 0,
        'mean_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
        'min_segment_duration': min(segment_durations) if segment_durations else 0,
        'max_segment_duration': max(segment_durations) if segment_durations else 0,
        'mean_segment_duration': np.mean(segment_durations) if segment_durations else 0,
        'total_segment_frames': sum(segment_lengths),
        'total_segment_duration': sum(segment_durations),
        'original_segment_count': original_segment_count,
        'segments_removed': original_segment_count - len(segments),
        'removal_percentage': ((original_segment_count - len(segments)) / original_segment_count * 100) if original_segment_count > 0 else 0,
        'movement_frames': int(total_movement_frames),
        'movement_percentage': movement_percentage,
        'parameters': {
            'max_segment_length': max_segment_length,
            'velocity_threshold': velocity_threshold,
            'acceleration_threshold': acceleration_threshold,
            'min_segment_length': min_segment_length,
            'fps': fps
        }
    }
    
    # Save summary
    output_dir = os.path.dirname(output_path)
    summary_file = os.path.join(output_dir, 'segmentation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Segmentation summary saved to: {summary_file}")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SEGMENTATION SUMMARY")
    print(f"{'='*80}")
    print(f"📊 INPUT DATA:")
    print(f"   Total frames: {summary['input_frames']:,}")
    print(f"   Total duration: {summary['input_duration_seconds']:.2f} seconds ({summary['input_duration_seconds']/60:.2f} minutes)")
    print(f"   Original segments: {summary['original_segment_count']} (243-frame segments)")
    print(f"   Movement frames detected: {summary['movement_frames']:,} ({summary['movement_percentage']:.1f}%)")
    
    print(f"\n📈 SEGMENTATION RESULTS:")
    print(f"   Final segments created: {summary['num_segments']}")
    print(f"   Segments removed: {summary['segments_removed']} ({summary['removal_percentage']:.1f}%)")
    print(f"   Total segment frames: {summary['total_segment_frames']:,}")
    print(f"   Total segment duration: {summary['total_segment_duration']:.2f} seconds")
    
    print(f"\n⏱️  DURATION STATISTICS:")
    print(f"   Min segment duration: {summary['min_segment_duration']:.2f} seconds ({summary['min_segment_length']} frames)")
    print(f"   Max segment duration: {summary['max_segment_duration']:.2f} seconds ({summary['max_segment_length']} frames)")
    print(f"   Mean segment duration: {summary['mean_segment_duration']:.2f} seconds ({summary['mean_segment_length']:.1f} frames)")
    
    print(f"\n⚙️  PARAMETERS USED:")
    print(f"   Max segment length: {max_segment_length} frames")
    print(f"   Velocity threshold: {velocity_threshold}")
    print(f"   Acceleration threshold: {acceleration_threshold}")
    print(f"   Min segment length: {min_segment_length} frames")
    print(f"   FPS: {fps}")
    
    print(f"\n💾 OUTPUT:")
    print(f"   Segmented poses saved to: {output_path}")
    if create_visualization:
        print(f"   Visualization saved to: {os.path.join(output_dir, 'segmentation_visualization.gif')}")
    print(f"   Summary saved to: {summary_file}")
    
    return summary

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Segment poses based on movement detection')
    parser.add_argument('poses_path', help='Path to NPZ file containing pose segments')
    parser.add_argument('output_path', help='Output path for segmented poses NPZ file')
    parser.add_argument('--max-segment-length', type=int, default=243, 
                       help='Maximum length of each segment in frames (default: 243)')
    parser.add_argument('--velocity-threshold', type=float, default=0.1,
                       help='Threshold for detecting significant movement velocity (default: 0.1)')
    parser.add_argument('--acceleration-threshold', type=float, default=0.1,
                       help='Threshold for detecting significant movement acceleration (default: 0.1)')
    parser.add_argument('--min-segment-length', type=int, default=30,
                       help='Minimum length of each segment in frames (default: 30)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization creation')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for animations (default: 30)')
    
    args = parser.parse_args()
    
    # Run segmentation
    summary = segment_poses(
        poses_path=args.poses_path,
        output_path=args.output_path,
        max_segment_length=args.max_segment_length,
        velocity_threshold=args.velocity_threshold,
        acceleration_threshold=args.acceleration_threshold,
        min_segment_length=args.min_segment_length,
        create_visualization=not args.no_viz,
        fps=args.fps
    )
    
    print(f"\nSegmentation completed successfully!")
    print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()
