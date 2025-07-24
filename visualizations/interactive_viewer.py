#!/usr/bin/env python3
"""
Interactive Pose Viewer

This script loads a hardcoded pickle file containing a single pose segment 
with 243 frames and displays an interactive 3D skeleton animation.

Controls:
- Spacebar: Play/Pause
- Left/Right arrows: Navigate frames manually
- R: Reset to first frame
- Q: Quit
- Mouse: Click and drag to rotate 3D view
- Mouse wheel: Zoom in/out
- Middle mouse: Pan the view
"""

import numpy as np
import matplotlib
import pickle
import sys
import os

# Try to set an interactive backend for 3D visualization
def setup_interactive_backend():
    """Try to set up an interactive matplotlib backend."""
    backends_to_try = ['Qt5Agg', 'GTK3Agg', 'Qt4Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            print(f"✅ Using matplotlib backend: {backend}")
            break
        except ImportError:
            continue
    else:
        print("⚠️ Warning: No interactive backend available, using default")
        print("   You may need to install: python3-tk or python3-pyqt5")

setup_interactive_backend()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Hardcoded pickle file path
PICKLE_FILE_PATH = "../clustered_poses/cluster_049/poses.pkl"
# Segment index to load (0 for first segment, or specify which one)
SEGMENT_INDEX = 0

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

class InteractivePoseViewer:
    def __init__(self, poses):
        """Initialize the interactive pose viewer."""
        self.poses = poses
        self.current_frame = 0
        self.total_frames = poses.shape[0]
        self.is_playing = False
        self.animation = None
        
        # Setup the figure and 3D axis
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Calculate axis limits from all poses
        self._calculate_axis_limits()
        
        # Setup the plot
        self._setup_plot()
        
        # Create line objects for skeleton connections
        self.lines = []
        self.joints = []
        
        # Create lines for bones
        for _ in H36M_CONNECTIONS:
            line, = self.ax.plot([], [], [], 'royalblue', linewidth=4, alpha=0.8)
            self.lines.append(line)
        
        # Create scatter plot for joints
        self.joint_scatter = self.ax.scatter([], [], [], c='red', s=80, alpha=0.9)
        
        # Add frame counter text
        self.frame_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        fontsize=12, fontweight='bold',
                                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add control instructions
        self.control_text = self.ax.text2D(0.02, 0.02, 
                                         '🎮 PRESS SPACEBAR TO PLAY/PAUSE 🎮\n' +
                                         'Controls: ←→=Navigate, R=Reset, Q=Quit\n' +
                                         'Mouse: Drag=Rotate, Wheel=Zoom, Middle=Pan\n' +
                                         'Views: 1=Front, 2=Side, 3=Top, 4=Iso, H=Help',
                                         transform=self.ax.transAxes, fontsize=8,
                                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Update the first frame
        self._update_frame()
        
        print(f"🎬 Interactive Pose Viewer loaded!")
        print(f"📊 Total frames: {self.total_frames}")
        print(f"📏 Pose data shape: {self.poses.shape}")
        print(f"🎮 Keyboard Controls:")
        print(f"   SPACEBAR: Play/Pause animation")
        print(f"   ← →: Navigate frames manually")
        print(f"   R: Reset to first frame") 
        print(f"   Q: Quit")
        print(f"   1-4: Quick view angles (Front/Side/Top/Isometric)")
        print(f"   H: Show help")
        print(f"🖱️  Mouse Controls (3D View):")
        print(f"   LEFT CLICK + DRAG: Rotate 3D view")
        print(f"   MOUSE WHEEL: Zoom in/out")
        print(f"   MIDDLE CLICK + DRAG: Pan the view")
        print(f"   RIGHT CLICK: Context menu (matplotlib)")
        print(f"💡 Tip: Use mouse to freely rotate and explore the 3D skeleton!")
        print(f"🎞️ Frame range: 0 to {self.total_frames - 1} (inclusive)")
        print(f"")
        print(f"🚨 IMPORTANT: Press SPACEBAR to start/pause animation!")
        print(f"▶️ Starting animation now...")
        
        # Start animation immediately
        self.play()
    
    def _calculate_axis_limits(self):
        """Calculate appropriate axis limits from pose data."""
        all_coords = self.poses.reshape(-1, 3)
        valid_coords = all_coords[np.any(np.abs(all_coords) > 1e-6, axis=1)]
        
        if len(valid_coords) > 0:
            ranges = np.ptp(valid_coords, axis=0)
            centers = np.mean(valid_coords, axis=0)
            max_range = np.max(ranges)
            padding = max_range * 0.1 + 0.1
            
            self.x_lim = [centers[0] - max_range/2 - padding, centers[0] + max_range/2 + padding]
            self.y_lim = [centers[1] - max_range/2 - padding, centers[1] + max_range/2 + padding]
            self.z_lim = [centers[2] - max_range/2 - padding, centers[2] + max_range/2 + padding]
        else:
            self.x_lim = [-1, 1]
            self.y_lim = [-1, 1]
            self.z_lim = [-1, 1]
    
    def _setup_plot(self):
        """Setup the 3D plot properties."""
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_zlim(self.z_lim)
        
        self.ax.set_xlabel('X (Left-Right)', fontsize=12)
        self.ax.set_ylabel('Y (Height)', fontsize=12)
        self.ax.set_zlabel('Z (Forward-Back)', fontsize=12)
        self.ax.set_title('Interactive 3D Pose Viewer - Click and Drag to Rotate!', fontsize=14, fontweight='bold')
        
        # Enable grid for better 3D perception
        self.ax.grid(True, alpha=0.3)
        
        # Set background color for better contrast
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Make pane edges more visible
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)
        
        # Set a nice viewing angle
        self.ax.view_init(elev=15, azim=45)
    
    def _update_frame(self):
        """Update the visualization with the current frame."""
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
        
        # Get current frame poses
        current_pose = self.poses[self.current_frame, :, :]
        x_coords = current_pose[:, 0]
        y_coords = current_pose[:, 1]
        z_coords = current_pose[:, 2]
        
        # Update skeleton connections
        for i, (start_joint, end_joint) in enumerate(H36M_CONNECTIONS):
            self.lines[i].set_data([x_coords[start_joint], x_coords[end_joint]], 
                                  [y_coords[start_joint], y_coords[end_joint]])
            self.lines[i].set_3d_properties([z_coords[start_joint], z_coords[end_joint]])
        
        # Update joint positions
        self.joint_scatter._offsets3d = (x_coords, y_coords, z_coords)
        
        # Update frame counter with more detailed information
        status = "▶ PLAYING" if self.is_playing else "⏸ PAUSED"
        progress = (self.current_frame + 1) / self.total_frames * 100
        self.frame_text.set_text(f'Frame: {self.current_frame + 1}/{self.total_frames} ({progress:.1f}%) | {status}')
        
        # Redraw
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == ' ':  # Spacebar - Play/Pause
            print(f"🎮 Spacebar pressed! Current state: {'PLAYING' if self.is_playing else 'PAUSED'}")
            self.toggle_playback()
        elif event.key == 'left':  # Left arrow - Previous frame
            self.is_playing = False
            self.current_frame = max(0, self.current_frame - 1)
            self._update_frame()
        elif event.key == 'right':  # Right arrow - Next frame
            self.is_playing = False
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            self._update_frame()
        elif event.key == 'r':  # R - Reset to first frame
            self.is_playing = False
            self.current_frame = 0
            self._update_frame()
        elif event.key == 'q':  # Q - Quit
            plt.close('all')
            sys.exit(0)
        # View angle shortcuts
        elif event.key == '1':  # Front view
            self.ax.view_init(elev=0, azim=0)
            self.fig.canvas.draw()
            print("📐 Front view")
        elif event.key == '2':  # Side view
            self.ax.view_init(elev=0, azim=90)
            self.fig.canvas.draw()
            print("📐 Side view")
        elif event.key == '3':  # Top view
            self.ax.view_init(elev=90, azim=0)
            self.fig.canvas.draw()
            print("📐 Top view")
        elif event.key == '4':  # Isometric view
            self.ax.view_init(elev=15, azim=45)
            self.fig.canvas.draw()
            print("📐 Isometric view (default)")
        elif event.key == 'h':  # Help
            self._print_help()
    
    def _print_help(self):
        """Print help information to console."""
        print("\n" + "="*60)
        print("🆘 INTERACTIVE POSE VIEWER - HELP")
        print("="*60)
        print("🎮 Keyboard Controls:")
        print("   SPACEBAR     → Play/Pause animation")
        print("   ← →          → Navigate frames manually")
        print("   R            → Reset to first frame")
        print("   Q            → Quit application")
        print("   1            → Front view (elev=0°, azim=0°)")
        print("   2            → Side view (elev=0°, azim=90°)")
        print("   3            → Top view (elev=90°, azim=0°)")
        print("   4            → Isometric view (elev=15°, azim=45°)")
        print("   H            → Show this help")
        print()
        print("🖱️  Mouse Controls (3D View):")
        print("   LEFT DRAG    → Rotate 3D view freely")
        print("   WHEEL        → Zoom in/out")
        print("   MIDDLE DRAG  → Pan the view")
        print("   RIGHT CLICK  → Matplotlib context menu")
        print()
        print("📊 Current Status:")
        print(f"   Frame: {self.current_frame + 1}/{self.total_frames}")
        print(f"   Status: {'▶ PLAYING' if self.is_playing else '⏸ PAUSED'}")
        print("="*60 + "\n")
    
    def toggle_playback(self):
        """Toggle between play and pause."""
        print(f"🔄 Toggling playback... Current: {self.is_playing}")
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start playing the animation."""
        if not self.is_playing:
            # Stop any existing animation first
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            
            self.is_playing = True
            # Create animation that continuously cycles through all frames
            self.animation = animation.FuncAnimation(
                self.fig, self._animate_frame, 
                frames=range(self.total_frames),  # Explicit frame range
                interval=33, repeat=True, blit=False  # 33ms ≈ 30fps
            )
            print(f"▶ Playing animation... ({self.total_frames} frames)")
    
    def pause(self):
        """Pause the animation."""
        if self.is_playing:
            self.is_playing = False
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            print("⏸ Animation paused")
    
    def _animate_frame(self, frame_num):
        """Animation update function."""
        if not self.is_playing:
            return self.lines + [self.joint_scatter, self.frame_text]
        
        # Ensure frame_num is within valid range
        self.current_frame = frame_num % self.total_frames
        
        # Debug print for every 30 frames (about once per second at 30fps)
        if frame_num % 30 == 0:
            print(f"🎬 Playing frame {self.current_frame + 1}/{self.total_frames}")
        
        self._update_frame()
        return self.lines + [self.joint_scatter, self.frame_text]
    
    def show(self):
        """Display the interactive viewer."""
        plt.show()

def load_pose_data(pickle_file_path, segment_index=0):
    """Load pose data from pickle file."""
    try:
        with open(pickle_file_path, 'rb') as f:
            pose_segments = pickle.load(f)
        
        print(f"✅ Loaded pickle file: {pickle_file_path}")
        print(f"📊 Found {len(pose_segments)} pose segment(s)")
        
        # Show information about all segments
        if len(pose_segments) > 1:
            print(f"📋 Available segments:")
            for i, segment in enumerate(pose_segments):
                frames = segment.shape[0] if len(segment.shape) >= 1 else "Unknown"
                marker = "👉" if i == segment_index else "  "
                print(f"   {marker} Segment {i}: {frames} frames")
        
        # Get the specified segment
        if len(pose_segments) == 0:
            raise ValueError("No pose segments found in pickle file")
        
        if segment_index >= len(pose_segments):
            raise ValueError(f"Segment index {segment_index} not found (available: 0-{len(pose_segments)-1})")
        
        pose_segment = pose_segments[segment_index]
        print(f"📏 Loading segment {segment_index}: {pose_segment.shape} (frames: {pose_segment.shape[0]})")
        
        # Validate the expected format
        if len(pose_segment.shape) != 3 or pose_segment.shape[1] != 17 or pose_segment.shape[2] != 3:
            raise ValueError(f"Expected shape (frames, 17, 3), got {pose_segment.shape}")
        
        return pose_segment
        
    except FileNotFoundError:
        print(f"❌ Error: Pickle file not found: {pickle_file_path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"💡 Available pickle files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pkl'):
                print(f"   - {file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        sys.exit(1)

def main():
    """Main function to run the interactive pose viewer."""
    print("🚀 Starting Interactive Pose Viewer...")
    print(f"📁 Pickle file: {PICKLE_FILE_PATH}")
    print(f"🎯 Target segment: {SEGMENT_INDEX}")
    
    # Load pose data
    poses = load_pose_data(PICKLE_FILE_PATH, SEGMENT_INDEX)
    
    # Create and show the interactive viewer
    viewer = InteractivePoseViewer(poses)
    viewer.show()

if __name__ == "__main__":
    main() 