import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_2d_poses(npz_path, output_dir="data/viz"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the saved keypoints
    data = np.load(npz_path)
    keypoints = data['keypoints']  # [frames, 17, 3] - x, y, confidence
    frame_count = data['frame_count']
    person_detected_frame = data['person_detected_frame']
    vid_size = data['vid_size'] if 'vid_size' in data else None
    
    print(f"Loaded keypoints with shape: {keypoints.shape}")
    print(f"Video size: {vid_size}")
    print(f"Frame count: {frame_count}")
    print(f"Person detected from frame: {person_detected_frame}")
    
    # Define the connections between keypoints for H36M format
    skeleton_connections = [
        [0, 7],       # root -> belly
        [7, 8],       # belly -> neck
        [8, 9],       # neck -> nose
        [9, 10],      # nose -> head
        [8, 11],      # neck -> left shoulder
        [11, 12],     # left shoulder -> left elbow
        [12, 13],     # left elbow -> left wrist
        [8, 14],      # neck -> right shoulder
        [14, 15],     # right shoulder -> right elbow
        [15, 16],     # right elbow -> right wrist
        [0, 1],       # root -> right hip
        [1, 2],       # right hip -> right knee
        [2, 3],       # right knee -> right ankle
        [0, 4],       # root -> left hip
        [4, 5],       # left hip -> left knee
        [5, 6],       # left knee -> left ankle
    ]
    
    # H36M keypoint names for reference
    h36m_names = [
        'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank',
        'belly', 'neck', 'nose', 'head', 'lsho', 'lelb', 'lwri',
        'rsho', 'relb', 'rwri'
    ]
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Visualize each frame
    for frame_idx in range(len(keypoints)):
        plt.clf()  # Clear the current figure
        
        # Get current frame's keypoints
        pose = keypoints[frame_idx]
        
        # Create plot area
        if vid_size is not None:
            plt.xlim(0, vid_size[0])
            plt.ylim(0, vid_size[1])
        
        # Filter keypoints by confidence
        confidence_threshold = 0.2
        valid_keypoints = pose[:, 2] > confidence_threshold
        
        # Plot the keypoints with confidence above threshold
        plt.scatter(pose[valid_keypoints, 0], pose[valid_keypoints, 1], 
                   c='red', s=50, alpha=0.8)
        
        # Add keypoint labels for easier debugging
        for i, name in enumerate(h36m_names):
            if pose[i, 2] > confidence_threshold:
                plt.annotate(name, (pose[i, 0], pose[i, 1]), fontsize=8)
        
        # Plot keypoints with lower confidence in a different color
        lower_conf = (pose[:, 2] > 0) & (pose[:, 2] <= confidence_threshold)
        if np.any(lower_conf):
            plt.scatter(pose[lower_conf, 0], pose[lower_conf, 1], 
                      c='orange', s=30, alpha=0.5)
        
        # Draw the skeleton connections
        for connection in skeleton_connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # Only draw if both keypoints have confidence above threshold
            if pose[start_idx, 2] > confidence_threshold and pose[end_idx, 2] > confidence_threshold:
                plt.plot([pose[start_idx, 0], pose[end_idx, 0]],
                        [pose[start_idx, 1], pose[end_idx, 1]], 'b-', alpha=0.7, linewidth=2)
            # Draw dashed line for lower confidence connections
            elif pose[start_idx, 2] > 0 and pose[end_idx, 2] > 0:
                plt.plot([pose[start_idx, 0], pose[end_idx, 0]],
                        [pose[start_idx, 1], pose[end_idx, 1]], 'b--', alpha=0.4, linewidth=1)
        
        # Invert y-axis since image coordinates start from top
        plt.gca().invert_yaxis()
        plt.title(f'Frame {frame_idx + person_detected_frame} - H36M Format')
        plt.axis('equal')
        
        # Save the frame as an image
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        
        # Add a small pause to create animation effect if displaying
        plt.pause(0.01)
    
    print(f"Saved {len(keypoints)} visualization frames to {output_dir}")
    
    # Create a video from the saved frames
    create_video_from_frames(output_dir, "data/pose_visualization.mp4")
    
    plt.show()

def create_video_from_frames(image_folder, output_video_path, fps=30):
    """Create a video from a sequence of images"""
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    
    if not images:
        print("No images found in the folder.")
        return
    
    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    # Release resources
    cv2.destroyAllWindows()
    video.release()
    
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    npz_path = "data/keypoints_2d.npz"
    visualize_2d_poses(npz_path) 