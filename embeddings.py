from pose_extractor.extract import extract_pose
from postprocess_poses.filter import process_poses_with_glitch_filtering
from postprocess_poses.normalize import normalize_pose_segments
from postprocess_poses.rotate import process_pose_segments
from action_recognition.ctrgcn.inference import extract_embeddings_from_segments
from clustering.pca import apply_pca_to_embeddings, save_pca_results

import os
import argparse

# Configuration for glitch filtering
GLITCH_FILTERING_ENABLED = True
VELOCITY_THRESHOLD = 0.25        # Adjust based on your data sensitivity
ACCELERATION_THRESHOLD = 0.5    # Adjust based on your data sensitivity
CREATE_VISUALIZATIONS = False    # Set to False to skip visualization generation
VISUALIZATION_DIR = "outputs/removed_glitches"

# Configuration for pose normalization
POSE_NORMALIZATION_ENABLED = True
TARGET_SKELETON_SCALE = 1.0      # Target scale for skeleton size
EMA_SMOOTHING_ALPHA = 0.3        # EMA smoothing factor (0 < alpha < 1)
USE_STANDARD_BONE_LENGTHS = True # Use predefined bone length ratios

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from video and save results.")
    parser.add_argument('--video_path', type=str, default='/home/shanaka/Desktop/thesis/pipeline-final/preprocess_videos/videos/004_t1_20230217_clip_10min.mp4', help='Path to input video file')
    parser.add_argument('--poses_output_path', type=str, default='./poses/004_t1_20230217_clip_10min', help='Directory to save pose results')
    parser.add_argument('--embeddings_output_path', type=str, default='./embeddings/004_t1_20230217_clip_10min', help='Directory to save embeddings results')
    args = parser.parse_args()

    video_path = args.video_path
    poses_output_path = args.poses_output_path
    embeddings_output_path = args.embeddings_output_path

    print(f"Using video_path: {video_path}")
    print(f"Using poses_output_path: {poses_output_path}")
    print(f"Using embeddings_output_path: {embeddings_output_path}")

    # Create output directories if they don't exist
    os.makedirs(poses_output_path, exist_ok=True)
    os.makedirs(embeddings_output_path, exist_ok=True)

    # Step 1: Extract poses from video
    print("Extracting poses from video...")
    raw_poses = extract_pose(video_path, poses_output_path)

    print(f"Pose extraction completed. Raw poses type: {type(raw_poses)}")

    # Step 2: Filter out glitched segments (if enabled)
    if GLITCH_FILTERING_ENABLED:
        print("Filtering out glitched segments...")
        
        clean_poses = process_poses_with_glitch_filtering(
            raw_poses=raw_poses,
            velocity_threshold=VELOCITY_THRESHOLD,
            acceleration_threshold=ACCELERATION_THRESHOLD,
            create_visualizations=CREATE_VISUALIZATIONS,
            output_dir=VISUALIZATION_DIR
        )
        
        print(f"Glitch filtering completed. Clean poses count: {len(clean_poses)}")
        poses_for_normalization = clean_poses
        
    else:
        poses_for_normalization = raw_poses

    # selected_indices = [0, 1, 2, 3, 4]

    # create_pose_segment_visualizations(
    #     pose_segments=poses_for_normalization,
    #     output_dir='visualizations/poses',
    #     segment_indices=selected_indices,
    #     fps=30,
    #     max_frames_per_segment=243,
    #     frame_skip=1,
    #     prefix="raw"
    # )

    # Step 3: Normalize poses (if enabled)
    if POSE_NORMALIZATION_ENABLED:
        print("Normalizing pose segments...")
        
        normalized_poses = normalize_pose_segments(
            pose_segments=poses_for_normalization,
            target_scale=TARGET_SKELETON_SCALE,
            ema_alpha=EMA_SMOOTHING_ALPHA
        )
        
        poses_to_save = normalized_poses


    # Step 4: Rotate poses to front-facing
    print("Rotating poses to front-facing...")
    rotated_poses = process_pose_segments(normalized_poses)

    # Extract embeddings for clustering analysis
    print("Extracting embeddings for clustering...")
    embeddings_results = extract_embeddings_from_segments(rotated_poses)
    print(f"Extracted {len(embeddings_results)} embeddings, each of shape: {embeddings_results[0]['embedding'].shape}")

    # Apply PCA to reduce dimensionality
    print("Applying PCA for dimensionality reduction...")
    pca_results = apply_pca_to_embeddings(embeddings_results, n_components=50, standardize=True)

    save_pca_results(pca_results, f'{embeddings_output_path}/pca_results.pkl')

if __name__ == "__main__":
    main()