from pose_extractor.extract import extract_pose
from postprocess_poses.filter import process_poses_with_glitch_filtering, filter_empty_skeleton_segments
from postprocess_poses.normalize import normalize_pose_segments
from postprocess_poses.rotate import process_pose_segments
from action_recognition.ctrgcn.inference import extract_embeddings_from_segments
from clustering.pca import apply_pca_to_embeddings, save_pca_results

import os
import argparse
import json
import cv2

# Configuration for glitch filtering
GLITCH_FILTERING_ENABLED = True
VELOCITY_THRESHOLD = 0.25        # Adjust based on your data sensitivity
ACCELERATION_THRESHOLD = 0.5    # Adjust based on your data sensitivity
CREATE_VISUALIZATIONS = True    # Set to False to skip visualization generation
VISUALIZATION_DIR = "outputs/removed_glitches"

# Configuration for pose normalization
POSE_NORMALIZATION_ENABLED = True
TARGET_SKELETON_SCALE = 1.0      # Target scale for skeleton size
EMA_SMOOTHING_ALPHA = 0.3        # EMA smoothing factor (0 < alpha < 1)
USE_STANDARD_BONE_LENGTHS = True # Use predefined bone length ratios

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from video and save results.")
    parser.add_argument('--video_path', type=str, default='/home/shanaka/Desktop/thesis/pipeline-final/preprocess_videos/videos/004_t1_20230217_clip_10min.mp4', help='Path to input video file')
    parser.add_argument('--poses_output_path', type=str, default='./poses/004_t1_20230217_clip_10min', help='Directory to save pose results')
    parser.add_argument('--embeddings_output_path', type=str, default='./embeddings/004_t1_20230217_clip_10min', help='Directory to save embeddings results')
    parser.add_argument('--summary_dir', type=str, default=None, help='Directory to save filter summary statistics (optional)')
    args = parser.parse_args()

    video_path = args.video_path
    poses_output_path = args.poses_output_path
    embeddings_output_path = args.embeddings_output_path
    summary_dir = args.summary_dir

    print(f"Using video_path: {video_path}")
    print(f"Using poses_output_path: {poses_output_path}")
    print(f"Using embeddings_output_path: {embeddings_output_path}")
    if summary_dir:
        print(f"Using summary_dir: {summary_dir}")

    # Create output directories if they don't exist
    os.makedirs(poses_output_path, exist_ok=True)
    os.makedirs(embeddings_output_path, exist_ok=True)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    # Step 1: Extract poses from video
    print("Extracting poses from video...")
    raw_poses = extract_pose(video_path, poses_output_path)

    print(f"Pose extraction completed. Raw poses type: {type(raw_poses)}")

    # Step 1.5: Filter out empty skeleton segments (new step)
    print("Filtering out empty skeleton segments...")
    poses_after_empty_filter, empty_filter_summary = filter_empty_skeleton_segments(
        pose_segments=raw_poses,
        summary_dir=summary_dir
    )
    
    print(f"Empty skeleton filtering completed. Segments after filtering: {len(poses_after_empty_filter)}")

    # Step 2: Filter out glitched segments (if enabled)
    if GLITCH_FILTERING_ENABLED:
        print("\n🔍 Filtering out glitched segments...")
        
        clean_poses = process_poses_with_glitch_filtering(
            raw_poses=poses_after_empty_filter,
            velocity_threshold=VELOCITY_THRESHOLD,
            acceleration_threshold=ACCELERATION_THRESHOLD,
            create_visualizations=CREATE_VISUALIZATIONS,
            output_dir=VISUALIZATION_DIR,
            summary_dir=summary_dir
        )
        
        print(f"Glitch filtering completed. Clean poses count: {len(clean_poses)}")
        poses_for_normalization = clean_poses
        
    else:
        poses_for_normalization = poses_after_empty_filter

    # Step 3: Normalize poses (if enabled)
    if POSE_NORMALIZATION_ENABLED:
        print("\nNormalizing pose segments...")
        
        normalized_poses = normalize_pose_segments(
            pose_segments=poses_for_normalization,
            target_scale=TARGET_SKELETON_SCALE,
            ema_alpha=EMA_SMOOTHING_ALPHA
        )

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

    # Create combined summary if summary_dir is provided
    if summary_dir:
        print("\n📊 Creating combined processing summary...")
        
        # Calculate video duration (assuming 30 FPS)
        fps = 30
        
        # Use actual video frame count for original_frames
        original_frames = get_video_frame_count(video_path)
        original_duration_seconds = original_frames / fps
        
        final_frames = sum(seg.shape[0] for seg in rotated_poses if hasattr(seg, 'shape'))
        final_duration_seconds = final_frames / fps
        
        # Sanity check
        if final_frames > original_frames:
            print(f"Warning: final_frames ({final_frames}) > original_frames ({original_frames}). Adjusting removed frames to 0.")
            total_removed_frames = 0
            total_time_removed_seconds = 0
            total_time_removed_percentage = 0
        else:
            total_removed_frames = original_frames - final_frames
            total_time_removed_seconds = original_duration_seconds - final_duration_seconds
            total_time_removed_percentage = (total_time_removed_seconds / original_duration_seconds) * 100 if original_duration_seconds > 0 else 0

        combined_summary = {
            'video_info': {
                'video_path': video_path,
                'original_frames': original_frames,
                'original_duration_seconds': original_duration_seconds,
                'final_frames': final_frames,
                'final_duration_seconds': final_duration_seconds,
                'total_removed_frames': total_removed_frames,
                'total_time_removed_seconds': total_time_removed_seconds,
                'total_time_removed_percentage': total_time_removed_percentage
            },
            'processing_steps': {
                'empty_skeleton_filter': empty_filter_summary,
                'glitch_filter_enabled': GLITCH_FILTERING_ENABLED,
                'normalization_enabled': POSE_NORMALIZATION_ENABLED,
                'final_segment_count': len(rotated_poses),
                'final_embedding_count': len(embeddings_results)
            },
            'configuration': {
                'velocity_threshold': VELOCITY_THRESHOLD,
                'acceleration_threshold': ACCELERATION_THRESHOLD,
                'target_skeleton_scale': TARGET_SKELETON_SCALE,
                'ema_smoothing_alpha': EMA_SMOOTHING_ALPHA,
                'pca_components': 50
            }
        }
        
        combined_summary_file = os.path.join(summary_dir, 'combined_processing_summary.json')
        with open(combined_summary_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"💾 Combined processing summary saved to: {combined_summary_file}")
        
        print(f"\n🎯 Final processing results:")
        print(f"   Original video duration: {combined_summary['video_info']['original_duration_seconds']:.1f}s")
        print(f"   Final video duration: {combined_summary['video_info']['final_duration_seconds']:.1f}s")
        print(f"   Total time removed: {combined_summary['video_info']['total_time_removed_seconds']:.1f}s ({combined_summary['video_info']['total_time_removed_percentage']:.1f}%)")
        print(f"   Final segments: {combined_summary['processing_steps']['final_segment_count']}")
        print(f"   Final embeddings: {combined_summary['processing_steps']['final_embedding_count']}")

if __name__ == "__main__":
    main()