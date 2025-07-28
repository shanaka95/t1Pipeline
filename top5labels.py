from pose_extractor.extract import extract_pose
from postprocess_poses.filter import process_poses_with_glitch_filtering, filter_empty_skeleton_segments
from postprocess_poses.normalize import normalize_pose_segments
from postprocess_poses.rotate import process_pose_segments
from action_recognition.ctrgcn.inference import extract_top5_labels_from_segments

import os
import argparse
import json
import cv2
import pickle

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

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    parser = argparse.ArgumentParser(description="Extract top 5 action labels from video and save results.")
    parser.add_argument('--video_path', type=str, default='/home/shanaka/Desktop/thesis/pipeline-final/preprocess_videos/videos/004_t1_20230217_clip_10min.mp4', help='Path to input video file')
    parser.add_argument('--poses_output_path', type=str, default='./poses/004_t1_20230217_clip_10min', help='Directory to save pose results')
    parser.add_argument('--labels_output_path', type=str, default='./top5_labels/004_t1_20230217_clip_10min', help='Directory to save top 5 labels results')
    parser.add_argument('--summary_dir', type=str, default=None, help='Directory to save filter summary statistics (optional)')
    
    # Component enable/disable flags
    parser.add_argument('--disable_pose_extraction', action='store_true', default=False, help='Disable pose extraction from video (use saved poses instead)')
    parser.add_argument('--disable_empty_filter', action='store_true', default=False, help='Disable empty skeleton segment filtering')
    parser.add_argument('--disable_glitch_filter', action='store_true', default=False, help='Disable glitch filtering')
    parser.add_argument('--disable_normalization', action='store_true', default=False, help='Disable pose normalization')
    parser.add_argument('--disable_rotation', action='store_true', default=False, help='Disable pose rotation to front-facing')
    
    # Path for reading saved poses (when pose extraction is disabled)
    parser.add_argument('--saved_poses_path', type=str, default=None, help='Path to saved poses file (when pose extraction is disabled)')
    
    args = parser.parse_args()

    video_path = args.video_path
    poses_output_path = args.poses_output_path
    labels_output_path = args.labels_output_path
    summary_dir = args.summary_dir
    
    # Component flags (invert the disable flags to get enable flags)
    enable_pose_extraction = not args.disable_pose_extraction
    enable_empty_filter = not args.disable_empty_filter
    enable_glitch_filter = not args.disable_glitch_filter
    enable_normalization = not args.disable_normalization
    enable_rotation = not args.disable_rotation
    saved_poses_path = args.saved_poses_path

    print(f"Using video_path: {video_path}")
    print(f"Using poses_output_path: {poses_output_path}")
    print(f"Using labels_output_path: {labels_output_path}")
    if summary_dir:
        print(f"Using summary_dir: {summary_dir}")
    
    print(f"\nComponent status:")
    print(f"  Pose extraction: {'ENABLED' if enable_pose_extraction else 'DISABLED'}")
    print(f"  Empty filter: {'ENABLED' if enable_empty_filter else 'DISABLED'}")
    print(f"  Glitch filter: {'ENABLED' if enable_glitch_filter else 'DISABLED'}")
    print(f"  Normalization: {'ENABLED' if enable_normalization else 'DISABLED'}")
    print(f"  Rotation: {'ENABLED' if enable_rotation else 'DISABLED'}")

    # Create output directories if they don't exist
    os.makedirs(poses_output_path, exist_ok=True)
    os.makedirs(labels_output_path, exist_ok=True)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    # Step 1: Extract poses from video or load saved poses
    if enable_pose_extraction:
        print("\nExtracting poses from video...")
        raw_poses = extract_pose(video_path, poses_output_path)
        print(f"Pose extraction completed. Raw poses type: {type(raw_poses)}")
    else:
        if saved_poses_path is None:
            raise ValueError("--saved_poses_path must be provided when pose extraction is disabled")
        print(f"\nLoading saved poses from: {saved_poses_path}")
        with open(saved_poses_path, 'rb') as f:
            raw_poses = pickle.load(f)
        print(f"Loaded saved poses. Raw poses type: {type(raw_poses)}")

    # Step 1.5: Filter out empty skeleton segments (if enabled)
    if enable_empty_filter:
        print("\nFiltering out empty skeleton segments...")
        poses_after_empty_filter, empty_filter_summary = filter_empty_skeleton_segments(
            pose_segments=raw_poses,
            summary_dir=summary_dir
        )
        print(f"Empty skeleton filtering completed. Segments after filtering: {len(poses_after_empty_filter)}")
    else:
        print("\nSkipping empty skeleton filtering...")
        poses_after_empty_filter = raw_poses
        empty_filter_summary = {"enabled": False, "segments_removed": 0}

    # Step 2: Filter out glitched segments (if enabled)
    if enable_glitch_filter and GLITCH_FILTERING_ENABLED:
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
        print("\nSkipping glitch filtering...")
        poses_for_normalization = poses_after_empty_filter

    # Step 3: Normalize poses (if enabled)
    if enable_normalization and POSE_NORMALIZATION_ENABLED:
        print("\nNormalizing pose segments...")
        
        normalized_poses = normalize_pose_segments(
            pose_segments=poses_for_normalization,
            target_scale=TARGET_SKELETON_SCALE,
            ema_alpha=EMA_SMOOTHING_ALPHA
        )
    else:
        print("\nSkipping pose normalization...")
        normalized_poses = poses_for_normalization

    # Step 4: Rotate poses to front-facing (if enabled)
    if enable_rotation:
        print("\nRotating poses to front-facing...")
        rotated_poses = process_pose_segments(normalized_poses)
    else:
        print("\nSkipping pose rotation...")
        rotated_poses = normalized_poses

    # Extract top 5 labels for each segment
    print("\nExtracting top 5 action labels...")
    top5_labels_results = extract_top5_labels_from_segments(rotated_poses)
    print(f"Extracted top 5 labels for {len(top5_labels_results)} segments")

    # Save top 5 labels results
    labels_file_path = os.path.join(labels_output_path, 'top5_labels.pkl')
    with open(labels_file_path, 'wb') as f:
        pickle.dump(top5_labels_results, f)
    print(f"Top 5 labels saved to: {labels_file_path}")

    # Print sample results
    if top5_labels_results:
        print(f"\nSample top 5 labels for first segment:")
        print(f"Sequence ID: {top5_labels_results[0]['sequence_id']}")
        print(f"Top 5 labels: {top5_labels_results[0]['top5_labels']}")

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
                'pose_extraction_enabled': enable_pose_extraction,
                'saved_poses_path': saved_poses_path if not enable_pose_extraction else None,
                'empty_skeleton_filter': empty_filter_summary,
                'glitch_filter_enabled': enable_glitch_filter and GLITCH_FILTERING_ENABLED,
                'normalization_enabled': enable_normalization and POSE_NORMALIZATION_ENABLED,
                'rotation_enabled': enable_rotation,
                'final_segment_count': len(rotated_poses),
                'final_labels_count': len(top5_labels_results)
            },
            'configuration': {
                'velocity_threshold': VELOCITY_THRESHOLD,
                'acceleration_threshold': ACCELERATION_THRESHOLD,
                'target_skeleton_scale': TARGET_SKELETON_SCALE,
                'ema_smoothing_alpha': EMA_SMOOTHING_ALPHA
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
        print(f"   Final labels: {combined_summary['processing_steps']['final_labels_count']}")

if __name__ == "__main__":
    main()