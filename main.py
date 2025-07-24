from pose_extractor.extract import extract_pose
from postprocess_poses.filter import process_poses_with_glitch_filtering
from postprocess_poses.normalize import normalize_pose_segments
from visualizations.visualize import create_pose_segment_visualizations
from postprocess_poses.rotate import process_pose_segments
from action_recognition.ctrgcn.inference import run_inference_on_segments, extract_embeddings_from_segments

import pickle, random

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


print("🎬 Starting pose extraction pipeline...")

# Step 1: Extract poses from video
print("📹 Extracting poses from video...")
raw_poses = extract_pose('/home/shanaka/Desktop/thesis/pipeline-final/preprocess_videos/videos/004_t1_20230217_clip_10min.mp4', './')

print(f"Pose extraction completed. Raw poses type: {type(raw_poses)}")

# Step 2: Filter out glitched segments (if enabled)
if GLITCH_FILTERING_ENABLED:
    print("\n🔍 Filtering out glitched segments...")
    
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
    print("\n🔧 Normalizing pose segments...")
    
    normalized_poses = normalize_pose_segments(
        pose_segments=poses_for_normalization,
        target_scale=TARGET_SKELETON_SCALE,
        ema_alpha=EMA_SMOOTHING_ALPHA
    )
    
    poses_to_save = normalized_poses


# Step 4: Rotate poses to front-facing
print("\n🔄 Rotating poses to front-facing...")
rotated_poses = process_pose_segments(normalized_poses)

# Extract embeddings for clustering analysis
print("\n🧠 Extracting embeddings for clustering...")
embeddings_results = extract_embeddings_from_segments(rotated_poses)
print(f"Extracted {len(embeddings_results)} embeddings, each of shape: {embeddings_results[0]['embedding'].shape}")

# Save embeddings for later analysis
embeddings_filename = 'pose_embeddings.pkl'
with open(embeddings_filename, 'wb') as f:
    pickle.dump(embeddings_results, f)
print(f"💾 Saved embeddings to {embeddings_filename}")

filename = 'poses_single_segment.pkl'
# Step 5: Save processed poses
print(f"\n💾 Saving poses to {filename}...")
with open(filename, 'wb') as f:
    pickle.dump(rotated_poses, f)