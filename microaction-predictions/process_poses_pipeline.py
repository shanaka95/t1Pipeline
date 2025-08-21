#!/usr/bin/env python3
"""
Complete Pose Processing Pipeline for Micro-Action Prediction

This script processes pose data through the complete pipeline:
1. Load poses from .pkl file
2. Concatenate all segments into a single long sequence
3. Apply filtering using postprocess_poses/filter.py
4. Apply normalization using postprocess_poses/normalize.py
5. Apply rotation using postprocess_poses/rotate.py
6. Convert to COCO body style format
7. Segment using postprocess_poses/segment.py with 0.05 threshold
8. Predict micro-actions for all segments using MMN model

Usage:
    python process_poses_pipeline.py --input poses/005_t1_20230519/poses_3D.pkl --output results/005_t1_20230519
"""

import os
import sys
import numpy as np
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add action recognition paths for MMN
sys.path.append('../action_recognition/MMN')

# Add postprocess_poses to path
sys.path.append('../postprocess_poses')

def load_poses_from_pkl(poses_path: str) -> np.ndarray:
    """
    Load and concatenate all pose segments from a PKL file.
    
    Args:
        poses_path (str): Path to the PKL file containing pose segments
        
    Returns:
        np.ndarray: Concatenated pose sequence with shape (total_frames, 17, 3)
    """
    print(f"Loading poses from: {poses_path}")
    
    import pickle
    
    # Load the PKL file
    with open(poses_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict):
        # If it's a dictionary with segment keys
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
            raise ValueError("No valid pose segments found in the PKL file")
    
    elif isinstance(data, list):
        # If it's a list of segments
        print(f"Found {len(data)} pose segments in list format")
        
        all_segments = []
        total_frames = 0
        
        for i, segment in enumerate(data):
            # Remove batch dimension if present
            if len(segment.shape) == 4 and segment.shape[0] == 1:
                segment = segment.squeeze(0)
            
            # Validate segment format
            if len(segment.shape) != 3 or segment.shape[1] != 17 or segment.shape[2] != 3:
                print(f"Warning: Segment {i} has unexpected shape {segment.shape}, skipping...")
                continue
            
            all_segments.append(segment)
            total_frames += segment.shape[0]
            print(f"  Segment {i}: {segment.shape[0]} frames")
        
        # Concatenate all segments
        if all_segments:
            concatenated_poses = np.concatenate(all_segments, axis=0)
            print(f"Concatenated {len(all_segments)} segments into {concatenated_poses.shape[0]} frames")
            return concatenated_poses
        else:
            raise ValueError("No valid pose segments found in the PKL file")
    
    elif isinstance(data, np.ndarray):
        # If it's already a numpy array
        print(f"Found numpy array with shape: {data.shape}")
        
        # Handle extra batch dimension if present
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        
        # Validate format
        if len(data.shape) != 3 or data.shape[1] != 17 or data.shape[2] != 3:
            raise ValueError(f"Invalid pose array shape: {data.shape}, expected (frames, 17, 3)")
        
        print(f"Using pose array with {data.shape[0]} frames")
        return data
    
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

def filter_poses(poses: np.ndarray, velocity_threshold: float = 0.3, 
                acceleration_threshold: float = 0.5) -> np.ndarray:
    """
    Filter poses using the glitch detection logic from postprocess_poses/filter.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        velocity_threshold: Threshold for velocity-based glitch detection
        acceleration_threshold: Threshold for acceleration-based glitch detection
        
    Returns:
        Filtered pose sequence
    """
    print(f"🔍 Filtering poses with velocity_threshold={velocity_threshold}, acceleration_threshold={acceleration_threshold}")
    
    # Import filter functions
    from filter import PoseGlitchDetector
    
    # Initialize detector
    detector = PoseGlitchDetector(velocity_threshold, acceleration_threshold)
    
    # Detect glitches
    glitch_info = detector.detect_glitches(poses)
    
    if glitch_info["has_glitches"]:
        print(f"⚠️  Found {glitch_info['num_glitch_frames']} glitch frames ({glitch_info['glitch_percentage']:.1f}%)")
        
        # Create mask for non-glitch frames
        glitch_mask = np.ones(poses.shape[0], dtype=bool)
        glitch_mask[glitch_info['glitch_frames']] = False
        
        # Filter out glitch frames
        filtered_poses = poses[glitch_mask]
        print(f"✅ Filtered poses: {poses.shape[0]} -> {filtered_poses.shape[0]} frames")
        
        return filtered_poses
    else:
        print(f"✅ No glitches detected, keeping all {poses.shape[0]} frames")
        return poses

def normalize_poses(poses: np.ndarray, target_scale: float = 1.0, 
                   ema_alpha: float = 0.3) -> np.ndarray:
    """
    Normalize poses using postprocess_poses/normalize.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        target_scale: Target skeleton scale
        ema_alpha: EMA smoothing factor
        
    Returns:
        Normalized pose sequence
    """
    print(f"🔧 Normalizing poses with target_scale={target_scale}, ema_alpha={ema_alpha}")
    
    # Import normalize functions
    from normalize import scale_skeleton_to_standard_size, center_at_origin, apply_ema_smoothing
    
    # Apply normalization steps
    normalized = scale_skeleton_to_standard_size(poses, target_scale)
    normalized = center_at_origin(normalized)
    normalized = apply_ema_smoothing(normalized, ema_alpha)
    
    print(f"✅ Normalization completed!")
    return normalized

def rotate_poses(poses: np.ndarray, target_angle_deg: float = -178.55) -> np.ndarray:
    """
    Rotate poses using postprocess_poses/rotate.py
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        target_angle_deg: Target hip angle in degrees
        
    Returns:
        Rotated pose sequence
    """
    print(f"🔄 Rotating poses to front-facing orientation (target angle: {target_angle_deg:.2f}°)")
    
    # Import rotate functions
    from rotate import rotate_skeleton_to_front_facing
    
    # Apply rotation to each frame
    rotated_poses = np.array([
        rotate_skeleton_to_front_facing(pose, target_angle_deg) 
        for pose in poses
    ])
    
    print(f"✅ Rotation completed!")
    return rotated_poses

def convert_h36m_to_coco_format(h36m_pose: np.ndarray) -> np.ndarray:
    """
    Convert H36M pose format (17 joints, 3D) to COCO format (44 joints, 2D)
    
    Args:
        h36m_pose: Shape (T, 17, 3) - H36M pose sequence
        
    Returns:
        Shape (T, 44, 2) - COCO pose sequence
    """
    T, V, C = h36m_pose.shape
    
    # Project 3D to 2D by taking X and Y coordinates
    h36m_2d = h36m_pose[:, :, :2]
    
    # Initialize COCO pose with zeros
    coco_pose = np.zeros((T, 44, 2))
    
    # Map H36M joints to COCO joints
    h36m_to_coco = {
        0: 18,   # root -> left_hip
        1: 19,   # rhip -> right_hip
        2: 21,   # rkne -> right_knee
        3: 23,   # rank -> right_ankle
        4: 18,   # lhip -> left_hip (duplicate)
        5: 20,   # lkne -> left_knee
        6: 22,   # lank -> left_ankle
        7: 18,   # belly -> left_hip (approximation)
        8: 18,   # neck -> left_hip (approximation)
        9: 0,    # nose -> nose
        10: 0,   # head -> nose (approximation)
        11: 12,  # lsho -> left_shoulder
        12: 14,  # lelb -> left_elbow
        13: 16,  # lwri -> left_wrist
        14: 13,  # rsho -> right_shoulder
        15: 15,  # relb -> right_elbow
        16: 17,  # rwri -> right_wrist
    }
    
    # Copy joints from H36M to COCO
    for h36m_idx, coco_idx in h36m_to_coco.items():
        coco_pose[:, coco_idx, :] = h36m_2d[:, h36m_idx, :]
    
    return coco_pose

def segment_poses(poses: np.ndarray, velocity_threshold: float = 0.05, 
                 acceleration_threshold: float = 0.05) -> List[np.ndarray]:
    """
    Segment poses using postprocess_poses/segment.py with specified thresholds
    
    Args:
        poses: Pose sequence with shape (frames, 17, 3)
        velocity_threshold: Threshold for movement detection
        acceleration_threshold: Threshold for movement detection
        
    Returns:
        List of pose segments
    """
    print(f"📊 Segmenting poses with velocity_threshold={velocity_threshold}, acceleration_threshold={acceleration_threshold}")
    
    # Import segment functions
    from segment import MovementSegmenter
    
    # Initialize segmenter
    segmenter = MovementSegmenter(
        max_segment_length=243,
        velocity_threshold=velocity_threshold,
        acceleration_threshold=acceleration_threshold,
        min_segment_length=30
    )
    
    # Segment poses
    segments = segmenter.segment_by_movement(poses)
    
    print(f"✅ Created {len(segments)} segments")
    for i, segment in enumerate(segments):
        print(f"  Segment {i+1}: {segment.shape[0]} frames")
    
    return segments

def create_mmn_data_files(segments: List[np.ndarray], output_dir: str, predictions: Dict[str, int] = None) -> str:
    """
    Create MMN data files in the exact format expected by the feeder
    
    Args:
        segments: List of pose segments
        output_dir: Output directory
        
    Returns:
        Path to the MMN data directory
    """
    print(f"💾 Creating MMN data files...")
    
    # Create MMN data directory structure exactly as expected
    mmn_data_dir = os.path.join(output_dir, 'data', 'MA52')
    os.makedirs(mmn_data_dir, exist_ok=True)
    
    # Prepare data for MMN format
    data_annotations = []
    label_data = []
    segment_labels = {}  # Dictionary to store segment labels
    
    for i, segment in enumerate(segments):
        # MMN expects data in format (T, V, C) where T=64, V=44, C=2
        # Our segment is (T, V, C) where T=variable, V=44, C=2
        
        # Sample to 64 frames if longer, pad if shorter
        if segment.shape[0] > 64:
            indices = np.linspace(0, segment.shape[0]-1, 64, dtype=int)
            segment_resized = segment[indices]
        elif segment.shape[0] < 64:
            # Pad with last frame
            padding = np.repeat(segment[-1:], 64 - segment.shape[0], axis=0)
            segment_resized = np.concatenate([segment, padding], axis=0)
        else:
            segment_resized = segment
        
        # Normalize to reasonable range
        segment_normalized = segment_resized / np.max(np.abs(segment_resized)) if np.max(np.abs(segment_resized)) > 0 else segment_resized
        
        # Create data annotation in the format expected by MMN
        frame_dir = f'segment_{i:03d}'
        data_annotation = {
            'frame_dir': frame_dir,
            'keypoint': [segment_normalized],  # MMN expects list with one element
            'label': 0  # Default label
        }
        data_annotations.append(data_annotation)
        
        # Create label entry
        label_entry = {
            'file_name': frame_dir,
            'label': 0  # Default label
        }
        label_data.append(label_entry)
        
        # Store segment label for JSON output
        segment_name = f'segment_{i:03d}'
        if predictions and segment_name in predictions:
            segment_labels[segment_name] = predictions[segment_name]
        else:
            segment_labels[segment_name] = 0  # Default label as integer
    
    # Create the data structure exactly as expected by MMN
    data_dict = {
        'annotations': data_annotations
    }
    
    # Save data files in the exact format MMN expects
    val_data_file = os.path.join(mmn_data_dir, 'val_data.pkl')
    val_label_file = os.path.join(mmn_data_dir, 'val_label.pkl')
    
    import pickle
    with open(val_data_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    with open(val_label_file, 'wb') as f:
        pickle.dump(label_data, f)
    
    # Save segment labels as JSON
    segment_labels_file = os.path.join(output_dir, 'segment_labels.json')
    import json
    with open(segment_labels_file, 'w') as f:
        json.dump(segment_labels, f, indent=2)
    
    print(f"✅ Created MMN data files:")
    print(f"   - Data file: {val_data_file}")
    print(f"   - Label file: {val_label_file}")
    print(f"   - Segment labels: {segment_labels_file}")
    print(f"   - {len(segments)} segments processed")
    
    return os.path.join(output_dir, 'data')

def extract_mmn_predictions(mmn_dir: str, output_dir: str) -> Dict[str, int]:
    """
    Extract predictions from MMN output files
    
    Args:
        mmn_dir: MMN directory where results are saved
        output_dir: Output directory for our results
        
    Returns:
        Dictionary mapping segment names to predicted labels
    """
    print(f"🔍 Extracting MMN predictions...")
    
    # Look for MMN result files
    import glob
    import pickle
    
    # MMN saves results in work_dir, let's find the latest results
    work_dirs = glob.glob(os.path.join(mmn_dir, 'work_dir', '*'))
    if not work_dirs:
        print("⚠️  No MMN work directories found")
        return {}
    
    # Get the most recent work directory
    latest_work_dir = max(work_dirs, key=os.path.getctime)
    print(f"Found MMN work directory: {latest_work_dir}")
    
    # Look for subdirectories (like MA52_J)
    subdirs = [d for d in os.listdir(latest_work_dir) if os.path.isdir(os.path.join(latest_work_dir, d))]
    if subdirs:
        # Use the first subdirectory
        score_dir = os.path.join(latest_work_dir, subdirs[0])
        print(f"Looking in subdirectory: {score_dir}")
    else:
        score_dir = latest_work_dir
    
    # Look for score.pkl files (these contain the actual predictions)
    score_files = glob.glob(os.path.join(score_dir, '*score.pkl'))
    print(f"Looking for score files in: {score_dir}")
    print(f"Found score files: {score_files}")
    if not score_files:
        print("⚠️  No MMN score files found")
        return {}
    
    # Read the score file
    score_file = score_files[0]
    print(f"Reading MMN scores from: {score_file}")
    
    predictions = {}
    try:
        with open(score_file, 'rb') as f:
            score_dict = pickle.load(f)
        
        # Convert scores to predictions (argmax of scores)
        for segment_idx, scores in score_dict.items():
            predicted_label = int(np.argmax(scores))
            segment_name = f"segment_{segment_idx:03d}"
            predictions[segment_name] = predicted_label
            
    except Exception as e:
        print(f"⚠️  Error reading MMN scores: {e}")
        return {}
    
    print(f"✅ Extracted {len(predictions)} predictions from MMN")
    return predictions



def run_mmn_inference(config_path: str, weights_path: str, data_dir: str, output_dir: str) -> str:
    """
    Run MMN inference using main.py and extract predictions
    
    Args:
        config_path: Path to MMN config file
        weights_path: Path to MMN model weights
        data_dir: Directory containing data files (should contain MA52 subdirectory)
        output_dir: Output directory for results
        
    Returns:
        Path to MMN results
    """
    print(f"🤖 Running MMN inference...")
    
    # Change to MMN directory
    mmn_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), '..', '..'))
    original_dir = os.getcwd()
    
    try:
        os.chdir(mmn_dir)
        
        # Create symbolic link to our data directory in MMN's expected location
        mmn_data_link = os.path.join(mmn_dir, 'data')
        # The data_dir is relative to the original working directory, not the MMN directory
        abs_data_dir = os.path.abspath(os.path.join(original_dir, data_dir))
        
        # Remove existing data link/directory if it exists
        if os.path.exists(mmn_data_link) or os.path.islink(mmn_data_link):
            if os.path.islink(mmn_data_link):
                os.unlink(mmn_data_link)
            elif os.path.isdir(mmn_data_link):
                import shutil
                shutil.rmtree(mmn_data_link)
            elif os.path.isfile(mmn_data_link):
                os.remove(mmn_data_link)
        
        # Create symbolic link to our data directory
        try:
            os.symlink(abs_data_dir, mmn_data_link)
        except FileExistsError:
            # If it still exists, force remove and try again
            if os.path.exists(mmn_data_link) or os.path.islink(mmn_data_link):
                try:
                    os.unlink(mmn_data_link)
                except:
                    import shutil
                    shutil.rmtree(mmn_data_link)
            os.symlink(abs_data_dir, mmn_data_link)
        print(f"Created data symlink: {mmn_data_link} -> {abs_data_dir}")
        
        # Add torchlight and torchpack to Python path and run MMN main.py
        torchlight_path = os.path.join(mmn_dir, 'torchlight')
        torchpack_path = os.path.join(mmn_dir, 'torchpack')
        # Use CPU version of MMN main.py
        cmd = f"PYTHONPATH={torchlight_path}:{torchpack_path}:$PYTHONPATH python main_cpu.py --config ./config/test/MA52_J.yaml --weights ./checkpoints/MMN_MA52_J.pt"
        print(f"Running command: {cmd}")
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Clean up symlink
        if os.path.exists(mmn_data_link):
            os.unlink(mmn_data_link)
        
        if result.returncode != 0:
            print(f"MMN inference failed: {result.stderr}")
            raise RuntimeError(f"MMN inference failed with return code {result.returncode}")
        
        print(f"MMN inference completed successfully")
        print(f"Output: {result.stdout}")
        
        # Extract predictions from MMN output files
        predictions = extract_mmn_predictions(mmn_dir, output_dir)
        
        return mmn_dir, predictions
        
    finally:
        os.chdir(original_dir)

def process_poses_pipeline(input_path: str, output_dir: str, 
                          config_path: str = '../action_recognition/MMN/config/test/MA52_J.yaml',
                          weights_path: str = '../action_recognition/MMN/checkpoints/MMN_MA52_J.pt',
                          device: str = 'cuda') -> Dict:
    """
    Complete pipeline for processing poses and predicting micro-actions
    
    Args:
        input_path: Path to input PKL file
        output_dir: Output directory for results
        config_path: Path to MMN config file
        weights_path: Path to MMN model weights
        device: Device to run inference on
        
    Returns:
        Dictionary with pipeline results
    """
    print("=" * 80)
    print("COMPLETE POSE PROCESSING PIPELINE")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load poses from PKL file
    print(f"\n📁 Step 1: Loading poses from {input_path}")
    poses = load_poses_from_pkl(input_path)
    print(f"✅ Loaded {poses.shape[0]} frames")
    
    # Step 2: Filter poses
    print(f"\n🔍 Step 2: Filtering poses")
    filtered_poses = filter_poses(poses, velocity_threshold=0.3, acceleration_threshold=0.5)
    print(f"✅ Filtered to {filtered_poses.shape[0]} frames")
    
    # Step 3: Normalize poses
    print(f"\n🔧 Step 3: Normalizing poses")
    normalized_poses = normalize_poses(filtered_poses, target_scale=1.0, ema_alpha=0.3)
    print(f"✅ Normalized {normalized_poses.shape[0]} frames")
    
    # Step 4: Rotate poses
    print(f"\n🔄 Step 4: Rotating poses")
    rotated_poses = rotate_poses(normalized_poses, target_angle_deg=-178.55)
    print(f"✅ Rotated {rotated_poses.shape[0]} frames")
    
    # Step 5: Convert to COCO format
    print(f"\n🔄 Step 5: Converting to COCO format")
    coco_poses = convert_h36m_to_coco_format(rotated_poses)
    print(f"✅ Converted to COCO format: {coco_poses.shape}")
    
    # Step 6: Segment poses
    print(f"\n📊 Step 6: Segmenting poses")
    segments = segment_poses(coco_poses, velocity_threshold=0.05, acceleration_threshold=0.05)
    print(f"✅ Created {len(segments)} segments")
    
    # Step 7: Create MMN data files (initially with default labels)
    print(f"\n💾 Step 7: Creating MMN data files")
    mmn_data_dir = create_mmn_data_files(segments, output_dir)
    
    # Step 8: Run MMN inference
    print(f"\n🤖 Step 8: Running MMN inference")
    mmn_results_dir, predictions = run_mmn_inference(config_path, weights_path, mmn_data_dir, output_dir)
    
    # Step 8.5: Update segment labels with MMN predictions
    if predictions:
        print(f"\n🔄 Step 8.5: Updating segment labels with MMN predictions")
        # Recreate data files with real predictions
        mmn_data_dir = create_mmn_data_files(segments, output_dir, predictions)
    
    # Step 9: Save results
    print(f"\n💾 Step 9: Saving results")
    
    # Save processed segments
    segments_file = os.path.join(output_dir, 'processed_segments.npz')
    segment_data = {}
    for i, segment in enumerate(segments):
        segment_data[f'segment_{i:03d}'] = segment
    segment_data['num_segments'] = len(segments)
    np.savez_compressed(segments_file, **segment_data)
    
    # Create summary
    summary = {
        'input_path': input_path,
        'output_dir': output_dir,
        'mmn_results_dir': mmn_results_dir,
        'pipeline_steps': {
            'original_frames': poses.shape[0],
            'filtered_frames': filtered_poses.shape[0],
            'normalized_frames': normalized_poses.shape[0],
            'rotated_frames': rotated_poses.shape[0],
            'coco_frames': coco_poses.shape[0],
            'num_segments': len(segments)
        }
    }
    
    summary_file = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}")
    print(f"  - Segments: {segments_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Segment labels: {os.path.join(output_dir, 'segment_labels.json')}")
    print(f"  - MMN data: {mmn_data_dir}")
    print(f"  - MMN results: {mmn_results_dir}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"📊 Pipeline Summary:")
    print(f"   Original frames: {summary['pipeline_steps']['original_frames']:,}")
    print(f"   Filtered frames: {summary['pipeline_steps']['filtered_frames']:,}")
    print(f"   Final segments: {summary['pipeline_steps']['num_segments']}")
    print(f"   MMN inference completed")
    
    print(f"\n📁 Output Files:")
    print(f"   - Processed segments: {segments_file}")
    print(f"   - Segment labels: {os.path.join(output_dir, 'segment_labels.json')}")
    print(f"   - MMN data directory: {mmn_data_dir}")
    print(f"   - MMN results directory: {mmn_results_dir}")
    
    return summary

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Complete pose processing pipeline for micro-action prediction')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input PKL file (e.g., poses/005_t1_20230519/poses_3D.pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory (e.g., results/005_t1_20230519)')
    parser.add_argument('--config', type=str, 
                       default='../action_recognition/MMN/config/test/MA52_J.yaml',
                       help='Path to MMN config file')
    parser.add_argument('--weights', type=str,
                       default='../action_recognition/MMN/checkpoints/MMN_MA52_J.pt',
                       help='Path to MMN model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Validate config and weights files exist
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights file {args.weights} does not exist")
        sys.exit(1)
    
    # Run pipeline
    try:
        summary = process_poses_pipeline(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config,
            weights_path=args.weights,
            device=args.device
        )
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
