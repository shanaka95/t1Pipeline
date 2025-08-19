#!/usr/bin/env python3
"""
Create a CSV file with video names and action class counts (0-51) based on top1 labels.
Counts how many segments have each action class as the first (top1) label per video.
"""

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

def load_top1_labels_from_test_dir(test_labels_dir):
    """
    Load top 1 labels data from the test directory structure
    Each video has its own folder with top5_labels.pkl, but we only take the first label
    
    Returns:
    - video_counts: dict with video_name as key and action_class_counts as value
    """
    video_counts = defaultdict(lambda: [0] * 52)  # Initialize counts for 52 action classes (0-51)
    
    # Walk through the test labels directory
    video_dirs = [d for d in os.listdir(test_labels_dir) if os.path.isdir(os.path.join(test_labels_dir, d))]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        video_path = os.path.join(test_labels_dir, video_dir)
        pkl_file = os.path.join(video_path, 'top5_labels.pkl')
        
        if os.path.exists(pkl_file):
            print(f"Processing labels from: {video_dir}")
            
            with open(pkl_file, 'rb') as f:
                labels_data = pickle.load(f)
            
            # Count top1 labels for this video
            for segment in labels_data:
                top1_label = segment['top5_labels'][0]  # Take only the first label
                if 0 <= top1_label <= 51:  # Ensure valid action class range
                    video_counts[video_dir][top1_label] += 1
                else:
                    print(f"Warning: Invalid action class {top1_label} in video {video_dir}")
    
    print(f"Processed {len(video_counts)} videos")
    return video_counts

def create_action_class_csv(video_counts, output_file):
    """
    Create CSV file with video names and action class counts
    """
    # Create DataFrame
    columns = ['video_name'] + [f'action_class_{i:02d}' for i in range(52)]
    
    data = []
    for video_name, counts in video_counts.items():
        row = [video_name] + counts
        data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    
    # Sort by video name for consistency
    df = df.sort_values('video_name').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"CSV file saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   • Total videos: {len(df)}")
    print(f"   • Total segments across all videos: {df.iloc[:, 1:].sum().sum():,}")
    
    # Show top 5 most frequent action classes across all videos
    action_class_totals = df.iloc[:, 1:].sum().sort_values(ascending=False)
    print(f"\n🏆 Top 5 Most Frequent Action Classes (across all videos):")
    for i, (action_class, count) in enumerate(action_class_totals.head().items()):
        print(f"   {i+1}. {action_class}: {count:,} segments")
    
    # Show videos with most segments
    total_segments_per_video = df.iloc[:, 1:].sum(axis=1)
    df_with_totals = df.copy()
    df_with_totals['total_segments'] = total_segments_per_video
    
    print(f"\n📹 Videos with Most Segments:")
    top_videos = df_with_totals.nlargest(5, 'total_segments')[['video_name', 'total_segments']]
    for _, row in top_videos.iterrows():
        print(f"   • {row['video_name']}: {row['total_segments']:,} segments")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create CSV file with action class counts based on top1 labels')
    parser.add_argument('--test_dir', type=str, default='test/top_5labels',
                        help='Path to test directory containing top5_labels (default: test/top_5labels)')
    parser.add_argument('--output_file', type=str, default='datasets/video_action_class_counts.csv',
                        help='Output CSV file path (default: datasets/video_action_class_counts.csv)')
    
    args = parser.parse_args()
    
    # Configuration
    test_labels_dir = args.test_dir
    output_file = args.output_file
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Step 1: Load and process top 1 labels data
    print("🚀 Loading top 1 labels data from test directory...")
    video_counts = load_top1_labels_from_test_dir(test_labels_dir)
    
    # Step 2: Create CSV file
    print(f"\n📝 Creating CSV file with action class counts...")
    df = create_action_class_csv(video_counts, output_file)
    
    print(f"\n✅ Action class counts CSV creation completed successfully!")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    main()

