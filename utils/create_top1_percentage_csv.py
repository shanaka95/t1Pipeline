#!/usr/bin/env python3
"""
Script to create CSV file with video action class percentages from top1 labels.
Each row represents a video, each column represents an action class percentage (0-51).
This is similar to cluster percentages but uses direct action class assignments.
"""

import json
import os
import csv
from collections import defaultdict

def load_grouping_data(grouping_file):
    """Load grouping data from JSON file."""
    with open(grouping_file, 'r') as f:
        return json.load(f)

def load_video_durations(video_data_dir):
    """Load video durations from combined_processing_summary.json files."""
    video_durations = {}
    
    for video_dir in os.listdir(video_data_dir):
        video_dir_path = os.path.join(video_data_dir, video_dir)
        if not os.path.isdir(video_dir_path):
            continue
            
        summary_file = os.path.join(video_dir_path, 'combined_processing_summary.json')
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    video_name = video_dir  # Use directory name as video name
                    final_duration = data['video_info']['final_duration_seconds']
                    video_durations[video_name] = final_duration
            except Exception as e:
                print(f"Error reading {summary_file}: {e}")
                
    return video_durations

def count_segments_per_video_per_action_class(grouping_data):
    """Count how many segments each video has in each action class."""
    n_groups = grouping_data['n_groups']  # Should be 52 for action classes
    
    # Initialize data structure: video -> action_class -> segment_count
    video_action_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each action class group
    for group_id in range(n_groups):
        group_data = grouping_data['group_summary'][str(group_id)]
        
        # For each video in this action class
        for video_name, segment_indices in group_data['videos'].items():
            # Count segments for this video in this action class
            video_action_counts[video_name][group_id] = len(segment_indices)
    
    return video_action_counts

def calculate_segment_duration(frames_per_segment=243, fps=30):
    """Calculate duration of one segment in seconds."""
    return frames_per_segment / fps

def calculate_percentages(video_action_counts, video_durations, frames_per_segment=243, fps=30):
    """Calculate percentage of each action class for each video."""
    segment_duration = calculate_segment_duration(frames_per_segment, fps)
    
    video_percentages = {}
    
    for video_name, action_counts in video_action_counts.items():
        if video_name not in video_durations:
            print(f"Warning: No duration data for video {video_name}")
            continue
            
        total_duration = video_durations[video_name]
        
        # Calculate total segments for this video
        total_segments = sum(action_counts.values())
        total_segment_duration = total_segments * segment_duration
        
        # Calculate percentages for each action class
        percentages = {}
        for action_class in range(52):  # 0 to 51 action classes
            segment_count = action_counts.get(action_class, 0)
            action_duration = segment_count * segment_duration
            percentage = action_duration / total_duration if total_duration > 0 else 0
            percentages[action_class] = percentage
            
        video_percentages[video_name] = percentages
        
        # Verify percentages sum to reasonable value
        total_percentage = sum(percentages.values())
        if total_percentage > 0:
            print(f"Video {video_name}: {total_segments} segments, "
                  f"Total duration: {total_duration:.1f}s, "
                  f"Segment duration: {total_segment_duration:.1f}s, "
                  f"Coverage: {total_percentage:.3f}")
    
    return video_percentages

def create_csv(video_percentages, output_file):
    """Create CSV file with video action class percentages."""
    
    # Prepare data for CSV
    rows = []
    for video_name, percentages in video_percentages.items():
        row = {'video_name': video_name}
        for action_class in range(52):
            row[f'action_class_{action_class:02d}'] = percentages.get(action_class, 0.0)
        rows.append(row)
    
    # Sort by video name for consistency
    rows.sort(key=lambda x: x['video_name'])
    
    # Write CSV
    if rows:
        fieldnames = ['video_name'] + [f'action_class_{i:02d}' for i in range(52)]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"CSV file created: {output_file}")
        print(f"Number of videos: {len(rows)}")
        
        # Validate percentages
        for row in rows[:5]:  # Check first 5 videos
            total = sum(row[f'action_class_{i:02d}'] for i in range(52))
            print(f"Video {row['video_name']}: Total percentage = {total:.6f}")
    else:
        print("No data to write to CSV")

def main():
    # File paths
    grouping_file = 'grouping_results_top1/grouping_summary.json'
    video_data_dir = 'test/video_data'
    output_file = 'video_action_class_percentages_top1.csv'
    
    print("Loading grouping data...")
    grouping_data = load_grouping_data(grouping_file)
    
    print("Loading video durations...")
    video_durations = load_video_durations(video_data_dir)
    print(f"Found duration data for {len(video_durations)} videos")
    
    print("Counting segments per video per action class...")
    video_action_counts = count_segments_per_video_per_action_class(grouping_data)
    print(f"Found grouping data for {len(video_action_counts)} videos")
    
    print("Calculating percentages...")
    video_percentages = calculate_percentages(video_action_counts, video_durations)
    
    print("Creating CSV file...")
    create_csv(video_percentages, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()