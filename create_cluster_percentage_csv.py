#!/usr/bin/env python3
"""
Script to create CSV file with video cluster percentages.
Each row represents a video, each column represents a cluster percentage.
"""

import json
import os
import csv
from collections import defaultdict

def load_clustering_data(clustering_file):
    """Load clustering data from JSON file."""
    with open(clustering_file, 'r') as f:
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

def count_segments_per_video_per_cluster(clustering_data):
    """Count how many segments each video has in each cluster."""
    n_clusters = clustering_data['n_clusters']
    
    # Initialize data structure: video -> cluster -> segment_count
    video_cluster_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each cluster
    for cluster_id in range(n_clusters):
        cluster_data = clustering_data['cluster_summary'][str(cluster_id)]
        
        # For each video in this cluster
        for video_name, segment_indices in cluster_data['videos'].items():
            # Count segments for this video in this cluster
            video_cluster_counts[video_name][cluster_id] = len(segment_indices)
    
    return video_cluster_counts

def calculate_segment_duration(frames_per_segment=243, fps=30):
    """Calculate duration of one segment in seconds."""
    return frames_per_segment / fps

def calculate_percentages(video_cluster_counts, video_durations, frames_per_segment=243, fps=30):
    """Calculate percentage of each cluster for each video."""
    segment_duration = calculate_segment_duration(frames_per_segment, fps)
    
    video_percentages = {}
    
    for video_name, cluster_counts in video_cluster_counts.items():
        if video_name not in video_durations:
            print(f"Warning: No duration data for video {video_name}")
            continue
            
        total_duration = video_durations[video_name]
        
        # Calculate total segments for this video
        total_segments = sum(cluster_counts.values())
        total_segment_duration = total_segments * segment_duration
        
        # Calculate percentages for each cluster
        percentages = {}
        for cluster_id in range(100):  # 0 to 99
            segment_count = cluster_counts.get(cluster_id, 0)
            cluster_duration = segment_count * segment_duration
            percentage = cluster_duration / total_duration if total_duration > 0 else 0
            percentages[cluster_id] = percentage
            
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
    """Create CSV file with video cluster percentages."""
    
    # Prepare data for CSV
    rows = []
    for video_name, percentages in video_percentages.items():
        row = {'video_name': video_name}
        for cluster_id in range(100):
            row[f'cluster_{cluster_id:03d}'] = percentages.get(cluster_id, 0.0)
        rows.append(row)
    
    # Sort by video name for consistency
    rows.sort(key=lambda x: x['video_name'])
    
    # Write CSV
    if rows:
        fieldnames = ['video_name'] + [f'cluster_{i:03d}' for i in range(100)]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"CSV file created: {output_file}")
        print(f"Number of videos: {len(rows)}")
        
        # Validate percentages
        for row in rows[:5]:  # Check first 5 videos
            total = sum(row[f'cluster_{i:03d}'] for i in range(100))
            print(f"Video {row['video_name']}: Total percentage = {total:.6f}")
    else:
        print("No data to write to CSV")

def main():
    # File paths
    clustering_file = 'test/clustering_info_with_top5_labels/100/clustering_summary.json'
    video_data_dir = 'test/video_data'
    output_file = 'video_cluster_percentages.csv'
    
    print("Loading clustering data...")
    clustering_data = load_clustering_data(clustering_file)
    
    print("Loading video durations...")
    video_durations = load_video_durations(video_data_dir)
    print(f"Found duration data for {len(video_durations)} videos")
    
    print("Counting segments per video per cluster...")
    video_cluster_counts = count_segments_per_video_per_cluster(clustering_data)
    print(f"Found clustering data for {len(video_cluster_counts)} videos")
    
    print("Calculating percentages...")
    video_percentages = calculate_percentages(video_cluster_counts, video_durations)
    
    print("Creating CSV file...")
    create_csv(video_percentages, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main() 