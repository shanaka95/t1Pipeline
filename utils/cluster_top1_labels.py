#!/usr/bin/env python3
"""
Apply direct label-based grouping to top 1 labels from test directory.
Since we know the labels are always 0-51 (52 action classes), we don't need k-means clustering.
We simply assign each segment to its corresponding group based on the first (top1) label.
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict

def load_top1_labels_from_test_dir(test_labels_dir):
    """
    Load top 1 labels data from the test directory structure
    Each video has its own folder with top5_labels.pkl, but we only take the first label
    
    Returns:
    - labels_data: numpy array of shape (n_segments,) containing top1 labels
    - segment_metadata: list of dicts with segment info including video_name
    """
    all_labels_data = []
    segment_metadata = []
    
    # Walk through the test labels directory
    video_dirs = [d for d in os.listdir(test_labels_dir) if os.path.isdir(os.path.join(test_labels_dir, d))]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        video_path = os.path.join(test_labels_dir, video_dir)
        pkl_file = os.path.join(video_path, 'top5_labels.pkl')
        
        if os.path.exists(pkl_file):
            print(f"Loading labels from: {video_dir}")
            
            with open(pkl_file, 'rb') as f:
                labels_data = pickle.load(f)
            
            # Extract only the first label (top1) from each segment and store metadata
            for segment in labels_data:
                top1_label = segment['top5_labels'][0]  # Take only the first label
                all_labels_data.append(top1_label)
                segment_metadata.append({
                    'video_name': video_dir,
                    'sequence_id': segment['sequence_id'],
                    'global_segment_id': len(segment_metadata),  # Unique ID across all videos
                    'top1_label': top1_label
                })
    
    print(f"Loaded {len(all_labels_data)} total segments from {len(video_dirs)} videos")
    return np.array(all_labels_data), segment_metadata

def apply_direct_labeling_to_top1(labels_data, segment_metadata, output_dir, visualization_dir):
    """
    Apply direct label-based grouping to top 1 labels data with comprehensive metadata tracking.
    No k-means needed - we have 52 predefined groups (0-51).
    """
    # Since labels are already the group assignments (0-51), we use them directly
    group_labels = labels_data  # Direct assignment - no clustering needed
    n_groups = 52  # Fixed number of action classes (0-51)
    
    # Verify all labels are within expected range
    min_label = np.min(group_labels)
    max_label = np.max(group_labels)
    print(f"Label range: {min_label} to {max_label}")
    
    if min_label < 0 or max_label > 51:
        print(f"WARNING: Found labels outside expected range [0, 51]")
    
    # Create comprehensive grouping results (similar structure to clustering results)
    grouping_results = {
        'labels_data': labels_data,
        'group_labels': group_labels,  # Same as labels_data for direct assignment
        'n_groups': n_groups,
        'segment_metadata': segment_metadata,
        'total_segments': len(labels_data),
        'method': 'direct_labeling',  # Indicate this is not k-means
        'label_distribution': np.bincount(group_labels, minlength=52).tolist()
    }
    
    # Add group assignment to metadata (redundant but consistent with clustering version)
    for i, metadata in enumerate(segment_metadata):
        metadata['group_id'] = int(group_labels[i])
    
    # Create group summary (similar to cluster summary)
    group_summary = {}
    for group_id in range(n_groups):
        group_indices = np.where(group_labels == group_id)[0]
        group_segments = [segment_metadata[i] for i in group_indices]
        
        # Group by video
        videos_in_group = {}
        for segment in group_segments:
            video_name = segment['video_name']
            if video_name not in videos_in_group:
                videos_in_group[video_name] = []
            videos_in_group[video_name].append(segment['sequence_id'])
        
        group_summary[group_id] = {
            'total_segments': len(group_segments),
            'num_videos': len(videos_in_group),
            'videos': videos_in_group,
            'action_class': group_id  # The action class this group represents
        }
    
    grouping_results['group_summary'] = group_summary
    
    # Save grouping results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'direct_top1_labels_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(grouping_results, f)
    
    # Save human-readable summary as JSON
    json_summary = {
        'n_groups': n_groups,
        'total_segments': len(labels_data),
        'method': 'direct_labeling',
        'label_range': f"{min_label}-{max_label}",
        'group_summary': group_summary,
        'label_distribution': grouping_results['label_distribution']
    }
    json_path = os.path.join(output_dir, 'grouping_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Grouping results saved to: {results_path}")
    print(f"Grouping summary saved to: {json_path}")
    
    # Create visualizations
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        generate_grouping_visualizations(grouping_results, visualization_dir)
    
    return grouping_results

def generate_grouping_visualizations(grouping_results, visualization_dir):
    """
    Generate comprehensive visualizations for the direct label grouping results
    """
    group_labels = grouping_results['group_labels']
    n_groups = grouping_results['n_groups']
    labels_data = grouping_results['labels_data']
    label_distribution = grouping_results['label_distribution']
    
    # 1. Label distribution bar plot
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(n_groups)
    counts = np.array(label_distribution)
    
    # Color bars based on frequency (more frequent = darker)
    colors = plt.cm.viridis(counts / (counts.max() if counts.max() > 0 else 1))
    
    bars = plt.bar(x_pos, counts, color=colors, edgecolor='navy', alpha=0.7)
    plt.title(f'Action Class Distribution (Direct Top1 Labeling, n_groups={n_groups})')
    plt.xlabel('Action Class ID')
    plt.ylabel('Number of Segments')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on top of bars (only for non-zero bars to avoid clutter)
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(i, count + counts.max()*0.01, str(count), ha='center', va='bottom', fontsize=8)
    
    # Set x-axis ticks (show every 5th label to avoid overcrowding)
    plt.xticks(x_pos[::5], x_pos[::5])
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'action_class_distribution_barplot_top1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Video distribution across action classes
    group_summary = grouping_results['group_summary']
    video_counts_per_group = [group_summary[i]['num_videos'] for i in range(n_groups)]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(x_pos, video_counts_per_group, color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.title('Number of Videos per Action Class')
    plt.xlabel('Action Class ID')
    plt.ylabel('Number of Videos')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on top of bars (only for non-zero bars)
    for i, count in enumerate(video_counts_per_group):
        if count > 0:
            plt.text(i, count + max(video_counts_per_group)*0.01, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.xticks(x_pos[::5], x_pos[::5])
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'videos_per_action_class_top1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top 20 most frequent action classes
    top_20_indices = np.argsort(counts)[-20:][::-1]  # Top 20 by frequency
    top_20_counts = counts[top_20_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(20), top_20_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Top 20 Most Frequent Action Classes')
    plt.xlabel('Rank')
    plt.ylabel('Number of Segments')
    plt.grid(axis='y', alpha=0.3)
    
    # Add action class IDs as labels
    for i, (idx, count) in enumerate(zip(top_20_indices, top_20_counts)):
        plt.text(i, count + top_20_counts.max()*0.01, f'Class {idx}\n({count})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'top20_action_classes_top1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Coverage statistics plot
    non_zero_classes = np.sum(counts > 0)
    total_classes = len(counts)
    coverage_percentage = (non_zero_classes / total_classes) * 100
    
    plt.figure(figsize=(10, 6))
    plt.pie([non_zero_classes, total_classes - non_zero_classes], 
            labels=[f'Used Classes\n({non_zero_classes})', f'Unused Classes\n({total_classes - non_zero_classes})'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title(f'Action Class Coverage\n({coverage_percentage:.1f}% of classes used)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'action_class_coverage_top1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {visualization_dir}")
    print(f"📊 Statistics:")
    print(f"   • Total action classes: {total_classes}")
    print(f"   • Used action classes: {non_zero_classes} ({coverage_percentage:.1f}%)")
    print(f"   • Most frequent class: {np.argmax(counts)} ({counts.max()} segments)")
    print(f"   • Least frequent class (non-zero): {np.argmin(counts[counts > 0])} ({counts[counts > 0].min()} segments)")

# Set up argument parser
parser = argparse.ArgumentParser(description='Apply direct label-based grouping to top 1 labels from test directory')
parser.add_argument('--test_dir', type=str, default='test/top_5labels',
                    help='Path to test directory containing top5_labels (default: test/top_5labels)')
parser.add_argument('--grouping_output_dir', type=str, default='./grouping_results_top1',
                    help='Directory to save grouping results (default: ./grouping_results_top1)')
parser.add_argument('--visualization_dir', type=str, default='./visualizations/clustering_top1',
                    help='Directory to save grouping visualizations (default: ./visualizations/clustering_top1)')

# Parse arguments
args = parser.parse_args()

# Configuration
test_labels_dir = args.test_dir
grouping_output_dir = args.grouping_output_dir
visualization_dir = args.visualization_dir

# Step 1: Load top 1 labels data from test directory
print("🚀 Loading top 1 labels data from test directory...")
labels_data, segment_metadata = load_top1_labels_from_test_dir(test_labels_dir)
print(f"Loaded {len(labels_data)} segments from test directory")

# Step 2: Apply direct label-based grouping to top 1 labels
print(f"\n🔍 Applying direct label-based grouping with 52 action classes...")
grouping_results = apply_direct_labeling_to_top1(
    labels_data=labels_data,
    segment_metadata=segment_metadata,
    output_dir=grouping_output_dir,
    visualization_dir=visualization_dir
)

# Step 3: Display summary statistics
print(f"\n📊 Grouping Summary:")
print(f"   • Total segments: {len(labels_data):,}")
print(f"   • Number of action classes: 52 (0-51)")
print(f"   • Method: Direct labeling (no clustering)")

group_summary = grouping_results['group_summary']
label_distribution = grouping_results['label_distribution']
used_classes = sum(1 for count in label_distribution if count > 0)

print(f"\n📈 Action Class Details:")
print(f"   • Used action classes: {used_classes}/52 ({used_classes/52*100:.1f}%)")
print(f"   • Most frequent class: {np.argmax(label_distribution)} ({max(label_distribution)} segments)")
print(f"   • Average segments per used class: {len(labels_data)/used_classes:.1f}")

# Show top 10 most frequent classes
top_classes = sorted(range(52), key=lambda x: label_distribution[x], reverse=True)[:10]
print(f"\n🏆 Top 10 Most Frequent Action Classes:")
for i, class_id in enumerate(top_classes[:10]):
    count = label_distribution[class_id]
    percentage = (count / len(labels_data)) * 100
    if count > 0:
        print(f"   {i+1:2d}. Class {class_id:2d}: {count:4d} segments ({percentage:5.1f}%) from {group_summary[class_id]['num_videos']} videos")

print(f"\n✅ Top 1 labels direct grouping pipeline completed successfully!")
print(f"📁 Results saved in: {grouping_output_dir}")
print(f"🎨 Visualizations saved in: {visualization_dir}")