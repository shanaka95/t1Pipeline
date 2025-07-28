from clustering.kmeans import apply_kmeans_clustering, extract_poses_by_clusters
from visualizations.visualize_cluster import visualize_cluster_poses
import argparse
import pickle
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_top5_labels_from_test_dir(test_labels_dir):
    """
    Load top 5 labels data from the test directory structure
    Each video has its own folder with top5_labels.pkl
    
    Returns:
    - labels_data: numpy array of shape (n_segments, 5) containing top5 labels
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
            
            # Extract top5_labels from each segment and store metadata
            for segment in labels_data:
                all_labels_data.append(segment['top5_labels'])
                segment_metadata.append({
                    'video_name': video_dir,
                    'sequence_id': segment['sequence_id'],
                    'global_segment_id': len(segment_metadata)  # Unique ID across all videos
                })
    
    print(f"Loaded {len(all_labels_data)} total segments from {len(video_dirs)} videos")
    return np.array(all_labels_data), segment_metadata

def apply_kmeans_to_top5_labels(labels_data, segment_metadata, n_clusters, output_dir, visualization_dir):
    """
    Apply k-means clustering to top 5 labels data with comprehensive metadata tracking
    """
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(labels_data)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(labels_data, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    # Create comprehensive clustering results
    clustering_results = {
        'labels_data': labels_data,
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'kmeans_model': kmeans,
        'segment_metadata': segment_metadata,
        'total_segments': len(labels_data)
    }
    
    # Add cluster assignment to metadata
    for i, metadata in enumerate(segment_metadata):
        metadata['cluster_id'] = int(cluster_labels[i])
    
    # Create cluster summary
    cluster_summary = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_segments = [segment_metadata[i] for i in cluster_indices]
        
        # Group by video
        videos_in_cluster = {}
        for segment in cluster_segments:
            video_name = segment['video_name']
            if video_name not in videos_in_cluster:
                videos_in_cluster[video_name] = []
            videos_in_cluster[video_name].append(segment['sequence_id'])
        
        cluster_summary[cluster_id] = {
            'total_segments': len(cluster_segments),
            'num_videos': len(videos_in_cluster),
            'videos': videos_in_cluster,
            'cluster_center': kmeans.cluster_centers_[cluster_id].tolist()
        }
    
    clustering_results['cluster_summary'] = cluster_summary
    
    # Save clustering results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'kmeans_top5_labels_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(clustering_results, f)
    
    # Save human-readable summary as JSON
    json_summary = {
        'n_clusters': n_clusters,
        'total_segments': len(labels_data),
        'silhouette_score': float(silhouette_avg),
        'cluster_summary': cluster_summary
    }
    json_path = os.path.join(output_dir, 'clustering_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Clustering results saved to: {results_path}")
    print(f"Clustering summary saved to: {json_path}")
    
    # Create visualizations
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        generate_clustering_visualizations(clustering_results, visualization_dir)
    
    return clustering_results

def generate_clustering_visualizations(clustering_results, visualization_dir):
    """
    Generate comprehensive visualizations for the clustering results
    """
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = clustering_results['n_clusters']
    silhouette_avg = clustering_results['silhouette_score']
    labels_data = clustering_results['labels_data']
    
    # 1. Cluster distribution bar plot
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title(f'Cluster Distribution (n_clusters={n_clusters}, Silhouette Score={silhouette_avg:.3f})')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Segments')
    plt.grid(axis='y', alpha=0.3)
    for i, count in enumerate(counts):
        plt.text(unique[i], count + max(counts)*0.01, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'cluster_distribution_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Silhouette analysis plot
    plt.figure(figsize=(12, 8))
    silhouette_vals = silhouette_samples(labels_data, cluster_labels)
    
    y_ticks = []
    y_lower, y_upper = 0, 0
    
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper += size_cluster_i
        color = plt.cm.tab10(float(i) / n_clusters)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                height=1.0, edgecolor='none', color=color, alpha=0.7)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += size_cluster_i
    
    plt.axvline(silhouette_avg, color="red", linestyle="--", linewidth=2, 
                label=f'Average Score: {silhouette_avg:.3f}')
    plt.yticks(y_ticks, [f'Cluster {i}' for i in range(n_clusters)])
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.title(f'Silhouette Analysis for {n_clusters} Clusters')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'silhouette_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Video distribution across clusters
    cluster_summary = clustering_results['cluster_summary']
    video_counts_per_cluster = [cluster_summary[i]['num_videos'] for i in range(n_clusters)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_clusters), video_counts_per_cluster, color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.title('Number of Videos per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Videos')
    plt.grid(axis='y', alpha=0.3)
    for i, count in enumerate(video_counts_per_cluster):
        plt.text(i, count + max(video_counts_per_cluster)*0.01, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'videos_per_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {visualization_dir}")

# Set up argument parser
parser = argparse.ArgumentParser(description='Apply k-means clustering to top 5 labels from test directory')
parser.add_argument('--n_clusters', type=int, default=10, 
                    help='Number of clusters for k-means clustering (default: 10)')
parser.add_argument('--test_dir', type=str, default='test/top_5labels',
                    help='Path to test directory containing top5_labels (default: test/top_5labels)')
parser.add_argument('--clustering_output_dir', type=str, default='./clustering_results',
                    help='Directory to save clustering results (default: ./clustering_results)')
parser.add_argument('--visualization_dir', type=str, default='./visualizations/clustering',
                    help='Directory to save clustering visualizations (default: ./visualizations/clustering)')

# Parse arguments
args = parser.parse_args()

# Configuration
test_labels_dir = args.test_dir
clustering_output_dir = args.clustering_output_dir
n_clusters = args.n_clusters
visualization_dir = args.visualization_dir

# Step 1: Load top 5 labels data from test directory
print("🚀 Loading top 5 labels data from test directory...")
labels_data, segment_metadata = load_top5_labels_from_test_dir(test_labels_dir)
print(f"Loaded {len(labels_data)} segments from test directory")

# Step 2: Apply k-means clustering to top 5 labels
print(f"\n🔍 Applying k-means clustering with {n_clusters} clusters...")
clustering_results = apply_kmeans_to_top5_labels(
    labels_data=labels_data,
    segment_metadata=segment_metadata,
    n_clusters=n_clusters,
    output_dir=clustering_output_dir,
    visualization_dir=visualization_dir
)

# Step 3: Display summary statistics
print(f"\n📊 Clustering Summary:")
print(f"   • Total segments: {len(labels_data):,}")
print(f"   • Number of clusters: {n_clusters}")
print(f"   • Silhouette score: {clustering_results['silhouette_score']:.6f}")

cluster_summary = clustering_results['cluster_summary']
print(f"\n📈 Cluster Details:")
for cluster_id in range(n_clusters):
    summary = cluster_summary[cluster_id]
    print(f"   • Cluster {cluster_id}: {summary['total_segments']} segments from {summary['num_videos']} videos")

print(f"\n✅ Top 5 labels clustering pipeline completed successfully!")
print(f"📁 Results saved in: {clustering_output_dir}")
print(f"🎨 Visualizations saved in: {visualization_dir}")
