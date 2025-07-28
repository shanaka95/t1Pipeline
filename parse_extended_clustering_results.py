#!/usr/bin/env python3
"""
Parse clustering results from 5 to 199 clusters and extract Silhouette scores.
This script analyzes the extended range of clustering results.
"""

import os
import json
import glob
from pathlib import Path

def extract_clustering_metrics(file_path):
    """
    Extract clustering metrics from a clustering_summary.json file.
    
    Parameters:
    file_path (str): Path to the clustering_summary.json file
    
    Returns:
    dict: Dictionary containing extracted metrics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract basic metrics
        metrics = {
            'n_clusters': data.get('n_clusters'),
            'total_segments': data.get('total_segments'),
            'silhouette_score': data.get('silhouette_score'),
            'file_path': file_path
        }
        
        # Extract cluster distribution statistics
        cluster_summary = data.get('cluster_summary', {})
        if cluster_summary:
            cluster_sizes = []
            video_counts = []
            
            for cluster_id, cluster_info in cluster_summary.items():
                if isinstance(cluster_info, dict):
                    cluster_sizes.append(cluster_info.get('size', 0))
                    video_counts.append(cluster_info.get('video_count', 0))
            
            if cluster_sizes:
                metrics.update({
                    'largest_cluster_size': max(cluster_sizes),
                    'smallest_cluster_size': min(cluster_sizes),
                    'average_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
                    'total_videos': sum(video_counts) if video_counts else 0
                })
        
        return metrics
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_all_clustering_results():
    """
    Parse all clustering_summary.json files from 5 to 199 clusters.
    
    Returns:
    list: List of dictionaries containing clustering metrics
    """
    base_dir = "test/clustering_info_with_top5_labels"
    results = []
    
    print("🔍 Parsing clustering results from 5 to 199 clusters...")
    
    # Get all cluster directories
    cluster_dirs = []
    for i in range(5, 200):  # 5 to 199
        cluster_dir = os.path.join(base_dir, str(i))
        if os.path.exists(cluster_dir):
            cluster_dirs.append(str(i))
    
    print(f"📊 Found {len(cluster_dirs)} cluster directories")
    
    # Process each cluster directory
    for cluster_num in sorted(cluster_dirs, key=int):
        clustering_summary_path = os.path.join(base_dir, cluster_num, "clustering_summary.json")
        
        if os.path.exists(clustering_summary_path):
            metrics = extract_clustering_metrics(clustering_summary_path)
            if metrics:
                results.append(metrics)
                print(f"✅ Processed cluster {cluster_num}: Silhouette Score = {metrics['silhouette_score']:.6f}")
        else:
            print(f"⚠️  Missing clustering_summary.json for cluster {cluster_num}")
    
    return results

def save_results(results, output_file="extended_clustering_analysis.json"):
    """
    Save the parsed results to a JSON file.
    
    Parameters:
    results (list): List of clustering metrics
    output_file (str): Output file path
    """
    output_data = {
        'total_clusters_analyzed': len(results),
        'cluster_range': '5-199',
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")

def main():
    """Main function to parse and save clustering results."""
    print("🚀 Starting extended clustering analysis (5-199 clusters)...")
    
    # Parse all clustering results
    results = parse_all_clustering_results()
    
    if not results:
        print("❌ No clustering results found!")
        return
    
    # Save results
    save_results(results)
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"   Total clusters analyzed: {len(results)}")
    
    # Find best and worst scores
    if results:
        best_result = max(results, key=lambda x: x['silhouette_score'])
        worst_result = min(results, key=lambda x: x['silhouette_score'])
        
        print(f"   Best Silhouette Score: {best_result['silhouette_score']:.6f} (Cluster {best_result['n_clusters']})")
        print(f"   Worst Silhouette Score: {worst_result['silhouette_score']:.6f} (Cluster {worst_result['n_clusters']})")
        
        # Calculate statistics
        scores = [r['silhouette_score'] for r in results]
        avg_score = sum(scores) / len(scores)
        print(f"   Average Silhouette Score: {avg_score:.6f}")
    
    print("✅ Extended clustering analysis completed!")

if __name__ == "__main__":
    main() 