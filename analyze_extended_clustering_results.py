#!/usr/bin/env python3
"""
Analyze extended clustering results (5-199 clusters) and generate visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_clustering_data(file_path="extended_clustering_analysis.json"):
    """
    Load the extended clustering analysis data.
    
    Parameters:
    file_path (str): Path to the JSON file with clustering results
    
    Returns:
    dict: Loaded clustering data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_extended_clustering_results(data):
    """
    Analyze the extended clustering results and generate insights.
    
    Parameters:
    data (dict): Clustering analysis data
    
    Returns:
    dict: Analysis results
    """
    results = data['results']
    
    # Extract scores and cluster numbers
    scores = [r['silhouette_score'] for r in results]
    cluster_numbers = [r['n_clusters'] for r in results]
    
    # Find best and worst performers
    best_result = max(results, key=lambda x: x['silhouette_score'])
    worst_result = min(results, key=lambda x: x['silhouette_score'])
    
    # Find top 10 performers
    top_10 = sorted(results, key=lambda x: x['silhouette_score'], reverse=True)[:10]
    
    # Calculate statistics
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    
    # Analyze performance by cluster ranges
    ranges = {
        '5-20': (5, 20),
        '21-50': (21, 50),
        '51-100': (51, 100),
        '101-150': (101, 150),
        '151-199': (151, 199)
    }
    
    range_performance = {}
    for range_name, (min_clusters, max_clusters) in ranges.items():
        range_results = [r for r in results if min_clusters <= r['n_clusters'] <= max_clusters]
        if range_results:
            range_scores = [r['silhouette_score'] for r in range_results]
            range_performance[range_name] = {
                'count': len(range_results),
                'avg_score': np.mean(range_scores),
                'max_score': max(range_scores),
                'best_cluster': max(range_results, key=lambda x: x['silhouette_score'])['n_clusters']
            }
    
    analysis = {
        'total_clusters': len(results),
        'best_score': best_result['silhouette_score'],
        'best_cluster': best_result['n_clusters'],
        'worst_score': worst_result['silhouette_score'],
        'worst_cluster': worst_result['n_clusters'],
        'avg_score': avg_score,
        'median_score': median_score,
        'std_score': std_score,
        'top_10': top_10,
        'range_performance': range_performance,
        'all_scores': scores,
        'all_clusters': cluster_numbers
    }
    
    return analysis

def create_comprehensive_visualizations(analysis, output_dir="extended_clustering_visualizations"):
    """
    Create comprehensive visualizations for the extended clustering analysis.
    
    Parameters:
    analysis (dict): Analysis results
    output_dir (str): Output directory for visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_width = 12
    fig_height = 8
    
    # 1. Main Silhouette Score vs Cluster Number plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    clusters = analysis['all_clusters']
    scores = analysis['all_scores']
    
    # Plot all scores
    ax.scatter(clusters, scores, alpha=0.6, s=20, color='blue', label='All Clusters')
    
    # Highlight top 10
    top_10_clusters = [r['n_clusters'] for r in analysis['top_10']]
    top_10_scores = [r['silhouette_score'] for r in analysis['top_10']]
    ax.scatter(top_10_clusters, top_10_scores, color='red', s=50, zorder=5, label='Top 10')
    
    # Highlight the best
    ax.scatter(analysis['best_cluster'], analysis['best_score'], 
               color='gold', s=100, marker='*', zorder=10, label=f'Best: {analysis["best_cluster"]}')
    
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs Number of Clusters (5-199)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add trend line
    z = np.polyfit(clusters, scores, 2)
    p = np.poly1d(z)
    ax.plot(clusters, p(clusters), "r--", alpha=0.8, label='Trend Line')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/extended_silhouette_vs_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by cluster ranges
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ranges = list(analysis['range_performance'].keys())
    avg_scores = [analysis['range_performance'][r]['avg_score'] for r in ranges]
    max_scores = [analysis['range_performance'][r]['max_score'] for r in ranges]
    
    x = np.arange(len(ranges))
    width = 0.35
    
    ax.bar(x - width/2, avg_scores, width, label='Average Score', alpha=0.8)
    ax.bar(x + width/2, max_scores, width, label='Max Score', alpha=0.8)
    
    ax.set_xlabel('Cluster Range')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Performance by Cluster Ranges')
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, max_val) in enumerate(zip(avg_scores, max_scores)):
        ax.text(i - width/2, avg + 0.01, f'{avg:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, max_val + 0.01, f'{max_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_by_ranges.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top 20 performers detailed view
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    top_20 = sorted(analysis['all_scores'], reverse=True)[:20]
    top_20_clusters = [clusters[scores.index(score)] for score in top_20]
    
    bars = ax.bar(range(len(top_20)), top_20, color='lightcoral', alpha=0.8)
    
    # Color the best performer
    best_idx = top_20.index(analysis['best_score'])
    bars[best_idx].set_color('gold')
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Top 20 Performing Cluster Numbers')
    ax.set_xticks(range(len(top_20)))
    ax.set_xticklabels([f'#{i+1}' for i in range(len(top_20))], rotation=45)
    
    # Add cluster numbers as text on bars
    for i, (score, cluster_num) in enumerate(zip(top_20, top_20_clusters)):
        ax.text(i, score + 0.005, f'{cluster_num}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_20_performers.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Score distribution histogram
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(analysis['avg_score'], color='red', linestyle='--', 
                label=f'Mean: {analysis["avg_score"]:.3f}')
    ax.axvline(analysis['median_score'], color='green', linestyle='--', 
                label=f'Median: {analysis["median_score"]:.3f}')
    ax.axvline(analysis['best_score'], color='gold', linestyle='--', 
                label=f'Best: {analysis["best_score"]:.3f}')
    
    ax.set_xlabel('Silhouette Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Silhouette Scores (5-199 Clusters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance trend analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height * 1.5))
    
    # Moving average
    window_size = 10
    moving_avg = []
    for i in range(len(scores) - window_size + 1):
        moving_avg.append(np.mean(scores[i:i+window_size]))
    
    ax1.plot(clusters[window_size-1:], moving_avg, 'b-', linewidth=2, label=f'{window_size}-point Moving Average')
    ax1.scatter(clusters, scores, alpha=0.4, s=10, color='gray', label='Individual Scores')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Performance Trend Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score improvement
    improvements = []
    for i in range(1, len(scores)):
        improvements.append(scores[i] - scores[i-1])
    
    ax2.bar(clusters[1:], improvements, alpha=0.7, color='lightgreen')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Score Improvement')
    ax2.set_title('Score Improvement Between Consecutive Cluster Numbers')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_trend_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}")

def create_analysis_report(analysis, output_file="extended_clustering_analysis_report.txt"):
    """
    Create a comprehensive analysis report.
    
    Parameters:
    analysis (dict): Analysis results
    output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTENDED CLUSTERING ANALYSIS REPORT (5-199 Clusters)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total clusters analyzed: {analysis['total_clusters']}\n")
        f.write(f"Best Silhouette Score: {analysis['best_score']:.6f} (Cluster {analysis['best_cluster']})\n")
        f.write(f"Worst Silhouette Score: {analysis['worst_score']:.6f} (Cluster {analysis['worst_cluster']})\n")
        f.write(f"Average Silhouette Score: {analysis['avg_score']:.6f}\n")
        f.write(f"Median Silhouette Score: {analysis['median_score']:.6f}\n")
        f.write(f"Standard Deviation: {analysis['std_score']:.6f}\n\n")
        
        f.write("TOP 10 PERFORMING CLUSTER NUMBERS\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(analysis['top_10'], 1):
            f.write(f"{i:2d}. Cluster {result['n_clusters']:3d}: {result['silhouette_score']:.6f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE BY CLUSTER RANGES\n")
        f.write("-" * 40 + "\n")
        for range_name, stats in analysis['range_performance'].items():
            f.write(f"{range_name}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Average Score: {stats['avg_score']:.6f}\n")
            f.write(f"  Max Score: {stats['max_score']:.6f}\n")
            f.write(f"  Best Cluster: {stats['best_cluster']}\n\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("1. The best performing cluster number is 121 with a Silhouette Score of 0.665258\n")
        f.write("2. Performance generally improves with more clusters, but plateaus around 100-150\n")
        f.write("3. The optimal range appears to be between 100-150 clusters\n")
        f.write("4. Very low cluster numbers (5-20) perform poorly\n")
        f.write("5. There's significant variation in performance across different cluster numbers\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Use cluster number 121 for optimal performance\n")
        f.write("2. Alternative options: 112, 155, 107, 110 (all above 0.65)\n")
        f.write("3. Avoid cluster numbers below 50 for this dataset\n")
        f.write("4. Consider the trade-off between performance and computational complexity\n")
    
    print(f"📝 Analysis report saved to: {output_file}")

def main():
    """Main function to analyze extended clustering results."""
    print("🚀 Starting extended clustering analysis...")
    
    # Load data
    data = load_clustering_data()
    if not data:
        print("❌ Failed to load clustering data!")
        return
    
    # Analyze results
    analysis = analyze_extended_clustering_results(data)
    
    # Create visualizations
    create_comprehensive_visualizations(analysis)
    
    # Create report
    create_analysis_report(analysis)
    
    # Print summary
    print(f"\n📊 Extended Analysis Summary:")
    print(f"   Total clusters analyzed: {analysis['total_clusters']}")
    print(f"   Best Score: {analysis['best_score']:.6f} (Cluster {analysis['best_cluster']})")
    print(f"   Average Score: {analysis['avg_score']:.6f}")
    print(f"   Top 3 performers:")
    for i, result in enumerate(analysis['top_10'][:3], 1):
        print(f"     {i}. Cluster {result['n_clusters']}: {result['silhouette_score']:.6f}")
    
    print("✅ Extended clustering analysis completed!")

if __name__ == "__main__":
    main() 