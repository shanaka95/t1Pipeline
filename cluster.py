from clustering.kmeans import apply_kmeans_clustering, extract_poses_by_clusters
from visualizations.visualize_cluster import visualize_cluster_poses
import argparse

# embeddings_master_dir = "/home/shanaka/Desktop/thesis/pipeline-final/embeddings"
# clustering_output_dir = "./clustering_results"
# poses_dir = "/home/shanaka/Desktop/thesis/pipeline-final/poses"
# clustered_poses_output_dir = "./clustered_poses"
# n_clusters = 50
# poses_per_cluster = 10
# visualization_dir = "./visualizations/clustering"
# Set up argument parser
parser = argparse.ArgumentParser(description='Apply k-means clustering to pose embeddings')
parser.add_argument('--n_clusters', type=int, default=10, 
                    help='Number of clusters for k-means clustering (default: 10)')

# Parse arguments
args = parser.parse_args()

embeddings_master_dir = "/home/janus/iwso-datasets/t1-embeddings-final"
clustering_output_dir = "/home/janus/iwso-datasets/t1-clusters-final"
poses_dir = "/home/janus/iwso-datasets/t1-body-poses-final"
clustered_poses_output_dir = "/home/janus/iwso-datasets/t1-clustered-poses-final"
n_clusters = args.n_clusters or 10
poses_per_cluster = 10
visualization_dir = "./visualizations/clustering"

# Step 1: Apply k-means clustering
print("🚀 Starting clustering process...")
clustering_results = apply_kmeans_clustering(
    embeddings_master_dir=embeddings_master_dir,
    output_dir=clustering_output_dir,
    n_clusters=n_clusters,
    visualization_dir=visualization_dir
)

clustering_data_path = f"{clustering_output_dir}/kmeans_clustering_results.pkl"

# Step 2: Extract poses by clusters
print("\n📦 Extracting poses by clusters...")
extraction_summary = extract_poses_by_clusters(
    poses_per_cluster=poses_per_cluster,
    poses_dir=poses_dir,
    clustering_data_path=clustering_data_path,
    output_dir=clustered_poses_output_dir
)

# Step 3: Visualize cluster poses
print("\n🎬 Generating cluster visualizations...")
vis_dir = visualize_cluster_poses(clustered_poses_output_dir, num_vis_per_cluster=5, fps=30)
print(f"\n🎥 Cluster visualizations saved in: {vis_dir}")

print("\n✅ Clustering pipeline completed successfully!")
