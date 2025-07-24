from clustering.kmeans import apply_kmeans_clustering, extract_poses_by_clusters
from visualizations.visualize_cluster import visualize_cluster_poses

embeddings_master_dir = "/home/shanaka/Desktop/thesis/pipeline-final/embeddings"
clustering_output_dir = "./clustering_results"
poses_dir = "/home/shanaka/Desktop/thesis/pipeline-final/poses"
clustered_poses_output_dir = "./clustered_poses"
n_clusters = 50
poses_per_cluster = 10
visualization_dir = "./visualizations/clustering"

# # Step 1: Apply k-means clustering
# clustering_results = apply_kmeans_clustering(
#     embeddings_master_dir=embeddings_master_dir,
#     output_dir=clustering_output_dir,
#     n_clusters=n_clusters,
#     visualization_dir=visualization_dir
# )

# clustering_data_path = f"{clustering_output_dir}/kmeans_clustering_results.pkl"

# # Step 2: Extract poses by clusters
# extraction_summary = extract_poses_by_clusters(
#     poses_per_cluster=poses_per_cluster,
#     poses_dir=poses_dir,
#     clustering_data_path=clustering_data_path,
#     output_dir=clustered_poses_output_dir
# )

# Step 3: Visualize cluster poses
vis_dir = visualize_cluster_poses(clustered_poses_output_dir, num_vis_per_cluster=5, fps=30)
print(f"\n🎥 Cluster visualizations saved in: {vis_dir}")
