import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob
from pathlib import Path
import json

def apply_kmeans_clustering(embeddings_master_dir, output_dir, n_clusters=50, random_state=42, visualization_dir=None):
    """
    Apply k-means clustering to all PCA results found in subdirectories.
    Optionally generate and save clustering visualizations.
    
    Parameters:
    embeddings_master_dir: Master directory containing subdirectories with pca_results.pkl files
    output_dir: Directory to save clustering results
    n_clusters: Number of clusters for k-means (default: 50)
    random_state: Random state for reproducibility
    visualization_dir: Optional directory to save clustering visualizations
    
    Returns:
    dict: Clustering results with metadata
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        umap_available = True
    except ImportError:
        umap_available = False
    import seaborn as sns

    print(f"🔍 Searching for PCA results in: {embeddings_master_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Find all pca_results.pkl files
    pca_files = glob.glob(os.path.join(embeddings_master_dir, "**/pca_results.pkl"), recursive=True)
    
    if not pca_files:
        raise ValueError(f"No pca_results.pkl files found in {embeddings_master_dir}")
    
    print(f"📁 Found {len(pca_files)} PCA result files:")
    for pca_file in pca_files:
        print(f"   - {pca_file}")
    
    # Load and combine all PCA results
    all_embeddings = []
    segment_metadata = []
    
    for pca_file in pca_files:
        # Extract video name from path
        video_name = os.path.basename(os.path.dirname(pca_file))
        
        print(f"📊 Loading PCA results from: {video_name}")
        with open(pca_file, 'rb') as f:
            pca_data = pickle.load(f)
        
        reduced_embeddings = pca_data['reduced_embeddings']
        sequence_ids = pca_data['sequence_ids']
        
        print(f"   - Shape: {reduced_embeddings.shape}")
        print(f"   - Segments: {len(sequence_ids)}")
        
        # Store embeddings
        all_embeddings.append(reduced_embeddings)
        
        # Store metadata for each segment
        for i, seq_id in enumerate(sequence_ids):
            segment_metadata.append({
                'video_name': video_name,
                'sequence_id': seq_id,
                'global_index': len(segment_metadata),
                'video_index': i,
                'pca_file': pca_file
            })
    
    # Combine all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    print(f"\n🎯 Combined embeddings shape: {combined_embeddings.shape}")
    print(f"📈 Total segments: {len(segment_metadata)}")
    
    # Apply k-means clustering
    print(f"\n🤖 Applying k-means clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(combined_embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(combined_embeddings, cluster_labels)
    print(f"📊 Silhouette score: {silhouette_avg:.4f}")
    
    # Analyze cluster distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n📋 Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"   Cluster {label}: {count} segments")
    
    # Organize results by cluster
    clusters_data = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters_data:
            clusters_data[cluster_id] = []
        
        segment_info = segment_metadata[i].copy()
        segment_info['cluster_id'] = int(cluster_id)
        clusters_data[cluster_id].append(segment_info)
    
    # Prepare final results
    clustering_results = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'cluster_labels': cluster_labels.tolist(),
        'segment_metadata': segment_metadata,
        'clusters_data': clusters_data,
        'kmeans_centers': kmeans.cluster_centers_,
        'video_files': pca_files,
        'total_segments': len(segment_metadata),
        'embeddings_shape': combined_embeddings.shape
    }
    
    # Save clustering results
    clustering_file = os.path.join(output_dir, 'kmeans_clustering_results.pkl')
    with open(clustering_file, 'wb') as f:
        pickle.dump(clustering_results, f)
    print(f"💾 Saved clustering results to: {clustering_file}")
    
    # Save human-readable summary
    summary_file = os.path.join(output_dir, 'clustering_summary.json')
    summary = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'total_segments': len(segment_metadata),
        'cluster_distribution': {int(k): len(v) for k, v in clusters_data.items()},
        'videos_processed': list(set([meta['video_name'] for meta in segment_metadata]))
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary to: {summary_file}")
    
    # --- Visualization Section ---
    if visualization_dir:
        print(f"\n🖼️ Generating clustering visualizations in: {visualization_dir}")
        # 1. Cluster distribution bar plot
        plt.figure(figsize=(10, 5))
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        sns.barplot(x=unique_labels, y=counts, palette="tab20")
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Segments')
        plt.title('Cluster Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'cluster_distribution_barplot.png'))
        plt.close()

        # 2. 2D PCA scatter plot colored by cluster
        pca_2d = PCA(n_components=2)
        emb_2d = pca_2d.fit_transform(combined_embeddings)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.7)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('2D PCA Scatter by Cluster')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'pca2d_scatter_by_cluster.png'))
        plt.close()

        # 3. t-SNE scatter plot colored by cluster
        try:
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
            emb_tsne = tsne.fit_transform(combined_embeddings)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.7)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE Scatter by Cluster')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'tsne_scatter_by_cluster.png'))
            plt.close()
        except Exception as e:
            print(f"⚠️ t-SNE visualization failed: {e}")

        # 4. UMAP scatter plot colored by cluster (if available)
        if umap_available:
            try:
                reducer = umap.UMAP(n_components=2, random_state=random_state)
                emb_umap = reducer.fit_transform(combined_embeddings)
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(emb_umap[:, 0], emb_umap[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.7)
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                plt.title('UMAP Scatter by Cluster')
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                plt.savefig(os.path.join(visualization_dir, 'umap_scatter_by_cluster.png'))
                plt.close()
            except Exception as e:
                print(f"⚠️ UMAP visualization failed: {e}")
        else:
            print("⚠️ UMAP not available, skipping UMAP visualization.")

        # 5. Silhouette plot
        from sklearn.metrics import silhouette_samples
        sample_silhouette_values = silhouette_samples(combined_embeddings, cluster_labels)
        plt.figure(figsize=(10, 6))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster label")
        plt.title(f"Silhouette Plot for {n_clusters} Clusters")
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'silhouette_plot.png'))
        plt.close()

        # 6. Cluster center heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(kmeans.cluster_centers_, cmap='viridis', cbar=True)
        plt.xlabel('Feature Index')
        plt.ylabel('Cluster')
        plt.title('Cluster Centers Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'cluster_centers_heatmap.png'))
        plt.close()

        print(f"✅ Saved all clustering visualizations to {visualization_dir}")

    return clustering_results

def extract_poses_by_clusters(poses_per_cluster, poses_dir, clustering_data_path, output_dir):
    """
    Extract and organize poses by cluster assignments.
    
    Parameters:
    poses_per_cluster: Number of poses to extract per cluster
    poses_dir: Directory containing pose files (poses_3D.pkl) for different videos
    clustering_data_path: Path to saved clustering results
    output_dir: Directory to save organized poses by cluster
    """
    print(f"📂 Loading clustering data from: {clustering_data_path}")
    
    # Load clustering results
    with open(clustering_data_path, 'rb') as f:
        clustering_results = pickle.load(f)
    
    clusters_data = clustering_results['clusters_data']
    n_clusters = clustering_results['n_clusters']
    
    print(f"🎯 Processing {n_clusters} clusters")
    print(f"📊 Extracting {poses_per_cluster} poses per cluster")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pose files in poses directory
    pose_files = glob.glob(os.path.join(poses_dir, "**/poses_3D.pkl"), recursive=True)
    
    if not pose_files:
        raise ValueError(f"No poses_3D.pkl files found in {poses_dir}")
    
    print(f"📁 Found {len(pose_files)} pose files:")
    for pose_file in pose_files:
        print(f"   - {pose_file}")
    
    # Load all pose data
    all_poses_data = {}
    for pose_file in pose_files:
        video_name = os.path.basename(os.path.dirname(pose_file))
        print(f"📊 Loading poses from: {video_name}")
        
        with open(pose_file, 'rb') as f:
            poses = pickle.load(f)
        
        print(f"   - Loaded {len(poses)} segments")
        all_poses_data[video_name] = poses
    
    # Extract poses for each cluster
    cluster_poses_summary = {}
    
    for cluster_id in range(n_clusters):
        print(f"\n🎯 Processing cluster {cluster_id}")
        
        if cluster_id not in clusters_data:
            print(f"   ⚠️ No segments found for cluster {cluster_id}")
            continue
        
        cluster_segments = clusters_data[cluster_id]
        print(f"   📈 {len(cluster_segments)} segments in cluster")
        
        # Collect poses for this cluster
        cluster_poses = []
        videos_used = set()
        
        for segment_info in cluster_segments:
            video_name = segment_info['video_name']
            sequence_id = segment_info['sequence_id']
            
            if video_name in all_poses_data:
                if sequence_id < len(all_poses_data[video_name]):
                    pose_data = all_poses_data[video_name][sequence_id]
                    cluster_poses.append({
                        'pose': pose_data,
                        'video_name': video_name,
                        'sequence_id': sequence_id,
                        'video_index': segment_info['video_index']
                    })
                    videos_used.add(video_name)
                else:
                    print(f"   ⚠️ Sequence {sequence_id} not found in {video_name}")
            else:
                print(f"   ⚠️ Video {video_name} not found in pose data")
        
        # Limit poses per cluster if needed
        if len(cluster_poses) > poses_per_cluster:
            # Try to sample from different videos if possible
            np.random.shuffle(cluster_poses)
            cluster_poses = cluster_poses[:poses_per_cluster]
        
        print(f"   ✅ Collected {len(cluster_poses)} poses from {len(videos_used)} videos")
        
        # Create cluster directory
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id:03d}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Save poses for this cluster
        cluster_poses_file = os.path.join(cluster_dir, "poses.pkl")
        poses_array = [item['pose'] for item in cluster_poses]
        
        cluster_data = {
            'poses': poses_array,
            'metadata': cluster_poses,
            'cluster_id': cluster_id,
            'num_poses': len(poses_array),
            'videos_represented': list(videos_used)
        }
        
        with open(cluster_poses_file, 'wb') as f:
            pickle.dump(cluster_data, f)
        
        print(f"   💾 Saved to: {cluster_poses_file}")
        
        # Store summary info
        cluster_poses_summary[cluster_id] = {
            'num_poses': len(poses_array),
            'videos_represented': list(videos_used),
            'pose_file': cluster_poses_file
        }
    
    # Save overall summary
    summary_file = os.path.join(output_dir, "extraction_summary.json")
    extraction_summary = {
        'poses_per_cluster_requested': poses_per_cluster,
        'total_clusters': n_clusters,
        'clusters_processed': len(cluster_poses_summary),
        'total_videos_available': len(all_poses_data),
        'cluster_details': cluster_poses_summary
    }
    
    with open(summary_file, 'w') as f:
        json.dump(extraction_summary, f, indent=2)
    
    print(f"\n✅ Pose extraction completed!")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"📂 Cluster poses saved to: {output_dir}")
    
    return extraction_summary
