import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca_to_embeddings(embeddings_data, n_components=50, standardize=True):
    """
    Apply PCA to reduce dimensionality of embedding vectors.
    
    Parameters:
    embeddings_data: List of dicts with 'embedding' and 'sequence_id' keys, or numpy array
    n_components: Number of principal components to keep (default: 50)
    standardize: Whether to standardize the data before PCA (default: True)
    
    Returns:
    dict containing:
        - 'reduced_embeddings': PCA-transformed embeddings
        - 'pca_object': Fitted PCA object
        - 'scaler': Fitted StandardScaler object (if standardize=True)
        - 'explained_variance_ratio': Explained variance ratio for each component
        - 'cumulative_variance': Cumulative explained variance
        - 'sequence_ids': List of sequence IDs (if input was dict format)
    """
    # Handle different input formats
    if isinstance(embeddings_data, list) and isinstance(embeddings_data[0], dict):
        # Extract embeddings and sequence IDs from dict format
        embeddings = np.array([item['embedding'] for item in embeddings_data])
        sequence_ids = [item['sequence_id'] for item in embeddings_data]
    else:
        # Assume it's already a numpy array
        embeddings = embeddings_data
        sequence_ids = list(range(len(embeddings)))
    
    print(f"Input embeddings shape: {embeddings.shape}")
    print(f"Reducing from {embeddings.shape[1]} to {n_components} dimensions")
    
    # Standardize the data if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        print("Applied standardization (mean=0, std=1)")
    else:
        embeddings_scaled = embeddings
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)
    
    # Calculate variance information
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"PCA Results:")
    print(f"  - Reduced shape: {reduced_embeddings.shape}")
    print(f"  - Explained variance (first 5 components): {explained_variance_ratio[:5]}")
    print(f"  - Total variance explained: {cumulative_variance[-1]:.4f}")
    print(f"  - Variance explained by top 10 components: {cumulative_variance[9]:.4f}")
    
    return {
        'reduced_embeddings': reduced_embeddings,
        'pca_object': pca,
        'scaler': scaler,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'sequence_ids': sequence_ids,
        'original_shape': embeddings.shape,
        'n_components': n_components
    }

def load_and_apply_pca(embeddings_file, n_components=50, standardize=True):
    """
    Load embeddings from pickle file and apply PCA.
    
    Parameters:
    embeddings_file: Path to pickle file containing embeddings
    n_components: Number of principal components to keep
    standardize: Whether to standardize the data before PCA
    
    Returns:
    PCA results dictionary (same as apply_pca_to_embeddings)
    """
    print(f"Loading embeddings from: {embeddings_file}")
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    return apply_pca_to_embeddings(embeddings_data, n_components, standardize)

def save_pca_results(pca_results, output_file):
    """
    Save PCA results to pickle file.
    
    Parameters:
    pca_results: Dictionary returned by apply_pca_to_embeddings
    output_file: Path to save the results
    """
    print(f"Saving PCA results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(pca_results, f)
    print(f"✅ PCA results saved successfully!")

def plot_variance_explained(pca_results, max_components=None):
    """
    Plot explained variance ratio and cumulative variance.
    
    Parameters:
    pca_results: Dictionary returned by apply_pca_to_embeddings
    max_components: Maximum number of components to plot (default: all)
    """
    try:
        import matplotlib.pyplot as plt
        
        explained_variance = pca_results['explained_variance_ratio']
        cumulative_variance = pca_results['cumulative_variance']
        
        if max_components is None:
            max_components = len(explained_variance)
        else:
            max_components = min(max_components, len(explained_variance))
        
        components = range(1, max_components + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot explained variance ratio
        ax1.bar(components, explained_variance[:max_components])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative variance
        ax2.plot(components, cumulative_variance[:max_components], 'o-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_variance_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Variance analysis plot saved as 'pca_variance_analysis.png'")
        
    except ImportError:
        print("⚠️ matplotlib not available, skipping variance plot")
