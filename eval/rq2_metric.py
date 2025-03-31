import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
import pacmap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import psutil
import gc
import os

def measure_memory_usage():
    """Measure current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    return memory_gb

def calculate_preservation(original_distances, embedded_distances):
    """
    Calculate how well the embedding preserves distances/relationships
    
    Args:
        original_distances: Pairwise distances in the original space
        embedded_distances: Pairwise distances in the embedded space
        
    Returns:
        Preservation percentage
    """
    # Normalize distances to [0,1] range
    orig_norm = original_distances / np.max(original_distances)
    emb_norm = embedded_distances / np.max(embedded_distances)
    
    # Calculate correlation between distance matrices
    correlation = np.corrcoef(orig_norm.flatten(), emb_norm.flatten())[0, 1]
    
    # Convert to percentage
    preservation = correlation * 100
    
    return preservation

def evaluate_visualization_method(embeddings, labels, method_name, method_func, **kwargs):
    """
    Evaluate a dimensionality reduction method
    
    Args:
        embeddings: Original high-dimensional embeddings
        labels: Ground truth labels for validation
        method_name: Name of the method for reporting
        method_func: Function to perform the dimensionality reduction
        **kwargs: Additional parameters for the method
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {'method': method_name}
    
    # Record starting memory
    mem_before = measure_memory_usage()
    
    # Start timing
    start_time = time.time()
    
    # Apply dimensionality reduction
    reduced_embedding = method_func(embeddings, **kwargs)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    results['processing_time'] = processing_time
    
    # Measure peak memory usage
    mem_after = measure_memory_usage()
    memory_used = mem_after - mem_before
    results['memory_usage'] = memory_used
    
    # Calculate cluster separation (silhouette score)
    cluster_separation = silhouette_score(reduced_embedding, labels)
    results['cluster_separation'] = cluster_separation
    
    # Calculate distance matrices
    from scipy.spatial.distance import pdist, squareform
    original_distances = squareform(pdist(embeddings))
    embedded_distances = squareform(pdist(reduced_embedding))
    
    # Calculate how well the embedding preserves QCMR-related information
    # For this demo, we'll simulate this with a subset of the dimensions
    qcmr_dims = embeddings[:, :embeddings.shape[1]//2]  # First half of dimensions as proxy for QCMR
    qcmr_orig_dist = squareform(pdist(qcmr_dims))
    qcmr_preservation = calculate_preservation(qcmr_orig_dist, embedded_distances)
    results['qcmr_preservation'] = qcmr_preservation
    
    # Calculate how well the embedding preserves REI-related information
    # For this demo, we'll simulate this with the remaining dimensions
    rei_dims = embeddings[:, embeddings.shape[1]//2:]  # Second half of dimensions as proxy for REI
    rei_orig_dist = squareform(pdist(rei_dims))
    rei_preservation = calculate_preservation(rei_orig_dist, embedded_distances)
    results['rei_preservation'] = rei_preservation
    
    # Clean up to free memory
    gc.collect()
    
    return results, reduced_embedding

def run_visualization_comparison(embeddings, labels):
    """
    Compare different visualization techniques
    
    Args:
        embeddings: Original high-dimensional embeddings
        labels: Ground truth labels for validation
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Test t-SNE
    tsne_results, tsne_embedding = evaluate_visualization_method(
        scaled_embeddings, 
        labels,
        't-SNE (baseline)',
        lambda x, **kwargs: TSNE(n_components=2, **kwargs).fit_transform(x),
        perplexity=30, 
        random_state=42
    )
    results.append(tsne_results)
    
    # Test UMAP with different configurations
    umap_configs = [
        ('UMAP', {'n_neighbors': 15, 'min_dist': 0.1}),
        ('n_neighbors=5', {'n_neighbors': 5, 'min_dist': 0.1}),
        ('n_neighbors=15', {'n_neighbors': 15, 'min_dist': 0.1}),
        ('n_neighbors=30', {'n_neighbors': 30, 'min_dist': 0.1})
    ]
    
    umap_embeddings = {}
    
    for name, config in umap_configs:
        method_name = 'UMAP' if name == 'UMAP' else name
        umap_result, umap_embedding = evaluate_visualization_method(
            scaled_embeddings,
            labels,
            method_name,
            lambda x, **kwargs: UMAP(n_components=2, **kwargs).fit_transform(x),
            **config
        )
        results.append(umap_result)
        umap_embeddings[name] = umap_embedding
    
    # Test PaCMAP
    pacmap_results, pacmap_embedding = evaluate_visualization_method(
        scaled_embeddings,
        labels,
        'PaCMAP',
        lambda x, **kwargs: pacmap.PaCMAP(n_components=2, **kwargs).fit_transform(x),
        n_neighbors=15,
        random_state=42
    )
    results.append(pacmap_results)
    
    # Test Isomap
    isomap_results, isomap_embedding = evaluate_visualization_method(
        scaled_embeddings,
        labels,
        'Isomap',
        lambda x, **kwargs: Isomap(n_components=2, **kwargs).fit_transform(x),
        n_neighbors=15
    )
    results.append(isomap_results)
    
    # Create a DataFrame of the results
    results_df = pd.DataFrame(results)
    
    # Generate visualizations
    methods_to_plot = {
        't-SNE (baseline)': tsne_embedding,
        'UMAP': umap_embeddings['UMAP'],
        'PaCMAP': pacmap_embedding,
        'Isomap': isomap_embedding
    }
    
    plot_visualization_comparison(methods_to_plot, labels)
    plot_umap_configs(umap_embeddings, labels)
    
    return results_df

def plot_visualization_comparison(embeddings_dict, labels):
    """
    Plot comparison of different visualization techniques
    
    Args:
        embeddings_dict: Dictionary of method names and their embedded representations
        labels: Ground truth labels for coloring
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (method_name, embedding) in enumerate(embeddings_dict.items()):
        ax = axes[i]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.7, s=30)
        ax.set_title(f"{method_name}", fontsize=14)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.colorbar(scatter, ax=axes, label='Phase Label')
    plt.tight_layout()
    plt.savefig('visualization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_umap_configs(umap_embeddings, labels):
    """
    Plot comparison of different UMAP configurations
    
    Args:
        umap_embeddings: Dictionary of UMAP configurations and their embedded representations
        labels: Ground truth labels for coloring
    """
    configs = ['n_neighbors=5', 'n_neighbors=15', 'n_neighbors=30']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, config in enumerate(configs):
        ax = axes[i]
        embedding = umap_embeddings[config]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.7, s=30)
        ax.set_title(f"UMAP ({config})", fontsize=14)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.colorbar(scatter, ax=axes, label='Phase Label')
    plt.tight_layout()
    plt.savefig('umap_config_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def format_results_for_latex(results_df):
    """
    Format results dataframe for LaTeX table
    
    Args:
        results_df: DataFrame with results
        
    Returns:
        Formatted string for LaTeX table
    """
    # Order the rows
    order = ['t-SNE (baseline)', 'UMAP', 'PaCMAP', 'Isomap', 
             'n_neighbors=5', 'n_neighbors=15', 'n_neighbors=30']
    
    results_df = results_df.set_index('method').loc[order].reset_index()
    
    # Format the values
    latex_rows = []
    for _, row in results_df.iterrows():
        method = row['method']
        qcmr = f"{row['qcmr_preservation']:.1f}"
        rei = f"{row['rei_preservation']:.1f}"
        cluster = f"{row['cluster_separation']:.3f}"
        time_s = f"{row['processing_time']:.1f}"
        memory = f"{row['memory_usage']:.1f}"
        
        latex_row = f"{method} & {qcmr} & {rei} & {cluster} & {time_s} & {memory} \\\\"
        latex_rows.append(latex_row)
    
    return "\n".join(latex_rows)

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=1000, n_features=100, n_phases=3, random_seed=42):
    """
    Generate synthetic data for visualization comparison
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_phases: Number of phases/classes
        random_seed: Random seed for reproducibility
        
    Returns:
        Embeddings and labels
    """
    np.random.seed(random_seed)
    
    # Generate cluster centers
    centers = np.random.randn(n_phases, n_features) * 10
    
    # Generate data points around centers
    embeddings = []
    labels = []
    
    samples_per_phase = n_samples // n_phases
    
    for i in range(n_phases):
        cluster_points = centers[i] + np.random.randn(samples_per_phase, n_features) * 2
        embeddings.append(cluster_points)
        labels.extend([i] * samples_per_phase)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Add some feature correlations to simulate QCMR and REI relationships
    # First half of features relate to QCMR, second half to REI
    for i in range(n_features // 2):
        # QCMR-related features have stronger correlations with each other
        if i > 0:
            embeddings[:, i] = 0.7 * embeddings[:, i-1] + 0.3 * embeddings[:, i]
        
        # REI-related features also correlate
        j = i + n_features // 2
        if j < n_features - 1:
            embeddings[:, j] = 0.6 * embeddings[:, j+1] + 0.4 * embeddings[:, j]
    
    return embeddings, labels

# Run the comparison
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    embeddings, labels = generate_synthetic_data(n_samples=1000, n_features=100)
    
    # Run visualization comparison
    print("Comparing visualization methods...")
    results_df = run_visualization_comparison(embeddings, labels)
    
    # Display results
    print("\nResults Summary:")
    print(results_df[['method', 'qcmr_preservation', 'rei_preservation', 
                      'cluster_separation', 'processing_time', 'memory_usage']])
    
    # Format for LaTeX
    print("\nFormatted for LaTeX table:")
    latex_format = format_results_for_latex(results_df)
    print(latex_format)
    
    print("\nVisualization comparison saved as 'visualization_comparison.png'")
    print("UMAP configuration comparison saved as 'umap_config_comparison.png'")