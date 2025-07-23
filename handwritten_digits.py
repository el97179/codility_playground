import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                           homogeneity_completeness_v_measure, silhouette_score,
                           calinski_harabasz_score, davies_bouldin_score)


def main():
    # Step 1: Load the digits dataset
    print("Loading digits dataset...")
    digits = load_digits()
    data = digits.data  # 64-dimensional data (8x8 flattened)
    target = digits.target  # true labels (0-9)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Number of classes: {len(np.unique(target))}")
    
    # Step 2: Reduce to 2-D with PCA for visualization
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=0)
    data_2d = pca.fit_transform(data)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Step 3: Fine-tuned K-Means clustering
    print("\nApplying fine-tuned K-Means clustering...")
    
    # Test different K-means configurations
    kmeans_configs = [
        {'n_clusters': 10, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 42},
        {'n_clusters': 10, 'init': 'random', 'n_init': 30, 'max_iter': 1000, 'random_state': 42},
        {'n_clusters': 10, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 13}
    ]
    
    best_kmeans = None
    best_ari = -1
    best_labels = None
    
    for i, config in enumerate(kmeans_configs):
        print(f"  Testing K-means config {i+1}: {config}")
        kmeans = KMeans(**config)
        labels = kmeans.fit_predict(data)
        ari = adjusted_rand_score(target, labels)
        print(f"    ARI: {ari:.3f}")
        
        if ari > best_ari:
            best_ari = ari
            best_kmeans = kmeans
            best_labels = labels
    
    cluster_labels = best_labels
    print(f"\nBest K-means ARI: {best_ari:.3f}")
    
    # Step 3b: Apply DBSCAN clustering
    print("\nApplying DBSCAN clustering...")
    
    # Standardize data for DBSCAN (important for distance-based clustering)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Test different DBSCAN configurations
    dbscan_configs = [
        {'eps': 0.5, 'min_samples': 5},
        {'eps': 0.7, 'min_samples': 3},
        {'eps': 0.8, 'min_samples': 4},
        {'eps': 1.0, 'min_samples': 5},
        {'eps': 1.2, 'min_samples': 3}
    ]
    
    best_dbscan = None
    best_dbscan_ari = -1
    best_dbscan_labels = None
    
    for i, config in enumerate(dbscan_configs):
        print(f"  Testing DBSCAN config {i+1}: {config}")
        dbscan = DBSCAN(**config)
        labels = dbscan.fit_predict(data_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"    Clusters: {n_clusters}, Noise points: {n_noise}")
        
        if n_clusters > 1:  # Only calculate ARI if we have clusters
            # Filter out noise points for ARI calculation
            mask = labels != -1
            if mask.sum() > 0:
                ari = adjusted_rand_score(target[mask], labels[mask])
                print(f"    ARI (excluding noise): {ari:.3f}")
                
                if ari > best_dbscan_ari:
                    best_dbscan_ari = ari
                    best_dbscan = dbscan
                    best_dbscan_labels = labels
            else:
                print("    All points are noise")
        else:
            print("    Too few clusters found")
    
    if best_dbscan_labels is not None:
        print(f"\nBest DBSCAN ARI: {best_dbscan_ari:.3f}")
        dbscan_cluster_labels = best_dbscan_labels
    else:
        print("\nNo valid DBSCAN configuration found, using default")
        dbscan = DBSCAN(eps=0.8, min_samples=4)
        dbscan_cluster_labels = dbscan.fit_predict(data_scaled)
    
    # Step 4: Comprehensive evaluation metrics
    print("\n" + "="*50)
    print("COMPREHENSIVE CLUSTERING EVALUATION")
    print("="*50)
    
    evaluate_clustering_comprehensive(data, target, cluster_labels)
    
    # Store results for combined visualization later
    clustering_results = {
        'kmeans': {'labels': cluster_labels, 'ari': best_ari, 'name': 'K-Means'},
        'dbscan': {'labels': dbscan_cluster_labels if best_dbscan_labels is not None else None, 
                  'ari': best_dbscan_ari if best_dbscan_labels is not None else None, 'name': 'DBSCAN'}
    }
    
    # Analyze cluster-digit mapping for K-means
    analyze_cluster_digit_mapping(target, cluster_labels)
    
    # Analyze DBSCAN results (if valid)
    if best_dbscan_labels is not None:
        print("\nDBSCAN Results:")
        analyze_cluster_digit_mapping(target[best_dbscan_labels != -1], 
                                    best_dbscan_labels[best_dbscan_labels != -1])
    
    # Stretch Goal 1: Test different values of k
    print("\n" + "="*50)
    print("STRETCH GOAL 1: Testing different values of k")
    print("="*50)
    
    k_values = range(8, 13)
    ari_scores = []
    
    for k in k_values:
        kmeans_k = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels_k = kmeans_k.fit_predict(data)
        ari_k = adjusted_rand_score(target, labels_k)
        ari_scores.append(ari_k)
        print(f"k={k}: ARI = {ari_k:.3f}")
    
    # Plot ARI vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, ari_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Adjusted Rand Index')
    plt.title('K-Means Performance vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    for i, (k, ari) in enumerate(zip(k_values, ari_scores)):
        plt.annotate(f'{ari:.3f}', (k, ari), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    plt.tight_layout()
    
    # Stretch Goal 2: Compare with Agglomerative Clustering
    print("\n" + "="*50)
    print("STRETCH GOAL 2: Agglomerative Clustering Comparison")
    print("="*50)
    
    linkage_methods = ['ward', 'complete', 'average', 'single']
    
    for linkage in linkage_methods:
        print(f"\nTesting Agglomerative Clustering with {linkage} linkage...")
        agg_clustering = AgglomerativeClustering(n_clusters=10, linkage=linkage)
        agg_labels = agg_clustering.fit_predict(data)
        agg_ari = adjusted_rand_score(target, agg_labels)
        print(f"ARI (Agglomerative-{linkage}): {agg_ari:.3f}")
    
    # Best agglomerative clustering for visualization
    best_agg = AgglomerativeClustering(n_clusters=10, linkage='ward')
    best_agg_labels = best_agg.fit_predict(data)
    best_agg_ari = adjusted_rand_score(target, best_agg_labels)
    
    # Add agglomerative results to clustering_results
    clustering_results['agglomerative'] = {
        'labels': best_agg_labels, 
        'ari': best_agg_ari, 
        'name': 'Agglomerative (Ward)'
    }
    
    # Stretch Goal 3: Compare with DBSCAN
    print("\n" + "="*50)
    print("STRETCH GOAL 3: DBSCAN Comparison")
    print("="*50)
    
    # Use the already computed DBSCAN results from earlier
    if best_dbscan_labels is not None:
        print(f"Best DBSCAN configuration found earlier:")
        print(f"ARI: {best_dbscan_ari:.3f}")
        n_clusters = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
        n_noise = list(best_dbscan_labels).count(-1)
        print(f"Number of clusters: {n_clusters}")
        print(f"Noise points: {n_noise}")
    
    # Try additional DBSCAN configurations on 2D data for comparison
    print("\nTesting DBSCAN on 2D PCA data...")
    eps_values = [0.3, 0.5, 0.7, 1.0]
    dbscan_ari_scores = []
    
    for eps in eps_values:
        print(f"\nTesting DBSCAN with eps={eps} on 2D data...")
        dbscan_2d = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels_2d = dbscan_2d.fit_predict(data_2d)
        
        n_clusters = len(set(dbscan_labels_2d)) - (1 if -1 in dbscan_labels_2d else 0)
        n_noise = list(dbscan_labels_2d).count(-1)
        print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
        
        if n_clusters > 1 and n_noise < len(data_2d) * 0.9:  # If we have clusters and not too much noise
            mask = dbscan_labels_2d != -1
            if mask.sum() > 0:
                ari_dbscan = adjusted_rand_score(target[mask], dbscan_labels_2d[mask])
                dbscan_ari_scores.append(ari_dbscan)
                print(f"  ARI (excluding noise): {ari_dbscan:.3f}")
            else:
                dbscan_ari_scores.append(-1)
        else:
            print("  → Too few clusters or too much noise")
            dbscan_ari_scores.append(-1)
    
    # Plot ARI for DBSCAN on 2D data
    valid_scores = [score for score in dbscan_ari_scores if score != -1]
    valid_eps = [eps for eps, score in zip(eps_values, dbscan_ari_scores) if score != -1]
    
    if valid_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_eps, valid_scores, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Epsilon (eps)')
        plt.ylabel('Adjusted Rand Index')
        plt.title('DBSCAN Performance vs Epsilon (2D PCA Data)')
        plt.grid(True, alpha=0.3)
        plt.xticks(valid_eps)
        for eps, ari in zip(valid_eps, valid_scores):
            plt.annotate(f'{ari:.3f}', (eps, ari), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        plt.tight_layout()
    
    # Summary comparison
    print("\n" + "="*50)
    print("SUMMARY COMPARISON")
    print("="*50)
    
    # Get ARI from comprehensive evaluation
    kmeans_metrics = evaluate_clustering_comprehensive(data, target, cluster_labels)
    ari = kmeans_metrics['ari']
    
    print(f"K-Means (k=10):           ARI = {ari:.3f}")
    print(f"Agglomerative (Ward):     ARI = {best_agg_ari:.3f}")
    print(f"Best k for K-Means:       k={k_values[np.argmax(ari_scores)]}, ARI = {max(ari_scores):.3f}")
    
    # DBSCAN summary
    for eps, ari_dbscan in zip(eps_values, dbscan_ari_scores):
        if ari_dbscan != -1:
            print(f"DBSCAN (eps={eps}):        ARI = {ari_dbscan:.3f}")
        else:
            print(f"DBSCAN (eps={eps}):        No valid clusters found")
    
    # Create combined visualization of all three algorithms
    print("\n" + "="*50)
    print("COMBINED VISUALIZATION")
    print("="*50)
    create_combined_visualization(data_2d, target, clustering_results)

def evaluate_clustering_comprehensive(data, true_labels, cluster_labels):
    """Comprehensive evaluation of clustering using multiple metrics"""
    
    print("EXTERNAL METRICS (require true labels):")
    print("-" * 40)
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI):     {ari:.3f}")
    print("  → Range: [-1, 1], Perfect=1.0, Random≈0.0")
    print("  → Use: General clustering quality measure")
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    print(f"\nNormalized Mutual Info (NMI):  {nmi:.3f}")
    print("  → Range: [0, 1], Perfect=1.0, Random≈0.0")
    print("  → Use: Information-theoretic similarity measure")
    
    # Homogeneity, Completeness, V-measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, cluster_labels)
    print(f"\nHomogeneity:                   {homogeneity:.3f}")
    print("  → How pure are clusters? (each cluster contains only one class)")
    print(f"Completeness:                  {completeness:.3f}")
    print("  → How complete are clusters? (all members of class in same cluster)")
    print(f"V-measure:                     {v_measure:.3f}")
    print("  → Harmonic mean of homogeneity and completeness")
    
    print("\nINTERNAL METRICS (no true labels needed):")
    print("-" * 40)
    
    # Silhouette Score
    silhouette = silhouette_score(data, cluster_labels)
    print(f"Silhouette Score:              {silhouette:.3f}")
    print("  → Range: [-1, 1], Higher=Better")
    print("  → Use: How well-separated are clusters?")
    
    # Calinski-Harabasz Index
    calinski = calinski_harabasz_score(data, cluster_labels)
    print(f"\nCalinski-Harabasz Index:       {calinski:.1f}")
    print("  → Range: [0, ∞], Higher=Better")
    print("  → Use: Ratio of between/within cluster variance")
    
    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(data, cluster_labels)
    print(f"\nDavies-Bouldin Index:          {davies_bouldin:.3f}")
    print("  → Range: [0, ∞], Lower=Better")
    print("  → Use: Average similarity between clusters")
    
    return {
        'ari': ari, 'nmi': nmi, 'homogeneity': homogeneity, 
        'completeness': completeness, 'v_measure': v_measure,
        'silhouette': silhouette, 'calinski': calinski, 
        'davies_bouldin': davies_bouldin
    }

def create_cluster_visualization(data_2d, cluster_labels, true_labels, title_suffix=""):
    """Create side-by-side visualization of clusters vs true labels"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot clusters
    scatter1 = ax1.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_labels, 
                          cmap='tab10', alpha=0.7, s=50)
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.set_title(f'Clusters - {title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    # Plot true labels
    scatter2 = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=true_labels, 
                          cmap='tab10', alpha=0.7, s=50)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.set_title('True Digit Labels')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    plt.colorbar(scatter2, ax=ax2, label='Digit')
    
    plt.tight_layout()

def analyze_cluster_digit_mapping(true_labels, cluster_labels):
    """Analyze how well clusters map to true digits"""
    
    print("\nCluster-to-Digit Mapping Analysis:")
    print("-" * 40)
    
    # Create confusion-like matrix
    n_clusters = len(np.unique(cluster_labels))
    n_digits = len(np.unique(true_labels))
    
    mapping_matrix = np.zeros((n_clusters, n_digits))
    
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        for digit in range(n_digits):
            digit_mask = true_labels == digit
            mapping_matrix[cluster, digit] = np.sum(cluster_mask & digit_mask)
    
    # Find dominant digit for each cluster
    for cluster in range(n_clusters):
        dominant_digit = np.argmax(mapping_matrix[cluster])
        dominant_count = mapping_matrix[cluster, dominant_digit]
        total_count = np.sum(mapping_matrix[cluster])
        purity = dominant_count / total_count if total_count > 0 else 0
        
        print(f"Cluster {cluster}: Mostly digit {dominant_digit} "
              f"({dominant_count}/{total_count} = {purity:.2%})")

def create_combined_visualization(data_2d, target, clustering_results):
    """Create a combined visualization showing all clustering algorithms and true labels"""
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot true labels (reference)
    scatter0 = axes[0].scatter(data_2d[:, 0], data_2d[:, 1], c=target, 
                              cmap='tab10', alpha=0.7, s=30)
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    axes[0].set_title('True Digit Labels (Reference)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter0, ax=axes[0], label='Digit')
    
    # Plot clustering results
    plot_idx = 1
    colors = ['viridis', 'plasma', 'coolwarm']
    
    for i, (algo_name, result) in enumerate(clustering_results.items()):
        if result['labels'] is not None and plot_idx < 4:
            labels = result['labels']
            ari = result['ari']
            name = result['name']
            
            # Handle noise points for DBSCAN (color them gray)
            if algo_name == 'dbscan' and -1 in labels:
                # Create custom colormap that includes gray for noise
                scatter_colors = labels.copy().astype(float)
                noise_mask = labels == -1
                scatter_colors[noise_mask] = np.max(labels) + 1  # Assign special value for noise
                
                scatter = axes[plot_idx].scatter(data_2d[:, 0], data_2d[:, 1], 
                                               c=scatter_colors, cmap='tab10', alpha=0.7, s=30)
                
                # Manually color noise points gray
                if noise_mask.any():
                    axes[plot_idx].scatter(data_2d[noise_mask, 0], data_2d[noise_mask, 1], 
                                         c='gray', alpha=0.7, s=30, label='Noise')
                    axes[plot_idx].legend()
            else:
                scatter = axes[plot_idx].scatter(data_2d[:, 0], data_2d[:, 1], 
                                               c=labels, cmap='tab10', alpha=0.7, s=30)
            
            axes[plot_idx].set_xlabel('First Principal Component')
            axes[plot_idx].set_ylabel('Second Principal Component')
            
            if ari is not None:
                axes[plot_idx].set_title(f'{name}\nARI: {ari:.3f}', fontweight='bold')
            else:
                axes[plot_idx].set_title(f'{name}', fontweight='bold')
            
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Add colorbar (skip for DBSCAN with noise to avoid confusion)
            if not (algo_name == 'dbscan' and -1 in labels):
                plt.colorbar(scatter, ax=axes[plot_idx], label='Cluster')
            
            plot_idx += 1
    
    # Hide unused subplot if less than 3 algorithms
    if plot_idx == 3:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    
    # Create a separate comparison bar chart
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    algorithm_names = []
    ari_scores = []
    
    for algo_name, result in clustering_results.items():
        if result['ari'] is not None:
            algorithm_names.append(result['name'])
            ari_scores.append(result['ari'])
    
    if algorithm_names:
        bars = ax.bar(algorithm_names, ari_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Adjusted Rand Index (ARI)')
        ax.set_title('Clustering Algorithm Performance Comparison', fontweight='bold')
        ax.set_ylim(0, max(ari_scores) * 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, ari_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    return fig, fig2

if __name__ == "__main__":
    main()
    plt.show()