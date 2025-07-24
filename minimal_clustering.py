from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic clustering data
X_synthetic, y_true_synthetic = make_blobs(n_samples=300, centers=4, n_features=2, 
                                         random_state=42, cluster_std=1.5)

# Load real dataset (Iris) for comparison
iris = load_iris()
X_iris = iris.data
y_true_iris = iris.target

# Standardize features
scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_iris_scaled = scaler.fit_transform(X_iris)

datasets = [
    ("Synthetic Data", X_synthetic_scaled, y_true_synthetic, 4),
    ("Iris Dataset", X_iris_scaled, y_true_iris, 3)
]

print("Clustering Results:")
print("=" * 50)

for dataset_name, X, y_true, n_clusters in datasets:
    print(f"\n{dataset_name}:")
    print("-" * 30)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred_kmeans = kmeans.fit_predict(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_pred_dbscan = dbscan.fit_predict(X)
    
    # Evaluate clustering
    algorithms = [
        ("K-Means", y_pred_kmeans),
        ("DBSCAN", y_pred_dbscan)
    ]
    
    for alg_name, y_pred in algorithms:
        if len(set(y_pred)) > 1:  # Check if clusters were found
            ari = adjusted_rand_score(y_true, y_pred)
            sil = silhouette_score(X, y_pred)
            n_clusters_found = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            print(f"{alg_name}: ARI={ari:.3f}, Silhouette={sil:.3f}, Clusters={n_clusters_found}")
        else:
            print(f"{alg_name}: No clusters found")

# Visualize synthetic data clustering
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data
axes[0].scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_true_synthetic, cmap='viridis')
axes[0].set_title('True Clusters (Synthetic)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-Means results
kmeans_synthetic = KMeans(n_clusters=4, random_state=42)
y_pred_synthetic = kmeans_synthetic.fit_predict(X_synthetic_scaled)
axes[1].scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_pred_synthetic, cmap='viridis')
axes[1].scatter(kmeans_synthetic.cluster_centers_[:, 0], kmeans_synthetic.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
axes[1].set_title('K-Means Clustering')
axes[1].set_xlabel('Feature 1')
axes[1].legend()

# DBSCAN results
dbscan_synthetic = DBSCAN(eps=0.5, min_samples=5)
y_pred_dbscan_synthetic = dbscan_synthetic.fit_predict(X_synthetic_scaled)
axes[2].scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_pred_dbscan_synthetic, cmap='viridis')
axes[2].set_title('DBSCAN Clustering')
axes[2].set_xlabel('Feature 1')

plt.tight_layout()
plt.show()

print(f"\nCluster Analysis Summary:")
print(f"K-Means found {len(set(y_pred_synthetic))} clusters")
print(f"DBSCAN found {len(set(y_pred_dbscan_synthetic)) - (1 if -1 in y_pred_dbscan_synthetic else 0)} clusters")
print(f"DBSCAN noise points: {sum(y_pred_dbscan_synthetic == -1)}")