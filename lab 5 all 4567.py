import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load the features dataset
file_path = "C:\\Users\\Administrator\\Documents\\ML_bird_project\\bird_species_features (1).xlsx"
features_df = pd.read_excel(file_path)

# Extract features for clustering (assuming columns to cluster start from the third column)
X = features_df.iloc[:, 2:].values  # Adjust indices if needed

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Define a range of k values to test
k_values = range(2, 11)  # Example: testing k values from 2 to 10

# Initialize lists to store metrics
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
distortions = []

for k in k_values:
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=100, n_init="auto")
    clusters = kmeans.fit_predict(X_normalized)
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_normalized, clusters)
    calinski_harabasz_avg = calinski_harabasz_score(X_normalized, clusters)
    davies_bouldin_avg = davies_bouldin_score(X_normalized, clusters)
    
    # Store the metrics
    silhouette_scores.append(silhouette_avg)
    calinski_harabasz_scores.append(calinski_harabasz_avg)
    davies_bouldin_scores.append(davies_bouldin_avg)
    
    # Store the distortion (inertia)
    distortions.append(kmeans.inertia_)
    
    # Print clustering results for the current k
    print(f"Number of clusters: {k}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg:.4f}")
    print()

# Optional: Save the clustered data for the last k value (e.g., k=10) to an Excel file
last_k = k_values[-1]
last_kmeans = KMeans(n_clusters=last_k, random_state=100, n_init="auto")
last_clusters = last_kmeans.fit_predict(X_normalized)
features_df['cluster'] = last_clusters
output_file = 'bird_species_clustered.xlsx'
features_df.to_excel(output_file, index=False)
print(f"Clustered data for k={last_k} saved to '{output_file}'.")

# Plot the metrics against k values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot(k_values, calinski_harabasz_scores, marker='o', linestyle='--')
plt.title('Calinski-Harabasz Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_scores, marker='o', linestyle='--')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

# Plot the elbow curve to determine the optimal k value
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), distortions, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.xticks(range(2, 11))
plt.show()
