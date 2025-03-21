import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

class KMeansClustering:
    def __init__(self):
        
        """Add the necessary logic as instructed in the Exercise instruction document"""

        self.dataset_path = "./test_data.csv"
        self.df = pd.read_csv(self.dataset_path)

    def preprocess_data(self):
        
        """Add the necessary logic as instructed in the Exercise instruction document"""

        self.df = self.df.dropna()
        
        self.df = self.df.drop_duplicates()

        # First, select only the numerical columns
        numerical_columns = self.df.select_dtypes(include='number').columns

        #Then iterate over every column, define its 1st and 99th percentiles, and then clip the column to these values.
        for col in numerical_columns:
            lower_bound = self.df[col].quantile(0.01) # 'quantile' returns the value at the given quantile
            upper_bound = self.df[col].quantile(0.99)

            # 'clip' limits the values in the column to the values between lower and upper bounds
            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)        

    def plot_elbow_curve(self):
        
        """Add the logic to plot the elbow curve."""

        plt.plot(self.k_values, self.inertia, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method for Optimal k')
        plt.show()
        

    def apply_kmeans(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""

        # Normalize the data for better clustering results
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.df)

        #The following line is just for demonstration purposes only to see what the statistics of the transformed data look like.
        pd.DataFrame(self.X_scaled).describe()

        self.inertia = [] # Sum of squared distances of data points to their closest cluster centroid

        # The following defines a range of number of clusters that we iterate over to minimize Inertia
        # Typically, tha range is from 1 to 10, but it can be adjusted based on the problem
        # # We use (1,11) to include 10 in the range. 
        self.k_values = range(1, 11) 

        for k in self.k_values:
            # The following line creates a KMeans model with k clusters. 
            # n_init=10 means that the algorithm will run 10 times
            # Each time, it will try to select a different centroid to minimize Inertia.
            # You can have a higher number but that will take longer to run.
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

            # The following line fits the model to the scaled data
            kmeans.fit(self.X_scaled)

            # The following line appends the inertia of the model to the inertia list.
            # This will be used to plot the Elbow Curve
            self.inertia.append(kmeans.inertia_)
        

        optimal_k = 3 

        # Then we apply the KMeans algorithm with the optimal number of clusters, which is 3 in this case.
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

        # We fit the model to the scaled data and predict the clusters
        # We add another column called 'Cluster' to the original dataframe to show the cluster of each data point
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)

    def evaluate_clustering(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""

        silhouette = silhouette_score(self.X_scaled, self.df['Cluster'])
        davies_bouldin = davies_bouldin_score(self.X_scaled, self.df['Cluster'])
        ch_index = calinski_harabasz_score(self.X_scaled, self.df['Cluster'])

        return silhouette, davies_bouldin, ch_index

    def visualize_clusters(self):
        """Add the necessary logic as instructed in the Exercise instruction document"""

        # Flatten your features into two dimensions using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        # Plot with PCA components
        # X_pca[:, 0] is the first component (feature) of the PCA and X_pca[:, 1] is the second one.
        # c=df['Cluster'] means that we color the data points based on the cluster they belong to.
        # cmap='viridis' is the color map we use to color the data points.
        # alpha=0.6 means that the transparency of the data points is 60%.
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.df['Cluster'], cmap='viridis', alpha=0.6)

        # The following line plots the centroids of the clusters in red.
        # kmeans.cluster_centers_[:, 0] is the X coordinate of each centroid and kmeans.cluster_centers_[:, 1] is the second one.
        # s=200 means that the size of the centroids is 200.
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], 
                    s=200, c='red', marker='X', label='Centroids')

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('K-Means Clustering (PCA Reduced)')
        plt.legend()
        plt.show()

        plt.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c=self.df['Cluster'], cmap='viridis', alpha=0.6)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], 
                    s=200, c='red', marker='X', label='Centroids')
        plt.xlabel('Feature 1')  
        plt.ylabel('Feature 2')  
        plt.title(f'K-Means Clustering (Original Features)')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    """Add the necessary logic as instructed in the Exercise instruction document"""
    km = KMeansClustering()
    km.preprocess_data()
    km.apply_kmeans()
    km.plot_elbow_curve()
    a,b,c = km.evaluate_clustering()
    km.visualize_clusters()
    print(a,b,c)
    
