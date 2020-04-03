
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import homogeneity_score
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def analyze(X, range_n_clusters, show_individual_graphs = True):
    x_plot = []
    silhuette_plot = []
    sse_plot = []
    print(f'Computing clusters from {min(range_n_clusters)} to {max(range_n_clusters)}')
    for n_clusters in tqdm(range_n_clusters):
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        sse_plot.append(clusterer.inertia_)

        #homo_score = homogeneity_score(true_labels, cluster_labels)

        #print(f"Clusters ={n_clusters} Silhouette: {silhouette_avg} SSE: {clusterer.inertia_}")
        x_plot.append(n_clusters)
        silhuette_plot.append(silhouette_avg)
        
        if(show_individual_graphs):        
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1,1)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        
     
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


            plt.title(("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"%n_clusters),fontsize=14, fontweight='bold')
            

    fig, ax1 = plt.subplots(1)
    ax1.set_title(("Silhouette score for each cluster number"),fontsize=16, fontweight='bold')
    fig.set_size_inches(18, 7)
    silhuette_plot = silhuette_plot/max(silhuette_plot)
    ax1.plot(x_plot , silhuette_plot)
    sse_plot = sse_plot/max(sse_plot)
    ax1.plot(x_plot , sse_plot)
    ax1.set_xticks([i+1 for i in range(max(range_n_clusters))])

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    return sihuetter_plot