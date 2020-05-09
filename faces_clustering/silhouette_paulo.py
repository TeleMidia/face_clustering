
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import homogeneity_score
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def silhuoette(X, range_n_clusters):
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
        

    fig, ax1 = plt.subplots(1)
    ax1.set_title(("Silhouette score for each cluster number"),fontsize=16, fontweight='bold')
    fig.set_size_inches(18, 7)
    silhuette_plot = silhuette_plot/max(silhuette_plot)
    ax1.plot(x_plot , silhuette_plot, label='silhuoette')
    sse_plot = sse_plot/max(sse_plot)
    ax1.plot(x_plot , sse_plot, label='inertia')
    ax1.set_xticks([i+1 for i in range(max(range_n_clusters))])
    ax1.legend()

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()