
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import homogeneity_score
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def silhuoette(X, alg = "kmeans", max_dec = 5):

    assert alg in ['agglomerative', 'kmeans'], "alg must be kmeans or agglomerative"

    x_plot = []
    silhuette_plot = []
    sse_plot = []

    max_silhouette = -1

    #x_plot.append(1)
    #silhuette_plot.append(max_silhouette)   

    num_dec = 0
    n_clusters = 1

    fig, ax1 = plt.subplots(1)
    ax1.set_title(("Silhouette score for each cluster number"),fontsize=16, fontweight='bold')
    fig.set_size_inches(18, 7)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    hl, = plt.plot([], [])
    best_labels = []

    while num_dec <= max_dec:
        n_clusters+=1
        if alg == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif alg == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)

        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        x_plot.append(n_clusters)
        silhuette_plot.append(silhouette_avg)        

        if silhouette_avg < max_silhouette:
            num_dec += 1
        else:
            best_labels = cluster_labels
            max_silhouette = silhouette_avg
            num_dec = 0
    
    if max_silhouette < 0.2:
        best_labels = [0]*len(X)   

    #silhuette_plot = silhuette_plot/max(silhuette_plot)
    ax1.plot(x_plot , silhuette_plot, label='silhuoette')
    #sse_plot = sse_plot/max(sse_plot)
    #ax1.plot(x_plot , sse_plot, label='inertia')
    ax1.set_xticks([i+1 for i in range(n_clusters)])
    ax1.legend()
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette score")
    plt.show()

    print(f"Best cluster number is {len(set(best_labels))}")

    return best_labels