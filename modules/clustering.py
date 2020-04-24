import pandas as pd
import numpy as np
import random
from modules.feature_extractor import FeatureExtractor
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42

class Clusterer():
	"""docstring for Clusterer"""

	def __init__(self, urls = None, face_embeddings = None, algs=['kmeans', 'gmm', 'affinity', 'agglomerative'], backbone='senet50', n_clusters=2):
		self.algs = algs
		self.n_clusters = n_clusters
		self.urls = urls

		if face_embeddings is None:
			extractor = FeatureExtractor(backbone)
			self.face_embeddings = extractor.extract(self.urls)
		else:
			self.face_embeddings = face_embeddings
		self.scaler = MinMaxScaler()

		self.models = {'kmeans': lambda n_clusters: KMeans(n_clusters= n_clusters, verbose = 0, random_state=RANDOM_SEED), 
						'gmm' : lambda n_clusters: GaussianMixture(n_components = n_clusters),
						'affinity': lambda n_clusters: AffinityPropagation(),
						'agglomerative': lambda n_clusters: AgglomerativeClustering(n_clusters=n_clusters)}

	def clusterize(self):
		features = pd.DataFrame(self.face_embeddings['embeddings'].values.tolist(), index = self.face_embeddings.index)
		features_scaled = self.scaler.fit_transform(features)

		for alg in self.algs:
			self.face_embeddings['cluster_'+alg] = self.models[alg](n_clusters = self.n_clusters).fit_predict(features_scaled)
		return self.face_embeddings
