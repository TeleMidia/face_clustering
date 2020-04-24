import os
import pandas as pd
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

class FeatureExtractor():
	"""docstring for FeatureExtractor"""
	#backbone=['vgg16', 'resnet', 'senet50']
	def __init__(self, backbone='senet50'):
		self.backbone = backbone
		self.detector = MTCNN()
		self.model = VGGFace(model=self.backbone, include_top=False, input_shape=(224, 224, 3), pooling='avg')
		
	def extract(self, urls):
		self.df_imgs = pd.DataFrame(urls, columns=['urls'])
		embeddings = self.df_imgs.urls.apply(lambda x: self.get_embeddings(x))
		self.df_imgs['embeddings'] = embeddings

		return self.df_imgs

	def extract_face(self, filename, required_size=(224, 224)):
		print(filename)
		pixels = cv2.imread(filename)
		if pixels is not None:
			pixels_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

			results = self.detector.detect_faces(pixels_rgb)

			if len(results)>0:

				x1, y1, width, height = results[0]['box']
				x2, y2 = x1 + width, y1 + height
				face = pixels_rgb[y1:y2, x1:x2]

				if face.shape[0]>0 and face.shape[1]>0:
					return cv2.resize(face, required_size)

		return np.zeros(2048)

	def get_embeddings(self, filename):
		face = self.extract_face(filename)
		sample = np.asarray(face, 'float32')
		sample = np.expand_dims(sample, axis=0)
		sample = preprocess_input(sample, version=2)
		
		embedding = self.model.predict(sample)
		
		return embedding[0]