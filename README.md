# face_clustering
This project aims at clustering people by face.
Author: Paulo Renato Conceição Mendes

## Install:

You can use pip to install.
```
pip install https://github.com/TeleMidia/face_clustering/releases/download/0.0.1/face_clustering-0.0.1-py3-none-any.whl
```

## Usage:

```
from faces_clustering import Clusterer

clusters = Clusterer(n_clusters=4, urls=complete_urls).clusterize()
```
### Basic Params

- *n_clusters*: number of clusters used

- *urls*: list of urls from the images you want to clusterize

### Additional Params

- *backbone*: string that specifies the backbone used for extracting face features. It can be one of the following: 
```
'senet50'(default), 'resnet50', 'vgg16'
```
```
clusters = Clusterer(n_clusters=4, urls=complete_urls, backbone='resnet50').clusterize()
```
- *algs*: list that specify which clustering algorithms you want to use. By default, all of them are used:
```
['kmeans', 'gmm', 'affinity', 'agglomerative']
```
```
clusters = Clusterer(n_clusters=4, urls=complete_urls, algs=['kmeans', 'gmm']).clusterize()
```
  - *kmeans* refers to sklearn kmeans
  - *gmm* refers to sklearn Gaussian Mixture Model
  - *affinity* refers to sklearn Affinity Propagation
  - *agglomerative* refers to sklearn Ward Agglomerarive 

## References

- Keras-vggface used for feature extraction: https://github.com/rcmalli/keras-vggface
- MTCNN for face detection: https://pypi.org/project/mtcnn/
- Sklearn for clustering algorithms: https://scikit-learn.org/stable/modules/clustering.html#clustering
