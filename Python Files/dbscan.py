from sklearn.cluster import dbscan
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

centers = [[1, 1], [-1, -1], [1, -1], [0, -2]]

X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4, random_state=0)

X = StandardScaler().fit_transform(X)

db = dbscan(X, eps=0.25, min_samples=10)

print('Core sample indices:', db[0])

n_noisy_points = list(db[1]).count(-1)
print('There are', n_noisy_points, 'noisy points')

core_samples_mask = np.zeros_like(db[1], dtype=bool)
core_samples_mask[db[0]] = True
labels = db[1]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))

print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
