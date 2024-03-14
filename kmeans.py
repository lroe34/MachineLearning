# write a function called doKmeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.stats import mode
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1])

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')


print(kmeans.cluster_centers_)

print(kmeans.inertia_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

def VCR (data):
    U, s, V = np.linalg.svd(data)
    vcr = s[0] / np.sum(s)
    return vcr

def resolve_masking(y_pred, y_true):
    pred_labels = np.zeros_like(y_true)
    for i in range(len(np.unique(y_pred))):
        mask = (y_pred == i)
        pred_labels[mask] = mode(y_true[mask])[0]
    return pred_labels

y_pred = kmeans.labels_
y_true = y
y_pred = resolve_masking(y_pred, y_true)


print(accuracy_score(y_true, y_pred))


def resolve_masking(y_pred, y_true):
    pred_labels = np.zeros_like(y_true)
    for i in range(len(np.unique(y_pred))):
        mask = (y_pred == i)
        pred_labels[mask] = mode(y_true[mask])[0]
    return pred_labels

def computeLearningHardIndex(acc, vcr):
    return (1 - acc) * vcr

print(accuracy_score(y_true, y_pred))

kmeans_acc_maxabs = accuracy_score(y_true, y_pred)

LHI_maxabs = computeLearningHardIndex(kmeans_acc_maxabs, VCR(X))

print(LHI_maxabs)

