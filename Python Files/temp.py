# import iris data, do kmeans with different scalars
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import mode

iris = load_iris()
X = iris.data
y = iris.target
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]

def resolve_masking(y_pred, y_true):
    pred_labels = np.zeros_like(y_true)
    for i in range(len(np.unique(y_pred))):
        mask = (y_pred == i)
        pred_labels[mask] = mode(y_true[mask], keepdims=True)[0]
    return pred_labels

for scaler in scalers:
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit_predict(X_scaled)
    y_pred = kmeans.labels_
    y_true = y
    y_pred = resolve_masking(y_pred, y_true)
    print(accuracy_score(y_true, y_pred))


def computeLearningHardIndex(acc, vcr):
    return (1 - acc) * vcr

def VCR (data):
    U, s, V = np.linalg.svd(data)
    vcr = s[0] / np.sum(s)
    return vcr

kmeans_acc_maxabs = accuracy_score(y, y_pred)
vcr_maxabs = VCR(X)
LHI_maxabs = computeLearningHardIndex(kmeans_acc_maxabs, vcr_maxabs)



