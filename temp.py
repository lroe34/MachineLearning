from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import mode

iris = load_iris()
X = iris.data
y = iris.target


def test_knn(metric='cosine', n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    return accuracy_score(y, y_pred)

print("cosine: " + str(test_knn()))

import numpy as np
import matplotlib.pyplot as plt
from mlp_toolkit.mplot3d import Axes3D

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(X, Y)

Z = X**2 + Y**2

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.colorbar(surf, ax=ax)

plt.show()