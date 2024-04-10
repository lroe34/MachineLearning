import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

load_iris = datasets.load_iris()

dir(load_iris)

training_data = load_iris.data[:, :2]
training_data_label = load_iris.target
cmap_light = ListedColormap(['#FFFAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FFAA', '#0000FF'])


x_min, x_max = training_data[:, 0].min() - 1, training_data[:, 0].max() + 1
y_min, y_max = training_data[:, 1].min() - 1, training_data[:, 1].max() + 1

h = .01

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
test_data = np.c_[xx.ravel(), yy.ravel()]

k_list = range(1, 25, 5)

for nb in k_list:
    kNN = KNeighborsClassifier(nb, weights='distance', algorithm='auto')
    kNN.fit(training_data, training_data_label)
    test_data_label = kNN.predict(test_data)

    test_data_labels = test_data_label.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, test_data_labels, cmap=cmap_light)
    plt.scatter(training_data[:, 0], training_data[:, 1], c=training_data_label, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % nb)

plt.show()







