from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
digits = load_digits()

print(dir(digits))

data = digits.data
print(data.shape)

print('\nimage is just reshaped from data: 64 --> 8x8')

images = digits.images
print(images.shape)

true_labels = digits.target
print("\ntrue_labels")
print(true_labels)

def plot_digits(images, n_rows, n_cols):
    plt.figure(figsize=(10, 8))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

# plot_digits(images, 20, 20)

k = 10


def doKmeans(X, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    label = kmeans.labels_
    return label, kmeans

pred_clusters, kmeans_digits = doKmeans(data, k)
centers = kmeans_digits.cluster_centers_

centers_r = centers.reshape(k, 8, 8)
plot_digits(centers_r, 2, 5)

kmeans_clustering_accuracy = accuracy_score(true_labels, pred_clusters)
# show the 1 digits
plot_digits(images[pred_clusters == 3], 5, 5)
# print(kmeans_clustering_accuracy)

