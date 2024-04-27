import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

X = [[0,0], [-2,0], [1,1], [10,1]]

y = [0, 0, 1, 1]

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']

clf = svm.SVC()
clf.fit(X, y)

pred = clf.predict([[2.5,8]])

predicted_labels = {}
support_vectors = {}

for kernel in kernel_list:
    clf = svm.SVC(kernel=kernel, tol = 0.0001, gamma = 0.5)
    clf.fit(X, y)
    predicted_labels[kernel] = pred
    support_vectors[kernel] = clf.support_vectors_

print(predicted_labels)
print(support_vectors)

iris = datasets.load_iris()
data = iris.data
labels = iris.target

kernel = 'rbf'
clf = svm.SVC(kernel=kernel, gamma = 0.5, C=1.0)

print(classification_report(labels, clf.fit(data, labels).predict(data)))

k = 10
scores = cross_val_score(clf, data, labels, cv=k)
print(scores)
print(np.mean(scores))

