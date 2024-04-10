import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = datasets.load_iris()
data = iris.data
labels = iris.target

test_size = 0.3

training_data, test_data, training_data_label, test_data_label = train_test_split(data, labels, test_size=test_size)

N = 7

svm_learning_machine = svm.SVC(kernel='rbf', C=1.0, tol=1e-4, gamma=0.5)
svm_learning_machine.fit(training_data, training_data_label)

predicted_labels = svm_learning_machine.predict(test_data)

print("Score: ", svm_learning_machine.score(test_data, test_data_label))
