import h5py
import numpy as np
from PercptronClass import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


PATH_DIRECTORY = "ImageClassifierPerceptron/"


def load_data():
    with h5py.File(PATH_DIRECTORY + "datasets/trainset.hdf5", "r") as f:
        X = np.array(f["X_train"])
        y = np.array(f["Y_train"])
    
    with h5py.File(PATH_DIRECTORY + "datasets/trainset.hdf5", "r") as f:
        X_test = np.array(f["X_train"])
        y_test = np.array(f["Y_train"])
    return X, y, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

X_train = X_train.dot(1/256)
X_test = X_test.dot(1/256)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


perceptronImage = Preceptron(X_train)

LogLoss = []
History = []

for i in range(5000):
    perceptronImage.model(X_train)
    LogLoss.append(perceptronImage.log_loss(y_train))
    perceptronImage.gradients(X_train,y_train)
    perceptronImage.update(0.001)
    if i % 50 == 0:
        History.append((perceptronImage.W.copy(), perceptronImage.b))

y_pre = perceptronImage.predict(X_test)
print(accuracy_score(y_test, y_pre))

plt.clf()
plt.plot(LogLoss)
plt.savefig(PATH_DIRECTORY + "logloss.png")
