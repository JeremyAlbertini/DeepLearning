import numpy as np
import math

class Preceptron:
    def __init__(self, X):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
        self.A = np.empty
        self.dW = np.empty
        self.db = np.empty

    def model(self, X):
        Z = X.dot(self.W) + self.b
        self.A = 1 / (1 + np.exp(-Z))

    def log_loss(self, y):
        return 1 / len(y) * np.sum(-y * np.log(self.A) - (1 - y) * np.log(1 - self.A))

    def gradients(self, X, y):
        self.dW = 1 / len(y) * np.dot(X.T, self.A - y)
        self.db = 1 / len(y) * np.sum(self.A - y)
    
    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
    
    def predict(self, X):
        return self.A >= 0.5






if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    percep = Preceptron(X)
    print(percep.b, percep.w)
