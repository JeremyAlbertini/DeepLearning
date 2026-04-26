import numpy as np

class NN2Layer:
    def __init__(self, n0, n1, n2):
        self.W1 = np.random.randn(n1, n0)
        self.b1 = np.zeros((n1, 1))
        self.W2 = np.random.randn(n2, n1)
        self.b2 = np.zeros((n2, 1))
        self.A1 = np.empty
        self.A2 = np.empty

    def forward_propagation(self, X):
        Z1 = self.W1.dot(X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))

        Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))

    def log_loss(self, y):
        A2 = np.clip(self.A2, 1e-15, 1 - 1e-15)
        return 1 / y.shape[1] * np.sum(-y * np.log(A2) - (1 - y) * np.log(1 - A2))

    def back_propagation(self, X, y):
        m = y.shape[1]

        dZ2 = self.A2 - y
        self.dW2 = 1 / m * dZ2.dot(self.A1.T)
        self.db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)
        self.dW1 = 1 / m * dZ1.dot(X.T)
        self.db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def predict(self, X):
        self.forward_propagation(X)
        return self.A2 >= 0.5
