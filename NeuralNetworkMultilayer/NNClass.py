import numpy as np

class NNetwork:
    def __init__(self, n0, n1, n2):
        self.W = []
        self.b = []
        self.A = []
        self.dW = []
        self.db = []
        self.W.append(np.random.randn(n1[0], n0))
        self.b.append(np.zeros((n1[0], 1)))
        for i in range(1, len(n1)):
            self.W.append(np.random.randn(n1[i], n1[i -1]))
            self.b.append(np.zeros((n1[i], 1)))
            self.A.append(np.empty)
            self.dW.append([])
            self.db.append([])
        self.W.append(np.random.randn(n2, n1[-1]))
        self.b.append(np.zeros((n2, 1)))
        self.A.append(np.empty)
        self.A.append(np.empty)
        self.dW.append([])
        self.dW.append([])
        self.db.append([])
        self.db.append([])

    def forward_propagation(self, X):
        Z = self.W[0].dot(X) + self.b[0]
        self.A[0] = 1 / (1 + np.exp(-Z))

        for i in range(1, len(self.W) - 1):
            Z = self.W[i].dot(self.A[i - 1]) + self.b[i]
            self.A[i] = 1 / (1 + np.exp(-Z))

        Z = self.W[-1].dot(self.A[-2]) + self.b[-1]
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        self.A[-1] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def log_loss(self, y):
        A = np.clip(self.A[-1], 1e-15, 1 - 1e-15)
        return -1 / y.shape[1] * np.sum(y * np.log(A))

    def back_propagation(self, X, y):
        m = y.shape[1]

        dZ = self.A[-1] - y
        self.dW[-1] = 1 / m * dZ.dot(self.A[-2].T)
        self.db[-1] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        for i in range(len(self.W) - 2, 0, -1):
            dZ = np.dot(self.W[i+1].T, dZ) * self.A[i] * (1 - self.A[i])
            self.dW[i] = 1 / m * dZ.dot(self.A[i - 1].T)
            self.db[i] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        dZ = np.dot(self.W[1].T, dZ) * self.A[0] * (1 - self.A[0])
        self.dW[0] = 1 / m * dZ.dot(X.T)
        self.db[0] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * self.dW[i]
            self.b[i] -= learning_rate * self.db[i]

    def predict(self, X):
        self.forward_propagation(X)
        return np.argmax(self.A[-1], axis=0)
