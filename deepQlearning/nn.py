import numpy as np


class QNetwork:
    def __init__(self, n_inputs, hidden_layers, n_actions):
        """
        n_inputs      : taille de l'état
        hidden_layers : liste des tailles des couches cachées
        n_actions     : nombre d'actions = nombre de sorties
        """
        layer_sizes = [n_inputs] + hidden_layers + [n_actions]

        self.W = []
        self.b = []
        self.A = []
        self.dW = []
        self.db = []

        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.W.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * scale)
            self.b.append(np.zeros((layer_sizes[i + 1], 1)))
            self.A += [0]
            self.dW += [0]
            self.db += [0]

    def forward(self, X):
        """
        X : état, shape (n_inputs, m) où m = taille du batch
        Retourne Q-values, shape (n_actions, m)
        """
        self.Z = []
        current = X
        for i in range(len(self.W) - 1):
            Z = self.W[i].dot(current) + self.b[i]
            self.Z.append(Z)
            self.A[i] = np.maximum(0, Z)
            current = self.A[i]

        Z_out = self.W[-1].dot(current) + self.b[-1]
        self.A[-1] = Z_out

        return self.A[-1]

    def huber_loss(self, Q_pred, Q_target, delta=1.0):
        err = Q_pred - Q_target
        loss = np.where(np.abs(err) <= delta,
                        0.5 * err ** 2,
                        delta * (np.abs(err) - 0.5 * delta))
        return np.mean(loss)

    def backward(self, X, Q_target, delta=1.0):
        """
        X        : batch d'états, shape (n_inputs, m)
        Q_target : cibles Q, shape (n_actions, m)
                   Seule l'action jouée diffère de Q_pred, les autres sont
                   identiques → gradient nul pour les actions non jouées.
        """
        m = X.shape[1]
        err = self.A[-1] - Q_target

        dZ = np.where(np.abs(err) <= delta, err, delta * np.sign(err)) / m

        self.dW[-1] = dZ.dot(self.A[-2].T)
        self.db[-1] = np.sum(dZ, axis=1, keepdims=True)

        for i in range(len(self.W) - 2, 0, -1):
            dA = self.W[i + 1].T.dot(dZ)
            dZ = dA * (self.Z[i] > 0)
            self.dW[i] = dZ.dot(self.A[i - 1].T)
            self.db[i] = np.sum(dZ, axis=1, keepdims=True)

        dA = self.W[1].T.dot(dZ)
        dZ = dA * (self.Z[0] > 0)
        self.dW[0] = dZ.dot(X.T)
        self.db[0] = np.sum(dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * self.dW[i]
            self.b[i] -= learning_rate * self.db[i]

    def predict_action(self, state):
        """
        state : vecteur 1D de taille n_inputs
        Retourne l'action avec la Q-value maximale (scalaire int).
        """
        X = state.reshape(-1, 1)
        Q = self.forward(X)
        return int(np.argmax(Q))

    def get_weights(self):
        """Retourne une copie des poids (pour le target network)."""
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_weights(self, weights, biases):
        """Charge des poids (utilisé pour synchroniser le target network)."""
        self.W = [w.copy() for w in weights]
        self.b = [b.copy() for b in biases]

    def get_all_activations(self):
        return self.A
