from PercptronClass import Preceptron
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import numpy as np

X, y = make_blobs(n_samples=500, n_features=3, centers=2, random_state=2)
y = y.reshape((y.shape[0], 1))

perceptron = Preceptron(X)

for i in range(500):
    perceptron.model(X)
    perceptron.gradients(X, y)
    perceptron.update(0.1)

# Scatter 3D des données brutes
fig = go.Figure(data=[go.Scatter3d(
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=X[:, 2].flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y.flatten(),
        colorscale='YlGn',
        opacity=0.8,
        reversescale=True
    )
)])

fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()

# Surface sigmoïde (en fixant x2 à sa moyenne)
X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
xx0, xx1 = np.meshgrid(X0, X1)
x2_mean = X[:, 2].mean()

Z = perceptron.W[0] * xx0 + perceptron.W[1] * xx1 + perceptron.W[2] * x2_mean + perceptron.b
A = 1 / (1 + np.exp(-Z))

fig = go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity=0.7, reversescale=True)])

fig.add_scatter3d(
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity=0.9, reversescale=True)
)

fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()
