from PercptronClass import Preceptron
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.animation import FuncAnimation

PATH_DIRECTORY = "PerceptronSimple/"

X, y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=1)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.savefig(PATH_DIRECTORY + "blobs.png")

perceptronSimple = Preceptron(X)

LogLoss = []
History = []

for i in range(2000):
    perceptronSimple.model(X)
    LogLoss.append(perceptronSimple.log_loss(y))
    perceptronSimple.gradients(X,y)
    perceptronSimple.update(0.1)
    if i % 50 == 0:
        History.append((perceptronSimple.W.copy(), perceptronSimple.b))


y_pre = perceptronSimple.predict(X)
print(accuracy_score(y, y_pre))


#drawing
plt.clf()
plt.plot(LogLoss)
plt.savefig(PATH_DIRECTORY + "logloss.png")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:,0], X[:, 1], c=y, cmap='summer')

x1 = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 500)
x2 = (-perceptronSimple.W[0] * x1 - perceptronSimple.b) / perceptronSimple.W[1]

ax.plot(x1, x2, c='orange', lw=3)

plt.savefig(PATH_DIRECTORY + "result.png")


#annimations
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:,0], X[:,1], c=y, cmap='summer')
x1 = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 500)
line, = ax.plot([], [], c='orange', lw=3)
title = ax.set_title('')

def update(frame):
    W, b = History[frame]
    x2 = (-W[0] * x1 - b) / W[1]
    line.set_data(x1, x2)
    title.set_text(f'Iteration {frame * 50}')
    return line, title

anim = FuncAnimation(fig, update, frames=len(History), interval=100, blit=True)
anim.save(PATH_DIRECTORY + 'evolution.gif', writer='pillow')

# animation with sigmoid background
x1_range = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 100)
x2_range = np.linspace(X[:,1].min() - 1, X[:,1].max() + 1, 100)
XX1, XX2 = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[XX1.ravel(), XX2.ravel()]

fig2, ax2 = plt.subplots(figsize=(9, 6))

perceptronSimple.W, perceptronSimple.b = History[0]
perceptronSimple.model(X_grid)
prob0 = perceptronSimple.A.reshape(XX1.shape)

mesh = ax2.pcolormesh(XX1, XX2, prob0, cmap='summer', alpha=0.8, vmin=0, vmax=1)
ax2.scatter(X[:,0], X[:,1], c=y, cmap='summer', edgecolors='k', linewidths=0.5, zorder=3)
line2, = ax2.plot([], [], c='orange', lw=3, zorder=4)
title2 = ax2.set_title('')

def update2(frame):
    W, b = History[frame]
    perceptronSimple.W = W
    perceptronSimple.b = b
    perceptronSimple.model(X_grid)
    prob = perceptronSimple.A.reshape(XX1.shape)
    mesh.set_array(prob.ravel())
    x2_line = (-W[0] * x1_range - b) / W[1]
    line2.set_data(x1_range, x2_line)
    title2.set_text(f'Iteration {frame * 50}')
    return mesh, line2, title2

anim2 = FuncAnimation(fig2, update2, frames=len(History), interval=50, blit=True)
anim2.save(PATH_DIRECTORY + 'evolution_sigmoid.gif', writer='pillow')