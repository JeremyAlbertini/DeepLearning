from NN2LayerClass import NN2Layer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score
from tqdm import tqdm

PATH_DIRECTORY = "NeuralNetwork2Layer/"

X, y = make_circles(n_samples=300, factor=0.5, noise=0.05)

# Moitié du cercle extérieur reçoit le même label que le cercle intérieur
outer_mask = y == 0
angles = np.arctan2(X[outer_mask, 1], X[outer_mask, 0])
outer_indices = np.where(outer_mask)[0]
y[outer_indices[angles > 0]] = 1

X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.savefig(PATH_DIRECTORY + "base_state")

train_loss = []
train_acc = []
History = []

model = NN2Layer(X.shape[0], 256, y.shape[0])

for i in tqdm(range(10000)):
    model.forward_propagation(X)

    train_loss.append(model.log_loss(y))
    y_pred = model.predict(X)
    train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

    if i % 50 == 0:
        History.append((model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy()))

    model.back_propagation(X, y)
    model.update(0.1)

plt.clf()
plt.plot(train_loss)
plt.savefig(PATH_DIRECTORY + "log_loss")

plt.clf()
plt.plot(train_acc)
plt.savefig(PATH_DIRECTORY + "accuracy")

print(train_acc[-1])

# Animation de la frontière de décision
x1_range = np.linspace(X[0, :].min() - 1, X[0, :].max() + 1, 100)
x2_range = np.linspace(X[1, :].min() - 1, X[1, :].max() + 1, 100)
XX1, XX2 = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[XX1.ravel(), XX2.ravel()].T  # shape (2, 10000)

fig_anim, ax_anim = plt.subplots(figsize=(9, 6))

ax_anim.scatter(X[0, :], X[1, :], c=y.flatten(), cmap='summer', edgecolors='k', linewidths=0.5, zorder=3)

W1, b1, W2, b2 = History[0]
model.W1, model.b1, model.W2, model.b2 = W1, b1, W2, b2
model.forward_propagation(X_grid)
prob0 = model.A2.reshape(XX1.shape)
contour = ax_anim.contour(XX1, XX2, prob0, levels=[0.5], colors='orange', linewidths=3, zorder=4)
title_anim = ax_anim.set_title('')

def update_anim(frame):
    global contour
    W1, b1, W2, b2 = History[frame]
    model.W1, model.b1, model.W2, model.b2 = W1, b1, W2, b2
    model.forward_propagation(X_grid)
    prob = model.A2.reshape(XX1.shape)
    contour.remove()
    contour = ax_anim.contour(XX1, XX2, prob, levels=[0.5], colors='orange', linewidths=3, zorder=4)
    title_anim.set_text(f'Iteration {frame * 50}')

anim = FuncAnimation(fig_anim, update_anim, frames=len(History), interval=100, blit=False)
anim.save(PATH_DIRECTORY + 'evolution_decision_boundary.gif', writer='pillow')