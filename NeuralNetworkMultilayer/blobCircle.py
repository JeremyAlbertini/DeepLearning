from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from NNClass import NNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tqdm import tqdm

X1, y1 = make_circles(n_samples=300, factor=0.5, noise=0.05)
X2, y2 = make_circles(n_samples=300, factor=0.7, noise=0.05)
x3, y3 = make_moons(n_samples=300, noise=0.05)

X2[:, 0] += 2
x3[:, 0] += 4

X_full = np.vstack([X1, X2, x3])
y_full = np.hstack([y1, y2, y3])

plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='summer')
plt.show()

model = NNetwork(2, [128, 128, 128 , 128, 128, 128], 2)

train_loss = []
train_acc = []
History = []

X_full = X_full.T

y_onehot = np.zeros((2, X_full.shape[1]))
y_onehot[y_full, np.arange(X_full.shape[1])] = 1

for i in tqdm(range(10000)):
    model.forward_propagation(X_full)

    train_loss.append(model.log_loss(y_onehot))
    y_pred = model.predict(X_full)
    train_acc.append(accuracy_score(y_full.flatten(), y_pred.flatten()))

    if i % 100 == 0:
        History.append([W.copy() for W in model.W] + [b.copy() for b in model.b])
    if i % 500 == 0:
        print(train_loss[-1], train_acc[-1])


    model.back_propagation(X_full, y_onehot)
    model.update(0.01)
    if train_acc[-1] == 1:
        break

y_pred = model.predict(X_full)

print("accuracy:", accuracy_score(y_full.flatten(), y_pred.flatten()))


plt.clf()
plt.plot(train_loss)
plt.show()

plt.clf()
plt.plot(train_acc)
plt.show()

# Animation frontière de décision
from matplotlib.animation import FuncAnimation

x1_range = np.linspace(X_full[0].min() - 0.5, X_full[0].max() + 0.5, 100)
x2_range = np.linspace(X_full[1].min() - 0.5, X_full[1].max() + 0.5, 100)
XX1, XX2 = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[XX1.ravel(), XX2.ravel()].T

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X_full[0], X_full[1], c=y_full, cmap='summer', edgecolors='k', linewidths=0.5, zorder=3)

n = len(model.W)

def restore(snapshot):
    for i in range(n):
        model.W[i] = snapshot[i]
        model.b[i] = snapshot[n + i]

restore(History[0])
model.forward_propagation(X_grid)
prob = model.A[-1][1].reshape(XX1.shape)
contour = ax.contourf(XX1, XX2, prob, levels=50, cmap='coolwarm', alpha=0.4, zorder=1)
contour_line = ax.contour(XX1, XX2, prob, levels=[0.5], colors='orange', linewidths=2, zorder=2)
title = ax.set_title('')

def update_anim(frame):
    global contour, contour_line
    restore(History[frame])
    model.forward_propagation(X_grid)
    prob = model.A[-1][1].reshape(XX1.shape)
    contour.remove()
    contour_line.remove()
    contour = ax.contourf(XX1, XX2, prob, levels=50, cmap='coolwarm', alpha=0.4, zorder=1)
    contour_line = ax.contour(XX1, XX2, prob, levels=[0.5], colors='orange', linewidths=2, zorder=2)
    title.set_text(f'Iteration {frame * 100}')

anim = FuncAnimation(fig, update_anim, frames=len(History), interval=100, blit=False)
import os
PATH_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
anim.save(os.path.join(PATH_DIRECTORY, 'decision_boundary_circles.gif'), writer='pillow')
plt.show()