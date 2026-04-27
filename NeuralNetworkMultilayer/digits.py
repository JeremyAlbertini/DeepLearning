from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NNClass import NNetwork
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {digits.target[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

X = digits.data.T / 16.0
y_raw = digits.target

print(X, y_raw)

# One-hot encoding (10, 1797)
y = np.zeros((10, X.shape[1]))
y[y_raw, np.arange(X.shape[1])] = 1

# Split train/test
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X.T, y.T, y_raw, test_size=0.2, random_state=42
)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T

model = NNetwork(64, [128, 128, 128, 128], 10)

losstotal = []
accuracytotal = []

# Entraînement
for i in tqdm(range(8000)):
    model.forward_propagation(X_train)
    model.back_propagation(X_train, y_train)
    model.update(0.01)
    if i % 200 == 0:
        loss = model.log_loss(y_train)
        losstotal.append(loss)
        y_pred = model.predict(X_train)
        accuracytotal.append(accuracy_score(y_raw_train, y_pred))

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_raw_test, y_pred):.4f}")

plt.plot(losstotal)
plt.savefig('lossdigits.png')

plt.clf()
plt.plot(accuracytotal)
plt.savefig('accuracydigits.png')

indices = np.random.choice(len(digits.images), size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in zip(indices, axes.flatten()):
    ax.imshow(digits.images[i], cmap='gray')
    pred = model.predict(digits.data[i].reshape(-1, 1) / 16.0)[0]
    ax.set_title(f'Vrai: {digits.target[i]} | Pred: {pred}')
    ax.axis('off')

plt.tight_layout()
plt.show()