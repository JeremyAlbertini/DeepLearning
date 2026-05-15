import kagglehub
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from NNClass import NNetwork
from tqdm import tqdm

CACHE_FILE = "dataset_cache_gray.npz"

if os.path.exists(CACHE_FILE):
    print("Chargement depuis le cache...")
    data = np.load(CACHE_FILE)
    X, y = data["X"], data["y"]
else:
    print("Construction du dataset...")
    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    dataset_path = os.path.join(path, "PetImages")

    IMG_SIZE = (64, 64)
    X, y = [], []

    for label, category in enumerate(["Cat", "Dog"]):
        folder = os.path.join(dataset_path, category)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert("L").resize(IMG_SIZE)  # "L" = niveaux de gris
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass

    X = np.array(X) / 255.0   # shape: (N, 64, 64)
    y = np.array(y)
    np.savez(CACHE_FILE, X=X, y=y)
    print(f"Cache sauvegardé dans {CACHE_FILE}")

# Préparation
X_flat = X.reshape(X.shape[0], -1).T       # (4096, N)
y_onehot = np.zeros((2, len(y)))
y_onehot[y, np.arange(len(y))] = 1         # (2, N)

# Train / test split (80/20)
m = X_flat.shape[1]
split = int(0.8 * m)
X_train, X_test = X_flat[:, :split], X_flat[:, split:]
y_train, y_test = y_onehot[:, :split], y_onehot[:, split:]

# Entraînement
nn = NNetwork(n0=4096, n1=[64, 64, 64, 64, 64], n2=2)
learning_rate = 0.1
epochs = 500
losses = []

for epoch in tqdm(range(epochs)):
    nn.forward_propagation(X_train)
    loss = nn.log_loss(y_train)
    nn.back_propagation(X_train, y_train)
    nn.update(learning_rate)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

# Précision sur le test
preds = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=0)
accuracy = np.mean(preds == y_test_labels)
print(f"\nPrécision sur le test : {accuracy * 100:.2f}%")

# Courbe de loss
plt.plot(losses)
plt.title("Loss pendant l'entraînement")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
