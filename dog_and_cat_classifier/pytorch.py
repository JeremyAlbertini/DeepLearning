import numpy as np
import os
import kagglehub
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

CACHE_FILE = "dataset_cache.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# --- Chargement des données ---
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
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass

    X = np.array(X) / 255.0
    y = np.array(y)
    np.savez(CACHE_FILE, X=X, y=y)
    print(f"Cache sauvegardé dans {CACHE_FILE}")

# --- Préparation des tenseurs ---
# PyTorch attend (N, C, H, W) donc on permute les canaux
X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 3, 64, 64)
y_tensor = torch.tensor(y, dtype=torch.long)                           # (N,)

# Shuffle avant le split pour avoir chats et chiens dans les deux sets
indices = torch.randperm(len(X_tensor))
X_tensor = X_tensor[indices]
y_tensor = y_tensor[indices]

# Train / test split (80/20)
m = len(X_tensor)
split = int(0.8 * m)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64)

# --- Architecture CNN ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (N, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (N, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (N, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 128, 8, 8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# --- Entraînement ---
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

epochs = 50
patience = 7
best_test_loss = float("inf")
no_improve = 0
train_losses, test_losses, accuracies = [], [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Loss et accuracy sur le test
    model.eval()
    test_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            test_loss += criterion(out, y_batch).item()
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss  = test_loss  / len(test_loader)
    accuracy = correct / total * 100
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    accuracies.append(accuracy)
    scheduler.step(avg_train_loss)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Early stopping sur la test loss
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping à l'epoch {epoch+1} — meilleure test loss : {best_test_loss:.4f}")
            break

# --- Courbes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train")
ax1.plot(test_losses,  label="Test")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(accuracies)
ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.show()
