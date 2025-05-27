import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import logging
import sys
import math

# ========= CUDA Check ========= #
assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")
print(f" Using device: {torch.cuda.get_device_name(0)}")

# ========= Logging Setup ========= #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"Using device: {torch.cuda.get_device_name(0)}")
logging.info("🚀 Starting HAR Transformer training...")

# ========= Load Dataset ========= #
processed_dir = Path("/cfs/earth/scratch/adlurmoh/har_nn/processed")
X_train = np.load(processed_dir / "X_train.npy")
y_train = np.load(processed_dir / "y_train.npy")
X_test = np.load(processed_dir / "X_test.npy")
y_test = np.load(processed_dir / "y_test.npy")

logging.info(f"Loaded shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# ========= Transform for Transformer: (batch, 128, 9) ========= #
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).transpose(1, 2)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# ========= Dataset & DataLoader ========= #
class HARTransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

full_train_dataset = HARTransformerDataset(X_train_tensor, y_train_tensor)

# 90% train, 10% validation
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(HARTransformerDataset(X_test_tensor, y_test_tensor), batch_size=64)

# ========= Positional Encoding ========= #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle_rates = 1 / torch.pow(10000, (i / d_model))
        angle_rads = pos * angle_rates
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle_rads)
        pe[:, 1::2] = torch.cos(angle_rads)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ========= Model Definition ========= #
class HARTransformer(nn.Module):
    def __init__(self, input_dim=9, model_dim=64, num_classes=6, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(d_model=model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)

model = HARTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========= Training Loop ========= #
epochs = 500
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            _, preds = torch.max(output, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    val_acc = correct / total
    val_accuracies.append(val_acc)
    logging.info(f"Epoch {epoch+1}/{epochs} – Loss: {avg_loss:.4f} – Val Acc: {val_acc:.4f}")

# ========= Save Artifacts ========= #
output_dir = Path("/cfs/earth/scratch/adlurmoh/har_nn/T500")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_dir / "har_transformer.pt")
np.save(output_dir / "loss_log.npy", np.array(train_losses))
np.save(output_dir / "val_accuracy.npy", np.array(val_accuracies))
logging.info("Model, loss, and validation accuracy saved.")

# ========= Test Evaluation ========= #
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

np.save(output_dir / "all_preds.npy", all_preds)
np.save(output_dir / "all_targets.npy", all_targets)

activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}
target_names = [activity_labels[i] for i in sorted(set(all_targets))]

logging.info("\nClassification Report:")
logging.info(classification_report(all_targets, all_preds, target_names=target_names))

# ========= Plots ========= #
# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png")
plt.show()

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, epochs + 1), val_accuracies, marker='x', label='Validation Accuracy')
plt.title("Training Loss & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "training_summary.png")
plt.show()

logging.info("Plots saved: confusion_matrix.png, training_summary.png")
