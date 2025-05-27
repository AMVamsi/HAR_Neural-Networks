import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import logging
import sys
import os

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
logging.info("Starting HAR CNN-LSTM training...")

# ========= Load Dataset ========= #
processed_dir = Path("/cfs/earth/scratch/adlurmoh/har_nn/processed")
X_train = np.load(processed_dir / "X_train.npy")
y_train = np.load(processed_dir / "y_train.npy")
X_test = np.load(processed_dir / "X_test.npy")
y_test = np.load(processed_dir / "y_test.npy")

logging.info(f"Loaded shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# ========= Dataset & DataLoader ========= #
class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

full_train_dataset = HARDataset(X_train, y_train)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(HARDataset(X_test, y_test), batch_size=64)

# ========= Model Definition ========= #
class CNNLSTM(nn.Module):
    def __init__(self, input_channels=9, lstm_hidden=128, num_classes=6):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat((h[0], h[1]), dim=1)
        return self.fc(h_cat)

model = CNNLSTM().to(device)
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
output_dir = Path("/cfs/earth/scratch/adlurmoh/har_nn/T500_CNNLSTM")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_dir / "cnn_lstm_model.pt")
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
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png")
plt.show()

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

logging.info(" Plots saved: confusion_matrix.png, training_summary.png")
