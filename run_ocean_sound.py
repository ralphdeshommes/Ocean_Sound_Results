import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

t0 = time.time()
BASE = r"C:\Users\Ralph\OneDrive\six_classes\Ocean_Sound_Classification"
AUDIO_DIR = os.path.join(BASE, "wmms_audio")
LABELS_CSV = os.path.join(BASE, "wmms_labels.csv")
OUT_DIR = r"C:\Users\Ralph\Desktop\Ocean_Sound_Results"

SR = 22050
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 130  # ~3 seconds of frames at this hop length
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
MIN_SAMPLES_PER_CLASS = 15  # drop species too rare to split train/val/test

device = torch.device("cpu")
print(f"Device: {device}")

# ---------- Load labels, drop species with too few samples ----------
df = pd.read_csv(LABELS_CSV)
counts = df["species"].value_counts()
keep_species = counts[counts >= MIN_SAMPLES_PER_CLASS].index
dropped = counts[counts < MIN_SAMPLES_PER_CLASS]
df = df[df["species"].isin(keep_species)].reset_index(drop=True)
print(f"Kept {df['species'].nunique()} species (>= {MIN_SAMPLES_PER_CLASS} samples each), "
      f"dropped {len(dropped)} rare species: {list(dropped.index)}")
print(f"Total usable clips: {len(df)}")

le = LabelEncoder()
df["label"] = le.fit_transform(df["species"])
num_classes = len(le.classes_)

# ---------- Feature extraction: mel-spectrograms ----------
def extract_mel(path):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :MAX_LEN]
    # normalize per-clip
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)

print("Extracting mel-spectrograms for all clips (this is the slow part)...")
X = np.zeros((len(df), N_MELS, MAX_LEN), dtype=np.float32)
for i, row in df.iterrows():
    X[i] = extract_mel(os.path.join(AUDIO_DIR, row["file"]))
    if i % 200 == 0:
        print(f"  {i}/{len(df)} clips processed ({time.time()-t0:.1f}s elapsed)")
y = df["label"].values
print(f"Feature extraction done in {time.time()-t0:.1f}s. X shape: {X.shape}")

# ---------- Split ----------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

class MelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)  # (N, 1, N_MELS, MAX_LEN)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MelDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MelDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(MelDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ---------- Models ----------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.fc(self.conv(x))

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, n_mels=N_MELS, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mels, hidden_size=hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        # x: (N, 1, N_MELS, MAX_LEN) -> (N, MAX_LEN, N_MELS)
        x = x.squeeze(1).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(model, name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        train_loss, train_acc = total_loss/n, correct/n

        model.eval()
        vloss, vcorrect, vn = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * len(yb)
                vcorrect += (out.argmax(1) == yb).sum().item()
                vn += len(yb)
        val_loss, val_acc = vloss/vn, vcorrect/vn
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"[{name}] Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")

    # Test evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    test_acc = accuracy_score(all_true, all_preds)
    print(f"[{name}] TEST ACCURACY: {test_acc:.4f}")
    report = classification_report(all_true, all_preds, target_names=le.classes_, zero_division=0)
    with open(os.path.join(OUT_DIR, f"{name}_classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    # Plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.title(f"{name} Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="train"); plt.plot(val_accs, label="val")
    plt.title(f"{name} Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_training_curves.png"))
    plt.close()

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix (test set, acc={test_acc:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    return test_acc

results = {}
results["CNN"] = train_model(CNN(num_classes), "CNN")
results["LSTM"] = train_model(LSTMClassifier(num_classes), "LSTM")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for k, v in results.items():
    print(f"{k}: {v:.4f} test accuracy")
print(f"Classes: {num_classes}  Total clips used: {len(df)}  Total elapsed: {time.time()-t0:.1f}s")

with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as f:
    f.write(f"Ocean Sound Classification - real run results\n")
    f.write(f"Species classes: {num_classes} (dropped {len(dropped)} species with < {MIN_SAMPLES_PER_CLASS} samples)\n")
    f.write(f"Total clips used: {len(df)}  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}\n\n")
    for k, v in results.items():
        f.write(f"{k} test accuracy: {v:.4f}\n")
