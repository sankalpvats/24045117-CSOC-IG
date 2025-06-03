import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix

# Quick data wrangling - drop unused cols & encode
def prep_data(df):
    df = df.copy()
    for col in ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']:
        if col in df:
            df.drop(col, axis=1, inplace=True)
    df.dropna(inplace=True)

    encs = {}
    if 'Gender' in df:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        encs['Gender'] = le

    if 'Neighbourhood' in df:
        df = pd.get_dummies(df, columns=['Neighbourhood'], drop_first=True)

    if 'No-show' in df:
        df['No-show'] = (df['No-show'] == 'Yes').astype(int)

    y = df['No-show'].values
    X = df.drop('No-show', axis=1).values

    scale = StandardScaler()
    X = scale.fit_transform(X)
    return X, y, scale, encs

# Load the data and preprocess
dataf = pd.read_csv("dataset.csv")
X, y, scaler, encs = prep_data(dataf)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Convert to tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

# Basic model definition
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hid = nn.Linear(input_dim, 64)
        self.act = nn.ReLU()
        self.out = nn.Linear(64, 2)

    def forward(self, z):
        return self.out(self.act(self.hid(z)))

# Model setup
net = MyModel(x_train.shape[1])
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.008)

# Training w/ early stop
best = float('inf')
stuck = 0
max_stuck = 180

for ep in range(3000):
    net.train()
    pred = net(x_train)
    loss = loss_fn(pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if loss.item() < best - 1e-3:
        best = loss.item()
        stuck = 0
    else:
        stuck += 1
        if stuck >= max_stuck:
            print(f"[Stop] Epoch {ep}, no improvement in {max_stuck}")
            break

    if ep % 75 == 0:
        acc = (torch.argmax(pred, dim=1) == y_train).float().mean().item()
        print(f"[ep:{ep}] loss={loss.item():.3f}, acc={acc:.2%}")

# ---------------- Evaluation Section ---------------- #
net.eval()
with torch.no_grad():
    test_preds = net(x_test)
    y_prob = torch.softmax(test_preds, dim=1)[:, 1].numpy()
    y_pred = torch.argmax(test_preds, dim=1).numpy()
    y_true = y_test.numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print("Confusion Matrix:")
print(conf)
