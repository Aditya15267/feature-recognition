import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from furniture_dataset import Furnituredata
from models.multilabel_model import MultiLabelResNet
from utils.label_encoder import LabelEncoder


BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
SCHEMA_PATH = 'data/feature_schema.json'
LABEL_PATH = 'data/annotations.json'
IMG_DIR = 'data/images'

def accuracy_thresh(y_pred, y_true, threshold=0.5):
    """
    Calculate accuracy with a threshold.
    """
    y_pred = (y_pred > threshold).float()
    return (y_pred == y_true).float().mean().item()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = LabelEncoder(SCHEMA_PATH)

    data = Furnituredata(IMG_DIR, LABEL_PATH, SCHEMA_PATH)
    train_len = int(len(data) * 0.8)
    val_len = len(data) - train_len
    train_set, val_set = random_split(data, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = MultiLabelResNet(output_dim=encoder.total_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        acc = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                acc.append(accuracy_thresh(outputs, labels))
        
        avg_acc = sum(acc) / len(acc)
        print(f"Validation Accuracy: {avg_acc:.4f}")

        # Save the model
        if avg_acc > 0.9:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/model_90acc.pth")
            print("Model reached 90% accuracy! Saved.")
            break

if __name__ == "__main__":
    train()
