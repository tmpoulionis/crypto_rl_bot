import os, glob, math, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ----- Casual CNN & Head -------
class CausalCNN(nn.Module):
    def __init__(self, input_size, embed_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            # nn.Dropout1d(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            # nn.Dropout1d(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            # nn.Dropout1d(0.2),
            nn.Conv1d(64, embed_size, kernel_size=1)
        )
        
    def forward(self, x):
        return self.cnn(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.head(x);


# ------ Training loop --------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} (device)")

    # load the dataset
    inputs_path = os.path.join("./data/cnn_pretrain", "hourly_outfile.npy")
    targets_path = os.path.join("./data/cnn_pretrain", "targets_outfile.npy")
    X = np.load(inputs_path).astype(np.float32)
    y = np.load(targets_path).astype(np.float32)
    
    N = len(X)
    split = int(0.9*N)

    # create tensors and dataloader
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    train_ds = TensorDataset(X_tensor[:split], y_tensor[:split])
    val_ds = TensorDataset(X_tensor[split:], y_tensor[split:])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    cnn = CausalCNN(X.shape[-1], embed_size=args.embed_size).to(device)
    head = ClassificationHead(args.embed_size).to(device)
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(head.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # training loop
    print("Start training...")
    best_val = float("inf")
    val_loss = float("inf")
    train_hist, val_hist = [], []
    wait = 0
    for epoch in range(args.epochs):
        # -------- train ----------
        train_loss = 0
        train_preds, train_labels = [], []
        cnn.train(); head.train()
        for X, y in tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            X, y = X.to(device), y.squeeze().long().to(device)
            feat_map = cnn.forward(X.permute(0, 2, 1))  # (B, E, T)
            pooled, _ = feat_map.max(dim=2) # (B, E)
            logits = head(pooled)  # (B, 2)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  
            # print("Grad mean:", cnn.cnn[0].weight.grad.abs().mean().item())
            # print("Grad std:", cnn.cnn[0].weight.grad.abs().std().item())

            train_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.append(y.cpu().numpy())

        train_acc = balanced_accuracy_score(np.concatenate(train_labels), np.concatenate(train_preds))
        train_loss /= len(train_dl)
        # -------- validation ----------
        val_loss = 0
        val_preds, val_labels = [], []
        cnn.eval(); head.eval()
        for X, y in val_dl:
            X, y = X.to(device), y.squeeze(1).long().to(device)
            feat_map = cnn.forward(X.permute(0, 2, 1))  # (B, E, T)
            pooled, _ = feat_map.max(dim=2) # (B, E)
            logits = head(pooled)  # (B, 2)
            val_loss += criterion(logits, y).item()
            
            val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.append(y.cpu().numpy())

        val_acc = balanced_accuracy_score(np.concatenate(val_labels), np.concatenate(val_preds))
        val_loss /= len(val_dl)

        train_hist.append(train_loss)
        val_hist.append(val_loss)  
        print(f"train_loss {train_loss:.4f} - train accuracy {train_acc:.4f} \nvalidation_loss {val_loss:4f} - validation accuracy {val_acc:.4f}")

        # ------- early stopping ---------
        if val_loss < best_val: 
            best_val = val_loss
            wait = 0
            torch.save({"cnn": cnn.state_dict(), "head": head.state_dict()}, args.out_file)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Saved CNN weights -> {args.out_file}')
                break

    print("validation balanced accuracy: ", balanced_accuracy_score(np.concatenate(val_labels), np.concatenate(val_preds)))
    print(classification_report(np.concatenate(val_labels), np.concatenate(val_preds)))

    plt.figure(figsize=(8,4))
    plt.plot(train_hist, label="train MSE")
    plt.plot(val_hist,   label="val MSE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/dataset")
    p.add_argument("--seq_len", type=int, default=168)
    p.add_argument("--embed_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--out_file", default="cnn_pretrain.pt")
    train(p.parse_args())