import os, glob, math, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ----- Casual CNN -------
class CausalCNN(nn.Module):
    def __init__(self, input_size, embed_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(64, embed_size, kernel_size=1)
        )
        
    def forward(self, x):
        return self.cnn(x)

# ------ Dataset ------
def WindowDataset(metrics_npy, price_npy, seq_len=168):
    seq_len = seq_len
    X, y = [], []
    
    metrics = np.load(metrics_npy).astype(np.float32)
    prices = np.load(price_npy).astype(np.float32)
    returns = np.diff(prices, axis=0) / prices[:-1]
    
    for t in range(len(metrics) - seq_len):
        X.append(metrics[t:t+seq_len])
        y.append(returns[t+seq_len-1])
    
    X = np.array(X) # (N, T-1, F)
    y = np.array(y) # (N, 1)
    return X, y            
                
# ------ Training loop --------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} (device)")
    
    # load and split data
    metrics_path = os.path.join(args.data_dir, "metrics_outfile.npy")
    price_path = os.path.join(args.data_dir, "price_outfile.npy")
    X, y = WindowDataset(metrics_npy=metrics_path, price_npy=price_path)
    N = len(X)
    split = int(0.9*N)
    
    # compute normalizers and normalize data
    mean = X.mean(axis=1, keepdims=True) # (N, 1, F)
    std= X.std(axis=1, keepdims=True) + 1e-8 # avoid zero division
    
    X_norm = (X - mean) / std
    y_scaled = y*100
    
    # create tensors and dataloader
    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y)
    
    train_ds = TensorDataset(X_tensor[:split], y_tensor[:split])
    val_ds = TensorDataset(X_tensor[split:], y_tensor[split:])
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # model
    cnn = CausalCNN(X.shape[-1], embed_size=args.embed_size).to(device)
    head = nn.Linear(args.embed_size, 1).to(device)
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(head.parameters()), lr=args.learning_rate)
    criterion = nn.HuberLoss(delta=0.2)
    
    best_val = float("inf")
    val_loss = float("inf")
    train_hist, val_hist = [], []
    wait = 0
    for epoch in range(args.epochs):
        # -------- train ----------
        train_loss = 0
        for X, y in tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            X, y = X.to(device), y.to(device)
            feat_map = cnn.forward(X.permute(0, 2, 1))  # (B, F, T)
            pred = head(feat_map[:, :, -1]) # (B, 1)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  
        train_loss /= len(train_dl)
        # -------- validation ----------
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                feat_map = cnn.forward(X.permute(0, 2, 1))
                pred = head(feat_map[:, :, -1])
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_dl)
        
        train_hist.append(train_loss)
        val_hist.append(val_loss)  
        print(f"epoch {epoch+1} train_loss {train_loss:.4f} validation_loss {val_loss:4f}")
    
        # ------- early stopping ---------
        if val_loss < best_val: 
            best_val = val_loss
            wait = 0
            torch.save({"cnn": cnn.state_dict(), "head": head.state_dict()}, args.out_file)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Saved CNN weights -> {args.out_file}")
                break
    
    # -------- evaluation -------------
    ckpt = torch.load(args.out_file, map_location=device, weights_only=True)
    cnn.load_state_dict(ckpt["cnn"])
    head.load_state_dict(ckpt["head"])
    preds, labels = [], []
    
    cnn.eval(); head.eval()
    with torch.no_grad():
        for X, y in val_dl:
            X = X.to(device)
            feat_map = cnn(X.permute(0, 2, 1))
            pred = head(feat_map[:, :, -1]).squeeze(-1).cpu().numpy()
            preds.append(pred)
            labels.append(y.squeeze(-1).cpu().numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    
    metrics_table(labels, preds)
    
    plt.figure(figsize=(8,4))
    plt.plot(train_hist, label="train MSE")
    plt.plot(val_hist,   label="val MSE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.plot(preds, label="prediction", color="orange")
    plt.plot(labels, label="ground-truth", color="blue",alpha=0.7)
    plt.title("close_1h_ETH Returns: Validation samples")
    plt.xlabel("sample index")
    plt.ylabel("return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def metrics_table(labels, preds):
    mse  = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(labels, preds)
    r2   = r2_score(labels, preds)
    dir_acc = (np.sign(labels) == np.sign(preds)).mean()
    
    metrics = pd.DataFrame(
        {
            "MSE":  [mse],
            "RMSE": [rmse],
            "MAE":  [mae],
            "R2":   [r2],
            "Dir-Acc": [dir_acc],
        }
    )

    print("\nValidation metrics:")
    print(metrics.to_string(index=False, float_format="%.4f"))
    
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