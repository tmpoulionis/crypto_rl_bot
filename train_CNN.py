import os, glob, math, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# ------ Dataset ------
class DatasetWindow(Dataset):
    def __init__(self, metrics_npy, price_npy, seq_len=168):
        self.seq_len = seq_len
        self.X = []
        self.y = []
        
        metrics = np.load(metrics_npy).astype(np.float32)
        prices = np.load(price_npy).astype(np.float32)
        returns = np.diff(prices, axis=0) / prices[:-1]

        for t in range(len(metrics) - seq_len):
            self.X.append(metrics[t:t+seq_len])
            self.y.append(returns[t+seq_len-1])
        
        self.X = np.array(self.X) # (N, T-1, F)
        self.y = np.array(self.y) # (N, 1)
    
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        return torch.as_tensor(self.X[idx], dtype=torch.float32), \
                torch.as_tensor(self.y[idx], dtype=torch.float32)
                
# ----- Casual CNN -------
class CausalCNN(nn.Module):
    def __init__(self, input_size, embed_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(64, embed_size, kernel_size=1)
        )
        
    def forward(self, x):
        return self.cnn(x)
    
# ------ Training loop --------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} (device)")
    metrics_path = os.path.join(args.data_dir, "metrics_outfile.npy")
    price_path = os.path.join(args.data_dir, "price_outfile.npy")
    ds = DatasetWindow(metrics_npy=metrics_path, price_npy=price_path)
    N = len(ds)
    split = int(0.9*N)
    train_ds = Subset(ds, list(range(split)))
    val_ds = Subset(ds, list(range(split, N)))
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    cnn = CausalCNN(ds.X.shape[-1], embed_size=args.embed_size).to(device)
    head = nn.Linear(args.embed_size, 1).to(device)
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(head.parameters()), lr=args.learning_rate)
    
    # ------- Early stopping ---------
    best_val = float("inf")
    val_loss = float("inf")
    wait = 0
    for epoch in range(args.epochs):
        if val_loss < best_val: 
            best_val = val_loss
            wait = 0
            torch.save(cnn.state_dict(), args.out_file)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Saved CNN weights -> {args.out_file}")
                plot(cnn, head, val_dl, device)
                return;
                
        train_loss = 0
        for X, y in tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            X, y = X.to(device), y.to(device)
            feat_map = cnn.forward(X.permute(0, 2, 1))  # (B, F, T)
            pred = head(feat_map[:, :, -1]).squeeze(-1) # (B, 1)
            loss = ((pred - y)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                feat_map = cnn.forward(X.permute(0, 2, 1))
                pred = head(feat_map[:, :, -1]).squeeze(-1)
                val_loss += ((pred - y)**2).mean().item()
            
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        print(f"epoch {epoch+1} train_loss {train_loss:.4f} validation_loss {val_loss:4f}")
    
    torch.save(cnn.state_dict(), args.out_file)
    print(f"Saved CNN weights -> {args.out_file}")
    plot(cnn, head, val_dl, device)
    
def plot(cnn, head, val_dl, device):
    cnn.eval(); head.eval();
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for X, y in val_dl:
            X = X.to(device)
            feat_map = cnn.forward(X.permute(0, 2, 1))
            pred = head(feat_map[:, :, -1]).squeeze(-1).cpu().numpy()
            
            last_price = X[:, -1, 41].cpu().numpy()
            pred_price = last_price*(1 + pred)
            true_price = last_price*(1 + y.cpu().numpy().flatten())
            pred_list.append(pred_price)
            label_list.append(true_price)
    
    labels = np.concatenate(label_list).flatten()
    pred = np.concatenate(pred_list).flatten()
    
    plt.figure(figsize=(10,4))
    plt.plot(pred, label="prediction", color="orange")
    plt.plot(labels, label="ground-truth", color="blue",alpha=0.7)
    plt.title("Return prediction: Validation samples")
    plt.xlabel("sample index")
    plt.ylabel("return")
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