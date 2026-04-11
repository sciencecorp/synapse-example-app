#!/usr/bin/env python3
"""
Train a PyTorch MLP: spike_counts [32] -> cursor_xy [2], export to ONNX.
Usage: python3 training/train_decoder.py --data training_data.npz --out decoder.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    print("pip install torch")
    sys.exit(1)


class JoystickDecoder(nn.Module):
    def __init__(self, n_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 2), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


def load_data(path):
    d = np.load(path)
    features = d['features'].astype(np.float32)
    labels   = d['labels'].astype(np.float32)
    print(f"Loaded {len(features)} samples")
    for i in range(2):
        vmin, vmax = labels[:, i].min(), labels[:, i].max()
        if vmax > 1.0 or vmin < -1.0:
            labels[:, i] = 2 * (labels[:, i] - vmin) / (vmax - vmin + 1e-8) - 1.0
            print(f"  Normalised label col {i}: [{vmin:.1f},{vmax:.1f}] -> [-1,1]")
    return features, labels


def train(model, train_dl, val_dl, epochs, lr, device):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.MSELoss()
    model.to(device)
    best_val, best_state = float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        tl = sum(loss_fn(model(xb.to(device)), yb.to(device)).item() * len(xb)
                 for xb, yb in train_dl) / len(train_dl.dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                vl += loss_fn(model(xb.to(device)), yb.to(device)).item() * len(xb)
        vl /= len(val_dl.dataset)
        sched.step()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  train={tl:.5f}  val={vl:.5f}  best={best_val:.5f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_directions(model, features, labels, threshold=0.30):
    def decode(x, y):
        if abs(x) < threshold and abs(y) < threshold: return 'NEUTRAL'
        if abs(x) >= abs(y): return 'RIGHT' if x > 0 else 'LEFT'
        return 'UP' if y > 0 else 'DOWN'
    model.eval().cpu()
    with torch.no_grad():
        preds = model(torch.from_numpy(features)).numpy()
    pairs = [(decode(labels[i,0], labels[i,1]), decode(preds[i,0], preds[i,1]))
             for i in range(len(labels))
             if decode(labels[i,0], labels[i,1]) != 'NEUTRAL']
    correct = sum(t == p for t, p in pairs)
    print(f"\nDirection accuracy: {correct}/{len(pairs)} = {100*correct/max(1,len(pairs)):.1f}%")
    for d in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        tp = sum(t == d and p == d for t, p in pairs)
        n  = sum(t == d             for t, p in pairs)
        print(f"  {d:6s}: {tp}/{n} = {100*tp/max(1,n):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--out',      default='decoder.onnx')
    parser.add_argument('--epochs',   type=int,   default=60)
    parser.add_argument('--lr',       type=float, default=3e-3)
    parser.add_argument('--batch',    type=int,   default=256)
    parser.add_argument('--channels', type=int,   default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    features, labels = load_data(args.data)
    X, Y = torch.from_numpy(features), torch.from_numpy(labels)
    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * 0.15)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch)

    model = JoystickDecoder(args.channels)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
    model = train(model, train_dl, val_dl, args.epochs, args.lr, device)
    evaluate_directions(model.cpu(), features, labels)

    model.eval().cpu()
    dummy = torch.zeros(1, args.channels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out_path), opset_version=18,
                      input_names=['spike_counts'], output_names=['cursor_xy'],
                      dynamic_axes={'spike_counts': {0: 'batch'}, 'cursor_xy': {0: 'batch'}})
    print(f"\nExported -> {out_path}")
    print(f"Next: synapsectl deploy-model {out_path} --name decoder -u <device-ip>")

if __name__ == '__main__':
    main()
