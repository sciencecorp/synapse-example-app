#!/usr/bin/env python3
"""
Train a 4-class direction classifier: spike_counts [32] -> direction [UP/DOWN/LEFT/RIGHT]
Exports to ONNX outputting (x, y) cursor position.
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

# Direction mapping — matches game's decode_direction logic
# x>0=RIGHT, x<0=LEFT, y>0=UP, y<0=DOWN
DIRECTION_VECTORS = {
    0: ( 0.0,  1.0),  # UP
    1: ( 0.0, -1.0),  # DOWN
    2: (-1.0,  0.0),  # LEFT
    3: ( 1.0,  0.0),  # RIGHT
    4: ( 0.0,  0.0),  # NEUTRAL
}

class DirectionClassifier(nn.Module):
    def __init__(self, n_channels=32, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def predict_xy(self, x):
        """Output (x, y) cursor position for ONNX export."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        # Weighted sum of direction vectors
        vectors = torch.tensor([
            [0.0,  1.0],   # UP
            [0.0, -1.0],   # DOWN
            [-1.0, 0.0],   # LEFT
            [1.0,  0.0],   # RIGHT
            [0.0,  0.0],   # NEUTRAL
        ], dtype=torch.float32)
        xy = torch.matmul(probs, vectors)
        return xy


class ONNXWrapper(nn.Module):
    """Outputs (x, y) using one-hot argmax — clean ±1 signals."""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        # Scale up probabilities so argmax winner dominates completely
        # This gives near-hard outputs while remaining ONNX-exportable
        scaled = probs * 10.0
        softmax_hard = torch.softmax(scaled, dim=-1)
        vectors = torch.tensor([
            [ 0.0,  1.0],   # UP
            [ 0.0, -1.0],   # DOWN
            [-1.0,  0.0],   # LEFT
            [ 1.0,  0.0],   # RIGHT
            [ 0.0,  0.0],   # NEUTRAL
        ])
        return torch.matmul(softmax_hard, vectors)


def label_to_class(lx, ly, threshold=0.3):
    """Convert normalised (x, y) label to class index."""
    ax, ay = abs(lx), abs(ly)
    if ax < threshold and ay < threshold:
        return 4  # NEUTRAL
    if ay >= ax:
        return 0 if ly > 0 else 1  # UP or DOWN
    else:
        return 2 if lx < 0 else 3  # LEFT or RIGHT


def load_data(path):
    d = np.load(path)
    features = d['features'].astype(np.float32)
    labels   = d['labels'].astype(np.float32)
    print(f"Loaded {len(features)} samples")

    # Normalise labels to [-1, 1]
    for i in range(2):
        vmin, vmax = labels[:, i].min(), labels[:, i].max()
        if vmax > 1.0 or vmin < -1.0:
            labels[:, i] = 2 * (labels[:, i] - vmin) / (vmax - vmin + 1e-8) - 1.0

    # Convert to class indices
    classes = np.array([label_to_class(labels[i,0], labels[i,1])
                        for i in range(len(labels))], dtype=np.int64)

    counts = np.bincount(classes, minlength=5)
    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
    print("Class distribution:")
    for i, (n, c) in enumerate(zip(names, counts)):
        print(f"  {n:8s}: {c}")

    return features, classes


def train(model, train_dl, val_dl, epochs, lr, device):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(train_dl.dataset)

        model.eval()
        vl, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl += loss_fn(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
        vl /= len(val_dl.dataset)
        acc = correct / len(val_dl.dataset)
        sched.step()

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  val_acc={acc:.1%}  best={best_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, features, classes, device):
    model.eval()
    X = torch.from_numpy(features).to(device)
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()

    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
    # Only evaluate on non-neutral
    mask = classes != 4
    correct = (preds[mask] == classes[mask]).sum()
    total = mask.sum()
    print(f"\nDirection accuracy (excl. neutral): {correct}/{total} = {100*correct/max(1,total):.1f}%")
    for i, name in enumerate(names[:4]):
        tp = ((preds == i) & (classes == i)).sum()
        n  = (classes == i).sum()
        print(f"  {name:8s}: {tp}/{n} = {100*tp/max(1,n):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True)
    parser.add_argument('--out',      default='decoder.onnx')
    parser.add_argument('--epochs',   type=int,   default=100)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--batch',    type=int,   default=128)
    parser.add_argument('--channels', type=int,   default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features, classes = load_data(args.data)

    X = torch.from_numpy(features)
    Y = torch.from_numpy(classes)
    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * 0.15)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch)

    model = DirectionClassifier(args.channels)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
    model = train(model, train_dl, val_dl, args.epochs, args.lr, device)
    evaluate(model, features, classes, device)

    # Export ONNX wrapper that outputs (x, y)
    wrapper = ONNXWrapper(model.cpu())
    wrapper.eval()
    dummy = torch.zeros(1, args.channels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        opset_version=18,
        input_names=['spike_counts'],
        output_names=['cursor_xy'],
        dynamic_axes={'spike_counts': {0: 'batch'}, 'cursor_xy': {0: 'batch'}}
    )
    print(f"\nExported -> {out_path}")
    print(f"Next: synapsectl deploy-model {out_path} --name decoder -u <device-ip>")


if __name__ == '__main__':
    main()