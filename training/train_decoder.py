#!/usr/bin/env python3
"""
Train 5-class direction classifier including A button.
Filters out ambiguous transition samples during training.
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


class ONNXWrapper(nn.Module):
    """Hard argmax output — clean ±1 signals for the game."""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        # Temperature scaling: high temp = hard argmax
        probs = torch.softmax(logits * 10.0, dim=-1)
        vectors = torch.tensor([
            [ 0.0,  1.0],   # UP
            [ 0.0, -1.0],   # DOWN
            [-1.0,  0.0],   # LEFT
            [ 1.0,  0.0],   # RIGHT
            [ 0.0,  0.0],   # NEUTRAL/A (center for now)
        ])
        return torch.matmul(probs, vectors)


def label_to_class(lx, ly, deflection_threshold=0.7):
    """
    Convert normalised (x, y) to class.
    Only assigns a direction if joystick is clearly deflected past threshold.
    Returns None for ambiguous/transition samples — these get filtered out.
    """
    ax, ay = abs(lx), abs(ly)
    # Clear neutral
    if ax < 0.2 and ay < 0.2:
        return 4  # NEUTRAL
    # Clear direction — must be past threshold
    if ay >= ax and ay > deflection_threshold:
        return 0 if ly < 0 else 1  # UP or DOWN — Y axis inverted in label channel
    if ax > ay and ax > deflection_threshold:
        return 2 if lx < 0 else 3  # LEFT or RIGHT
    # Ambiguous — filter out
    return None


def load_and_filter_data(path, deflection_threshold=0.7):
    d = np.load(path)
    features = d['features'].astype(np.float32)
    labels   = d['labels'].astype(np.float32)
    print(f"Loaded {len(features)} raw samples")

    # Normalise labels to [-1, 1]
    for i in range(2):
        vmin, vmax = labels[:, i].min(), labels[:, i].max()
        if vmax > 1.0 or vmin < -1.0:
            labels[:, i] = 2 * (labels[:, i] - vmin) / (vmax - vmin + 1e-8) - 1.0

    # Assign classes and filter ambiguous samples
    keep_feat, keep_class = [], []
    filtered = 0
    for i in range(len(features)):
        c = label_to_class(labels[i, 0], labels[i, 1], deflection_threshold)
        if c is None:
            filtered += 1
            continue
        keep_feat.append(features[i])
        keep_class.append(c)

    print(f"Filtered out {filtered} ambiguous transition samples")
    features = np.array(keep_feat, dtype=np.float32)
    classes  = np.array(keep_class, dtype=np.int64)

    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
    counts = np.bincount(classes, minlength=5)
    print("Class distribution after filtering:")
    for n, c in zip(names, counts):
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
                  f"train={tl:.4f}  val={vl:.4f}  val_acc={acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, features, classes, device):
    model.eval()
    X = torch.from_numpy(features).to(device)
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()
    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
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
    parser.add_argument('--data',      required=True)
    parser.add_argument('--out',       default='decoder.onnx')
    parser.add_argument('--epochs',    type=int,   default=100)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--batch',     type=int,   default=128)
    parser.add_argument('--channels',  type=int,   default=32)
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Joystick deflection threshold for clean samples (default: 0.7)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features, classes = load_and_filter_data(args.data, args.threshold)

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


if __name__ == '__main__':
    main()