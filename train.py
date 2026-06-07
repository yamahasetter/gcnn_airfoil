"""
Training loop for AirfoilGCNN.

Loads surface.npz files from a dataset directory, builds graphs,
trains with per-field normalised MSE loss, and saves the best checkpoint.

Usage:
    python train.py --datadir data/ --outdir runs/run_001/
    python train.py --datadir data/ --outdir runs/run_001/ --epochs 500 --hidden 128
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm

from gcnn import AirfoilGCNN, build_airfoil_graph, sort_surface_nodes


# ── Dataset ───────────────────────────────────────────────────────────────────

class AirfoilDataset(torch.utils.data.Dataset):
    """
    Loads surface.npz files and builds PyG graphs.

    Each graph has:
        data.x       (N, 4)   node features  [x, y, κ, φ]
        data.y       (N, 3)   targets        [Cp, Cfx, Cfy]
        data.normals (N, 2)   outward normals
        data.ds      (N,)     nodal arc-length weights
    """

    def __init__(self, sample_dirs, normalizer=None):
        self.graphs = []
        for d in sample_dirs:
            npz = np.load(d / "surface.npz")
            x, y = npz["x"], npz["y"]

            # Sort nodes into consistent closed-loop ordering
            order = sort_surface_nodes(x, y)
            x, y  = x[order], y[order]

            Cp  = npz["Cp"][order]  if "Cp"  in npz else np.zeros_like(x)
            Cfx = npz["Cfx"][order] if "Cfx" in npz else np.zeros_like(x)
            Cfy = npz["Cfy"][order] if "Cfy" in npz else np.zeros_like(x)

            graph   = build_airfoil_graph(x, y)
            targets = torch.tensor(
                np.stack([Cp, Cfx, Cfy], axis=1).astype(np.float32)
            )
            graph.y = targets
            self.graphs.append(graph)

        self.normalizer = normalizer

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.normalizer is not None:
            g = g.clone()
            g.y = self.normalizer.normalize(g.y)
        return g


# ── Per-field normaliser ──────────────────────────────────────────────────────

class FieldNormalizer:
    """
    Standardises each of (Cp, Cfx, Cfy) independently.

    Cp is O(1–3), Cfx/Cfy is O(1e-3).  Without normalisation the MSE
    loss is dominated by Cp and the network ignores Cf entirely.
    """

    def __init__(self, graphs):
        # Collect all target tensors
        all_y = torch.cat([g.y for g in graphs], dim=0)   # (total_nodes, 3)
        self.mean = all_y.mean(dim=0)                       # (3,)
        self.std  = all_y.std(dim=0).clamp(min=1e-8)       # (3,)

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y_norm):
        return y_norm * self.std + self.mean

    def save(self, path):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        d   = torch.load(path, map_location="cpu")
        obj.mean = d["mean"]
        obj.std  = d["std"]
        return obj


# ── Training utilities ────────────────────────────────────────────────────────

def rel_l2(pred, target, normalizer):
    """
    Relative L2 error computed in physical (de-normalised) space.
    Normalised targets have near-zero mean so their norm is tiny,
    making the ratio explode — always de-normalise first.
    """
    pred_phys   = normalizer.denormalize(pred)
    target_phys = normalizer.denormalize(target)
    return (
        torch.norm(pred_phys - target_phys, dim=0) /
        torch.norm(target_phys,             dim=0).clamp(min=1e-8)
    ).mean().item()


def run_epoch(model, loader, optimizer, device, normalizer, train=True):
    model.train(train)
    total_loss = 0.0
    total_rel  = 0.0
    criterion  = nn.MSELoss()

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            pred  = model(batch)

            loss = criterion(pred, batch.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_rel  += rel_l2(pred.detach(), batch.y, normalizer)

    n = len(loader)
    return total_loss / n, total_rel / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AirfoilGCNN surrogate")
    parser.add_argument("--datadir",      type=str,   default="data",       help="Dataset root (default: data/)")
    parser.add_argument("--outdir",       type=str,   default="runs/run_001", help="Output dir for checkpoints and logs")
    parser.add_argument("--train-frac",   type=float, default=0.8,          help="Train/val split fraction (default: 0.8)")
    parser.add_argument("--epochs",       type=int,   default=500,          help="Max epochs (default: 500)")
    parser.add_argument("--batch-size",   type=int,   default=16,           help="Batch size (default: 16)")
    parser.add_argument("--lr",           type=float, default=1e-3,         help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--hidden",       type=int,   default=128,          help="Hidden dim (default: 128)")
    parser.add_argument("--n-layers",     type=int,   default=4,            help="GCN layers (default: 4)")
    parser.add_argument("--patience",     type=int,   default=50,           help="Early stopping patience (default: 50)")
    parser.add_argument("--seed",         type=int,   default=0,            help="Random seed (default: 0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load samples ──────────────────────────────────────────────────────────
    datadir     = Path(args.datadir)
    sample_dirs = sorted([d for d in datadir.iterdir()
                          if d.is_dir() and (d / "surface.npz").exists()])
    if not sample_dirs:
        raise FileNotFoundError(f"No surface.npz files found under {datadir}")
    print(f"Found {len(sample_dirs)} samples")

    # ── Train / val split ─────────────────────────────────────────────────────
    n_train = int(len(sample_dirs) * args.train_frac)
    rng     = np.random.default_rng(args.seed)
    idx     = rng.permutation(len(sample_dirs))
    train_dirs = [sample_dirs[i] for i in idx[:n_train]]
    val_dirs   = [sample_dirs[i] for i in idx[n_train:]]
    print(f"Train: {len(train_dirs)}   Val: {len(val_dirs)}")

    # ── Normaliser (fit on raw training targets) ──────────────────────────────
    raw_train = AirfoilDataset(train_dirs, normalizer=None)
    normalizer = FieldNormalizer(raw_train.graphs)
    normalizer.save(outdir / "normalizer.pt")

    train_ds = AirfoilDataset(train_dirs, normalizer=normalizer)
    val_ds   = AirfoilDataset(val_dirs,   normalizer=normalizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AirfoilGCNN(hidden=args.hidden, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    log = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_loss, train_rel = run_epoch(model, train_loader, optimizer, device, normalizer, train=True)
        val_loss,   val_rel   = run_epoch(model, val_loader,   optimizer, device, normalizer, train=False)
        scheduler.step()

        log.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_rel": train_rel,
            "val_loss":   val_loss,   "val_rel":   val_rel,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "args":        vars(args),
            }, outdir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

        if epoch % 10 == 0:
            tqdm.write(
                f"  epoch {epoch:4d}  "
                f"train {train_loss:.4e} (rel {train_rel:.3f})  "
                f"val {val_loss:.4e} (rel {val_rel:.3f})  "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(outdir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Loss plot ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs      = [e["epoch"]      for e in log]
    train_loss  = [e["train_loss"] for e in log]
    val_loss_   = [e["val_loss"]   for e in log]
    train_rel   = [e["train_rel"]  for e in log]
    val_rel_    = [e["val_rel"]    for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(epochs, train_loss, label="Train")
    ax1.semilogy(epochs, val_loss_,  label="Val")
    ax1.axvline(log[np.argmin(val_loss_)]["epoch"], color="gray",
                linestyle="--", lw=1, label="Best val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE loss (log scale)")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_rel, label="Train")
    ax2.plot(epochs, val_rel_,  label="Val")
    ax2.axvline(log[np.argmin(val_loss_)]["epoch"], color="gray",
                linestyle="--", lw=1, label="Best val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Relative L2 error")
    ax2.set_title("Relative L2 error"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nBest val loss: {best_val_loss:.4e}")
    print(f"Checkpoint:    {outdir}/best_model.pt")
    print(f"Normalizer:    {outdir}/normalizer.pt")
    print(f"Loss plot:     {outdir}/loss_curves.png")


if __name__ == "__main__":
    main()
