"""
Geodesic Convolutional Neural Network for NACA airfoil surface fields.

Graph structure
---------------
Nodes  : surface points with features (x, y, κ, φ)
         x, y     : chord-normalised coordinates
         κ        : discrete curvature  (dθ/ds)
         φ        : outward normal angle  atan2(ny, nx)
Edges  : bidirectional ring — each node connects to its two arc-length neighbours
Output : per-node (Cp, Cfx, Cfy)  shape (N, 3)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm

"""
══════════════════════════════════════════════════════════════════════════════
WORKFLOW OVERVIEW
══════════════════════════════════════════════════════════════════════════════

This module defines the graph representation of an airfoil surface and the
GCNN model that maps surface geometry to surface aerodynamic fields.

STEP 1 — Surface ordering (sort_surface_nodes)
  Raw surface points from SU2/NACA may not be in a consistent loop order.
  sort_surface_nodes() reconstructs the closed boundary by nearest-neighbor
  traversal starting from the leading edge (leftmost x). This produces a
  consistent clockwise ordering: upper surface LE→TE then lower surface TE→LE.

STEP 2 — Graph construction (build_airfoil_graph)
  Given ordered (x, y) surface coordinates, compute per-node geometric
  features and build a PyG Data object:
    - Forward and backward tangent vectors are averaged at each node to get
      a smooth nodal tangent, from which the outward normal is derived.
    - Discrete curvature κ = Δθ/Δs is computed from the tangent angle change
      per unit arc length along the boundary.
    - Normal angle φ = atan2(ny, nx) is included as a directional feature.
    - Node feature matrix: [x, y, κ, φ]  shape (N, 4)
    - Edges form a bidirectional ring: each node i connects to i+1 and i-1
      (with periodic wrap), encoding the 1D geodesic structure of the curve.
    - Additional tensors stored on the graph: normals (N,2), ds (N,) arc-
      length weights, and pos (N,2) for downstream use.

STEP 3 — Residual GCN block (ResGCNBlock)
  A single layer of message-passing: GCNConv aggregates neighbor features,
  followed by BatchNorm and ReLU. A residual (skip) connection adds the
  input back after a linear projection if channel dimensions differ. This
  stabilizes training and allows deeper networks without vanishing gradients.

STEP 4 — Full model forward pass (AirfoilGCNN)
  1. Linear input projection: 4 node features → hidden dim (128).
  2. N residual GCN layers (default 4) performing geodesic message-passing
     along the boundary ring graph.
  3. MLP output head: hidden → hidden/2 → 3, producing per-node predictions
     of (Cp, Cfx, Cfy) — the pressure coefficient and two friction components.

This module is consumed by:
  - train.py         for supervised training on SU2-generated data
  - surrogate_model.py  for end-to-end differentiable L/D prediction

══════════════════════════════════════════════════════════════════════════════
"""

# ── Graph construction ────────────────────────────────────────────────────────

def sort_surface_nodes(x, y):
    """
    Reconstruct a closed surface loop by nearest-neighbor traversal.

    Returns indices that order (x, y) along the boundary.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    pts = np.column_stack([x, y])
    n = len(pts)

    # distance matrix (OK for typical surface sizes; can optimize if needed)
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff**2, axis=2)

    # start from a deterministic point: left-most (leading edge for airfoils)
    start = np.argmin(x)

    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True

    current = start

    for _ in range(n - 1):
        # mask already visited points
        d = dist2[current].copy()
        d[visited] = np.inf

        # pick nearest unvisited neighbor
        nxt = np.argmin(d)

        # safety: if we get stuck (numerical or disconnected), break
        if visited[nxt]:
            break

        order.append(nxt)
        visited[nxt] = True
        current = nxt

    order = np.array(order)

    # Enforce a consistent CLOCKWISE loop. Nearest-neighbor traversal may come
    # out CW or CCW depending on the data; downstream code (build_airfoil_graph
    # normals and integrate_forces) assumes CW. CW ⇔ signed (shoelace) area < 0.
    xo, yo = x[order], y[order]
    signed_area = 0.5 * np.sum(xo * np.roll(yo, -1) - np.roll(xo, -1) * yo)
    if signed_area > 0:                       # CCW → reverse to CW
        order = order[::-1]

    return order

# def sort_surface_nodes(x, y):
#     """
#     Sort surface nodes into a consistent closed loop:
#       upper surface  LE → TE  (sorted by x ascending,  y >= 0)
#       lower surface  TE → LE  (sorted by x descending, y <  0)

#     Returns indices that reorder the input arrays.
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     upper = np.where(y >= 0)[0]
#     lower = np.where(y <  0)[0]
#     upper_sorted = upper[np.argsort( x[upper])]
#     lower_sorted = lower[np.argsort(-x[lower])]
#     return np.concatenate([upper_sorted, lower_sorted])


def build_airfoil_graph(x_surf, y_surf):
    """
    Build a PyG Data object from ordered airfoil surface coordinates.

    Nodes are expected as a closed loop (upper LE→TE then lower TE→LE).
    Call sort_surface_nodes() first if ordering is not guaranteed.

    Parameters
    ----------
    x_surf, y_surf : array-like, shape (N,)

    Returns
    -------
    torch_geometric.data.Data with:
        x          (N, 4)   node features  [x, y, κ, φ]
        edge_index (2, 2N)  bidirectional ring edges
        normals    (N, 2)   outward unit normals  [nx, ny]
        ds         (N,)     nodal arc-length weight
        pos        (N, 2)   2-D positions
    """
    x_s = np.asarray(x_surf, dtype=np.float64)
    y_s = np.asarray(y_surf, dtype=np.float64)
    N   = len(x_s)

    # Segment vectors (periodic wrap)
    dx      = np.roll(x_s, -1) - x_s
    dy      = np.roll(y_s, -1) - y_s
    seg_len = np.hypot(dx, dy)

    # Forward tangent
    tx_fwd = dx / (seg_len + 1e-12)
    ty_fwd = dy / (seg_len + 1e-12)

    # Node tangent (average of forward and backward edge tangents)
    tx_bwd = np.roll(tx_fwd, 1)
    ty_bwd = np.roll(ty_fwd, 1)
    tx = 0.5 * (tx_fwd + tx_bwd)
    ty = 0.5 * (ty_fwd + ty_bwd)
    tn = np.hypot(tx, ty) + 1e-12
    tx /= tn
    ty /= tn

    # Outward normal
    # NACA airfoil nodes are ordered clockwise → outward normal = (-ty, tx)
    nx = -ty
    ny =  tx

    # Discrete curvature  κ = Δθ/Δs
    theta_fwd = np.arctan2(ty_fwd, tx_fwd)
    d_theta   = np.diff(np.unwrap(theta_fwd))
    d_theta   = np.append(d_theta, 0.0)
    kappa     = d_theta / (seg_len + 1e-12)

    # Normal angle feature
    phi = np.arctan2(ny, nx)

    # Nodal arc-length weight  Δsᵢ = ½(lᵢ₋₁ + lᵢ)
    ds_node = 0.5 * (seg_len + np.roll(seg_len, 1))

    # Node features: [x, y, κ, φ]
    node_feats = np.stack([x_s, y_s, kappa, phi], axis=1).astype(np.float32)

    # Bidirectional ring edges
    idx        = np.arange(N)
    edge_index = np.stack([
        np.concatenate([idx, idx]),
        np.concatenate([(idx + 1) % N, (idx - 1) % N])
    ], axis=0)

    return Data(
        x          = torch.from_numpy(node_feats),
        edge_index = torch.from_numpy(edge_index.astype(np.int64)),
        normals    = torch.from_numpy(np.stack([nx, ny], axis=1).astype(np.float32)),
        ds         = torch.from_numpy(ds_node.astype(np.float32)),
        pos        = torch.from_numpy(np.stack([x_s, y_s], axis=1).astype(np.float32)),
    )


# ── Model blocks ──────────────────────────────────────────────────────────────

class ResGCNBlock(nn.Module):
    """GCNConv + BatchNorm + ReLU with residual projection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.bn   = BatchNorm(out_ch)
        self.proj = nn.Linear(in_ch, out_ch, bias=False) \
                    if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index):
        out = F.relu(self.bn(self.conv(x, edge_index)))
        return out + self.proj(x)


# ── GCNN ──────────────────────────────────────────────────────────────────────

class AirfoilGCNN(nn.Module):
    """
    Geodesic CNN on the airfoil boundary curve.

    Input : PyG Data object (from build_airfoil_graph)
    Output: (N, 3)  per-node  [Cp, Cfx, Cfy]
    """

    def __init__(self, in_channels=4, hidden=128, n_layers=4, out_channels=3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.layers = nn.ModuleList([
            ResGCNBlock(hidden, hidden) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_channels),
        )

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))
        for layer in self.layers:
            x = layer(x, data.edge_index)
        return self.head(x)      # (N, 3)
