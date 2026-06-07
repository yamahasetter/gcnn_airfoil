"""
Differentiable surrogate model: NACA params → L/D.

This module wraps AirfoilGCNN with:
  1. Differentiable NACA 4-digit equations (PyTorch) so ∂(L/D)/∂(m,p) exists
  2. Graph construction from NACA surface coordinates
  3. Discrete surface integration of Cp/Cf → CL, CD

The full pipeline is differentiable end-to-end, enabling gradient-based
shape optimisation via backpropagation through the surrogate.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path

from gcnn import AirfoilGCNN, sort_surface_nodes, build_airfoil_graph
from train import FieldNormalizer

"""
══════════════════════════════════════════════════════════════════════════════
WORKFLOW OVERVIEW
══════════════════════════════════════════════════════════════════════════════

This module provides a fully differentiable surrogate from NACA parameters
(m, p) to L/D ratio, enabling gradient-based shape optimisation without
calling SU2. Gradients flow end-to-end via PyTorch autograd.

STEP 1 — Differentiable geometry (naca4_torch, naca4_closed_torch)
  The NACA 4-digit equations are re-implemented in PyTorch so that ∂(x,y)/∂(m,p)
  exists. Cosine-spaced x coordinates are mapped to thickness yt and camber yc
  using the standard polynomial formulas. Surface coordinates are computed as
  offsets from the camber line along the local normal (via atan of the camber
  slope). naca4_closed_torch returns the closed loop ordered upper LE→TE then
  lower interior TE→LE, matching the graph convention in gcnn.py.
  NOTE: thickness t is not a gradient target (fixed); only m and p are.

STEP 2 — Graph construction (build_airfoil_graph — imported from gcnn.py)
  The closed surface coordinates are detached to numpy for the numpy-based
  geometry computations (curvature, normal angle, edge construction). The
  resulting graph's x,y node features are then replaced with the differentiable
  PyTorch tensors so that the GCNN output remains connected to (m,p) in the
  autograd graph. κ and φ features are kept from the numpy path (non-differentiable,
  but acceptable — they are secondary features).

STEP 3 — GCNN inference (AirfoilGCNN)
  The graph is passed through the trained GCNN to obtain per-node normalised
  predictions of shape (N, 3): [Cp_norm, Cfx_norm, Cfy_norm].

STEP 4 — De-normalisation (FieldNormalizer.denormalize)
  Field predictions are mapped back to physical units using the training-set
  mean and std stored in the FieldNormalizer. This is necessary because Cp is
  O(1) while Cfx/Cfy are O(1e-3); normalisation prevents loss domination.

STEP 5 — Force integration (integrate_forces)
  Discrete surface integration over the closed boundary loop:
    - Panel vectors (Δx, Δy) → panel lengths ds and outward normals (nx, ny)
      using the CW loop convention (outward = (-dy, dx)/ds).
    - Panel midpoint field values are averaged from adjacent node values.
    - CL = Σ (-Cp_mid·ny + Cfy_mid)·ds  (lift, y-direction)
    - CD = Σ (-Cp_mid·nx + Cfx_mid)·ds  (drag, x-direction)
  All tensor ops are differentiable → ∂(L/D)/∂(m,p) via autograd.

LOADING — AirfoilSurrogate.load()
  Convenience classmethod: loads GCNN weights from a checkpoint file and
  FieldNormalizer from a separate .pt file, reconstructs the surrogate, and
  moves everything to the requested device.

USAGE (optimisation loop, external):
  m = torch.tensor(0.04, requires_grad=True)
  p = torch.tensor(0.40, requires_grad=True)
  surrogate = AirfoilSurrogate.load(ckpt, norm)
  ld = surrogate(m, p)
  (-ld).backward()           # maximise L/D
  # m.grad, p.grad now available for gradient descent

══════════════════════════════════════════════════════════════════════════════
"""

# ── Differentiable NACA equations ─────────────────────────────────────────────

def naca4_torch(m, p, t=0.15, n=100):
    """
    NACA 4-digit surface coordinates implemented in PyTorch.

    Gradients flow through m and p → enables backprop to NACA params.

    Parameters
    ----------
    m : scalar Tensor   max camber  (e.g. 0.04)
    p : scalar Tensor   position of max camber  (e.g. 0.4)
    t : float           max thickness (fixed, not optimised)
    n : int             points per surface

    Returns
    -------
    xu, yu, xl, yl : Tensor (n,)   upper and lower surface coordinates
    """
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015

    beta = torch.linspace(0.0, torch.pi, n, dtype=m.dtype, device=m.device)
    x    = 0.5 * (1.0 - torch.cos(beta))

    yt = (t / 0.2) * (
        a0 * torch.sqrt(x.clamp(min=1e-8))
        + a1 * x
        + a2 * x ** 2
        + a3 * x ** 3
        + a4 * x ** 4
    )

    # Camber line and slope (handle symmetric case m==0)
    yc = torch.where(
        x < p,
        (m / p.pow(2).clamp(min=1e-8)) * (2 * p * x - x ** 2),
        (m / (1 - p).pow(2).clamp(min=1e-8)) * (1 - 2 * p + 2 * p * x - x ** 2),
    )
    dyc_dx = torch.where(
        x < p,
        (2 * m / p.pow(2).clamp(min=1e-8))       * (p - x),
        (2 * m / (1 - p).pow(2).clamp(min=1e-8)) * (p - x),
    )

    theta = torch.atan(dyc_dx)

    xu = x  - yt * torch.sin(theta)
    yu = yc + yt * torch.cos(theta)
    xl = x  + yt * torch.sin(theta)
    yl = yc - yt * torch.cos(theta)

    return xu, yu, xl, yl


def naca4_closed_torch(m, p, t=0.15, n=100):
    """
    Return closed surface coordinates ordered for graph construction:
    upper LE→TE then lower interior TE→LE.

    Returns
    -------
    x_closed, y_closed : Tensor (2n-2,)
    """
    xu, yu, xl, yl = naca4_torch(m, p, t, n)
    x_closed = torch.cat([xu, xl.flip(0)[1:-1]])
    y_closed = torch.cat([yu, yl.flip(0)[1:-1]])
    return x_closed, y_closed


# ── Differentiable L/D integration ───────────────────────────────────────────

def integrate_forces(Cp, Cfx, Cfy, x, y):
    """
    Discrete surface integration of Cp and Cf → CL, CD.

    For AoA = 0:
      Lift (y-direction): CL = Σᵢ (-Cp_mid · n̂y + Cfy_mid) · Δsᵢ
      Drag (x-direction): CD = Σᵢ (-Cp_mid · n̂x + Cfx_mid) · Δsᵢ

    All inputs are Tensors so gradients flow through to (m, p).

    Parameters
    ----------
    Cp, Cfx, Cfy : Tensor (N,)   per-node surface fields
    x, y         : Tensor (N,)   surface coordinates (closed loop)

    Returns
    -------
    CL, CD : scalar Tensors
    """
    # Panel vectors
    dx = torch.roll(x, -1) - x     # Δxᵢ = x_{i+1} - xᵢ
    dy = torch.roll(y, -1) - y
    ds = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-12)

    # Orientation guard: (-dy, dx)/ds is the OUTWARD normal only for a CW loop.
    # sort_surface_nodes() does not guarantee CW ordering, so detect the loop
    # orientation from the signed (shoelace) area and flip if needed. CW → A<0.
    # The sign is detached so it acts as a discrete orientation correction and
    # does not perturb gradients to (m, p).
    signed_area = 0.5 * (x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()
    orient = -torch.sign(signed_area).detach()      # +1 if CW, -1 if CCW

    # Outward unit normals
    nx = orient * (-dy / ds)
    ny = orient * ( dx / ds)

    # Panel midpoint field values
    Cp_m  = 0.5 * (Cp  + torch.roll(Cp,  -1))
    Cfx_m = 0.5 * (Cfx + torch.roll(Cfx, -1))
    Cfy_m = 0.5 * (Cfy + torch.roll(Cfy, -1))

    CL = ((-Cp_m * ny + Cfy_m) * ds).sum()
    CD = ((-Cp_m * nx + Cfx_m) * ds).sum()

    return CL, CD


# ── Surrogate model ───────────────────────────────────────────────────────────

class AirfoilSurrogate(nn.Module):
    """
    End-to-end differentiable surrogate:  (m, p) → L/D

    Pipeline
    --------
    1. naca4_torch(m, p)   → surface coordinates  (differentiable)
    2. build_airfoil_graph → PyG graph             (node features from coords)
    3. AirfoilGCNN         → per-node Cp, Cfx, Cfy
    4. integrate_forces    → CL, CD → L/D          (differentiable)

    The normaliser is applied internally; predictions are de-normalised
    before the force integration so physical units are preserved.
    """

    def __init__(self, gcnn: AirfoilGCNN, normalizer: FieldNormalizer, t=0.15, n_surf=100):
        super().__init__()
        self.gcnn       = gcnn
        self.normalizer = normalizer
        self.t          = t
        self.n_surf     = n_surf

    def forward(self, m, p):
        """
        Parameters
        ----------
        m, p : scalar Tensors with requires_grad=True

        Returns
        -------
        ld : scalar Tensor   L/D ratio
        """
        device = m.device

        # 1. Surface coordinates (differentiable)
        x_c, y_c = naca4_closed_torch(m, p, self.t, self.n_surf)

        # 2. Graph (node features depend on x_c, y_c but edge structure is fixed)
        #    We detach for numpy-based graph construction, then re-attach coords
        #    as node features so gradients flow through x and y.
        x_np = x_c.detach().cpu().numpy()
        y_np = y_c.detach().cpu().numpy()
        graph = build_airfoil_graph(x_np, y_np).to(device)

        # Replace x[:, :2] with differentiable coords so ∂output/∂(m,p) exists
        graph.x = torch.cat([
            torch.stack([x_c, y_c], dim=1),   # differentiable x, y
            graph.x[:, 2:],                    # κ, φ (non-differentiable, ok)
        ], dim=1)

        # 3. GCNN → normalised predictions
        pred_norm = self.gcnn(graph)           # (N, 3)  normalised

        # 4. De-normalise
        pred = self.normalizer.denormalize(pred_norm.to(self.normalizer.mean.device))
        Cp, Cfx, Cfy = pred[:, 0], pred[:, 1], pred[:, 2]

        # 5. Integrate → L/D
        CL, CD = integrate_forces(Cp, Cfx, Cfy, x_c, y_c)
        return CL / (CD.abs().clamp(min=1e-8))

    @classmethod
    def load(cls, checkpoint_path, normalizer_path, device="cpu", **gcnn_kwargs):
        """Load a trained surrogate from checkpoint and normaliser files."""
        ckpt       = torch.load(checkpoint_path, map_location=device)
        normalizer = FieldNormalizer.load(normalizer_path)
        normalizer.mean = normalizer.mean.to(device)
        normalizer.std  = normalizer.std.to(device)

        args   = ckpt.get("args", {})
        hidden   = gcnn_kwargs.get("hidden",   args.get("hidden",   128))
        n_layers = gcnn_kwargs.get("n_layers", args.get("n_layers", 4))

        gcnn = AirfoilGCNN(hidden=hidden, n_layers=n_layers).to(device)
        gcnn.load_state_dict(ckpt["model_state"])
        gcnn.eval()

        return cls(gcnn, normalizer).to(device)
