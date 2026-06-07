"""
Gradient-based shape optimisation via the differentiable surrogate, plus
run-summary figure compilation.

Subcommands
-----------
optimise   Gradient ascent on L/D w.r.t. (m, p) via backprop through the surrogate.
compile    Compile an optimisation run directory into a 1×2 summary figure
           (streamlines + L/D landscape).

Usage
-----
    python find_optim_shape.py optimise \\
        --checkpoint runs/run_001/best_model.pt \\
        --normalizer  runs/run_001/normalizer.pt \\
        --outdir      runs/run_001/optim/

    python find_optim_shape.py compile runs/run_001 \\
        --optim-subdir optim --landscape-n 120
"""

import argparse
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)
from pathlib import Path
from scipy.interpolate import griddata

from naca_foil import naca4_coords, naca4_label
from surrogate_model import AirfoilSurrogate


# ── Bounded reparametrisation ──────────────────────────────────────────────────
# We optimise unconstrained variables u and map them into the feasible box with a
# smooth tanh. Unlike clamp(), this keeps the gradient alive at the box edges, so
# the optimiser no longer welds itself to m=0.09 and "rides the wall".

def _atanh_init(v, lo, hi):
    """Invert the tanh map so that map_param(u) == v at start."""
    z = 2.0 * (v - lo) / (hi - lo) - 1.0
    z = min(max(z, -0.999), 0.999)          # keep atanh finite
    return 0.5 * math.log((1 + z) / (1 - z))


def map_param(u, lo, hi):
    """Smooth, bounded, differentiable map  R -> (lo, hi)."""
    return lo + (hi - lo) * 0.5 * (1.0 + torch.tanh(u))


# ── Optimisation ──────────────────────────────────────────────────────────────

def optimise(surrogate, m0, p0, lr=1e-2, n_steps=500,
             m_bounds=(0.0, 0.09), p_bounds=(0.2, 0.5)):
    """
    Gradient ascent on L/D w.r.t. (m, p) using a smooth tanh reparametrisation.

    Two correctness fixes vs the original implementation:
      * tanh reparam instead of clamp()  -> gradient survives at the box edges
        (no more "riding the m=0.09 wall"); see map_param above.
      * each history row stores the (m, p) that *produced* its ld -- the original
        logged the post-step (m, p) against the pre-step ld (off-by-one).

    The per-step gradients ∂(L/D)/∂m and ∂(L/D)/∂p are also recorded so the
    convergence diagnostics can inspect for a vanishing / anisotropic gradient.

    Parameters
    ----------
    surrogate : AirfoilSurrogate
    m0, p0    : float   initial NACA parameters
    lr        : float   learning rate
    n_steps   : int     max gradient steps
    m_bounds  : tuple   (min, max) for m
    p_bounds  : tuple   (min, max) for p

    Returns
    -------
    history : list of dicts  {step, m, p, ld, grad_m, grad_p, grad_norm}
    """
    device = next(surrogate.parameters()).device

    u_m = torch.tensor(_atanh_init(m0, *m_bounds), dtype=torch.float,
                       device=device, requires_grad=True)
    u_p = torch.tensor(_atanh_init(p0, *p_bounds), dtype=torch.float,
                       device=device, requires_grad=True)

    optimizer = torch.optim.Adam([u_m, u_p], lr=lr)
    history   = []

    for step in range(n_steps):
        optimizer.zero_grad()

        m = map_param(u_m, *m_bounds)
        p = map_param(u_p, *p_bounds)

        ld   = surrogate(m, p)
        loss = -ld              # minimise negative L/D = maximise L/D
        loss.backward()

        # Convert the gradient back to physical (m, p) space via the chain rule
        # so the diagnostics report ∂(L/D)/∂m and ∂(L/D)/∂p directly.
        with torch.no_grad():
            dm_du = (m_bounds[1] - m_bounds[0]) * 0.5 * (1 - torch.tanh(u_m) ** 2)
            dp_du = (p_bounds[1] - p_bounds[0]) * 0.5 * (1 - torch.tanh(u_p) ** 2)
            g_m = float(-u_m.grad / dm_du.clamp(min=1e-12))   # ∂(L/D)/∂m
            g_p = float(-u_p.grad / dp_du.clamp(min=1e-12))   # ∂(L/D)/∂p

        history.append({
            "step":      step,
            "m":         m.item(),
            "p":         p.item(),
            "ld":        ld.item(),
            "grad_m":    g_m,
            "grad_p":    g_p,
            "grad_norm": math.hypot(g_m, g_p),
        })

        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step:4d}  m={m.item():.4f}  p={p.item():.4f}  "
                  f"L/D={ld.item():.4f}  |grad|={history[-1]['grad_norm']:.3e}")

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(history, outdir):
    """Plot L/D convergence and (m,p) trajectory."""
    steps = [h["step"] for h in history]
    ld    = [h["ld"]   for h in history]
    ms    = [h["m"]    for h in history]
    ps    = [h["p"]    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, ld, "b-", lw=1.5)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("L/D")
    axes[0].set_title("L/D during optimisation")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, ms, "g-", lw=1.5, label="m")
    axes[1].plot(steps, ps, "r-", lw=1.5, label="p")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Parameter value")
    axes[1].set_title("NACA parameter trajectory")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(ms, ps, c=steps, cmap="viridis", s=8)
    axes[2].scatter([ms[0]],  [ps[0]],  c="green", s=80,  zorder=5, label="Start")
    axes[2].scatter([ms[-1]], [ps[-1]], c="red",   s=80,  zorder=5, label="End")
    axes[2].set_xlabel("m"); axes[2].set_ylabel("p")
    axes[2].set_xlim(0.0, 0.09); axes[2].set_ylim(0.2, 0.5)
    axes[2].set_title("(m, p) trajectory")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "optimisation_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_optimal_foil(m_opt, p_opt, outdir):
    """Plot the optimised airfoil shape."""
    label = naca4_label(m_opt, p_opt, 0.15)
    xu, yu, xl, yl = naca4_coords(m_opt, p_opt, 0.15)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xu, yu, "b-", lw=2, label="Upper")
    ax.plot(xl, yl, "r-", lw=2, label="Lower")
    ax.fill(np.concatenate([xu, xl[::-1]]),
            np.concatenate([yu, yl[::-1]]),
            alpha=0.15, color="steelblue")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(f"Optimal aerofoil — {label}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = outdir / "optimal_foil.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def compute_landscape(surrogate, n_grid=120,
                      m_bounds=(0.0, 0.09), p_bounds=(0.2, 0.5)):
    """Brute-force the surrogate over a fine (m, p) grid. Returns ms, ps, LD."""
    print(f"Computing L/D landscape (brute force, {n_grid}x{n_grid}) …")
    device = next(surrogate.parameters()).device

    ms = np.linspace(m_bounds[0], m_bounds[1], n_grid)
    ps = np.linspace(p_bounds[0], p_bounds[1], n_grid)
    LD = np.full((n_grid, n_grid), np.nan)

    surrogate.eval()
    with torch.no_grad():
        for i, m_val in enumerate(ms):
            for j, p_val in enumerate(ps):
                m = torch.tensor(m_val, dtype=torch.float, device=device)
                p = torch.tensor(p_val, dtype=torch.float, device=device)
                try:
                    LD[j, i] = surrogate(m, p).item()
                except Exception:
                    LD[j, i] = float("nan")
            if i % max(1, n_grid // 10) == 0:
                print(f"  column {i+1}/{n_grid}")
    return ms, ps, LD


def grid_argmax(ms, ps, LD):
    """Return (m_star, p_star, ld_star) at the grid maximum."""
    j, i = np.unravel_index(np.nanargmax(LD), LD.shape)
    return float(ms[i]), float(ps[j]), float(LD[j, i])


def plot_landscape(surrogate, outdir, n_grid=120, history=None,
                   ms=None, ps=None, LD=None):
    """
    Filled-contour L/D landscape with the true grid maximum starred and, if a
    trajectory is given, the optimisation path overlaid (start, best, end).
    Pass a precomputed (ms, ps, LD) to avoid recomputing the grid.
    """
    if LD is None:
        ms, ps, LD = compute_landscape(surrogate, n_grid=n_grid)

    m_star, p_star, ld_star = grid_argmax(ms, ps, LD)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cf = ax.contourf(ms, ps, LD, levels=80, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="L/D")

    if history is not None:
        ms_h = np.array([h["m"] for h in history])
        ps_h = np.array([h["p"] for h in history])
        best = max(range(len(history)), key=lambda k: history[k]["ld"])
        ax.plot(ms_h, ps_h, "w.-", ms=2, lw=0.9, alpha=0.8, zorder=3,
                label="path")
        ax.scatter([ms_h[0]],  [ps_h[0]],  c="deepskyblue", s=90, zorder=5,
                   edgecolor="k", label="start")
        ax.scatter([ms_h[best]], [ps_h[best]], c="blue", s=90, zorder=5,
                   edgecolor="white", label="best")
        ax.scatter([ms_h[-1]], [ps_h[-1]], c="navy", marker="X", s=80, zorder=5,
                   edgecolor="white", label="end")

    ax.scatter([m_star], [p_star], marker="*", s=300, c="black",
               edgecolor="white", zorder=6, label=f"grid max ({ld_star:.3f})")
    ax.set_xlabel("m"); ax.set_ylabel("p")
    ax.set_xlim(ms.min(), ms.max()); ax.set_ylim(ps.min(), ps.max())
    ax.set_title("Surrogate L/D landscape" +
                 (" — optimisation path" if history is not None else ""))
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fname = "ld_landscape_path.png" if history is not None else "ld_landscape.png"
    out   = outdir / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    return ms, ps, LD


def plot_landscape_3d(ms, ps, LD, outdir, history=None):
    """
    3D surface of the L/D landscape. If a trajectory is given, the optimisation
    path is drawn riding on the surface (offset slightly above for visibility),
    with start/best/end markers and the grid maximum.
    """
    M, P = np.meshgrid(ms, ps)
    m_star, p_star, ld_star = grid_argmax(ms, ps, LD)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(M, P, LD, cmap="RdYlGn", linewidth=0,
                           antialiased=True, alpha=0.9, rstride=1, cstride=1)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="L/D")

    # contour projected on the floor for reference
    zfloor = np.nanmin(LD) - 0.05 * (np.nanmax(LD) - np.nanmin(LD))
    ax.contourf(M, P, LD, zdir="z", offset=zfloor, levels=40,
                cmap="RdYlGn", alpha=0.6)

    if history is not None:
        ms_h = np.array([h["m"] for h in history])
        ps_h = np.array([h["p"] for h in history])
        ld_h = np.array([h["ld"] for h in history])
        bump = 0.01 * (np.nanmax(LD) - np.nanmin(LD))
        best = int(np.argmax(ld_h))
        ax.plot(ms_h, ps_h, ld_h + bump, "k.-", ms=3, lw=1.2, zorder=10,
                label="path")
        ax.scatter([ms_h[0]], [ps_h[0]], [ld_h[0] + bump], c="deepskyblue",
                   s=70, depthshade=False, zorder=11, label="start")
        ax.scatter([ms_h[best]], [ps_h[best]], [ld_h[best] + bump], c="blue",
                   s=70, depthshade=False, zorder=11, label="best")
        ax.scatter([ms_h[-1]], [ps_h[-1]], [ld_h[-1] + bump], c="navy",
                   marker="X", s=70, depthshade=False, zorder=11, label="end")

    ax.scatter([m_star], [p_star], [ld_star], marker="*", s=260, c="black",
               depthshade=False, zorder=12, label=f"grid max ({ld_star:.3f})")
    ax.set_xlabel("m"); ax.set_ylabel("p"); ax.set_zlabel("L/D")
    ax.set_title("Surrogate L/D landscape (3D)")
    ax.view_init(elev=28, azim=-120)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = outdir / "ld_landscape_3d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_foil_progression(history, outdir, n_foils=60):
    """
    Overlay a subset of NACA foils sampled from the optimisation trajectory.

    Foils are sorted by L/D ascending and drawn in that order so the optimal
    foil renders on top. Colour encodes L/D (red → green); the optimal foil
    gets a heavier line weight. A legend identifies each foil.
    """
    n        = len(history)
    best_idx = max(range(n), key=lambda i: history[i]["ld"])

    # Farthest-point sampling in (m, p) space so we pick maximally diverse shapes
    # rather than evenly-spaced steps (which mostly cluster near convergence).
    mp      = np.array([[h["m"], h["p"]] for h in history])
    mp_span = mp.ptp(0) + 1e-8
    mp_norm = (mp - mp.min(0)) / mp_span   # normalise each axis to [0,1]

    selected = [0]
    for _ in range(n_foils - 1):
        dists = np.min(
            np.linalg.norm(mp_norm[np.array(selected)][:, None] - mp_norm[None], axis=2),
            axis=0,
        )
        dists[selected] = -1
        nxt = int(np.argmax(dists))
        if dists[nxt] < 1e-6:
            break
        selected.append(nxt)

    # Always include the best-L/D step
    if best_idx not in selected:
        selected[-1] = best_idx
    indices = sorted(set(selected))

    # Worst → best so optimal is drawn last (highest z-order)
    samples = sorted([history[i] for i in indices], key=lambda h: h["ld"])

    ld_vals = np.array([s["ld"] for s in samples])
    cmap    = plt.cm.rainbow
    norm_ld = plt.Normalize(ld_vals.min(), ld_vals.max())

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, s in enumerate(samples):
        xu, yu, xl, yl = (np.asarray(a) for a in naca4_coords(s["m"], s["p"], 0.15))
        color = cmap(norm_ld(s["ld"]))
        lw    = 2.2 if i == len(samples) - 1 else 1.2

        ax.plot(xu, yu, color=color, lw=lw, zorder=i + 2)
        ax.plot(xl, yl, color=color, lw=lw, zorder=i + 2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil progression during optimisation")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = outdir / "foil_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Convergence diagnostics ────────────────────────────────────────────────────

def _ld_grad(surrogate, m, p, device):
    """L/D and its gradient (∂/∂m, ∂/∂p) at a scalar point (autograd, no reparam)."""
    mt = torch.tensor(float(m), dtype=torch.float, device=device, requires_grad=True)
    pt = torch.tensor(float(p), dtype=torch.float, device=device, requires_grad=True)
    ld = surrogate(mt, pt)
    ld.backward()
    return ld.item(), float(mt.grad), float(pt.grad)


def analyze_convergence(surrogate, history, ms, ps, LD, outdir,
                        m_bounds=(0.0, 0.09), p_bounds=(0.2, 0.5), n_quiver=16):
    """
    Investigate *why* gradient ascent stalls short of the true max.

    Produces:
      * gradient_field.png — quiver of ∇(L/D) over the landscape (direction +
        magnitude), revealing the anisotropic ridge where ∂(L/D)/∂p ≈ 0.
      * grad_along_path.png — L/D and |∇| vs step for the actual run.
    Prints a plateau analysis and a "good enough" verdict comparing the
    optimiser's best against the brute-force grid maximum.

    Returns a dict summary (also written to convergence_report.json).
    """
    device = next(surrogate.parameters()).device
    print("\n" + "=" * 68)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 68)

    m_star, p_star, ld_star = grid_argmax(ms, ps, LD)

    # ---- 1. gradient field (anisotropy / vanishing-gradient evidence) ---------
    gm = np.linspace(m_bounds[0], m_bounds[1], n_quiver)
    gp = np.linspace(p_bounds[0], p_bounds[1], n_quiver)
    GM, GP = np.meshgrid(gm, gp)
    dM = np.zeros_like(GM); dP = np.zeros_like(GP)
    for a in range(n_quiver):
        for b in range(n_quiver):
            try:
                _, dm_, dp_ = _ld_grad(surrogate, GM[b, a], GP[b, a], device)
            except Exception:
                dm_, dp_ = np.nan, np.nan
            dM[b, a] = dm_; dP[b, a] = dp_

    mag = np.hypot(dM, dP)
    # normalise arrows to direction only (so the quiver shows where flow points)
    eps = 1e-12
    dMn, dPn = dM / (mag + eps), dP / (mag + eps)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(ms, ps, LD, levels=60, cmap="RdYlGn", alpha=0.85)
    fig.colorbar(cf, ax=ax, label="L/D")
    q = ax.quiver(GM, GP, dMn, dPn, mag, cmap="inferno", scale=28,
                  width=0.004, zorder=4)
    fig.colorbar(q, ax=ax, label="|∇ L/D|")
    if history is not None:
        ax.plot([h["m"] for h in history], [h["p"] for h in history],
                "b.-", ms=2, lw=0.9, zorder=5, label="path")
    ax.scatter([m_star], [p_star], marker="*", s=300, c="black",
               edgecolor="white", zorder=6, label="grid max")
    ax.set_xlabel("m"); ax.set_ylabel("p")
    ax.set_title("Gradient field of L/D (arrows = ascent direction)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "gradient_field.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {outdir / 'gradient_field.png'}")

    # anisotropy: how much weaker is the p-gradient than the m-gradient?
    med_gm = float(np.nanmedian(np.abs(dM)))
    med_gp = float(np.nanmedian(np.abs(dP)))
    aniso  = med_gm / (med_gp + eps)
    print(f"  median |∂L/∂m| = {med_gm:.3e}")
    print(f"  median |∂L/∂p| = {med_gp:.3e}")
    print(f"  anisotropy ratio (m:p) = {aniso:.1f}x  "
          f"-> p-gradient is ~{aniso:.0f}x weaker, so p barely migrates")

    # ---- 2. gradient magnitude along the actual run ---------------------------
    summary = {"grid_max": {"m": m_star, "p": p_star, "ld": ld_star},
               "anisotropy_ratio": aniso,
               "median_grad_m": med_gm, "median_grad_p": med_gp}

    if history is not None and "grad_norm" in history[0]:
        steps = [h["step"] for h in history]
        gnorm = [h["grad_norm"] for h in history]
        ldp   = [h["ld"] for h in history]
        fig, ax2 = plt.subplots(1, 2, figsize=(12, 4))
        ax2[0].plot(steps, ldp, "b-"); ax2[0].axhline(ld_star, ls="--", c="k",
                    label=f"grid max {ld_star:.3f}")
        ax2[0].set_xlabel("step"); ax2[0].set_ylabel("L/D")
        ax2[0].set_title("L/D vs step"); ax2[0].legend(); ax2[0].grid(alpha=0.3)
        ax2[1].semilogy(steps, gnorm, "r-")
        ax2[1].set_xlabel("step"); ax2[1].set_ylabel("|∇ L/D| (log)")
        ax2[1].set_title("Gradient magnitude vs step (vanishing?)")
        ax2[1].grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(outdir / "grad_along_path.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {outdir / 'grad_along_path.png'}")
        print(f"  |∇| start={gnorm[0]:.3e}  ->  end={gnorm[-1]:.3e}  "
              f"({gnorm[0] / (gnorm[-1] + eps):.0f}x decay)")
        summary["grad_norm_start"] = gnorm[0]
        summary["grad_norm_end"]   = gnorm[-1]

    # ---- 3. plateau analysis + "good enough" ---------------------------------
    finite = LD[np.isfinite(LD)]
    for frac in (0.999, 0.995, 0.99, 0.95):
        share = float(np.mean(finite >= frac * ld_star))
        print(f"  {frac*100:5.1f}% of max ({frac*ld_star:.4f}): "
              f"{share*100:5.1f}% of (m,p) space qualifies")
        summary[f"area_ge_{frac}"] = share

    if history is not None:
        best = max(history, key=lambda h: h["ld"])
        gap   = ld_star - best["ld"]
        pct   = 100.0 * best["ld"] / ld_star
        lbl_opt   = naca4_label(best["m"], best["p"], 0.15)
        lbl_true  = naca4_label(m_star,   p_star,   0.15)
        print("\n  GOOD-ENOUGH VERDICT")
        print(f"    optimiser best : m={best['m']:.4f} p={best['p']:.4f} "
              f"L/D={best['ld']:.4f}  ({lbl_opt})")
        print(f"    grid true max  : m={m_star:.4f} p={p_star:.4f} "
              f"L/D={ld_star:.4f}  ({lbl_true})")
        print(f"    reaches {pct:.2f}% of the true max  (ΔL/D = {gap:.4f})")
        print(f"    discrete NACA label match: "
              f"{'YES' if lbl_opt == lbl_true else f'no ({lbl_opt} vs {lbl_true})'}")
        verdict = ("GOOD ENOUGH — within surrogate noise / on the top plateau"
                   if pct >= 98.0 else
                   "MARGINAL — within a few % but distinct foil"
                   if pct >= 95.0 else
                   "NOT converged — meaningfully short of the optimum")
        print(f"    -> {verdict}")
        summary["best"] = {k: best[k] for k in ("m", "p", "ld")}
        summary["pct_of_max"] = pct
        summary["delta_ld"]   = gap
        summary["label_opt"]  = lbl_opt
        summary["label_true"] = lbl_true
        summary["verdict"]    = verdict

    with open(outdir / "convergence_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {outdir / 'convergence_report.json'}")
    return summary


# ── compile: SU2 (high-fidelity) helpers ─────────────────────────────────────

def read_su2_cl_cd(history_csv):
    """
    Return the converged (CL, CD, L/D) from an SU2 history.csv (last row).

    SU2 writes quoted, space-padded headers; strip and match case-insensitively.
    L/D is taken from the CEff column when present, else computed as CL/CD.
    """
    import pandas as pd

    df = pd.read_csv(history_csv, skipinitialspace=True)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    def find(name):
        for c in df.columns:
            if c.strip().strip('"').strip().lower() == name.lower():
                return c
        return None

    cl_col, cd_col = find("CL"), find("CD")
    if cl_col is None or cd_col is None:
        raise KeyError(f"CL/CD columns not found in {history_csv}; cols={list(df.columns)}")

    cl = float(df[cl_col].iloc[-1])
    cd = float(df[cd_col].iloc[-1])

    ceff_col = find("CEff")
    ld = float(df[ceff_col].iloc[-1]) if ceff_col is not None else cl / cd
    return cl, cd, ld


def read_su2_surface_fields(highfi_dir):
    """
    Return the SU2 airfoil-surface fields: (x, y, Cp, Cf_mag).

    Surface node coordinates come from surface_flow.csv (MARKER_PLOTTING=airfoil);
    Cp and the skin-friction-coefficient vector are read from the volume .vtk and
    sampled at those surface nodes via nearest-node matching (exact in practice).
    """
    import meshio
    import pandas as pd
    from scipy.spatial import cKDTree

    highfi_dir = Path(highfi_dir)
    vtk_path = next((highfi_dir / c for c in ("flow.vtu", "flow.vtk")
                     if (highfi_dir / c).exists()), None)
    if vtk_path is None:
        raise FileNotFoundError(f"No flow.vtu/flow.vtk in {highfi_dir}")

    mesh = meshio.read(str(vtk_path))
    pts = mesh.points[:, :2]
    cp = _get_field(mesh.point_data, "Pressure_Coefficient", "Pressure")
    sf = mesh.point_data.get("Skin_Friction_Coefficient")
    if cp is None:
        raise KeyError("Pressure_Coefficient not found in vtk point_data")
    cf_mag = (np.hypot(np.asarray(sf)[:, 0], np.asarray(sf)[:, 1])
              if sf is not None else None)

    surf_csv = highfi_dir / "surface_flow.csv"
    if surf_csv.exists():
        df = pd.read_csv(surf_csv, skipinitialspace=True)
        df.columns = [c.strip().strip('"').strip() for c in df.columns]
        sx = df["x"].to_numpy()
        sy = df["y"].to_numpy()
        idx = cKDTree(pts).query(np.column_stack([sx, sy]))[1]
    else:
        idx = np.arange(len(pts))
        sx, sy = pts[:, 0], pts[:, 1]

    cf_surf = cf_mag[idx] if cf_mag is not None else None
    return sx, sy, np.asarray(cp)[idx], cf_surf


def _split_upper_lower(x, y, m, p):
    """Classify surface points as upper/lower via the NACA camber line at x."""
    x = np.clip(x, 0.0, 1.0)
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            (m / p ** 2) * (2 * p * x - x ** 2),
            (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x ** 2),
        )
    return y >= yc


def _surface_mean_abs_diff(xa, ya, fa, xb, yb, fb, m, p, n=200):
    """
    Mean |Δfield| between two surface samplings (a=surrogate, b=SU2).

    Both are split into upper/lower via camber, sorted by x, and interpolated onto
    a common cosine-spaced x grid per surface before differencing. Returns the
    mean (over both surfaces) absolute difference, or NaN if a field is missing.
    """
    if fa is None or fb is None:
        return float("nan")

    beta = np.linspace(0.0, np.pi, n)
    xg = 0.5 * (1.0 - np.cos(beta))   # cosine spacing in [0, 1]

    ua = _split_upper_lower(xa, ya, m, p)
    ub = _split_upper_lower(xb, yb, m, p)

    diffs = []
    for mask_a, mask_b in ((ua, ub), (~ua, ~ub)):
        xa_s, fa_s = xa[mask_a], np.asarray(fa)[mask_a]
        xb_s, fb_s = xb[mask_b], np.asarray(fb)[mask_b]
        if len(xa_s) < 2 or len(xb_s) < 2:
            continue
        oa, ob = np.argsort(xa_s), np.argsort(xb_s)
        fa_i = np.interp(xg, xa_s[oa], fa_s[oa])
        fb_i = np.interp(xg, xb_s[ob], fb_s[ob])
        diffs.append(np.abs(fa_i - fb_i))

    if not diffs:
        return float("nan")
    return float(np.mean(np.concatenate(diffs)))


def _get_field(point_data, *names):
    for name in names:
        if name in point_data:
            return np.asarray(point_data[name]).squeeze()
    return None


def load_flow_field(vtk_path):
    """Read an SU2 .vtk/.vtu flow file → points, gridded U, V, P on a regular grid."""
    import meshio

    mesh = meshio.read(str(vtk_path))
    pts = mesh.points[:, :2]
    pd_ = mesh.point_data

    pressure = _get_field(pd_, "Pressure", "Pressure_Coefficient")
    if "Velocity" in pd_:
        vel = np.asarray(pd_["Velocity"])
        vel_x, vel_y = vel[:, 0], vel[:, 1]
    else:
        vel_x = _get_field(pd_, "Velocity_x", "x-Velocity")
        vel_y = _get_field(pd_, "Velocity_y", "y-Velocity")

    xi = np.linspace(-0.5, 2.0, 300)
    yi = np.linspace(-0.8, 0.8, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    U = griddata(pts, vel_x, (Xi, Yi), method="linear") if vel_x is not None else None
    V = griddata(pts, vel_y, (Xi, Yi), method="linear") if vel_y is not None else None
    P = griddata(pts, pressure, (Xi, Yi), method="linear") if pressure is not None else None
    return xi, yi, U, V, P


# ── compile: surrogate-predicted fields ───────────────────────────────────────

def predict_surface(surrogate, m, p):
    """
    Run the surrogate pipeline at (m, p) and return the predicted surface fields
    and integrated forces.

    Returns a dict with keys:
        x, y      surface node coordinates (closed loop, numpy)
        Cp        per-node pressure coefficient
        Cf_mag    per-node skin-friction-coefficient magnitude sqrt(Cfx^2+Cfy^2)
        CL, CD    integrated force coefficients
        LD        L/D ratio (CL / |CD|)
    """
    from gcnn import build_airfoil_graph
    from surrogate_model import naca4_closed_torch, integrate_forces

    device = next(surrogate.parameters()).device
    m_t = torch.as_tensor(float(m), dtype=torch.float, device=device)
    p_t = torch.as_tensor(float(p), dtype=torch.float, device=device)

    surrogate.eval()
    with torch.no_grad():
        x_c, y_c = naca4_closed_torch(m_t, p_t, surrogate.t, surrogate.n_surf)
        graph = build_airfoil_graph(x_c.cpu().numpy(), y_c.cpu().numpy()).to(device)
        graph.x = torch.cat(
            [torch.stack([x_c, y_c], dim=1), graph.x[:, 2:]], dim=1
        )
        pred_norm = surrogate.gcnn(graph)
        pred = surrogate.normalizer.denormalize(
            pred_norm.to(surrogate.normalizer.mean.device)
        )
        Cp, Cfx, Cfy = pred[:, 0], pred[:, 1], pred[:, 2]
        CL, CD = integrate_forces(Cp, Cfx, Cfy, x_c, y_c)
        ld = float(CL) / max(abs(float(CD)), 1e-8)

    return {
        "x": x_c.cpu().numpy(),
        "y": y_c.cpu().numpy(),
        "Cp": Cp.cpu().numpy(),
        "Cf_mag": np.hypot(Cfx.cpu().numpy(), Cfy.cpu().numpy()),
        "CL": float(CL),
        "CD": float(CD),
        "LD": ld,
    }


# ── compile: panel drawing ─────────────────────────────────────────────────────

def draw_streamlines(ax, xi, yi, U, V, P, label, m_opt, p_opt, t, diffs=None):
    """
    Streamlines + pressure contour panel for the optimal foil, with an optional
    surrogate-vs-SU2 comparison box (keys: 'cp', 'cf', 'ld').
    """
    Xi, Yi = np.meshgrid(xi, yi)
    if P is not None:
        ax.contourf(Xi, Yi, P, levels=50, cmap="RdBu_r", alpha=0.8)
    if U is not None and V is not None:
        ax.streamplot(xi, yi, U, V, density=1.5, color="k",
                      linewidth=0.5, arrowsize=0.8)

    xu, yu, xl, yl = naca4_coords(m_opt, p_opt, t)
    x_loop = np.concatenate([xu, xl[::-1]])
    y_loop = np.concatenate([yu, yl[::-1]])
    ax.fill(x_loop, y_loop, color="0.15", zorder=4)
    ax.plot(x_loop, y_loop, color="k", lw=1.2, zorder=5)

    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"Optimal foil discovered — {label}", fontsize=12, fontweight="bold")

    if diffs is not None:
        def fmt(v):
            return "n/a" if v is None or np.isnan(v) else f"{v:.4f}"
        txt = (
            "Surrogate vs SU2 (abs. difference)\n"
            f"mean |ΔCp| = {fmt(diffs.get('cp'))}\n"
            f"mean |ΔCf| = {fmt(diffs.get('cf'))}\n"
            f"|Δ L/D|    = {fmt(diffs.get('ld'))}"
        )
        ax.text(
            0.015, 0.025, txt, transform=ax.transAxes, fontsize=8.5,
            family="monospace", va="bottom", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.4", alpha=0.9),
        )


def draw_landscape_panel(ax, fig, surrogate, n_grid=120,
                         m_bounds=(0.0, 0.09), p_bounds=(0.2, 0.5),
                         save_dir=None):
    """
    Surrogate L/D landscape panel (filled contour, grid maximum starred).
    If `save_dir` is given, writes ld_landscape_data.npz and .csv for reuse.
    Returns (ms, ps, LD).
    """
    ms, ps, LD = compute_landscape(
        surrogate, n_grid=n_grid, m_bounds=m_bounds, p_bounds=p_bounds
    )
    m_star, p_star, ld_star = grid_argmax(ms, ps, LD)

    cf = ax.contourf(ms, ps, LD, levels=80, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="L/D")
    ax.scatter([m_star], [p_star], marker="*", s=300, c="black",
               edgecolor="white", zorder=6, label=f"grid max ({ld_star:.3f})")
    ax.set_xlabel("m")
    ax.set_ylabel("p")
    ax.set_xlim(ms.min(), ms.max())
    ax.set_ylim(ps.min(), ps.max())
    ax.set_title("Surrogate L/D landscape", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    if save_dir is not None:
        save_dir = Path(save_dir)
        npz = save_dir / "ld_landscape_data.npz"
        np.savez(npz, ms=ms, ps=ps, LD=LD,
                 m_star=m_star, p_star=p_star, ld_star=ld_star)
        M, Pm = np.meshgrid(ms, ps)
        np.savetxt(
            save_dir / "ld_landscape_data.csv",
            np.column_stack([M.ravel(), Pm.ravel(), LD.ravel()]),
            delimiter=",", header="m,p,ld", comments="",
        )
        print(f"Saved → {npz} (+ .csv)")

    return ms, ps, LD


# ── compile: top-level ────────────────────────────────────────────────────────

def compile_figure(run_dir, optim_subdir="optim", landscape_n=120, outfile=None):
    """
    Compile a 1×2 summary figure for a completed optimisation run:
      col 1 — streamlines + pressure field around the optimal foil
      col 2 — surrogate L/D landscape (grid max starred)
    """
    run_dir   = Path(run_dir)
    optim_dir = run_dir / optim_subdir
    highfi_dir = optim_dir / "highfi"

    with open(optim_dir / "result.json") as f:
        result = json.load(f)
    m_opt = result["m_opt"]
    p_opt = result["p_opt"]
    t     = 0.15
    label = result.get("label") or naca4_label(m_opt, p_opt, t)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = AirfoilSurrogate.load(
        run_dir / "best_model.pt", run_dir / "normalizer.pt", device=device
    )
    surrogate.eval()

    pred  = predict_surface(surrogate, m_opt, p_opt)
    diffs = {"cp": float("nan"), "cf": float("nan"), "ld": float("nan")}

    try:
        sx, sy, su2_cp, su2_cf = read_su2_surface_fields(highfi_dir)
        diffs["cp"] = _surface_mean_abs_diff(
            pred["x"], pred["y"], pred["Cp"], sx, sy, su2_cp, m_opt, p_opt)
        diffs["cf"] = _surface_mean_abs_diff(
            pred["x"], pred["y"], pred["Cf_mag"], sx, sy, su2_cf, m_opt, p_opt)
    except Exception as e:
        print(f"  ! Could not compute surface-field diffs: {e}")

    hist_csv = highfi_dir / "history.csv"
    if hist_csv.exists():
        _, _, su2_ld = read_su2_cl_cd(hist_csv)
        diffs["ld"] = abs(pred["LD"] - su2_ld)
    else:
        print(f"  ! {hist_csv} not found — skipping L/D comparison")
        su2_ld = float("nan")

    vtk_path = None
    for cand in ("flow.vtu", "flow.vtk"):
        if (highfi_dir / cand).exists():
            vtk_path = highfi_dir / cand
            break
    if vtk_path is None:
        raise FileNotFoundError(f"No flow.vtu/flow.vtk in {highfi_dir}")
    xi, yi, U, V, P = load_flow_field(vtk_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    draw_streamlines(ax1, xi, yi, U, V, P, label, m_opt, p_opt, t, diffs=diffs)
    draw_landscape_panel(ax2, fig, surrogate, n_grid=landscape_n, save_dir=optim_dir)

    fig.suptitle(
        f"Optimisation summary — {run_dir.name}   "
        f"(L/D = {result.get('ld_opt', float('nan')):.4f})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if outfile is None:
        outfile = optim_dir / "run_summary.png"
    outfile = Path(outfile)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outfile}")
    print(f"  L/D  pred/SU2 = {pred['LD']:.4f} / {su2_ld:.4f}  |Δ|={diffs['ld']:.4f}")
    print(f"  mean |ΔCp| = {diffs['cp']:.4f}   mean |ΔCf| = {diffs['cf']:.4f}")
    return outfile


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NACA shape optimisation and run-summary compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python find_optim_shape.py optimise \\\n"
            "      --checkpoint runs/run_001/best_model.pt \\\n"
            "      --normalizer  runs/run_001/normalizer.pt \\\n"
            "      --outdir      runs/run_001/optim/\n\n"
            "  python find_optim_shape.py compile runs/run_001\n"
            "      --optim-subdir optim --landscape-n 120\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── optimise subcommand ───────────────────────────────────────────────────
    opt_p = sub.add_parser("optimise", help="Gradient-ascent optimisation of L/D")
    opt_p.add_argument("--checkpoint",  type=str, required=True,
                       help="Path to best_model.pt")
    opt_p.add_argument("--normalizer",  type=str, required=True,
                       help="Path to normalizer.pt")
    opt_p.add_argument("--outdir",      type=str, default="runs/optim",
                       help="Output directory (default: runs/optim)")
    opt_p.add_argument("--m0",          type=float, default=0.04,
                       help="Initial m (default: 0.04)")
    opt_p.add_argument("--p0",          type=float, default=0.35,
                       help="Initial p (default: 0.35)")
    opt_p.add_argument("--lr",          type=float, default=1e-2,
                       help="Learning rate (default: 0.01)")
    opt_p.add_argument("--n-steps",     type=int,   default=500,
                       help="Gradient steps (default: 500)")
    opt_p.add_argument("--landscape",   action="store_true",
                       help="Also compute L/D landscape plots (2D + 3D)")
    opt_p.add_argument("--landscape-n", type=int,   default=120,
                       help="Grid size for landscape (default: 120)")
    opt_p.add_argument("--analyze",     action="store_true",
                       help="Run convergence / gradient-field diagnostics")

    # ── compile subcommand ────────────────────────────────────────────────────
    cmp_p = sub.add_parser("compile",
                           help="Compile an optimisation run into a summary figure")
    cmp_p.add_argument("run_dir",        type=str,
                       help="Path to runs/run_xxx/")
    cmp_p.add_argument("--optim-subdir", type=str, default="optim",
                       help="Sub-directory holding optimisation outputs (default: optim)")
    cmp_p.add_argument("--landscape-n",  type=int, default=120,
                       help="Grid size for the L/D landscape (default: 120)")
    cmp_p.add_argument("--outfile",      type=str, default=None,
                       help="Output PNG path "
                            "(default: <run_dir>/<optim-subdir>/run_summary.png)")

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.cmd == "optimise":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        surrogate = AirfoilSurrogate.load(
            args.checkpoint, args.normalizer, device=device
        )
        surrogate.eval()

        print(f"\nOptimising from m={args.m0}, p={args.p0} …\n")
        history = optimise(
            surrogate, args.m0, args.p0,
            lr=args.lr, n_steps=args.n_steps
        )

        best   = max(history, key=lambda h: h["ld"])
        m_opt  = best["m"]
        p_opt  = best["p"]
        ld_opt = best["ld"]

        print(f"\nOptimal:  m={m_opt:.4f}  p={p_opt:.4f}  L/D={ld_opt:.4f}")
        print(f"Label:    {naca4_label(m_opt, p_opt, 0.15)}")

        result = {"m_opt": m_opt, "p_opt": p_opt, "ld_opt": ld_opt,
                  "label": naca4_label(m_opt, p_opt, 0.15), "history": history}
        with open(outdir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        plot_trajectory(history, outdir)
        plot_optimal_foil(m_opt, p_opt, outdir)
        plot_foil_progression(history, outdir)

        if args.landscape or args.analyze:
            ms, ps, LD = compute_landscape(surrogate, n_grid=args.landscape_n)
            plot_landscape(surrogate, outdir, history=history, ms=ms, ps=ps, LD=LD)
            plot_landscape_3d(ms, ps, LD, outdir, history=history)
            if args.analyze:
                analyze_convergence(surrogate, history, ms, ps, LD, outdir)

        print(f"\nDone. Results in {outdir}/")

    elif args.cmd == "compile":
        compile_figure(
            args.run_dir,
            optim_subdir=args.optim_subdir,
            landscape_n=args.landscape_n,
            outfile=args.outfile,
        )


if __name__ == "__main__":
    main()
