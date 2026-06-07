#!/usr/bin/env python3
"""
NACA 4-digit airfoil geometry generator.

Importable module:
    from naca_foil import naca4_coords, naca4_label

CLI:
    python naca_foil.py --m 0.04 --p 0.4 --t 0.15 --plot --save foil.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def naca4_coords(m, p, t, n=100):
    """
    Compute NACA 4-digit airfoil surface coordinates.

    Parameters
    ----------
    m : float   Max camber as fraction of chord (e.g. 0.04)
    p : float   Chordwise position of max camber (e.g. 0.4)
    t : float   Max thickness as fraction of chord (e.g. 0.15)
    n : int     Number of points per surface (cosine-spaced, includes LE and TE)

    Returns
    -------
    xu, yu : ndarray   Upper surface x, y  (LE -> TE)
    xl, yl : ndarray   Lower surface x, y  (LE -> TE)
    """
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015

    beta = np.linspace(0, np.pi, n)
    x    = 0.5 * (1 - np.cos(beta))          # cosine spacing in [0, 1]

    yt = (t / 0.2) * (
        a0 * np.sqrt(np.maximum(x, 0))
        + a1 * x
        + a2 * x**2
        + a3 * x**3
        + a4 * x**4
    )

    if m == 0 or p == 0:
        yc     = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            (m / p**2) * (2*p*x - x**2),
            (m / (1 - p)**2) * (1 - 2*p + 2*p*x - x**2),
        )
        dyc_dx = np.where(
            x < p,
            (2*m / p**2)       * (p - x),
            (2*m / (1 - p)**2) * (p - x),
        )

    theta = np.arctan(dyc_dx)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def naca4_label(m, p, t):
    """Return NACA 4-digit label string, e.g. 'NACA 4415'."""
    return f"NACA {int(round(m * 100)):1d}{int(round(p * 10)):1d}{int(round(t * 100)):02d}"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NACA 4-digit airfoil coordinates")
    parser.add_argument("--m",    type=float, default=0.04,  help="Max camber, fraction of chord (default: 0.04)")
    parser.add_argument("--p",    type=float, default=0.4,   help="Position of max camber, fraction of chord (default: 0.4)")
    parser.add_argument("--t",    type=float, default=0.15,  help="Max thickness, fraction of chord (default: 0.15)")
    parser.add_argument("--n",    type=int,   default=100,   help="Points per surface (default: 100)")
    parser.add_argument("--plot", action="store_true",       help="Display plot interactively")
    parser.add_argument("--save", type=str,   default=None,  help="Save plot to path (e.g. foil.png)")
    args = parser.parse_args()

    xu, yu, xl, yl = naca4_coords(args.m, args.p, args.t, args.n)
    label = naca4_label(args.m, args.p, args.t)
    print(f"Generated {label}  (m={args.m}, p={args.p}, t={args.t}, n={args.n})")

    if args.plot or args.save:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xu, yu, "b-", lw=1.5, label="Upper")
        ax.plot(xl, yl, "r-", lw=1.5, label="Lower")
        ax.fill(
            np.concatenate([xu, xl[::-1]]),
            np.concatenate([yu, yl[::-1]]),
            alpha=0.15, color="steelblue",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.save}")
        if args.plot:
            plt.show()
