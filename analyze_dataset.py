#!/usr/bin/env python3
"""
Analyze NACA airfoil dataset.

For each sample reads params.json, history.csv, surface.npz, and full_field.npz,
computes aerodynamic metrics, saves them to <datadir>/analysis/, and generates
landscape plots over the (p, m) parameter space.

Usage:
    python analyze_dataset.py --datadir data/
    python analyze_dataset.py --datadir data/ --output-dir runs/exp1/analysis
    python analyze_dataset.py --datadir data/ --no-plots
"""

import argparse
import json
import warnings
import numpy as np

# numpy 2.0 renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", np.trapz)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata


# ── Surface split ─────────────────────────────────────────────────────────────
# generate_dataset.py stores 100 upper (LE→TE) + 98 lower (TE→LE) points.
N_UPPER = 100


# ── Per-sample loader ─────────────────────────────────────────────────────────

def load_sample(sample_dir: Path) -> dict | None:
    """
    Load one sample and return a flat dict of metrics, or None if incomplete.

    Metrics computed
    ----------------
    From history.csv (last converged row):
      CL, CD, LD           — lift, drag, lift-to-drag ratio
      CMz                  — pitching moment about z
      CEff                 — solver's own CL/CD (cross-check)
      rms_rho_final        — density residual at last iter (convergence quality)

    From surface.npz (Cp/Cf on NACA-equation nodes):
      Cp_min               — suction peak value (min Cp on upper surface)
      x_suction_peak       — chord-fraction location of suction peak
      CL_surf              — ∫ΔCp dx = ∫(Cp_lower – Cp_upper) dx  (independent CL estimate)
      x_separation         — first x/c where Cfx_upper < 0 (flow reversal / separation onset)
      separated            — bool: any upper-surface flow reversal
      Cf_mean_upper        — mean |Cfx| on upper surface (viscous drag indicator)
      Cp_recovery          — Cp_TE_upper – Cp_min  (strength of adverse pressure gradient)

    From full_field.npz:
      Mach_max_field       — peak Mach number in the domain (shock / compressibility indicator)
    """
    params_path  = sample_dir / "params.json"
    history_path = sample_dir / "history.csv"
    surface_path = sample_dir / "surface.npz"
    field_path   = sample_dir / "full_field.npz"

    if not params_path.exists() or not history_path.exists():
        return None

    # ── params ────────────────────────────────────────────────────────────────
    with open(params_path) as f:
        params = json.load(f)

    # ── history (final converged row) ─────────────────────────────────────────
    try:
        hist = pd.read_csv(history_path)
        hist.columns = hist.columns.str.strip().str.strip('"')
        row = hist.iloc[-1]
        CL  = float(row["CL"])
        CD  = float(row["CD"])
        CMz = float(row["CMz"]) if "CMz" in row.index else np.nan
        CEff = float(row["CEff"]) if "CEff" in row.index else np.nan
        rms_rho = float(row["rms[Rho]"]) if "rms[Rho]" in row.index else np.nan
    except Exception as e:
        warnings.warn(f"{sample_dir.name}: history parse failed — {e}")
        return None

    rec: dict = {
        "idx":           int(params["idx"]),
        "label":         params["label"],
        "m":             float(params["m"]),
        "p":             float(params["p"]),
        "t":             float(params.get("t", 0.15)),
        "re":            float(params.get("re", np.nan)),
        "mach":          float(params.get("mach", np.nan)),
        "aoa":           float(params.get("aoa", np.nan)),
        # ── aerodynamic coefficients ──────────────────────────────────────────
        "CL":            CL,
        "CD":            CD,
        "LD":            CL / CD if abs(CD) > 1e-12 else np.nan,
        "CMz":           CMz,
        "CEff":          CEff,
        "rms_rho_final": rms_rho,
    }

    # ── surface metrics ───────────────────────────────────────────────────────
    if surface_path.exists():
        try:
            s = np.load(surface_path)
            x, Cp, Cfx = s["x"], s["Cp"], s["Cfx"]

            # Upper: indices 0:N_UPPER  (LE→TE, x ~ 0→1)
            # Lower: indices N_UPPER:   (TE→LE, x ~ 1→0) — reverse for ascending x
            x_up   = x[:N_UPPER];    Cp_up  = Cp[:N_UPPER];  Cfx_up = Cfx[:N_UPPER]
            x_lo_r = x[N_UPPER:][::-1]                        # ascending x (LE→TE)
            Cp_lo_r = Cp[N_UPPER:][::-1]

            # Suction peak
            i_min            = int(np.argmin(Cp_up))
            rec["Cp_min"]          = float(Cp_up[i_min])
            rec["x_suction_peak"]  = float(x_up[i_min])

            # Independent CL from surface pressure
            # Interpolate lower surface onto upper-surface x grid, then integrate
            Cp_lo_at_up = np.interp(x_up, x_lo_r, Cp_lo_r,
                                    left=Cp_lo_r[0], right=Cp_lo_r[-1])
            delta_Cp    = Cp_lo_at_up - Cp_up          # positive → lift
            rec["CL_surf"] = float(_trapz(delta_Cp, x_up))

            # Separation onset (upper surface Cfx goes negative)
            sep_mask = Cfx_up < 0
            if sep_mask.any():
                rec["x_separation"] = float(x_up[sep_mask][0])
                rec["separated"]    = True
            else:
                rec["x_separation"] = np.nan
                rec["separated"]    = False

            # Mean upper-surface skin friction
            rec["Cf_mean_upper"] = float(np.mean(np.abs(Cfx_up)))

            # Adverse pressure gradient strength: Cp at TE minus suction peak
            rec["Cp_recovery"] = float(Cp_up[-1]) - rec["Cp_min"]

        except Exception as e:
            warnings.warn(f"{sample_dir.name}: surface metrics failed — {e}")

    # ── flow-field metrics ────────────────────────────────────────────────────
    if field_path.exists():
        try:
            ff = np.load(field_path)
            if "mach" in ff:
                rec["Mach_max_field"] = float(np.nanmax(ff["mach"]))
        except Exception as e:
            warnings.warn(f"{sample_dir.name}: field metrics failed — {e}")

    return rec


# ── Plotting ──────────────────────────────────────────────────────────────────

def _interp_landscape(p, m, z, n=300):
    """Return (Mi, Pi, Zi) grids for contour plotting with x=m, y=p."""
    mi = np.linspace(m.min(), m.max(), n)
    pi = np.linspace(p.min(), p.max(), n)
    Mi, Pi = np.meshgrid(mi, pi)
    Zi = griddata((m, p), z, (Mi, Pi), method="cubic")
    Zi_lin = griddata((m, p), z, (Mi, Pi), method="linear")
    Zi = np.where(np.isnan(Zi), Zi_lin, Zi)
    return Mi, Pi, Zi


def plot_landscape(df: pd.DataFrame, col: str, title: str, outpath: Path,
                   cmap: str = "viridis", best: str = "max") -> None:
    """Scatter + cubic-interpolated contour of `col`.
    x-axis = m (max-camber fraction), y-axis = p (max-camber position).
    """
    valid = df[["p", "m", col]].dropna()
    if len(valid) < 4:
        warnings.warn(f"Too few valid samples for {col} landscape — skipping.")
        return

    p, m, z = valid["p"].values, valid["m"].values, valid[col].values
    Mi, Pi, Zi = _interp_landscape(p, m, z)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(Mi, Pi, Zi, levels=50, cmap=cmap, alpha=0.85)
    fig.colorbar(cf, ax=ax, label=col)
    ax.scatter(m, p, c=z, cmap=cmap, edgecolors="k", linewidths=0.4,
               s=40, zorder=5, vmin=z.min(), vmax=z.max())

    ax.set_xlabel("m  (max-camber fraction)",      fontsize=11)
    ax.set_ylabel("p  (max-camber position)", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_landscape_3d(df: pd.DataFrame, col: str, title: str, outpath: Path,
                      cmap: str = "RdYlGn") -> None:
    """
    3-D surface of `col` over (m, p) with a contourf projected onto the floor,
    matching the style of find_optim_shape.plot_landscape_3d.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)

    valid = df[["p", "m", col]].dropna()
    if len(valid) < 4:
        warnings.warn(f"Too few valid samples for 3-D {col} landscape — skipping.")
        return

    p, m, z = valid["p"].values, valid["m"].values, valid[col].values
    Mi, Pi, Zi = _interp_landscape(p, m, z, n=100)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Mi, Pi, Zi, cmap=cmap, linewidth=0,
                           antialiased=True, alpha=0.9, rstride=1, cstride=1)
    fig.colorbar(surf, ax=ax, shrink=0.6, label=col)

    # contourf projected onto the floor
    zfloor = float(np.nanmin(Zi)) - 0.05 * (float(np.nanmax(Zi)) - float(np.nanmin(Zi)))
    ax.contourf(Mi, Pi, Zi, zdir="z", offset=zfloor, levels=40,
                cmap=cmap, alpha=0.6)

    # actual CFD sample points
    ax.scatter(m, p, z, c=z, cmap=cmap, edgecolors="k",
               linewidths=0.3, s=30, zorder=5,
               vmin=z.min(), vmax=z.max())

    ax.set_xlabel("m  (max-camber fraction)",  fontsize=10, labelpad=8)
    ax.set_ylabel("p  (max-camber position)",  fontsize=10, labelpad=8)
    ax.set_zlabel(col, fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=28, azim=-120)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cl_cd_polar(df: pd.DataFrame, outpath: Path) -> None:
    """CL–CD polar colored by L/D."""
    valid = df[["CL", "CD", "LD"]].dropna()
    if len(valid) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(valid["CD"], valid["CL"], c=valid["LD"],
                    cmap="plasma", edgecolors="k", linewidths=0.3, s=45)
    plt.colorbar(sc, ax=ax, label="L/D")
    ax.set_xlabel("CD", fontsize=11)
    ax.set_ylabel("CL", fontsize=11)
    ax.set_title("CL–CD polar  (colored by L/D)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze NACA airfoil dataset: metrics + L/D landscape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datadir",    default="data",
                        help="Dataset root directory")
    parser.add_argument("--output-dir", default=None,
                        help="Analysis output dir  [default: <datadir>/analysis]")
    parser.add_argument("--pattern",    default="sample_*",
                        help="Glob pattern for sample folders")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    datadir    = Path(args.datadir)
    output_dir = Path(args.output_dir) if args.output_dir else datadir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {output_dir}/")

    # ── load ──────────────────────────────────────────────────────────────────
    samples = sorted(datadir.glob(args.pattern))
    print(f"Found {len(samples)} sample folders in {datadir}/")
    if not samples:
        print("No samples matched. Check --datadir and --pattern.")
        return

    records, failed = [], []
    for sd in samples:
        rec = load_sample(sd)
        if rec is None:
            failed.append(sd.name)
        else:
            records.append(rec)

    if failed:
        print(f"  Skipped {len(failed)} incomplete/unreadable samples"
              f": {failed[:5]}{'…' if len(failed) > 5 else ''}")

    df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
    print(f"Loaded {len(df)} samples.")

    # ── save metrics ──────────────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False, float_format="%.6g")
    print(f"Saved {metrics_path.name}")

    # ── summary stats ─────────────────────────────────────────────────────────
    stat_cols = [c for c in
                 ["CL", "CD", "LD", "CMz", "Cp_min", "x_suction_peak",
                  "CL_surf", "x_separation", "Cf_mean_upper",
                  "Cp_recovery", "Mach_max_field", "rms_rho_final"]
                 if c in df.columns]
    df[stat_cols].describe().to_csv(output_dir / "summary_stats.csv",
                                    float_format="%.6g")
    print(f"Saved summary_stats.csv")

    # ── print top / bottom performers ─────────────────────────────────────────
    if "LD" in df.columns:
        show_cols = [c for c in ["idx", "label", "m", "p", "CL", "CD", "LD"]
                     if c in df.columns]
        fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
        print("\nTop 5 L/D:")
        print(df.nlargest(5, "LD")[show_cols].to_string(index=False))
        print("\nBottom 5 L/D:")
        print(df.nsmallest(5, "LD")[show_cols].to_string(index=False))

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots…")

        plot_landscape(df, "LD",
                       "Lift-to-drag ratio  (t = 0.15)",
                       output_dir / "landscape_LD.png",
                       cmap="plasma", best="max")

        plot_landscape_3d(df, "LD",
                          "L/D landscape  (t = 0.15)",
                          output_dir / "landscape_LD_3d.png")

        plot_landscape(df, "CL",
                       "Lift coefficient CL  (t = 0.15)",
                       output_dir / "landscape_CL.png",
                       cmap="Blues", best="max")

        plot_landscape(df, "CD",
                       "Drag coefficient CD  (t = 0.15)",
                       output_dir / "landscape_CD.png",
                       cmap="Reds_r", best="min")

        if "Cp_min" in df.columns:
            plot_landscape(df, "Cp_min",
                           "Suction peak Cp_min  (t = 0.15)",
                           output_dir / "landscape_Cp_min.png",
                           cmap="coolwarm_r", best="min")

        if "x_suction_peak" in df.columns:
            plot_landscape(df, "x_suction_peak",
                           "Suction peak location x/c  (t = 0.15)",
                           output_dir / "landscape_x_suction.png",
                           cmap="viridis", best="max")

        if "Cp_recovery" in df.columns:
            plot_landscape(df, "Cp_recovery",
                           "Pressure recovery  Cp_TE – Cp_min  (t = 0.15)",
                           output_dir / "landscape_Cp_recovery.png",
                           cmap="RdYlGn", best="max")

        if "x_separation" in df.columns:
            df_sep = df[df["separated"] == True].copy()
            if len(df_sep) >= 4:
                plot_landscape(df_sep, "x_separation",
                               "Separation onset x/c  (separated samples only)",
                               output_dir / "landscape_x_separation.png",
                               cmap="RdYlGn", best="max")
            else:
                pct = 100 * df["separated"].sum() / len(df)
                print(f"  {df['separated'].sum()} separated samples ({pct:.0f}%) — skipping separation landscape")

        plot_cl_cd_polar(df, output_dir / "polar_CL_CD.png")

        print(f"Plots saved to {output_dir}/")

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
