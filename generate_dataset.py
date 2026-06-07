#!/usr/bin/env python3
"""
Generate NACA airfoil training dataset.

Samples NACA 4-digit airfoils uniformly over the parameter space,
runs SU2 for each, and saves three levels of training data per sample:

  A  surface.npz       Surface Cp and Cf at boundary nodes
  B  subdomain.npz     Near-wall regular grid (interpolated, 128x64)
  C  full_field.npz    Full unstructured flow field (nodes + triangles + fields)

Each sample also gets: foil.png, flow_field.png, streamlines.png, surface_cp.png

Output structure:
  data/
    sample_0000/
      params.json
      surface.npz       (A)
      subdomain.npz     (B)
      full_field.npz    (C)
      foil.png
      flow_field.png
      streamlines.png
      surface_cp.png

Usage:
    python generate_dataset.py --n-samples 200 --datadir data/ --su2-bin ./bin/SU2_CFD
    python generate_dataset.py --n-samples 200 --datadir data/ --su2-bin ./bin/SU2_CFD --jobs 4
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for parallel runs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from naca_foil import naca4_coords, naca4_label
from flow_solver import generate_mesh, write_config, run_su2


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_parameters(n, seed=42):
    """Sample n airfoils uniformly over p ∈ [0.2, 0.5], m ∈ [0.0, 0.09], t fixed at 0.15."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.00, 0.09, n)
    p = rng.uniform(0.20, 0.50, n)
    t = np.full(n, 0.15)
    return list(zip(m, p, t))


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_surface(surface_csv, vtk_point_data, out_path, m, p, t=0.15, n_naca=100):
    """
    Option A: surface Cp and Cf interpolated onto NACA-equation coordinates.

    SU2 outputs ~400 irregular mesh nodes on the surface. The surrogate model
    uses 198 cosine-spaced NACA-equation nodes at inference time. To avoid a
    train/inference mismatch, we interpolate onto NACA nodes.

    IMPORTANT: upper and lower surfaces are interpolated SEPARATELY using x as
    the variable. Arc-length parameterisation is avoided because sort_surface_nodes
    produces a criss-crossing path (arc ~15 instead of ~2) for irregular SU2 mesh
    nodes, corrupting the interpolation.
    """
    import pandas as pd
    from scipy.interpolate import interp1d
    from naca_foil import naca4_coords

    df = pd.read_csv(surface_csv)
    df.columns = df.columns.str.strip().str.strip('"')

    point_ids = df["PointID"].values.astype(int)
    x_su2 = df["x"].values
    y_su2 = df["y"].values

    # Pull Cp and Cf from VTK at SU2 surface nodes
    if "Pressure_Coefficient" not in vtk_point_data:
        raise ValueError("Pressure_Coefficient not found in VTK output")
    if "Skin_Friction_Coefficient" not in vtk_point_data:
        raise ValueError("Skin_Friction_Coefficient not found in VTK output")

    cp_vtk  = np.asarray(vtk_point_data["Pressure_Coefficient"]).squeeze()
    cf_vtk  = np.asarray(vtk_point_data["Skin_Friction_Coefficient"])
    Cp_su2  = cp_vtk[point_ids]
    Cfx_su2 = cf_vtk[point_ids, 0]
    Cfy_su2 = cf_vtk[point_ids, 1]

    # NACA-equation target nodes (same as surrogate inference)
    xu, yu, xl, yl = naca4_coords(m, p, t, n_naca)
    x_naca = np.concatenate([xu, xl[-2:0:-1]])
    y_naca = np.concatenate([yu, yl[-2:0:-1]])

    # Split SU2 nodes relative to the camber line (not y=0).
    # Using y=0 misclassifies nodes near the LE for cambered airfoils,
    # corrupting ΔCp = Cp_lower - Cp_upper and hence CL.
    yc_su2 = np.where(
        x_su2 < p,
        (m / max(p, 1e-8)**2)       * (2*p*x_su2 - x_su2**2),
        (m / max(1-p, 1e-8)**2)     * (1 - 2*p + 2*p*x_su2 - x_su2**2),
    )
    # Include camber-line nodes in BOTH groups so the stagnation point (y=yc at LE)
    # contributes to both upper and lower surface interpolations.
    upper = y_su2 >= yc_su2
    lower = y_su2 <= yc_su2

    def interp_x(x_src, vals, x_tgt):
        """1-D interpolation using x as variable. Removes duplicate x values."""
        idx = np.argsort(x_src)
        xs, vs = x_src[idx], vals[idx]
        _, uniq = np.unique(xs, return_index=True)
        xs, vs = xs[uniq], vs[uniq]
        return interp1d(xs, vs, kind="linear",
                        bounds_error=False,
                        fill_value=(vs[0], vs[-1]))(x_tgt)

    # NOTE: do NOT override `upper`/`lower` with a y=0 split here.
    # The camber-line split computed above (y_su2 >= / <= yc_su2) is required:
    # for cambered airfoils a y=0 split misclassifies leading-edge nodes,
    # mixing upper/lower Cp near the LE and corrupting ΔCp (hence CL).

    def interp_all(mask, x_tgt):
        return (
            interp_x(x_su2[mask], Cp_su2[mask],  x_tgt),
            interp_x(x_su2[mask], Cfx_su2[mask], x_tgt),
            interp_x(x_su2[mask], Cfy_su2[mask], x_tgt),
        )

    # Upper surface: target x = xu (LE→TE, ascending)
    Cp_u, Cfx_u, Cfy_u = interp_all(upper, xu)

    # Lower surface: target x = xl[1:-1] (LE→TE interior, ascending),
    # then reverse to match the TE→LE ordering of the lower part of x_naca
    xl_interior = xl[1:-1]
    Cp_l, Cfx_l, Cfy_l = interp_all(lower, xl_interior)
    Cp_l, Cfx_l, Cfy_l = Cp_l[::-1], Cfx_l[::-1], Cfy_l[::-1]  # → TE→LE order

    np.savez(out_path,
             x   = x_naca,
             y   = y_naca,
             Cp  = np.concatenate([Cp_u,  Cp_l]),
             Cfx = np.concatenate([Cfx_u, Cfx_l]),
             Cfy = np.concatenate([Cfy_u, Cfy_l]))


def extract_subdomain(pts, fields, out_path,
                      x_range=(-0.2, 1.4), y_range=(-0.5, 0.5),
                      nx=128, ny=64):
    """Option B: interpolate onto a regular near-wall grid."""
    xi = np.linspace(*x_range, nx)
    yi = np.linspace(*y_range, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    data = {"x": xi, "y": yi}
    for name, values in fields.items():
        if values is not None and values.ndim == 1:
            data[name] = griddata(pts, values, (Xi, Yi), method="linear")

    np.savez(out_path, **data)


def extract_full_field(pts, tris, fields, out_path):
    """Option C: full unstructured field — nodes, triangles, and all scalar fields."""
    data = {"x": pts[:, 0], "y": pts[:, 1], "triangles": tris}
    for name, values in fields.items():
        if values is not None:
            data[name] = values
    np.savez(out_path, **data)


# ── Figures ───────────────────────────────────────────────────────────────────

def save_figures(pts, tris, fields, surface_csv, foil_coords, outdir, label):
    import pandas as pd

    xu, yu, xl, yl = foil_coords
    triang  = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    pressure = fields.get("pressure")
    density  = fields.get("density")
    vel_mag  = fields.get("vel_mag")
    mach     = fields.get("mach")
    vel_x    = fields.get("vel_x")
    vel_y    = fields.get("vel_y")

    # ── Foil geometry ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xu, yu, "b-", lw=1.5, label="Upper")
    ax.plot(xl, yl, "r-", lw=1.5, label="Lower")
    ax.fill(np.concatenate([xu, xl[::-1]]),
            np.concatenate([yu, yl[::-1]]),
            alpha=0.15, color="steelblue")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(label); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "foil.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Flow field contours ───────────────────────────────────────────────────
    field_map = {
        "Pressure":           pressure,
        "Density":            density,
        "Velocity Magnitude": vel_mag,
        "Mach":               mach,
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (name, data) in zip(axes.flatten(), field_map.items()):
        if data is None:
            ax.set_title(f"{name} (n/a)")
            continue
        tcf = ax.tricontourf(triang, data, levels=50, cmap="RdBu_r")
        fig.colorbar(tcf, ax=ax, shrink=0.8)
        ax.set_title(name); ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
        ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.8, 0.8); ax.set_aspect("equal")
    fig.suptitle(label, fontsize=13)
    plt.tight_layout()
    plt.savefig(outdir / "flow_field.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Streamlines ───────────────────────────────────────────────────────────
    if vel_x is not None and vel_y is not None:
        xi = np.linspace(-0.5, 2.0, 300)
        yi = np.linspace(-0.8, 0.8, 150)
        Xi, Yi = np.meshgrid(xi, yi)
        U = griddata(pts, vel_x, (Xi, Yi), method="linear")
        V = griddata(pts, vel_y, (Xi, Yi), method="linear")
        P = griddata(pts, pressure, (Xi, Yi), method="linear") if pressure is not None else None

        fig, ax = plt.subplots(figsize=(12, 5))
        if P is not None:
            ax.contourf(Xi, Yi, P, levels=50, cmap="RdBu_r", alpha=0.8)
        ax.streamplot(xi, yi, U, V, density=1.5, color="k",
                      linewidth=0.5, arrowsize=0.8)
        ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.8, 0.8); ax.set_aspect("equal")
        ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
        ax.set_title(f"Streamlines — {label}")
        plt.tight_layout()
        plt.savefig(outdir / "streamlines.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Surface Cp and Cf ─────────────────────────────────────────────────────
    if surface_csv.exists():
        df = pd.read_csv(surface_csv)
        df.columns = df.columns.str.strip()

        def find_col(*keywords):
            for c in df.columns:
                if all(k in c.lower() for k in keywords):
                    return c
            return None

        x_col   = find_col("x") or df.columns[0]
        cp_col  = find_col("pressure_coefficient")
        cfx_col = find_col("skin_friction", "_x")

        plots = [(cp_col, "Cp", "b"), (cfx_col, "Cfx", "r")]
        plots = [(c, n, col) for c, n, col in plots if c is not None]

        if plots:
            fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 4))
            if len(plots) == 1:
                axes = [axes]
            for ax, (col, name, color) in zip(axes, plots):
                ax.plot(df[x_col], df[col], f"{color}.", ms=2)
                if name == "Cp":
                    ax.invert_yaxis()
                ax.set_xlabel("x/c"); ax.set_ylabel(name)
                ax.set_title(f"Surface {name} — {label}")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "surface_cp.png", dpi=150, bbox_inches="tight")
            plt.close()


# ── Process one sample ────────────────────────────────────────────────────────

def process_sample(args):
    idx, m, p, t, datadir, su2_bin, re, mach, aoa = args
    import meshio

    sample_dir = Path(datadir) / f"sample_{idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    label = naca4_label(m, p, t)

    # Params
    (sample_dir / "params.json").write_text(json.dumps(
        {"idx": idx, "m": m, "p": p, "t": t,
         "re": re, "mach": mach, "aoa": aoa, "label": label},
        indent=2
    ))

    # Mesh + solve
    mesh_path   = sample_dir / "mesh.su2"
    config_path = sample_dir / "su2.cfg"
    generate_mesh(m, p, t, mesh_path)
    write_config(config_path, mesh_filename=str(mesh_path.resolve()),
                 reynolds=re, mach=mach, aoa=aoa)
    run_su2(config_path, Path(su2_bin), workdir=sample_dir)

    # Find VTK output
    vtk_candidates = sorted(sample_dir.glob("flow*.vtu")) or \
                     sorted(sample_dir.glob("flow*.vtk"))
    if not vtk_candidates:
        raise FileNotFoundError(f"No VTK output in {sample_dir}")

    # Read mesh output
    mesh   = meshio.read(str(vtk_candidates[-1]))
    pts    = mesh.points[:, :2]
    tris   = np.vstack([cb.data for cb in mesh.cells if cb.type == "triangle"])
    pd_    = mesh.point_data

    def scalar(name):
        return np.asarray(pd_[name]).squeeze() if name in pd_ else None

    pressure = scalar("Pressure")
    density  = scalar("Density")
    mach_f   = scalar("Mach")

    vel_x, vel_y = None, None
    if "Velocity" in pd_:
        vel  = np.asarray(pd_["Velocity"])
        vel_x, vel_y = vel[:, 0], vel[:, 1]
    else:
        vel_x = scalar("Velocity_x") or scalar("x-Velocity")
        vel_y = scalar("Velocity_y") or scalar("y-Velocity")

    vel_mag = (np.sqrt(vel_x**2 + vel_y**2)
               if vel_x is not None and vel_y is not None else None)

    fields = {
        "pressure": pressure, "density":  density,  "mach":    mach_f,
        "vel_x":    vel_x,    "vel_y":    vel_y,     "vel_mag": vel_mag,
    }
    scalar_fields = {k: v for k, v in fields.items()
                     if v is not None and np.asarray(v).ndim == 1}

    surface_csv = sample_dir / "surface_flow.csv"

    # A — surface quantities: Cp/Cf from VTK, interpolated onto NACA nodes
    if surface_csv.exists():
        extract_surface(surface_csv, pd_, sample_dir / "surface.npz", m=m, p=p, t=t)

    # B — near-wall subdomain
    extract_subdomain(pts, scalar_fields, sample_dir / "subdomain.npz")

    # C — full unstructured field
    extract_full_field(pts, tris, scalar_fields, sample_dir / "full_field.npz")

    # Figures
    save_figures(pts, tris, fields, surface_csv,
                 naca4_coords(m, p, t), sample_dir, label)

    return idx, label


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate NACA airfoil training dataset")
    parser.add_argument("--n-samples", type=int,   default=200,             help="Number of samples (default: 200)")
    parser.add_argument("--datadir",   type=str,   default="data",          help="Output directory (default: data/)")
    parser.add_argument("--su2-bin",   type=str,   default="./bin/SU2_CFD", help="Path to SU2_CFD binary")
    parser.add_argument("--seed",      type=int,   default=42,              help="Random seed (default: 42)")
    parser.add_argument("--jobs",      type=int,   default=1,               help="Parallel workers (default: 1)")
    parser.add_argument("--re",        type=float, default=500.0,           help="Reynolds number (default: 500)")
    parser.add_argument("--mach",      type=float, default=0.5,             help="Mach number (default: 0.5)")
    parser.add_argument("--aoa",       type=float, default=0.0,             help="Angle of attack (default: 0)")
    args = parser.parse_args()

    su2_bin = Path(args.su2_bin).resolve()
    if not su2_bin.exists():
        raise FileNotFoundError(f"SU2_CFD not found at {su2_bin}")

    # Find the next available sample index
    datadir = Path(args.datadir)
    existing = sorted(datadir.glob("sample_*")) if datadir.exists() else []
    start_idx = 0
    if existing:
        last_idx = int(existing[-1].name.split("_")[1])
        start_idx = last_idx + 1
        print(f"Found {len(existing)} existing samples. Starting from index {start_idx}.")

    params   = sample_parameters(args.n_samples, seed=args.seed + start_idx)
    job_args = [
        (start_idx + i, m, p, t, args.datadir, str(su2_bin), args.re, args.mach, args.aoa)
        for i, (m, p, t) in enumerate(params)
    ]

    print(f"\nGenerating {args.n_samples} samples → {args.datadir}/  (indices {start_idx}–{start_idx + args.n_samples - 1})")
    print(f"Re={args.re}  Ma={args.mach}  AoA={args.aoa}°  jobs={args.jobs}\n")

    if args.jobs == 1:
        for job in tqdm(job_args, unit="sample"):
            idx, label = process_sample(job)
            tqdm.write(f"  ✓ {idx:04d}  {label}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(process_sample, job): job[0] for job in job_args}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="sample"):
                idx, label = fut.result()
                tqdm.write(f"  ✓ {idx:04d}  {label}")

    print(f"\nDone. Dataset in {args.datadir}/")


if __name__ == "__main__":
    main()
