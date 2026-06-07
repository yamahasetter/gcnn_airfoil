#!/usr/bin/env python3
"""
Generate mesh, run SU2, and plot flow field for a NACA 4-digit airfoil.

CLI:
    python flow_solver.py --m 0.04 --p 0.4 --t 0.15 --outdir results/ --su2-bin ./bin/SU2_CFD
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).parent))
from naca_foil import naca4_coords, naca4_label

"""
══════════════════════════════════════════════════════════════════════════════
WORKFLOW OVERVIEW
══════════════════════════════════════════════════════════════════════════════

This script is the single-airfoil CFD pipeline: given NACA 4-digit parameters
and flow conditions, it meshes the domain, runs SU2, and produces plots.
Invoked as a CLI tool and also called in a loop by generate_dataset.py.

STEP 1 — Mesh generation (generate_mesh)
  NACA 4-digit surface coordinates are generated via naca4_coords() and
  assembled into a closed loop (upper LE→TE + lower interior TE→LE). Gmsh
  creates a body-fitted 2D triangular mesh:
    - Airfoil boundary: fine mesh (lc_foil=0.005, ~0.5% chord) for resolving
      the boundary layer.
    - Farfield box [-3,11]×[-3,3]: coarse mesh (lc_far=1.0).
    - Physical groups named "airfoil", "farfield", "fluid" must match the
      SU2 config MARKER_ entries.
  Output: a .su2 format mesh file.

STEP 2 — SU2 configuration (write_config)
  A config file is written from a template string (SU2_CONFIG_TEMPLATE)
  parameterized by Mach number, Reynolds number, angle of attack, and the
  mesh file path. Key solver settings:
    - Laminar Navier-Stokes (no turbulence model)
    - Roe upwind spatial scheme with MUSCL reconstruction and
      Venkatakrishnan slope limiter
    - FGMRES+ILU implicit time-marching to steady state
    - Convergence criterion: RMS(ρ) ≤ 1e-8 or max 10,000 iterations
  Output files requested: restart, ParaView VTU, and surface CSV.

STEP 3 — SU2 execution (run_su2)
  SU2_CFD is called via subprocess with the config file as argument, running
  in the output directory. Raises RuntimeError on non-zero exit code.
  Produces flow.vtu (volumetric field) and surface_flow.csv (boundary data).

STEP 4 — Visualization (plot_flow, plot_mesh)
  plot_flow() reads the VTU output via meshio and produces three figures:
    a) 2×2 contour plots: Pressure, Density, Velocity Magnitude, Mach number
       (zoomed to near-airfoil region, ±0.8 chord)
    b) Streamlines overlaid on pressure background (interpolated to regular
       grid via scipy griddata)
    c) Surface Cp distribution from the surface_flow.csv file
  plot_mesh() is optional (--savemesh flag) and shows full-domain and zoomed
  mesh triangulation for mesh quality inspection.

ENTRY POINT — main()
  Parses CLI arguments and runs steps 1–4 in sequence, printing progress.
  All outputs (mesh.su2, su2.cfg, flow.vtu, *.png) are written to --outdir.

══════════════════════════════════════════════════════════════════════════════
"""

# ── Mesh generation ───────────────────────────────────────────────────────────

def generate_mesh(m, p, t, mesh_path, n_foil=100, lc_foil=0.005, lc_far=1.0):
    """
    Generate a body-fitted triangular mesh around a NACA airfoil using Gmsh.
    Writes a .su2 mesh file to mesh_path.

    Domain: [-3, 11] x [-3, 3]  (matching the paper)
    Boundary tags: 'airfoil', 'farfield'
    """
    import gmsh

    xu, yu, xl, yl = naca4_coords(m, p, t, n_foil)
    n = len(xu)  # points per surface, including LE and TE

    # Closed surface: upper LE->TE, then lower interior TE->LE (no duplicate endpoints)
    all_x = np.concatenate([xu, xl[-2:0:-1]])
    all_y = np.concatenate([yu, yl[-2:0:-1]])

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("airfoil")

    # Airfoil points
    foil_pts = [
        gmsh.model.geo.addPoint(float(all_x[i]), float(all_y[i]), 0.0, lc_foil)
        for i in range(len(all_x))
    ]

    # Upper spline LE->TE, lower spline TE->LE
    upper_spline = gmsh.model.geo.addSpline(foil_pts[:n])
    lower_spline = gmsh.model.geo.addSpline(foil_pts[n - 1:] + [foil_pts[0]])
    foil_loop    = gmsh.model.geo.addCurveLoop([upper_spline, lower_spline])

    # Farfield box
    fp = [
        gmsh.model.geo.addPoint(-3.0, -3.0, 0.0, lc_far),
        gmsh.model.geo.addPoint(11.0, -3.0, 0.0, lc_far),
        gmsh.model.geo.addPoint(11.0,  3.0, 0.0, lc_far),
        gmsh.model.geo.addPoint(-3.0,  3.0, 0.0, lc_far),
    ]
    fl = [
        gmsh.model.geo.addLine(fp[0], fp[1]),
        gmsh.model.geo.addLine(fp[1], fp[2]),
        gmsh.model.geo.addLine(fp[2], fp[3]),
        gmsh.model.geo.addLine(fp[3], fp[0]),
    ]
    far_loop = gmsh.model.geo.addCurveLoop(fl)

    # Fluid surface — farfield outer boundary, airfoil is a hole
    surface = gmsh.model.geo.addPlaneSurface([far_loop, foil_loop])

    gmsh.model.geo.synchronize()

    # Physical groups — names must match MARKER_ entries in SU2 config
    gmsh.model.addPhysicalGroup(1, [upper_spline, lower_spline], name="airfoil")
    gmsh.model.addPhysicalGroup(1, fl,                           name="farfield")
    gmsh.model.addPhysicalGroup(2, [surface],                    name="fluid")

    gmsh.model.mesh.generate(2)
    gmsh.write(str(mesh_path))
    gmsh.finalize()
    print(f"  Mesh written → {mesh_path}")


# ── SU2 config ────────────────────────────────────────────────────────────────

SU2_CONFIG_TEMPLATE = """\
SOLVER= NAVIER_STOKES
KIND_TURB_MODEL= NONE
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
REYNOLDS_NUMBER= {reynolds}
REYNOLDS_LENGTH= 1.0

REF_ORIGIN_MOMENT_X= ( 0.25 )
REF_ORIGIN_MOMENT_Y= ( 0.00 )
REF_ORIGIN_MOMENT_Z= ( 0.00 )
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_DIMENSIONALIZATION= FREESTREAM_VEL_EQ_ONE

MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
CFL_NUMBER= 0.5
CFL_ADAPT= NO
ITER= 10000

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-10
LINEAR_SOLVER_ITER= 20

CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10

MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2
SOLUTION_FILENAME= solution_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_FILES= ( RESTART, PARAVIEW_ASCII, SURFACE_CSV )
VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )
TABULAR_FORMAT= CSV
OUTPUT_WRT_FREQ= 1000
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )
"""


def write_config(config_path, mesh_filename, reynolds=500.0, mach=0.5, aoa=0.0):
    text = SU2_CONFIG_TEMPLATE.format(
        mach=mach, aoa=aoa, reynolds=reynolds,
        mesh_filename=mesh_filename,
    )
    config_path.write_text(text)
    print(f"  Config written → {config_path}")


# ── Run SU2 ───────────────────────────────────────────────────────────────────

def run_su2(config_path, su2_bin, workdir):
    cmd = [str(su2_bin), config_path.name]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(workdir))
    if result.returncode != 0:
        raise RuntimeError(f"SU2_CFD exited with code {result.returncode}")
    print("  SU2 complete.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_mesh(mesh_path, out_path, label=""):
    """Plot the triangular mesh, with inset zoomed to the airfoil region."""
    import meshio

    mesh = meshio.read(str(mesh_path))
    pts  = mesh.points[:, :2]
    tris = np.vstack([cb.data for cb in mesh.cells if cb.type == "triangle"])
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, xlim, ylim, title in [
        (ax_full, (-3.0, 11.0), (-3.0, 3.0),  "Full domain"),
        (ax_zoom, (-0.2,  1.4), (-0.4, 0.4),  "Near-wall"),
    ]:
        ax.triplot(triang, color="steelblue", lw=0.3, alpha=0.7)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
        ax.set_title(f"{title} — {label}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

def _get_field(point_data, *candidates):
    """Return the first matching array from point_data as a 1D array, or None."""
    for name in candidates:
        if name in point_data:
            return np.asarray(point_data[name]).squeeze()
    return None


def plot_flow(vtu_path, surface_csv_path, outdir, label=""):
    import meshio
    import pandas as pd

    print("  Reading VTU output …")
    mesh = meshio.read(str(vtu_path))
    pts  = mesh.points[:, :2]

    tris = np.vstack([cb.data for cb in mesh.cells if cb.type == "triangle"])
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    pd_ = mesh.point_data
    pressure = _get_field(pd_, "Pressure", "Pressure_Coefficient")
    density  = _get_field(pd_, "Density")
    mach     = _get_field(pd_, "Mach")

    # Velocity may be stored as a (N,3) vector field
    vel_x, vel_y = None, None
    if "Velocity" in pd_:
        vel = np.asarray(pd_["Velocity"])
        vel_x, vel_y = vel[:, 0], vel[:, 1]
    else:
        vel_x = _get_field(pd_, "Velocity_x", "x-Velocity")
        vel_y = _get_field(pd_, "Velocity_y", "y-Velocity")
    vel_mag = np.sqrt(vel_x**2 + vel_y**2) if (vel_x is not None and vel_y is not None) else None

    fields = {
        "Pressure":           pressure,
        "Density":            density,
        "Velocity Magnitude": vel_mag,
        "Mach":               mach,
    }

    # ── Contour plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (name, data) in zip(axes.flatten(), fields.items()):
        if data is None:
            ax.set_title(f"{name} (not available)")
            continue
        tcf = ax.tricontourf(triang, data, levels=50, cmap="RdBu_r")
        fig.colorbar(tcf, ax=ax, shrink=0.8)
        ax.set_title(name)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")

    fig.suptitle(label, fontsize=13)
    plt.tight_layout()
    out = outdir / "flow_field.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # ── Streamlines ───────────────────────────────────────────────────────────
    if vel_x is not None and vel_y is not None:
        xi = np.linspace(-0.5, 2.0, 300)
        yi = np.linspace(-0.8, 0.8, 150)
        Xi, Yi = np.meshgrid(xi, yi)

        U = griddata(pts, vel_x,    (Xi, Yi), method="linear")
        V = griddata(pts, vel_y,    (Xi, Yi), method="linear")
        P = griddata(pts, pressure, (Xi, Yi), method="linear") if pressure is not None else None

        fig, ax = plt.subplots(figsize=(12, 5))
        if P is not None:
            ax.contourf(Xi, Yi, P, levels=50, cmap="RdBu_r", alpha=0.8)
        ax.streamplot(xi, yi, U, V, density=1.5, color="k",
                      linewidth=0.5, arrowsize=0.8)
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(f"Streamlines — {label}")
        plt.tight_layout()
        out = outdir / "streamlines.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")

    # ── Surface Cp ────────────────────────────────────────────────────────────
    if surface_csv_path.exists():
        df = pd.read_csv(surface_csv_path)
        x_col  = next((c for c in df.columns if "x"                    in c.lower()), None)
        cp_col = next((c for c in df.columns if "pressure_coefficient"  in c.lower()
                                              or c.strip() == "C_p"), None)
        if x_col and cp_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df[x_col], df[cp_col], "b.", ms=2)
            ax.invert_yaxis()
            ax.set_xlabel("x/c")
            ax.set_ylabel("Cp")
            ax.set_title(f"Surface Pressure Coefficient — {label}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = outdir / "surface_cp.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mesh, solve, and plot flow around a NACA 4-digit airfoil"
    )
    parser.add_argument("--m",       type=float, default=0.04,           help="Max camber (default: 0.04)")
    parser.add_argument("--p",       type=float, default=0.4,            help="Position of max camber (default: 0.4)")
    parser.add_argument("--t",       type=float, default=0.15,           help="Max thickness (default: 0.15)")
    parser.add_argument("--re",      type=float, default=500.0,          help="Reynolds number (default: 500)")
    parser.add_argument("--mach",    type=float, default=0.5,            help="Mach number (default: 0.5)")
    parser.add_argument("--aoa",     type=float, default=0.0,            help="Angle of attack in degrees (default: 0)")
    parser.add_argument("--outdir",  type=str,   default="results",      help="Output directory (default: results/)")
    parser.add_argument("--su2-bin", type=str,   default="./bin/SU2_CFD",help="Path to SU2_CFD binary")
    parser.add_argument("--n-foil",   type=int,   default=100,            help="Airfoil surface points (default: 100)")
    parser.add_argument("--lc-foil",  type=float, default=0.005,          help="Target cell size at the airfoil (default: 0.005)")
    parser.add_argument("--lc-far",   type=float, default=1.0,            help="Target cell size at the farfield (default: 1.0)")
    parser.add_argument("--savemesh", action="store_true",                help="Save mesh plot (mesh.png)")
    args = parser.parse_args()

    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    su2_bin = Path(args.su2_bin).resolve()
    if not su2_bin.exists():
        raise FileNotFoundError(f"SU2_CFD not found at {su2_bin}")

    label = naca4_label(args.m, args.p, args.t)
    print(f"\n{'='*55}")
    print(f"  {label}  |  Re={args.re}  Ma={args.mach}  AoA={args.aoa}°")
    print(f"{'='*55}\n")

    mesh_path   = outdir / "mesh.su2"
    config_path = outdir / "su2.cfg"
    surface_csv = outdir / "surface_flow.csv"

    # 1. Mesh
    print("[1/4] Generating mesh …")
    generate_mesh(args.m, args.p, args.t, mesh_path, n_foil=args.n_foil,
                  lc_foil=args.lc_foil, lc_far=args.lc_far)
    if args.savemesh:
        plot_mesh(mesh_path, outdir / "mesh.png", label=label)

    # 2. Config
    print("[2/4] Writing SU2 config …")
    write_config(config_path, mesh_filename=str(mesh_path.resolve()),
                 reynolds=args.re, mach=args.mach, aoa=args.aoa)

    # 3. Solve
    print("[3/4] Running SU2 …")
    run_su2(config_path, su2_bin, workdir=outdir)

    # 4. Plot — SU2 may write flow.vtu or flow_00001.vtu
    print("[4/4] Plotting results …")
    vtu_candidates = sorted(outdir.glob("flow*.vtu")) or sorted(outdir.glob("flow*.vtk"))
    if not vtu_candidates:
        raise FileNotFoundError("No .vtu/.vtk output found — check SU2 ran successfully.")
    plot_flow(vtu_candidates[-1], surface_csv, outdir, label=label)

    print(f"\nDone. Results in {outdir}/")


if __name__ == "__main__":
    main()
