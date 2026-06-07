In this project I recreate the results of Shukla, Oommen and Peyvan et al
but with Geodesic Convolutional Neural Networks instead of the DeepONet approach.

# Problem Statement

Find the NACA 4-digit parameters `(p, m)` that maximize the lift-to-drag ratio `L/D` of a 2D aerofoil in steady, subsonic, viscous flow.

The flow is governed by the 2D compressible Navier-Stokes equations:

```
∂ρ/∂t + ∇·(ρu) = 0                                          (continuity)
∂(ρu)/∂t + ∇·(ρu⊗u + pI) = (1/Re) ∇·τ                     (momentum)
∂E/∂t + ∇·((E+p)u) = (1/Re) ∇·(τu + κ∇T)                  (energy)
```

where `ρ` is density, `u` the velocity vector, `p` pressure, `E` total energy, `τ` the viscous stress tensor, and `κ` thermal conductivity. Lift and drag are obtained by integrating pressure and wall shear stress over the aerofoil surface.

## Workflow

1. **Data Generation**
   Sample NACA 4-digit airfoils over the parameter space `p ∈ [0.2, 0.5]`, `m ∈ [0.0, 0.09]` (thickness fixed at `t = 0.15`). Run each through SU2 to obtain the full flow field and surface Cp/Cf distributions.

2. **Surrogate Model**
   Build a GCNN on the airfoil boundary curve using PyTorch Geometric. Each node represents a surface point with features (x, y, curvature, normal angle). Output: surface Cp and Cf distributions, integrated to give L/D.

3. **Optimisation**
   Backpropagate gradients through the differentiable surrogate directly to the NACA parameters (p, m). Gradient descent finds the optimal shape without a black-box external optimizer.

4. **Validation**
   - Surrogate L/D predictions vs. SU2 ground truth
   - Optimization landscape over (p, m) parameter space
   - Surrogate inference time vs. SU2 wall time (speedup)

## Conda Setup

```bash
conda env create -f environment.yml
conda activate gcnn_aerofoil

# Add SU2 binaries to PATH
export PATH="$PWD/bin:$PATH"

# Verify SU2 is accessible
SU2_CFD --help
```

The `bin/` directory contains the SU2 executables (`SU2_CFD`, `SU2_DEF`, `SU2_SOL`, etc.). The `export PATH` line above adds them for the current shell session; to make it permanent add it to your `~/.zshrc` or `~/.bashrc`.

## CFD Solver Usage

**Generate and plot an airfoil geometry:**
```bash
python naca_foil.py --m 0.04 --p 0.4 --t 0.15 --plot --save foil.png
```

**Run the full pipeline (mesh → solve → plots):**
```bash
python flow_solver.py --m 0.04 --p 0.4 --t 0.15 --outdir results/ --su2-bin ./bin/SU2_CFD
```

Key flags for `flow_solver.py`:

| Flag | Default | Description |
|---|---|---|
| `--m` | 0.04 | Max camber |
| `--p` | 0.4 | Position of max camber |
| `--t` | 0.15 | Max thickness |
| `--re` | 500 | Reynolds number |
| `--mach` | 0.5 | Mach number |
| `--aoa` | 0.0 | Angle of attack (degrees) |
| `--outdir` | results/ | Output directory for mesh, config, and plots |
| `--su2-bin` | ./bin/SU2_CFD | Path to SU2_CFD binary |

Outputs saved to `--outdir`: `flow_field.png`, `streamlines.png`, `surface_cp.png`.

**Generate training dataset:**
```bash
python generate_dataset.py --n-samples 200 --datadir data/ --su2-bin ./bin/SU2_CFD
```

With parallel workers:
```bash
python generate_dataset.py --n-samples 200 --datadir data/ --su2-bin ./bin/SU2_CFD --jobs 4
```

Key flags for `generate_dataset.py`:

| Flag | Default | Description |
|---|---|---|
| `--n-samples` | 200 | Number of airfoils to generate |
| `--datadir` | data/ | Root output directory |
| `--seed` | 42 | Random seed for parameter sampling |
| `--jobs` | 1 | Parallel SU2 workers |

Each sample is saved to `data/sample_XXXX/` containing:

| File              | Contents                                                |
| ----------------- | ------------------------------------------------------- |
| `params.json`     | NACA parameters and flow conditions                     |
| `surface.npz`     | Surface Cp, Cfx, Cfy at boundary nodes                  |
| `subdomain.npz`   | Near-wall regular grid (128×64), interpolated fields    |
| `full_field.npz`  | Full unstructured field — nodes, triangles, all scalars |
| `foil.png`        | Airfoil geometry                                        |
| `flow_field.png`  | Pressure, density, velocity, Mach contours              |
| `streamlines.png` | Streamlines with pressure background                    |
| `surface_cp.png`  | Surface Cp and Cfx distributions                        |





## Solving with SU2

SU2 is an open-source CFD solver that solves the compressible Navier-Stokes equations on unstructured meshes. It produces the full volumetric flow field (u, v, p, ρ, T) as well as surface quantities (Cp, Cf), making it suitable both for generating training data and for visualizing flow around the aerofoil.

The pipeline for each airfoil is:
1. Generate surface coordinates from NACA equations
2. Mesh the flow domain with Gmsh (body-fitted unstructured triangular mesh)
3. Run SU2 with a config file specifying `Re`, `Ma`, `AoA`, and boundary conditions
4. Extract surface Cp, Cf and integrated CL, CD from SU2 output

SU2 is scripted in Python via subprocess, so the full 200-sample dataset is generated automatically. Each run takes ~1-5 minutes depending on mesh resolution.

```python
import subprocess

def run_su2(config_path):
    subprocess.run(['SU2_CFD', config_path], check=True)
```

## Surrogate Model

The surrogate is a differentiable approximation to SU2. Given NACA shape parameters, it predicts surface Cp and Cf distributions in milliseconds, from which L/D is integrated. Because the surrogate is a PyTorch graph network, gradients of L/D with respect to (p, m) are available via backpropagation — enabling direct gradient-based shape optimisation.

## Training Data

- **Samples:** 100 NACA 4-digit airfoils drawn from a uniform distribution over `p ∈ [0.2, 0.5]`, `m ∈ [0.0, 0.09]`, `t = 0.15` (fixed)
- **Split:** 60 train / 40 test
- **Geometry:** 100 cosine-spaced surface points per airfoil from the NACA equations
- **Labels:** Surface Cp and Cf distributions from SU2 at `Re = 500`, `Ma = 0.5`, `AoA = 0°`

Each airfoil is stored as a graph: nodes are surface points, edges connect adjacent nodes along the boundary curve.

## SU2 Solver Configuration

| Parameter | Value | Notes |
|---|---|---|
| Solver | Laminar Navier-Stokes | No turbulence model (`KIND_TURB_MODEL= NONE`) |
| Spatial scheme | Roe upwind | 2nd order with MUSCL reconstruction |
| Slope limiter | Venkatakrishnan | Stabilises solution near shocks/gradients |
| Gradient method | Weighted least squares | Standard for unstructured meshes |
| Time integration | Implicit (FGMRES + ILU) | Marched to steady state |
| CFL number | 0.5 | Conservative; suitable for low Re |
| Max iterations | 10,000 | With early stopping at residual ≤ 1e-8 |
| Convergence criterion | RMS(ρ) ≤ 1e-8 | |
| Mesh element size (surface) | `lc_foil = 0.005` | 0.5% chord |
| Mesh element size (farfield) | `lc_far = 1.0` | |
| Domain | `[-3, 11] × [-3, 3]` | Matching Shukla et al. |
| Boundary conditions | Adiabatic no-slip wall; farfield | |

## Model Architecture

A graph convolutional network operating on the 1D boundary curve. Each node is a surface point; edges connect geometrically adjacent nodes (geodesic convolution = message-passing along arc length).

| Component | Detail |
|---|---|
| Node features | x, y, κ (curvature), φ (normal angle) — 4 features |
| GCN layers | 4 × GCNConv with residual connections and BatchNorm |
| Hidden dim | 128 |
| Activation | ReLU |
| Output | Per-node (Cp, Cfx, Cfy) — 3 values per node |

## L/D from Surrogate Output

Let Φ_θ denote the trained GCNN and **x** = (m, p) the NACA shape parameters. The surrogate pipeline is:

```
G(x) = graph with N surface nodes, node features fᵢ = (xᵢ, yᵢ, κᵢ, φᵢ)

{C̃p, C̃fx, C̃fy} = Φ_θ(G(x))          # per-node field predictions
```

L/D is then recovered by discrete surface integration. For each panel between adjacent nodes i and i+1:

```
Δsᵢ   = √((xᵢ₊₁ - xᵢ)² + (yᵢ₊₁ - yᵢ)²)     # panel arc length

n̂ᵢ   = (-Δyᵢ, Δxᵢ) / Δsᵢ                    # outward normal

C̃Lᵢ  = (-C̃p_mid · n̂y + C̃fy_mid) · Δsᵢ       # lift contribution
C̃Dᵢ  = (-C̃p_mid · n̂x + C̃fx_mid) · Δsᵢ       # drag contribution
```

where `_mid` denotes the panel midpoint average of adjacent node values. Summing over all panels:

```
C̃L = Σᵢ C̃Lᵢ
C̃D = Σᵢ C̃Dᵢ
L̃/D̃ = C̃L / C̃D
```

Because Φ_θ is implemented in PyTorch and the integration is differentiable, ∂(L̃/D̃)/∂(m, p) is available via autograd — enabling direct gradient-based shape optimisation.

## Loss Function

Each field `f ∈ {Cp, Cfx, Cfy}` is standardised independently before training:
```
ŷ_f = (y_f - μ_f) / σ_f
```
where `μ_f` and `σ_f` are the mean and std computed over all nodes in the training set. This is necessary because Cp is O(1–3) and Cfx/Cfy is O(1e-3) — without it, the loss is dominated by Cp and the network never learns Cf.

The training loss is then MSE on the standardised values:
```
L = (1 / 3N) Σ_f Σ_i (Φ̂_f(i) - ŷ_f(i))²
```
where `N` is the number of surface nodes, `i` indexes nodes, `f` indexes fields, and `Φ̂_f(i)` is the normalised model prediction.

The **relative L2 error** ("rel") is a more interpretable metric computed in physical (de-normalised) units after training:
```
rel = (1/3) Σ_f  ‖Φ_f - y_f‖ / ‖y_f‖
```
A value of `0.05` means the model predictions are on average 5% off from the SU2 ground truth. Values much larger than 1 indicate the metric is being computed incorrectly (e.g. on normalised values where the denominator `‖y_f‖` is near zero).

## Training

| Hyperparameter | Value |
|---|---|
| Loss | MSE on standardised (Cp, Cfx, Cfy) |
| Optimiser | Adam |
| Learning rate | 1e-3 with cosine decay |
| Batch size | 16 |
| Epochs | 500 |
| Early stopping | patience = 50 |
