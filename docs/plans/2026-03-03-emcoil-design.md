# emcoil — Electromagnetic Coil External Field Solver

## Purpose

Approximate the magnetic field intensity (B-field) of a finite solenoid with configurable core material at arbitrary 3D points outside the coil volume. Designed to support MHD thruster magnet placement and field characterization.

## Repository

https://github.com/rbhale-git/emcoil.git

## Physics Model

Two superposed contributions compute B_total at any evaluation point:

### 1. B_coil — Air-Core Solenoid via Biot-Savart

Discretize the solenoid into N circular current loops evenly spaced along its length. For each loop, compute the off-axis magnetic field at the evaluation point using the exact elliptic integral formulation (scipy `ellipk`, `ellipe`). Sum contributions from all loops.

### 2. B_core — Magnetized Cylinder

Approximate the internal H-field from the solenoid formula. Compute magnetization M = (mu_r - 1) * H. Model the core as a uniformly magnetized cylinder with bound surface currents. Compute the external field contribution using the magnetic scalar potential / equivalent surface charge method at each end face.

### 3. Superposition

B_total = B_coil + B_core

## Inputs

- **Coil geometry:** radius R (mm), length L (mm), total amp-turns NI
- **Core material:** preset name or custom mu_r value
- **Evaluation point(s):** arbitrary (x, y, z) in mm

## Material Presets

| Name           | mu_r       |
|----------------|------------|
| air            | 1          |
| water          | 0.999992   |
| soft-iron      | 800        |
| silicon-steel  | 4000       |
| ferrite        | 1000       |
| mu-metal       | 20000      |

Custom mu_r: pass `--mu-r <float>` to override any preset.

## Project Structure

```
emcoil/
├── emcoil/
│   ├── __init__.py
│   ├── coil.py          # Biot-Savart field from discretized solenoid loops
│   ├── core.py          # Magnetized cylinder external field contribution
│   ├── solver.py        # Combines coil + core, evaluates B_total at points
│   ├── materials.py     # Material presets and custom mu_r handling
│   └── plotting.py      # Matplotlib field maps (contour, quiver, streamlines)
├── cli.py               # argparse CLI entry point
├── requirements.txt     # numpy, scipy, matplotlib
└── README.md
```

## CLI Interface

Dimensions in mm, output in Tesla.

### Single point evaluation

```bash
python cli.py --radius 25 --length 100 --amp-turns 500 --core soft-iron \
              --point 0 0 150
```

### Custom permeability

```bash
python cli.py --radius 25 --length 100 --amp-turns 500 --mu-r 1200 \
              --point 0 0 150
```

### Field map (2D cross-section)

```bash
python cli.py --radius 25 --length 100 --amp-turns 500 --core air \
              --plot-rz --rmax 200 --zmax 300
```

### Transverse slice

```bash
python cli.py --radius 25 --length 100 --amp-turns 500 --core air \
              --plot-xy --z-slice 150 --rmax 200
```

## Output

### Numeric (per evaluation point)

- Bx, By, Bz components in Tesla
- |B| magnitude
- Breakdown: |B_coil| and |B_core| contributions separately
- Results saved to results.json

### Visualization

- `--plot-rz`: Axisymmetric r-z cross-section with |B| contour fill, quiver arrows for direction, solenoid outline overlay
- `--plot-xy`: Transverse x-y slice at given z with |B| contour fill
- Colorbar in Tesla

## Dependencies

- numpy
- scipy (elliptic integrals)
- matplotlib (plotting)
