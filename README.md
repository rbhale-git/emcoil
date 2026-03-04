# emcoil

Electromagnetic coil external field solver. Computes the magnetic field (B-field) of a finite solenoid with configurable core material at arbitrary 3D points using Biot-Savart (elliptic integrals) and equivalent magnetization superposition.

**Live demo:** [huggingface.co/spaces/rbhale/emcoil](https://huggingface.co/spaces/rbhale/emcoil)

## Physics Model

Two superposed contributions compute **B_total** at any evaluation point:

1. **B_coil** — Discretized solenoid loops with off-axis field via elliptic integrals (`scipy.special.ellipk`, `ellipe`)
2. **B_core** — Magnetized cylinder bound surface currents, modeled as an equivalent solenoid with NI_core = (mu_r - 1) * NI
3. **B_total = B_coil + B_core**

## Material Presets

| Name          | mu_r     |
|---------------|----------|
| air           | 1        |
| water         | 0.999992 |
| soft-iron     | 800      |
| silicon-steel | 4000     |
| ferrite       | 1000     |
| mu-metal      | 20000    |

Custom mu_r values can also be provided directly.

## Web Application

Interactive Dash app with:

- Side-by-side comparison of two core configurations
- Interactive field maps (r-z and x-y planes) with origin-centered solenoid
- Click-to-probe exact field values (Bx, By, Bz, |B|, |B_coil|, |B_core|)
- Linked slider + numeric input controls
- Dark precision-instrument theme

```bash
python app.py
# Opens at http://localhost:8050
```

## CLI

Dimensions in mm, output in Tesla.

```bash
# Single point evaluation
python cli.py --radius 25 --length 100 --amp-turns 500 --core soft-iron --point 0 0 150

# Custom permeability
python cli.py --radius 25 --length 100 --amp-turns 500 --mu-r 1200 --point 0 0 150

# Field map (r-z cross-section)
python cli.py --radius 25 --length 100 --amp-turns 500 --core air --plot-rz

# Transverse slice (x-y plane)
python cli.py --radius 25 --length 100 --amp-turns 500 --core air --plot-xy --z-slice 150
```

## Project Structure

```
emcoil/
├── emcoil/
│   ├── coil.py          # Biot-Savart via elliptic integrals
│   ├── core.py          # Magnetized cylinder contribution
│   ├── solver.py        # Superposition solver
│   ├── materials.py     # Material presets and custom mu_r
│   └── plotting.py      # Matplotlib field maps
├── app.py               # Dash web application
├── cli.py               # CLI entry point
├── assets/style.css     # Web app theme
├── Dockerfile           # HF Spaces deployment
├── requirements.txt
└── tests/
```

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```
