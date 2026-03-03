"""CLI entry point for emcoil electromagnetic coil field solver."""

import argparse
import json
import numpy as np
from datetime import datetime

from emcoil.materials import get_mu_r
from emcoil.solver import compute_field


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute the external B-field of a finite solenoid with configurable core."
    )

    # Coil geometry (input in mm, converted to m internally)
    p.add_argument("--radius", type=float, required=True, help="Solenoid radius (mm)")
    p.add_argument("--length", type=float, required=True, help="Solenoid length (mm)")
    p.add_argument("--amp-turns", type=float, required=True, help="Total amp-turns (N*I)")

    # Core material
    p.add_argument("--core", type=str, default=None,
                   help="Core material preset (air, water, soft-iron, silicon-steel, ferrite, mu-metal)")
    p.add_argument("--mu-r", type=float, default=None,
                   help="Custom relative permeability (overrides --core)")

    # Evaluation
    p.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Evaluation point in mm (x y z)")

    # Plotting
    p.add_argument("--plot-rz", action="store_true",
                   help="Generate r-z cross-section field map")
    p.add_argument("--plot-xy", action="store_true",
                   help="Generate x-y transverse slice field map")
    p.add_argument("--z-slice", type=float, default=None,
                   help="Z position for --plot-xy slice (mm)")
    p.add_argument("--rmax", type=float, default=None,
                   help="Max radial extent for plots (mm)")
    p.add_argument("--zmax", type=float, default=None,
                   help="Max axial extent for plots (mm)")

    # Numerical
    p.add_argument("--n-loops", type=int, default=200,
                   help="Number of loops for numerical integration (default 200)")

    # Output
    p.add_argument("--output", type=str, default="results.json",
                   help="Output JSON file path (default results.json)")

    return p.parse_args()


def _format_number(value):
    """Format a number: show as int if it's a whole number, otherwise as float."""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def main():
    args = parse_args()

    # Convert mm to m
    R = args.radius * 1e-3
    L = args.length * 1e-3
    NI = args.amp_turns

    # Resolve permeability
    mu_r = get_mu_r(args.core, args.mu_r)

    core_label = args.core if args.core else "custom"
    if args.mu_r is not None:
        core_label = "custom"

    print("emcoil — Electromagnetic Coil Field Solver")
    print("=" * 44)
    print(f"Radius: {args.radius} mm")
    print(f"Length: {args.length} mm")
    print(f"Amp-turns: {NI}")
    print(f"Core: {core_label}")
    print(f"mu_r: {_format_number(mu_r)}")
    print(f"N_loops: {args.n_loops}")

    if args.point is not None:
        x_mm, y_mm, z_mm = args.point
        x, y, z = x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3

        result = compute_field(x, y, z, R, L, NI, mu_r, args.n_loops)

        print(f"\nEvaluation point: ({x_mm}, {y_mm}, {z_mm}) mm")
        print(f"\nField components (Tesla):")
        print(f"  Bx:       {result['Bx']:.6e}")
        print(f"  By:       {result['By']:.6e}")
        print(f"  Bz:       {result['Bz']:.6e}")
        print(f"  |B|:      {result['|B|']:.6e}")
        print(f"  |B_coil|: {result['|B_coil|']:.6e}")
        print(f"  |B_core|: {result['|B_core|']:.6e}")

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "radius_mm": args.radius,
                "length_mm": args.length,
                "amp_turns": NI,
                "core": core_label,
                "mu_r": mu_r,
                "point_mm": [x_mm, y_mm, z_mm],
                "n_loops": args.n_loops,
            },
            "field": {
                "Bx": result["Bx"],
                "By": result["By"],
                "Bz": result["Bz"],
                "|B|": result["|B|"],
                "|B_coil|": result["|B_coil|"],
                "|B_core|": result["|B_core|"],
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.plot_rz or args.plot_xy:
        from emcoil.plotting import plot_rz, plot_xy

        rmax = (args.rmax * 1e-3) if args.rmax else R * 8
        zmax = (args.zmax * 1e-3) if args.zmax else L * 3

        if args.plot_rz:
            plot_rz(R, L, NI, mu_r, rmax, zmax, args.n_loops)

        if args.plot_xy:
            z_slice = (args.z_slice * 1e-3) if args.z_slice else L
            plot_xy(R, L, NI, mu_r, rmax, z_slice, args.n_loops)

    if args.point is None and not args.plot_rz and not args.plot_xy:
        print("\nNo action specified. Use --point, --plot-rz, or --plot-xy.")


if __name__ == "__main__":
    main()
