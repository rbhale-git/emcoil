"""Field map visualization for emcoil."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from emcoil.solver import compute_field_grid


def plot_rz(R, L, NI, mu_r, rmax, zmax, N_loops=200, n_grid=80):
    """Plot axisymmetric r-z cross-section of |B| with field direction arrows.

    Parameters
    ----------
    R : float - Solenoid radius (m).
    L : float - Solenoid length (m).
    NI : float - Total amp-turns.
    mu_r : float - Relative permeability.
    rmax : float - Max radial extent (m).
    zmax : float - Max axial extent (m).
    N_loops : int - Number of loops for integration.
    n_grid : int - Grid resolution per axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r_arr = np.linspace(0, rmax, n_grid)
    z_arr = np.linspace(-zmax, zmax, n_grid)
    R_grid, Z_grid = np.meshgrid(r_arr, z_arr)

    Br, Bz, B_mag = compute_field_grid(r_arr, z_arr, R, L, NI, mu_r, N_loops)

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Contour fill of |B|
    levels = np.linspace(0, np.percentile(B_mag, 95), 50)
    cf = ax.contourf(R_grid * 1e3, Z_grid * 1e3, B_mag, levels=levels,
                     cmap="viridis", extend="max")
    plt.colorbar(cf, ax=ax, label="|B| (T)")

    # Quiver arrows (subsample for clarity)
    skip = max(1, n_grid // 15)
    ax.quiver(
        R_grid[::skip, ::skip] * 1e3,
        Z_grid[::skip, ::skip] * 1e3,
        Br[::skip, ::skip],
        Bz[::skip, ::skip],
        color="white", alpha=0.6, scale_units="inches",
        scale=np.percentile(B_mag, 90) * 4
    )

    # Solenoid outline
    sol_rect = Rectangle(
        (0, -L / 2 * 1e3), R * 1e3, L * 1e3,
        linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(sol_rect)

    ax.set_xlabel("r (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"emcoil: |B| field map (r-z plane)\n"
                 f"R={R*1e3:.1f}mm, L={L*1e3:.1f}mm, NI={NI}, \u03bc_r={mu_r}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig


def plot_xy(R, L, NI, mu_r, rmax, z_slice, N_loops=200, n_grid=80):
    """Plot transverse x-y slice of |B| at a given z.

    Parameters
    ----------
    R, L, NI, mu_r, N_loops : same as plot_rz.
    rmax : float - Max radial extent (m).
    z_slice : float - Z position of the slice (m).
    n_grid : int - Grid resolution per axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from emcoil.solver import compute_field

    x_arr = np.linspace(-rmax, rmax, n_grid)
    y_arr = np.linspace(-rmax, rmax, n_grid)
    X_grid, Y_grid = np.meshgrid(x_arr, y_arr)

    B_mag = np.zeros_like(X_grid)
    Bx_grid = np.zeros_like(X_grid)
    By_grid = np.zeros_like(X_grid)

    for i in range(n_grid):
        for j in range(n_grid):
            result = compute_field(X_grid[i, j], Y_grid[i, j], z_slice,
                                   R, L, NI, mu_r, N_loops)
            B_mag[i, j] = result["|B|"]
            Bx_grid[i, j] = result["Bx"]
            By_grid[i, j] = result["By"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    levels = np.linspace(0, np.percentile(B_mag, 95), 50)
    cf = ax.contourf(X_grid * 1e3, Y_grid * 1e3, B_mag, levels=levels,
                     cmap="viridis", extend="max")
    plt.colorbar(cf, ax=ax, label="|B| (T)")

    # Quiver
    skip = max(1, n_grid // 15)
    ax.quiver(
        X_grid[::skip, ::skip] * 1e3,
        Y_grid[::skip, ::skip] * 1e3,
        Bx_grid[::skip, ::skip],
        By_grid[::skip, ::skip],
        color="white", alpha=0.6
    )

    # Solenoid cross-section circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R * 1e3 * np.cos(theta), R * 1e3 * np.sin(theta),
            "r--", linewidth=2, label="Solenoid")
    ax.legend()

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"emcoil: |B| field map (x-y plane at z={z_slice*1e3:.1f}mm)\n"
                 f"R={R*1e3:.1f}mm, L={L*1e3:.1f}mm, NI={NI}, \u03bc_r={mu_r}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig
