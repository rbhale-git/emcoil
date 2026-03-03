"""Combined solenoid + core field solver."""

import numpy as np
from emcoil.coil import solenoid_field
from emcoil.core import core_field


def compute_field(x, y, z, R, L, NI, mu_r=1.0, N_loops=200):
    """Compute total B-field at a single point.

    Parameters
    ----------
    x, y, z : float
        Evaluation point coordinates (m).
    R : float
        Solenoid radius (m).
    L : float
        Solenoid length (m).
    NI : float
        Total amp-turns.
    mu_r : float
        Relative permeability of core (default 1.0 = air).
    N_loops : int
        Number of discrete loops for integration (default 200).

    Returns
    -------
    dict with keys: Bx, By, Bz, |B|, |B_coil|, |B_core|
    """
    Bx_c, By_c, Bz_c = solenoid_field(x, y, z, R, L, NI, N_loops)
    Bx_m, By_m, Bz_m = core_field(x, y, z, R, L, NI, mu_r, N_loops)

    Bx = Bx_c + Bx_m
    By = By_c + By_m
    Bz = Bz_c + Bz_m

    return {
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
        "|B|": np.sqrt(Bx**2 + By**2 + Bz**2),
        "|B_coil|": np.sqrt(Bx_c**2 + By_c**2 + Bz_c**2),
        "|B_core|": np.sqrt(Bx_m**2 + By_m**2 + Bz_m**2),
    }


def compute_field_grid(r_arr, z_arr, R, L, NI, mu_r=1.0, N_loops=200):
    """Compute field on a 2D axisymmetric grid for plotting.

    Parameters
    ----------
    r_arr : array-like
        1D array of radial distances (m).
    z_arr : array-like
        1D array of axial positions (m).
    R, L, NI, mu_r, N_loops : same as compute_field.

    Returns
    -------
    Br, Bz, B_mag : ndarray
        2D arrays of shape (len(z_arr), len(r_arr)).
    """
    nr = len(r_arr)
    nz = len(z_arr)
    Br = np.zeros((nz, nr))
    Bz = np.zeros((nz, nr))
    B_mag = np.zeros((nz, nr))

    for i, zv in enumerate(z_arr):
        for j, rv in enumerate(r_arr):
            result = compute_field(rv, 0.0, zv, R, L, NI, mu_r, N_loops)
            # Br = Bx when y=0 (since phi=0)
            Br[i, j] = result["Bx"]
            Bz[i, j] = result["Bz"]
            B_mag[i, j] = result["|B|"]

    return Br, Bz, B_mag
