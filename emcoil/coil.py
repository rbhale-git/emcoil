"""Magnetic field of a finite solenoid via Biot-Savart (elliptic integral formulation)."""

import numpy as np
from scipy.special import ellipk, ellipe


MU_0 = 4e-7 * np.pi


def loop_field_cylindrical(rho, z, a, z0, I):
    """Compute B_rho, B_z of a single current loop using elliptic integrals.

    Parameters
    ----------
    rho : float
        Radial distance from axis (m).
    z : float
        Axial position of evaluation point (m).
    a : float
        Loop radius (m).
    z0 : float
        Axial position of the loop (m).
    I : float
        Current in the loop (A).

    Returns
    -------
    B_rho, B_z : float
        Magnetic field components in cylindrical coordinates (T).
    """
    dz = z - z0

    # On-axis special case (avoids division by zero in rho)
    if abs(rho) < 1e-15:
        B_rho = 0.0
        B_z = MU_0 * I * a**2 / (2.0 * (a**2 + dz**2) ** 1.5)
        return B_rho, B_z

    alpha_sq = a**2 + rho**2 + dz**2 - 2.0 * a * rho
    beta_sq = a**2 + rho**2 + dz**2 + 2.0 * a * rho
    beta = np.sqrt(beta_sq)

    k_sq = 1.0 - alpha_sq / beta_sq

    K = ellipk(k_sq)
    E = ellipe(k_sq)

    C = MU_0 * I / (2.0 * np.pi)

    B_rho = (C * dz / (2.0 * alpha_sq * beta * rho)) * (
        (a**2 + rho**2 + dz**2) * E - alpha_sq * K
    )

    B_z = C / (2.0 * alpha_sq * beta) * (
        (a**2 - rho**2 - dz**2) * E + alpha_sq * K
    )

    return B_rho, B_z


def solenoid_field(x, y, z, R, L, NI, N_loops=200):
    """Compute B-field of a finite solenoid at an arbitrary 3D point.

    The solenoid is centered at the origin, extending from -L/2 to +L/2
    along the z-axis, with radius R.

    Parameters
    ----------
    x, y, z : float
        Evaluation point coordinates (m).
    R : float
        Solenoid radius (m).
    L : float
        Solenoid length (m).
    NI : float
        Total amp-turns (N * I).
    N_loops : int
        Number of discrete loops for numerical integration (default 200).

    Returns
    -------
    Bx, By, Bz : float
        Magnetic field components in Cartesian coordinates (T).
    """
    rho = np.sqrt(x**2 + y**2)
    I_per_loop = NI / N_loops

    # Loop positions evenly spaced along solenoid
    z_positions = np.linspace(-L / 2, L / 2, N_loops)

    B_rho_total = 0.0
    B_z_total = 0.0

    for z0 in z_positions:
        Br, Bz_loop = loop_field_cylindrical(rho, z, R, z0, I_per_loop)
        B_rho_total += Br
        B_z_total += Bz_loop

    # Convert cylindrical (B_rho, B_z) to Cartesian (Bx, By, Bz)
    if rho > 1e-15:
        phi = np.arctan2(y, x)
        Bx = B_rho_total * np.cos(phi)
        By = B_rho_total * np.sin(phi)
    else:
        Bx = 0.0
        By = 0.0

    return Bx, By, B_z_total
