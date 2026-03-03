"""Magnetized cylinder field contribution for a solenoid core."""

from emcoil.coil import solenoid_field


def core_field(x, y, z, R, L, NI, mu_r, N_loops=200):
    """Compute the external field contribution from a magnetized core.

    The core magnetization produces a field equivalent to a solenoid
    with effective amp-turns NI_core = (mu_r - 1) * NI.

    Parameters
    ----------
    x, y, z : float
        Evaluation point coordinates (m).
    R : float
        Core radius (same as solenoid radius) (m).
    L : float
        Core length (same as solenoid length) (m).
    NI : float
        Total amp-turns of the solenoid.
    mu_r : float
        Relative permeability of the core material.
    N_loops : int
        Number of discrete loops for numerical integration (default 200).

    Returns
    -------
    Bx, By, Bz : float
        Core field contribution in Cartesian coordinates (T).
    """
    if mu_r == 1.0:
        return 0.0, 0.0, 0.0

    NI_core = (mu_r - 1.0) * NI
    return solenoid_field(x, y, z, R, L, NI_core, N_loops)
