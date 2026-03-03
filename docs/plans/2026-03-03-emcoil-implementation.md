# emcoil Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI tool that computes the external B-field of a finite solenoid with configurable core permeability at arbitrary 3D points, with numeric output and field map plotting.

**Architecture:** Biot-Savart numerical integration (elliptic integrals) for the air-core coil field, superposed with a magnetized cylinder model for the core contribution. Modular design: separate physics modules for coil, core, and combined solver, plus CLI and plotting layers.

**Tech Stack:** Python 3.10+, numpy, scipy (elliptic integrals), matplotlib (field maps), argparse (CLI)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `emcoil/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```
numpy
scipy
matplotlib
pytest
```

**Step 2: Create package init**

```python
# emcoil/__init__.py
```

Empty file, just marks the package.

**Step 3: Create tests init**

```python
# tests/__init__.py
```

**Step 4: Install dependencies**

Run: `pip install -r requirements.txt`

**Step 5: Commit**

```bash
git add requirements.txt emcoil/__init__.py tests/__init__.py
git commit -m "chore: scaffold project with package structure and dependencies"
```

---

### Task 2: Materials Module

**Files:**
- Create: `emcoil/materials.py`
- Test: `tests/test_materials.py`

**Step 1: Write the failing tests**

```python
# tests/test_materials.py
import pytest
from emcoil.materials import get_mu_r, MATERIAL_PRESETS


def test_presets_exist():
    assert "air" in MATERIAL_PRESETS
    assert "water" in MATERIAL_PRESETS
    assert "soft-iron" in MATERIAL_PRESETS
    assert "silicon-steel" in MATERIAL_PRESETS
    assert "ferrite" in MATERIAL_PRESETS
    assert "mu-metal" in MATERIAL_PRESETS


def test_air_mu_r():
    assert get_mu_r("air") == 1.0


def test_water_mu_r():
    assert get_mu_r("water") == pytest.approx(0.999992)


def test_soft_iron_mu_r():
    assert get_mu_r("soft-iron") == 800.0


def test_custom_mu_r_overrides_preset():
    result = get_mu_r("soft-iron", custom_mu_r=1200.0)
    assert result == 1200.0


def test_custom_mu_r_without_preset():
    result = get_mu_r(None, custom_mu_r=42.0)
    assert result == 42.0


def test_no_material_no_custom_defaults_to_air():
    result = get_mu_r(None)
    assert result == 1.0


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="Unknown material"):
        get_mu_r("unobtainium")


def test_custom_mu_r_must_be_positive():
    with pytest.raises(ValueError, match="mu_r must be positive"):
        get_mu_r(None, custom_mu_r=-1.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_materials.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# emcoil/materials.py

MATERIAL_PRESETS = {
    "air": 1.0,
    "water": 0.999992,
    "soft-iron": 800.0,
    "silicon-steel": 4000.0,
    "ferrite": 1000.0,
    "mu-metal": 20000.0,
}


def get_mu_r(material_name=None, custom_mu_r=None):
    """Resolve relative permeability from material name and/or custom value.

    custom_mu_r overrides the preset if both are provided.
    If neither is provided, defaults to air (mu_r=1).
    """
    if custom_mu_r is not None:
        if custom_mu_r <= 0:
            raise ValueError("mu_r must be positive")
        return float(custom_mu_r)

    if material_name is None:
        return 1.0

    if material_name not in MATERIAL_PRESETS:
        raise ValueError(
            f"Unknown material '{material_name}'. "
            f"Available: {', '.join(sorted(MATERIAL_PRESETS.keys()))}"
        )

    return MATERIAL_PRESETS[material_name]
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_materials.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add emcoil/materials.py tests/test_materials.py
git commit -m "feat: add materials module with presets and custom mu_r support"
```

---

### Task 3: Single Current Loop Field (Elliptic Integral Formulation)

**Files:**
- Create: `emcoil/coil.py`
- Test: `tests/test_coil.py`

This is the core physics. A single circular current loop at position z0 on the axis, carrying current I, with radius a. The off-axis field at point (r, z) in cylindrical coordinates uses complete elliptic integrals.

**Reference formulas (cylindrical coordinates, SI units):**

For a loop of radius `a` at axial position `z0`, carrying current `I`, the field at point `(rho, z)`:

```
dz = z - z0
alpha^2 = a^2 + rho^2 + dz^2 - 2*a*rho
beta^2 = a^2 + rho^2 + dz^2 + 2*a*rho
k^2 = 1 - alpha^2 / beta^2
C = mu_0 * I / (2 * pi)

B_rho = (C * dz) / (2 * alpha^2 * beta * rho) * ((a^2 + rho^2 + dz^2) * E(k^2) - alpha^2 * K(k^2))
B_z   = C / (2 * alpha^2 * beta) * ((a^2 - rho^2 - dz^2) * E(k^2) + alpha^2 * K(k^2))
```

Where `K` and `E` are complete elliptic integrals of the first and second kind. On-axis (rho=0) this reduces to the textbook formula.

**Step 1: Write the failing tests**

```python
# tests/test_coil.py
import numpy as np
import pytest
from emcoil.coil import loop_field_cylindrical, solenoid_field


MU_0 = 4e-7 * np.pi


class TestLoopFieldCylindrical:
    """Test single current loop field using elliptic integral formulation."""

    def test_on_axis_center(self):
        """B_z at center of loop should match textbook: mu_0 * I / (2 * a)."""
        a = 0.05  # 50 mm radius in meters
        I = 10.0
        Br, Bz = loop_field_cylindrical(rho=0.0, z=0.0, a=a, z0=0.0, I=I)
        expected_Bz = MU_0 * I / (2 * a)
        assert Bz == pytest.approx(expected_Bz, rel=1e-6)
        assert Br == pytest.approx(0.0, abs=1e-15)

    def test_on_axis_offset(self):
        """B_z on axis at distance z from loop center: mu_0*I*a^2 / (2*(a^2+z^2)^(3/2))."""
        a = 0.05
        I = 10.0
        z = 0.1  # 100 mm away
        Br, Bz = loop_field_cylindrical(rho=0.0, z=z, a=a, z0=0.0, I=I)
        expected_Bz = MU_0 * I * a**2 / (2 * (a**2 + z**2) ** 1.5)
        assert Bz == pytest.approx(expected_Bz, rel=1e-6)
        assert Br == pytest.approx(0.0, abs=1e-15)

    def test_symmetry_z(self):
        """Field magnitude should be symmetric about the loop plane."""
        a = 0.05
        I = 10.0
        _, Bz_pos = loop_field_cylindrical(rho=0.0, z=0.08, a=a, z0=0.0, I=I)
        _, Bz_neg = loop_field_cylindrical(rho=0.0, z=-0.08, a=a, z0=0.0, I=I)
        assert Bz_pos == pytest.approx(Bz_neg, rel=1e-10)

    def test_off_axis_nonzero_Br(self):
        """Off-axis points should have nonzero radial component."""
        a = 0.05
        I = 10.0
        Br, Bz = loop_field_cylindrical(rho=0.03, z=0.05, a=a, z0=0.0, I=I)
        assert abs(Br) > 0
        assert abs(Bz) > 0

    def test_field_decays_with_distance(self):
        """Field should decay as we move further from the loop."""
        a = 0.05
        I = 10.0
        _, Bz_near = loop_field_cylindrical(rho=0.0, z=0.1, a=a, z0=0.0, I=I)
        _, Bz_far = loop_field_cylindrical(rho=0.0, z=0.5, a=a, z0=0.0, I=I)
        assert abs(Bz_near) > abs(Bz_far)


class TestSolenoidField:
    """Test finite solenoid field by summing loop contributions."""

    def test_center_field_matches_infinite_solenoid(self):
        """At the center of a long solenoid, B_z ~ mu_0 * n * I."""
        R = 0.01  # 10 mm radius
        L = 1.0   # 1 m length (long solenoid)
        NI = 5000  # amp-turns
        n = NI / L  # turns per meter (treating NI as N*I combined)
        # For solenoid_field, NI is total amp-turns, so I_per_loop = NI / N_loops
        Bx, By, Bz = solenoid_field(x=0.0, y=0.0, z=0.0, R=R, L=L, NI=NI, N_loops=1000)
        expected = MU_0 * NI / L
        # Long solenoid center should be close to ideal, within ~1%
        assert Bz == pytest.approx(expected, rel=0.01)

    def test_field_outside_weaker_than_inside(self):
        """Field outside the solenoid should be much weaker than inside."""
        R = 0.025
        L = 0.1
        NI = 500
        _, _, Bz_inside = solenoid_field(x=0.0, y=0.0, z=0.0, R=R, L=L, NI=NI)
        _, _, Bz_outside = solenoid_field(x=0.0, y=0.0, z=L, R=R, L=L, NI=NI)
        assert abs(Bz_inside) > abs(Bz_outside)

    def test_returns_3d_components(self):
        """Should return Bx, By, Bz tuple."""
        result = solenoid_field(x=0.03, y=0.04, z=0.1, R=0.025, L=0.1, NI=500)
        assert len(result) == 3

    def test_off_axis_has_transverse_components(self):
        """Off-axis points should have Bx, By components."""
        Bx, By, Bz = solenoid_field(x=0.03, y=0.04, z=0.15, R=0.025, L=0.1, NI=500)
        # At least one transverse component should be nonzero
        assert abs(Bx) > 0 or abs(By) > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_coil.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# emcoil/coil.py
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
        Br, Bz = loop_field_cylindrical(rho, z, R, z0, I_per_loop)
        B_rho_total += Br
        B_z_total += Bz

    # Convert cylindrical (B_rho, B_z) to Cartesian (Bx, By, Bz)
    if rho > 1e-15:
        phi = np.arctan2(y, x)
        Bx = B_rho_total * np.cos(phi)
        By = B_rho_total * np.sin(phi)
    else:
        Bx = 0.0
        By = 0.0

    return Bx, By, B_z_total
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_coil.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add emcoil/coil.py tests/test_coil.py
git commit -m "feat: add coil module with single loop and solenoid field computation"
```

---

### Task 4: Core Module (Magnetized Cylinder)

**Files:**
- Create: `emcoil/core.py`
- Test: `tests/test_core.py`

The core is modeled as a uniformly magnetized cylinder. A uniformly magnetized cylinder is equivalent to a surface current distribution — specifically, bound currents on the lateral surface. This is equivalent to a solenoid with `n*M` effective current, which we can compute by treating the magnetized core as a solenoid with `NI_eff = M * L`.

Alternatively and more directly: the core amplifies the solenoid's own field. The internal H-field of the solenoid is approximately `H = NI/L`. The magnetization is `M = (mu_r - 1) * H`. The magnetized core produces an external field identical to a solenoid of the same geometry with effective amp-turns `NI_core = M * L = (mu_r - 1) * NI`.

We reuse `solenoid_field` from coil.py to compute this contribution.

**Step 1: Write the failing tests**

```python
# tests/test_core.py
import numpy as np
import pytest
from emcoil.core import core_field


MU_0 = 4e-7 * np.pi


class TestCoreField:
    """Test magnetized cylinder field contribution."""

    def test_air_core_returns_zero(self):
        """Air core (mu_r=1) should contribute zero additional field."""
        Bx, By, Bz = core_field(x=0.0, y=0.0, z=0.15, R=0.025, L=0.1, NI=500, mu_r=1.0)
        assert Bx == pytest.approx(0.0, abs=1e-20)
        assert By == pytest.approx(0.0, abs=1e-20)
        assert Bz == pytest.approx(0.0, abs=1e-20)

    def test_iron_core_nonzero(self):
        """Iron core should produce nonzero field contribution."""
        Bx, By, Bz = core_field(x=0.0, y=0.0, z=0.15, R=0.025, L=0.1, NI=500, mu_r=800.0)
        assert abs(Bz) > 0

    def test_higher_mu_r_gives_stronger_field(self):
        """Higher permeability should give stronger core contribution."""
        _, _, Bz_low = core_field(x=0.0, y=0.0, z=0.15, R=0.025, L=0.1, NI=500, mu_r=100.0)
        _, _, Bz_high = core_field(x=0.0, y=0.0, z=0.15, R=0.025, L=0.1, NI=500, mu_r=800.0)
        assert abs(Bz_high) > abs(Bz_low)

    def test_core_field_scales_with_mu_r_minus_1(self):
        """Core contribution should scale linearly with (mu_r - 1)."""
        _, _, Bz_a = core_field(x=0.0, y=0.0, z=0.2, R=0.025, L=0.1, NI=500, mu_r=101.0)
        _, _, Bz_b = core_field(x=0.0, y=0.0, z=0.2, R=0.025, L=0.1, NI=500, mu_r=201.0)
        # Bz_b / Bz_a should be 200/100 = 2.0
        assert (Bz_b / Bz_a) == pytest.approx(2.0, rel=1e-6)

    def test_returns_3d_components(self):
        """Should return Bx, By, Bz tuple."""
        result = core_field(x=0.03, y=0.04, z=0.15, R=0.025, L=0.1, NI=500, mu_r=800.0)
        assert len(result) == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_core.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# emcoil/core.py
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
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_core.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add emcoil/core.py tests/test_core.py
git commit -m "feat: add core module for magnetized cylinder field contribution"
```

---

### Task 5: Solver Module (Superposition)

**Files:**
- Create: `emcoil/solver.py`
- Test: `tests/test_solver.py`

**Step 1: Write the failing tests**

```python
# tests/test_solver.py
import numpy as np
import pytest
from emcoil.solver import compute_field, compute_field_grid


MU_0 = 4e-7 * np.pi


class TestComputeField:
    """Test combined coil + core field computation."""

    def test_air_core_matches_coil_only(self):
        """With air core, total field should equal coil field alone."""
        from emcoil.coil import solenoid_field

        point = (0.0, 0.0, 0.15)
        R, L, NI = 0.025, 0.1, 500

        result = compute_field(*point, R=R, L=L, NI=NI, mu_r=1.0)
        coil_only = solenoid_field(*point, R=R, L=L, NI=NI)

        assert result["Bx"] == pytest.approx(coil_only[0], rel=1e-10)
        assert result["By"] == pytest.approx(coil_only[1], rel=1e-10)
        assert result["Bz"] == pytest.approx(coil_only[2], rel=1e-10)

    def test_iron_core_stronger_than_air(self):
        """Iron core should produce stronger total field than air."""
        R, L, NI = 0.025, 0.1, 500
        point = (0.0, 0.0, 0.15)

        air = compute_field(*point, R=R, L=L, NI=NI, mu_r=1.0)
        iron = compute_field(*point, R=R, L=L, NI=NI, mu_r=800.0)

        assert iron["|B|"] > air["|B|"]

    def test_result_has_all_keys(self):
        """Result dict should contain all expected field components."""
        result = compute_field(0.0, 0.0, 0.15, R=0.025, L=0.1, NI=500, mu_r=1.0)
        for key in ["Bx", "By", "Bz", "|B|", "|B_coil|", "|B_core|"]:
            assert key in result

    def test_magnitude_consistent(self):
        """Magnitude should equal sqrt(Bx^2 + By^2 + Bz^2)."""
        result = compute_field(0.03, 0.04, 0.15, R=0.025, L=0.1, NI=500, mu_r=100.0)
        expected_mag = np.sqrt(result["Bx"]**2 + result["By"]**2 + result["Bz"]**2)
        assert result["|B|"] == pytest.approx(expected_mag, rel=1e-10)


class TestComputeFieldGrid:
    """Test grid-based field computation for plotting."""

    def test_returns_correct_shape(self):
        """Grid output arrays should match input grid shape."""
        r = np.linspace(0, 0.2, 5)
        z = np.linspace(-0.2, 0.2, 7)
        Br, Bz, B_mag = compute_field_grid(r, z, R=0.025, L=0.1, NI=500, mu_r=1.0)
        assert Br.shape == (7, 5)
        assert Bz.shape == (7, 5)
        assert B_mag.shape == (7, 5)

    def test_on_axis_Br_is_zero(self):
        """Radial field on-axis should be zero."""
        r = np.array([0.0])
        z = np.linspace(-0.2, 0.2, 5)
        Br, Bz, B_mag = compute_field_grid(r, z, R=0.025, L=0.1, NI=500, mu_r=1.0)
        assert np.allclose(Br[:, 0], 0.0, atol=1e-15)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_solver.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# emcoil/solver.py
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
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_solver.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add emcoil/solver.py tests/test_solver.py
git commit -m "feat: add solver module combining coil and core field contributions"
```

---

### Task 6: CLI Entry Point

**Files:**
- Create: `cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing tests**

```python
# tests/test_cli.py
import json
import subprocess
import sys
import pytest


def run_cli(*args):
    """Run cli.py and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True, text=True, cwd="."
    )
    return result.returncode, result.stdout, result.stderr


class TestCLI:
    def test_point_evaluation_runs(self):
        """Basic point evaluation should exit 0."""
        code, out, err = run_cli(
            "--radius", "25", "--length", "100", "--amp-turns", "500",
            "--core", "air", "--point", "0", "0", "150"
        )
        assert code == 0
        assert "Bz" in out

    def test_custom_mu_r(self):
        """--mu-r should override --core."""
        code, out, err = run_cli(
            "--radius", "25", "--length", "100", "--amp-turns", "500",
            "--mu-r", "1200", "--point", "0", "0", "150"
        )
        assert code == 0
        assert "mu_r: 1200" in out

    def test_json_output(self):
        """Should write results.json with field data."""
        code, out, err = run_cli(
            "--radius", "25", "--length", "100", "--amp-turns", "500",
            "--core", "soft-iron", "--point", "0", "0", "150",
            "--output", "test_results.json"
        )
        assert code == 0
        with open("test_results.json") as f:
            data = json.load(f)
        assert "Bz" in data["field"]
        assert "|B|" in data["field"]

    def test_missing_required_args(self):
        """Should fail if coil params are missing."""
        code, out, err = run_cli("--point", "0", "0", "150")
        assert code != 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_cli.py -v`
Expected: FAIL — `cli.py` not found

**Step 3: Write implementation**

```python
# cli.py
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
    print(f"mu_r: {mu_r}")
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
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_cli.py -v`
Expected: 3 of 4 tests PASS (test_json_output may need cleanup of test file)

**Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add CLI entry point with point evaluation and JSON output"
```

---

### Task 7: Plotting Module

**Files:**
- Create: `emcoil/plotting.py`
- Test: `tests/test_plotting.py`

**Step 1: Write the failing tests**

```python
# tests/test_plotting.py
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

from emcoil.plotting import plot_rz, plot_xy


class TestPlotting:
    def test_plot_rz_returns_figure(self):
        """plot_rz should return a matplotlib figure."""
        fig = plot_rz(R=0.025, L=0.1, NI=500, mu_r=1.0,
                      rmax=0.2, zmax=0.3, N_loops=50)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_plot_xy_returns_figure(self):
        """plot_xy should return a matplotlib figure."""
        fig = plot_xy(R=0.025, L=0.1, NI=500, mu_r=1.0,
                      rmax=0.2, z_slice=0.1, N_loops=50)
        assert fig is not None
        assert hasattr(fig, "savefig")
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_plotting.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# emcoil/plotting.py
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
                 f"R={R*1e3:.1f}mm, L={L*1e3:.1f}mm, NI={NI}, μ_r={mu_r}")
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
                 f"R={R*1e3:.1f}mm, L={L*1e3:.1f}mm, NI={NI}, μ_r={mu_r}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_plotting.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add emcoil/plotting.py tests/test_plotting.py
git commit -m "feat: add plotting module with r-z and x-y field map visualizations"
```

---

### Task 8: Integration Test & Cleanup

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests for emcoil."""

import json
import subprocess
import sys
import os
import pytest


def run_cli(*args):
    result = subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True, text=True, cwd="."
    )
    return result.returncode, result.stdout, result.stderr


class TestIntegration:
    def test_full_pipeline_air_core(self):
        """Full pipeline: air core, point evaluation, JSON output."""
        code, out, err = run_cli(
            "--radius", "25", "--length", "100", "--amp-turns", "500",
            "--core", "air", "--point", "0", "0", "150",
            "--output", "test_integration.json"
        )
        assert code == 0
        assert "|B|" in out

        with open("test_integration.json") as f:
            data = json.load(f)
        assert data["inputs"]["mu_r"] == 1.0
        assert data["field"]["|B|"] > 0
        os.remove("test_integration.json")

    def test_full_pipeline_iron_core(self):
        """Iron core should give stronger field than air at same point."""
        run_cli("--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "air", "--point", "0", "0", "150",
                "--output", "test_air.json")
        run_cli("--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "soft-iron", "--point", "0", "0", "150",
                "--output", "test_iron.json")

        with open("test_air.json") as f:
            air = json.load(f)
        with open("test_iron.json") as f:
            iron = json.load(f)

        assert iron["field"]["|B|"] > air["field"]["|B|"]

        os.remove("test_air.json")
        os.remove("test_iron.json")

    def test_custom_mu_r_override(self):
        """--mu-r should override --core preset."""
        code, out, err = run_cli(
            "--radius", "25", "--length", "100", "--amp-turns", "500",
            "--core", "air", "--mu-r", "500",
            "--point", "0", "0", "150",
            "--output", "test_override.json"
        )
        assert code == 0

        with open("test_override.json") as f:
            data = json.load(f)
        assert data["inputs"]["mu_r"] == 500.0
        os.remove("test_override.json")
```

**Step 2: Run all tests**

Run: `cd ~/Projects/emcoil && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full CLI pipeline"
```

---

### Task 9: Push to Remote

**Step 1: Push all commits**

```bash
git push -u origin main
```
