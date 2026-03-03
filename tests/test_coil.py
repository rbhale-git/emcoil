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
