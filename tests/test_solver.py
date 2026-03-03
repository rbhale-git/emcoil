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
