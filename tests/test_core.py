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
