# tests/test_plotting.py
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

from emcoil.plotting import plot_rz, plot_xy


class TestPlotting:
    def test_plot_rz_returns_figure(self):
        """plot_rz should return a matplotlib figure."""
        fig = plot_rz(R=0.025, L=0.1, NI=500, mu_r=1.0,
                      rmax=0.2, zmax=0.3, N_loops=50, n_grid=10)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_plot_xy_returns_figure(self):
        """plot_xy should return a matplotlib figure."""
        fig = plot_xy(R=0.025, L=0.1, NI=500, mu_r=1.0,
                      rmax=0.2, z_slice=0.1, N_loops=50, n_grid=10)
        assert fig is not None
        assert hasattr(fig, "savefig")
