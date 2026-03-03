"""Tests for the Dash web application."""

import pytest
import numpy as np
import plotly.graph_objects as go

from app import app, _build_heatmap_rz, _build_heatmap_xy


# ---------------------------------------------------------------------------
# Shared test parameters (small grid for speed)
# ---------------------------------------------------------------------------
R_M = 0.01       # 10 mm radius
L_M = 0.05       # 50 mm length
NI = 1000        # amp-turns
MU_R = 1.0       # air core
N_GRID = 5       # very coarse for fast tests
Z_SLICE_M = 0.0  # centre slice
LABEL = "Test"


class TestBuildFieldMapRZ:
    """Tests for the r-z heatmap builder."""

    def test_returns_plotly_figure(self):
        """_build_heatmap_rz should return a Plotly Figure."""
        fig, _ = _build_heatmap_rz(R_M, L_M, NI, MU_R, N_GRID, LABEL)
        assert isinstance(fig, go.Figure)

    def test_figure_contains_heatmap_trace(self):
        """The returned figure should contain at least one Heatmap trace."""
        fig, _ = _build_heatmap_rz(R_M, L_M, NI, MU_R, N_GRID, LABEL)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) >= 1

    def test_returns_b_mag_array(self):
        """_build_heatmap_rz should return a 2-D B_mag array as second element."""
        _, bmag = _build_heatmap_rz(R_M, L_M, NI, MU_R, N_GRID, LABEL)
        assert isinstance(bmag, np.ndarray)
        assert bmag.ndim == 2
        assert bmag.shape == (N_GRID, N_GRID)

    def test_b_mag_non_negative(self):
        """All field magnitudes should be non-negative."""
        _, bmag = _build_heatmap_rz(R_M, L_M, NI, MU_R, N_GRID, LABEL)
        assert np.all(bmag >= 0)


class TestBuildFieldMapXY:
    """Tests for the x-y heatmap builder."""

    def test_returns_plotly_figure(self):
        """_build_heatmap_xy should return a Plotly Figure."""
        fig, _ = _build_heatmap_xy(R_M, L_M, NI, MU_R, N_GRID, Z_SLICE_M, LABEL)
        assert isinstance(fig, go.Figure)

    def test_figure_contains_heatmap_trace(self):
        """The returned figure should contain at least one Heatmap trace."""
        fig, _ = _build_heatmap_xy(R_M, L_M, NI, MU_R, N_GRID, Z_SLICE_M, LABEL)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) >= 1

    def test_returns_b_mag_array(self):
        """_build_heatmap_xy should return a 2-D B_mag array as second element."""
        _, bmag = _build_heatmap_xy(R_M, L_M, NI, MU_R, N_GRID, Z_SLICE_M, LABEL)
        assert isinstance(bmag, np.ndarray)
        assert bmag.ndim == 2
        assert bmag.shape == (N_GRID, N_GRID)

    def test_b_mag_non_negative(self):
        """All field magnitudes should be non-negative."""
        _, bmag = _build_heatmap_xy(R_M, L_M, NI, MU_R, N_GRID, Z_SLICE_M, LABEL)
        assert np.all(bmag >= 0)


class TestAppLayout:
    """Tests for the Dash app layout."""

    def test_app_has_layout(self):
        """The Dash app should have a layout defined."""
        assert app.layout is not None

    def test_layout_contains_compute_button(self):
        """The layout should contain the 'compute-btn' button."""
        ids = _collect_ids(app.layout)
        assert "compute-btn" in ids

    def test_layout_contains_field_map_a(self):
        """The layout should contain the 'field-map-A' graph."""
        ids = _collect_ids(app.layout)
        assert "field-map-A" in ids

    def test_layout_contains_field_map_b(self):
        """The layout should contain the 'field-map-B' graph."""
        ids = _collect_ids(app.layout)
        assert "field-map-B" in ids


# ---------------------------------------------------------------------------
# Helper: recursively collect component IDs from a Dash layout tree
# ---------------------------------------------------------------------------

def _collect_ids(component):
    """Walk the Dash component tree and return a set of all 'id' values."""
    ids = set()
    if hasattr(component, "id") and component.id is not None:
        ids.add(component.id)
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                ids |= _collect_ids(child)
        elif children is not None:
            ids |= _collect_ids(children)
    return ids
