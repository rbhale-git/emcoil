"""Dash web application for the emcoil electromagnetic coil field solver."""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, State, no_update

from emcoil.materials import MATERIAL_PRESETS, get_mu_r
from emcoil.solver import compute_field, compute_field_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATERIAL_OPTIONS = [{"label": k.replace("-", " ").title(), "value": k}
                    for k in MATERIAL_PRESETS] + [{"label": "Custom", "value": "custom"}]

SLIDER_INPUT_PAIRS = [
    {"prefix": "radius",    "label": "Radius (mm)",     "min": 1,  "max": 100,  "step": 1,  "value": 10},
    {"prefix": "length",    "label": "Length (mm)",      "min": 10, "max": 500,  "step": 1,  "value": 50},
    {"prefix": "ampturns",  "label": "Amp-turns",        "min": 10, "max": 10000,"step": 10, "value": 1000},
    {"prefix": "zslice",    "label": "Z-slice (mm)",     "min": -250, "max": 250, "step": 1, "value": 0},
    {"prefix": "gridres",   "label": "Grid resolution",  "min": 20, "max": 100,  "step": 1,  "value": 60},
]

# ---------------------------------------------------------------------------
# Helpers to build slider+input rows
# ---------------------------------------------------------------------------

def _slider_input_row(prefix, label, mn, mx, step, value, hidden=False):
    """Return a div with a label, slider, and numeric input side by side."""
    return html.Div(
        id=f"{prefix}-row",
        style={"display": "none" if hidden else "flex",
               "alignItems": "center", "gap": "10px", "marginBottom": "10px"},
        children=[
            html.Label(label, style={"width": "130px", "flexShrink": "0"}),
            dcc.Slider(
                id=f"{prefix}-slider",
                min=mn, max=mx, step=step, value=value,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            dcc.Input(
                id=f"{prefix}-input",
                type="number", min=mn, max=mx, step=step, value=value,
                style={"width": "80px"},
                debounce=True,
            ),
        ],
    )


def _config_column(suffix, default_material):
    """Return a config column (A or B) with material dropdown and custom mu_r input."""
    return html.Div(style={"flex": "1", "padding": "10px",
                           "border": "1px solid #ccc", "borderRadius": "6px"}, children=[
        html.H4(f"Config {suffix}"),
        html.Label("Core material"),
        dcc.Dropdown(
            id=f"material-{suffix}",
            options=MATERIAL_OPTIONS,
            value=default_material,
            clearable=False,
        ),
        html.Div(
            id=f"custom-mu-row-{suffix}",
            style={"display": "none", "marginTop": "8px"},
            children=[
                html.Label("Custom \u03bc\u1d63: "),
                dcc.Input(id=f"custom-mu-{suffix}", type="number",
                          min=0.001, step=0.1, value=1.0,
                          style={"width": "100px"}, debounce=True),
            ],
        ),
    ])


# ---------------------------------------------------------------------------
# App & Layout
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "emcoil \u2014 EM Coil Field Solver"

app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px",
                              "margin": "0 auto", "padding": "20px"}, children=[
    # Title bar
    html.H1("emcoil \u2014 Electromagnetic Coil Field Solver",
            style={"textAlign": "center", "marginBottom": "5px"}),
    html.Hr(),

    # ---- Control panel ----
    html.Div(style={"display": "flex", "gap": "30px", "flexWrap": "wrap"}, children=[

        # Left column: shared params + view controls
        html.Div(style={"flex": "1", "minWidth": "320px"}, children=[
            html.H3("Coil Parameters"),
            _slider_input_row("radius",   "Radius (mm)",    1,    100,  1,  10),
            _slider_input_row("length",   "Length (mm)",     10,   500,  1,  50),
            _slider_input_row("ampturns", "Amp-turns",       10,   10000,10, 1000),

            html.H3("View Controls", style={"marginTop": "20px"}),
            html.Div(style={"marginBottom": "10px"}, children=[
                html.Label("View: "),
                dcc.RadioItems(
                    id="view-radio",
                    options=[
                        {"label": "r-z plane", "value": "rz"},
                        {"label": "x-y plane", "value": "xy"},
                    ],
                    value="rz",
                    inline=True,
                    style={"display": "inline-block", "marginLeft": "10px"},
                ),
            ]),
            _slider_input_row("zslice", "Z-slice (mm)", -250, 250, 1, 0, hidden=True),
            _slider_input_row("gridres", "Grid resolution", 20, 100, 1, 60),
        ]),

        # Right column: config A & B
        html.Div(style={"flex": "1", "minWidth": "320px"}, children=[
            html.H3("Core Configurations"),
            html.Div(style={"display": "flex", "gap": "15px"}, children=[
                _config_column("A", "air"),
                _config_column("B", "soft-iron"),
            ]),
        ]),
    ]),

    # Compute button + spinner
    html.Div(style={"textAlign": "center", "margin": "20px 0"}, children=[
        html.Button("Compute", id="compute-btn",
                    style={"fontSize": "18px", "padding": "10px 40px",
                           "cursor": "pointer", "backgroundColor": "#2196F3",
                           "color": "white", "border": "none", "borderRadius": "6px"}),
    ]),
    dcc.Loading(id="loading-spinner", type="circle", children=[
        html.Div(id="loading-target"),
    ]),

    # Hidden store for shared params (used by probe callback)
    dcc.Store(id="shared-params-store"),

    # Field maps: side by side
    html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
        html.Div(style={"flex": "1", "minWidth": "400px"}, children=[
            html.H4("Config A", style={"textAlign": "center"}),
            dcc.Graph(id="field-map-A", config={"scrollZoom": False}),
        ]),
        html.Div(style={"flex": "1", "minWidth": "400px"}, children=[
            html.H4("Config B", style={"textAlign": "center"}),
            dcc.Graph(id="field-map-B", config={"scrollZoom": False}),
        ]),
    ]),

    # Probe readout
    html.Div(id="probe-readout", style={"marginTop": "20px", "padding": "15px",
                                         "border": "1px solid #ddd", "borderRadius": "6px",
                                         "backgroundColor": "#fafafa"},
             children=[html.P("Click on a field map to probe field values.",
                              style={"color": "#888"})]),
])


# ---------------------------------------------------------------------------
# Callbacks: slider <-> input sync
# ---------------------------------------------------------------------------

for _pair in SLIDER_INPUT_PAIRS:
    _p = _pair["prefix"]

    @callback(
        Output(f"{_p}-input", "value"),
        Input(f"{_p}-slider", "value"),
        prevent_initial_call=True,
    )
    def _sync_slider_to_input(val, _prefix=_p):  # noqa: E303
        return val

    @callback(
        Output(f"{_p}-slider", "value"),
        Input(f"{_p}-input", "value"),
        prevent_initial_call=True,
    )
    def _sync_input_to_slider(val, _prefix=_p):  # noqa: E303
        return val


# ---------------------------------------------------------------------------
# Callbacks: show/hide custom mu_r
# ---------------------------------------------------------------------------

for _suffix in ("A", "B"):

    @callback(
        Output(f"custom-mu-row-{_suffix}", "style"),
        Input(f"material-{_suffix}", "value"),
    )
    def _toggle_custom_mu(material, _s=_suffix):  # noqa: E303
        if material == "custom":
            return {"display": "block", "marginTop": "8px"}
        return {"display": "none", "marginTop": "8px"}


# ---------------------------------------------------------------------------
# Callback: show/hide z-slice row based on view selection
# ---------------------------------------------------------------------------

@callback(
    Output("zslice-row", "style"),
    Input("view-radio", "value"),
)
def _toggle_zslice_row(view):
    if view == "xy":
        return {"display": "flex", "alignItems": "center",
                "gap": "10px", "marginBottom": "10px"}
    return {"display": "none", "alignItems": "center",
            "gap": "10px", "marginBottom": "10px"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
