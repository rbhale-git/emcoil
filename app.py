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
# Callback: Compute field maps
# ---------------------------------------------------------------------------

def _build_heatmap_rz(R_m, L_m, NI, mu_r, n_grid, label):
    """Build a Plotly figure with r-z field magnitude heatmap + quiver arrows."""
    rmax = R_m * 8
    zmax = L_m * 3

    r_arr = np.linspace(0, rmax, n_grid)
    z_arr = np.linspace(-zmax, zmax, n_grid)

    Br, Bz, B_mag = compute_field_grid(r_arr, z_arr, R_m, L_m, NI, mu_r)

    # Convert axes to mm for display
    r_mm = r_arr * 1e3
    z_mm = z_arr * 1e3

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=r_mm, y=z_mm, z=B_mag,
        colorscale="Viridis",
        colorbar=dict(title="|B| (T)"),
        hovertemplate="r = %{x:.1f} mm<br>z = %{y:.1f} mm<br>|B| = %{z:.4e} T<extra></extra>",
    ))

    # Quiver-style arrows (subsampled)
    skip = max(1, n_grid // 12)
    for i in range(0, n_grid, skip):
        for j in range(0, n_grid, skip):
            mag = np.sqrt(Br[i, j]**2 + Bz[i, j]**2)
            if mag < 1e-20:
                continue
            # Normalize arrow length for visibility
            arrow_scale = min(rmax * 1e3 * 0.06, zmax * 1e3 * 0.06)
            dr = Br[i, j] / mag * arrow_scale
            dz = Bz[i, j] / mag * arrow_scale
            fig.add_annotation(
                x=r_mm[j] + dr, y=z_mm[i] + dz,
                ax=r_mm[j], ay=z_mm[i],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor="white",
            )

    # Solenoid outline (rectangle in r-z plane)
    R_mm = R_m * 1e3
    L_mm = L_m * 1e3
    fig.add_shape(
        type="rect",
        x0=0, y0=-L_mm / 2, x1=R_mm, y1=L_mm / 2,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.update_layout(
        title=f"{label}",
        xaxis_title="r (mm)",
        yaxis_title="z (mm)",
        width=600, height=550,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig, B_mag


def _build_heatmap_xy(R_m, L_m, NI, mu_r, n_grid, z_slice_m, label):
    """Build a Plotly figure with x-y field magnitude heatmap at a z-slice."""
    rmax = R_m * 8
    x_arr = np.linspace(-rmax, rmax, n_grid)
    y_arr = np.linspace(-rmax, rmax, n_grid)

    B_mag = np.zeros((n_grid, n_grid))
    Bx_arr = np.zeros((n_grid, n_grid))
    By_arr = np.zeros((n_grid, n_grid))

    for i, yv in enumerate(y_arr):
        for j, xv in enumerate(x_arr):
            result = compute_field(xv, yv, z_slice_m, R_m, L_m, NI, mu_r)
            B_mag[i, j] = result["|B|"]
            Bx_arr[i, j] = result["Bx"]
            By_arr[i, j] = result["By"]

    # Convert to mm
    x_mm = x_arr * 1e3
    y_mm = y_arr * 1e3

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_mm, y=y_mm, z=B_mag,
        colorscale="Viridis",
        colorbar=dict(title="|B| (T)"),
        hovertemplate="x = %{x:.1f} mm<br>y = %{y:.1f} mm<br>|B| = %{z:.4e} T<extra></extra>",
    ))

    # Quiver arrows
    skip = max(1, n_grid // 12)
    arrow_scale = rmax * 1e3 * 0.06
    for i in range(0, n_grid, skip):
        for j in range(0, n_grid, skip):
            mag = np.sqrt(Bx_arr[i, j]**2 + By_arr[i, j]**2)
            if mag < 1e-20:
                continue
            dx = Bx_arr[i, j] / mag * arrow_scale
            dy = By_arr[i, j] / mag * arrow_scale
            fig.add_annotation(
                x=x_mm[j] + dx, y=y_mm[i] + dy,
                ax=x_mm[j], ay=y_mm[i],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor="white",
            )

    # Solenoid circle outline
    R_mm = R_m * 1e3
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=R_mm * np.cos(theta), y=R_mm * np.sin(theta),
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"{label} (z = {z_slice_m * 1e3:.1f} mm)",
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=600, height=550,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig, B_mag


@callback(
    Output("field-map-A", "figure"),
    Output("field-map-B", "figure"),
    Output("shared-params-store", "data"),
    Output("loading-target", "children"),
    Input("compute-btn", "n_clicks"),
    State("radius-input", "value"),
    State("length-input", "value"),
    State("ampturns-input", "value"),
    State("material-A", "value"),
    State("custom-mu-A", "value"),
    State("material-B", "value"),
    State("custom-mu-B", "value"),
    State("view-radio", "value"),
    State("zslice-input", "value"),
    State("gridres-input", "value"),
    prevent_initial_call=True,
)
def _compute(n_clicks, radius_mm, length_mm, ampturns,
             mat_a, custom_mu_a, mat_b, custom_mu_b,
             view, zslice_mm, gridres):
    """Compute and display field maps for both configurations."""
    if n_clicks is None:
        return no_update, no_update, no_update, no_update

    # Convert mm -> m
    R_m = radius_mm * 1e-3
    L_m = length_mm * 1e-3
    NI = ampturns
    n_grid = int(gridres)

    # Resolve mu_r for each config
    mu_r_a = get_mu_r(mat_a, None) if mat_a != "custom" else get_mu_r(None, custom_mu_a)
    mu_r_b = get_mu_r(mat_b, None) if mat_b != "custom" else get_mu_r(None, custom_mu_b)

    if view == "rz":
        fig_a, bmag_a = _build_heatmap_rz(R_m, L_m, NI, mu_r_a, n_grid,
                                           f"Config A ({mat_a}, \u03bc\u1d63={mu_r_a:.4g})")
        fig_b, bmag_b = _build_heatmap_rz(R_m, L_m, NI, mu_r_b, n_grid,
                                           f"Config B ({mat_b}, \u03bc\u1d63={mu_r_b:.4g})")
    else:
        z_slice_m = zslice_mm * 1e-3
        fig_a, bmag_a = _build_heatmap_xy(R_m, L_m, NI, mu_r_a, n_grid, z_slice_m,
                                           f"Config A ({mat_a}, \u03bc\u1d63={mu_r_a:.4g})")
        fig_b, bmag_b = _build_heatmap_xy(R_m, L_m, NI, mu_r_b, n_grid, z_slice_m,
                                           f"Config B ({mat_b}, \u03bc\u1d63={mu_r_b:.4g})")

    # Shared colorscale: same zmin/zmax for both
    global_min = min(float(bmag_a.min()), float(bmag_b.min()))
    global_max = max(float(bmag_a.max()), float(bmag_b.max()))
    for fig in (fig_a, fig_b):
        fig.data[0].zmin = global_min
        fig.data[0].zmax = global_max

    # Store params for probe callback
    store_data = {
        "R_m": R_m, "L_m": L_m, "NI": NI,
        "mu_r_a": mu_r_a, "mu_r_b": mu_r_b,
        "mat_a": mat_a, "mat_b": mat_b,
        "view": view,
        "zslice_mm": zslice_mm,
    }

    return fig_a, fig_b, store_data, ""


# ---------------------------------------------------------------------------
# Callback: Click-to-probe
# ---------------------------------------------------------------------------

@callback(
    Output("probe-readout", "children"),
    Input("field-map-A", "clickData"),
    Input("field-map-B", "clickData"),
    State("shared-params-store", "data"),
    prevent_initial_call=True,
)
def _probe(click_a, click_b, params):
    """Probe field at clicked point on either map and show side-by-side readout."""
    if params is None:
        return no_update

    # Determine which click fired
    click = click_a or click_b
    if click is None:
        return no_update

    point = click["points"][0]
    coord1 = point["x"]  # r or x in mm
    coord2 = point["y"]  # z or y in mm

    R_m = params["R_m"]
    L_m = params["L_m"]
    NI = params["NI"]
    mu_r_a = params["mu_r_a"]
    mu_r_b = params["mu_r_b"]
    view = params["view"]

    # Convert clicked mm coords to m and compute field
    if view == "rz":
        # coord1 = r (mm), coord2 = z (mm)
        r_m = coord1 * 1e-3
        z_m = coord2 * 1e-3
        # In r-z view, evaluate at (x=r, y=0, z=z)
        res_a = compute_field(r_m, 0.0, z_m, R_m, L_m, NI, mu_r_a)
        res_b = compute_field(r_m, 0.0, z_m, R_m, L_m, NI, mu_r_b)
        loc_str = f"r = {coord1:.1f} mm, z = {coord2:.1f} mm"
    else:
        # coord1 = x (mm), coord2 = y (mm)
        x_m = coord1 * 1e-3
        y_m = coord2 * 1e-3
        z_m = params["zslice_mm"] * 1e-3
        res_a = compute_field(x_m, y_m, z_m, R_m, L_m, NI, mu_r_a)
        res_b = compute_field(x_m, y_m, z_m, R_m, L_m, NI, mu_r_b)
        loc_str = f"x = {coord1:.1f} mm, y = {coord2:.1f} mm, z = {params['zslice_mm']:.1f} mm"

    # Build readout table
    fields = ["Bx", "By", "Bz", "|B|", "|B_coil|", "|B_core|"]

    header = html.Tr([
        html.Th("Component", style={"padding": "6px 12px"}),
        html.Th(f"Config A ({params['mat_a']})", style={"padding": "6px 12px"}),
        html.Th(f"Config B ({params['mat_b']})", style={"padding": "6px 12px"}),
    ])

    rows = []
    for f in fields:
        rows.append(html.Tr([
            html.Td(f, style={"padding": "4px 12px", "fontWeight": "bold"}),
            html.Td(f"{res_a[f]:.6e} T", style={"padding": "4px 12px", "fontFamily": "monospace"}),
            html.Td(f"{res_b[f]:.6e} T", style={"padding": "4px 12px", "fontFamily": "monospace"}),
        ]))

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"borderCollapse": "collapse", "width": "100%",
               "border": "1px solid #ccc"},
    )

    return html.Div([
        html.H4(f"Probe readout at {loc_str}"),
        table,
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
