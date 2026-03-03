"""Dash web application for the emcoil electromagnetic coil field solver."""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, State, ctx, no_update

from emcoil.materials import MATERIAL_PRESETS, get_mu_r
from emcoil.solver import compute_field, compute_field_grid

# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------
MATERIAL_OPTIONS = [{"label": k.replace("-", " ").title(), "value": k}
                    for k in MATERIAL_PRESETS] + [{"label": "Custom", "value": "custom"}]

SLIDER_INPUT_PAIRS = [
    {"prefix": "radius",   "label": "RADIUS",     "unit": "mm",  "min": 1,    "max": 100,   "step": 1,  "value": 10},
    {"prefix": "length",   "label": "LENGTH",      "unit": "mm",  "min": 10,   "max": 500,   "step": 1,  "value": 50},
    {"prefix": "ampturns", "label": "AMP-TURNS",   "unit": "NI",  "min": 10,   "max": 10000, "step": 10, "value": 1000},
    {"prefix": "zslice",   "label": "Z-SLICE",     "unit": "mm",  "min": -250, "max": 250,   "step": 1,  "value": 0},
    {"prefix": "gridres",  "label": "GRID RES",    "unit": "pts", "min": 20,   "max": 100,   "step": 1,  "value": 60},
]

# Plotly dark theme template
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(11,17,32,0)",
        plot_bgcolor="#0f1729",
        font=dict(family="IBM Plex Mono, Consolas, monospace", color="#8494a7", size=11),
        title=dict(font=dict(size=13, color="#dfe6f0")),
        xaxis=dict(
            gridcolor="#1a2744", zerolinecolor="#1a2744",
            title=dict(font=dict(color="#64748b")),
            tickfont=dict(color="#506380"),
        ),
        yaxis=dict(
            gridcolor="#1a2744", zerolinecolor="#1a2744",
            title=dict(font=dict(color="#64748b")),
            tickfont=dict(color="#506380"),
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color="#506380"),
                title=dict(font=dict(color="#64748b")),
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
        ),
    ),
)

# Custom colorscale: deep navy -> teal -> green -> amber -> white
FIELD_COLORSCALE = [
    [0.0,  "#0a0e17"],
    [0.05, "#0b1a2e"],
    [0.15, "#0d3b5e"],
    [0.3,  "#0e7c6b"],
    [0.5,  "#00e5b0"],
    [0.7,  "#7dde92"],
    [0.85, "#f0b429"],
    [0.95, "#f5d67b"],
    [1.0,  "#fffdf0"],
]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _slider_input_row(prefix, label, unit, mn, mx, step, value, hidden=False):
    """Instrument-style parameter row: label | slider | value+unit."""
    return html.Div(
        id=f"{prefix}-row",
        style={
            "display": "none" if hidden else "flex",
            "alignItems": "center",
            "gap": "12px",
            "marginBottom": "8px",
            "padding": "6px 0",
        },
        children=[
            # Label
            html.Span(
                label,
                style={
                    "width": "90px", "flexShrink": "0",
                    "fontFamily": "var(--font-data)", "fontSize": "10px",
                    "fontWeight": "500", "letterSpacing": "0.1em",
                    "color": "var(--text-muted)",
                },
            ),
            # Slider
            html.Div(
                dcc.Slider(
                    id=f"{prefix}-slider",
                    min=mn, max=mx, step=step, value=value,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="mouseup",
                ),
                style={"flex": "1"},
            ),
            # Numeric input + unit
            html.Div([
                dcc.Input(
                    id=f"{prefix}-input",
                    type="number", min=mn, max=mx, step=step, value=value,
                    style={"width": "80px"},
                    debounce=True,
                ),
                html.Span(
                    unit,
                    style={
                        "fontFamily": "var(--font-data)", "fontSize": "10px",
                        "color": "var(--text-muted)", "marginLeft": "6px",
                        "letterSpacing": "0.05em",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "flexShrink": "0"}),
        ],
    )


def _config_column(suffix, default_material, accent_color):
    """Instrument-style config column with colored indicator."""
    return html.Div(
        className="panel",
        style={"flex": "1", "position": "relative", "overflow": "hidden"},
        children=[
            # Colored top-edge indicator
            html.Div(style={
                "position": "absolute", "top": "0", "left": "0", "right": "0",
                "height": "2px",
                "background": f"linear-gradient(90deg, transparent, {accent_color}, transparent)",
            }),
            # Header with config badge
            html.Div([
                html.Div(
                    suffix,
                    style={
                        "width": "28px", "height": "28px", "borderRadius": "6px",
                        "background": accent_color, "color": "#0a0e17",
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "fontFamily": "var(--font-data)", "fontWeight": "600",
                        "fontSize": "14px",
                    },
                ),
                html.Span(
                    f"CONFIG {suffix}",
                    className="panel-label",
                    style={"marginBottom": "0", "marginLeft": "10px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}),

            # Material dropdown
            html.Div([
                html.Span("MATERIAL", style={
                    "fontFamily": "var(--font-data)", "fontSize": "9px",
                    "letterSpacing": "0.12em", "color": "var(--text-muted)",
                    "marginBottom": "4px", "display": "block",
                }),
                dcc.Dropdown(
                    id=f"material-{suffix}",
                    options=MATERIAL_OPTIONS,
                    value=default_material,
                    clearable=False,
                ),
            ], style={"marginBottom": "10px"}),

            # Custom mu_r (hidden by default)
            html.Div(
                id=f"custom-mu-row-{suffix}",
                style={"display": "none", "marginTop": "8px"},
                children=[
                    html.Span("CUSTOM \u03bc\u1d63", style={
                        "fontFamily": "var(--font-data)", "fontSize": "9px",
                        "letterSpacing": "0.12em", "color": "var(--text-muted)",
                        "marginBottom": "4px", "display": "block",
                    }),
                    dcc.Input(
                        id=f"custom-mu-{suffix}", type="number",
                        min=0.001, step=0.1, value=1.0,
                        style={"width": "100%"},
                        debounce=True,
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# App & Layout
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "emcoil \u2014 EM Coil Field Solver"

app.layout = html.Div(
    style={"maxWidth": "1500px", "margin": "0 auto", "padding": "24px 32px",
           "position": "relative", "zIndex": "1"},
    children=[
        # ---- Header ----
        html.Div([
            html.Div([
                html.Div([
                    # Logo mark
                    html.Div(
                        "\u2300",  # diameter symbol
                        style={
                            "width": "40px", "height": "40px", "borderRadius": "10px",
                            "background": "linear-gradient(135deg, var(--accent), var(--accent-dim))",
                            "color": "var(--bg-deep)", "display": "flex",
                            "alignItems": "center", "justifyContent": "center",
                            "fontSize": "22px", "fontWeight": "700",
                            "boxShadow": "0 0 20px var(--accent-glow)",
                        },
                    ),
                    html.Div([
                        html.H1("emcoil", style={
                            "fontSize": "24px", "fontWeight": "700",
                            "color": "var(--text)", "marginBottom": "0",
                            "letterSpacing": "-0.02em",
                        }),
                        html.Span("ELECTROMAGNETIC COIL FIELD SOLVER", style={
                            "fontFamily": "var(--font-data)", "fontSize": "9px",
                            "letterSpacing": "0.15em", "color": "var(--text-muted)",
                        }),
                    ], style={"marginLeft": "14px"}),
                ], style={"display": "flex", "alignItems": "center"}),

                # Status indicator
                html.Div([
                    html.Div(style={
                        "width": "8px", "height": "8px", "borderRadius": "50%",
                        "background": "var(--accent)",
                        "boxShadow": "0 0 8px var(--accent)",
                        "animation": "pulseGlow 3s ease-in-out infinite",
                    }),
                    html.Span("READY", style={
                        "fontFamily": "var(--font-data)", "fontSize": "10px",
                        "letterSpacing": "0.1em", "color": "var(--accent)",
                        "marginLeft": "8px",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center"}),
        ], style={"marginBottom": "24px"}),

        # Thin separator line
        html.Div(style={
            "height": "1px", "marginBottom": "24px",
            "background": "linear-gradient(90deg, transparent, var(--border-active), var(--accent-glow), var(--border-active), transparent)",
        }),

        # ---- Control Panel ----
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px",
                   "flexWrap": "wrap"},
            children=[
                # Coil Parameters panel
                html.Div(
                    className="panel",
                    style={"flex": "1.6", "minWidth": "380px"},
                    children=[
                        html.Span("COIL PARAMETERS", className="panel-label"),
                        _slider_input_row("radius",   "RADIUS",    "mm",  1,    100,  1,  10),
                        _slider_input_row("length",   "LENGTH",    "mm",  10,   500,  1,  50),
                        _slider_input_row("ampturns", "AMP-TURNS", "NI",  10,   10000, 10, 1000),

                        # Divider
                        html.Div(style={
                            "height": "1px", "margin": "14px 0",
                            "background": "var(--border)",
                        }),

                        html.Span("VIEW CONTROLS", className="panel-label"),
                        html.Div([
                            dcc.RadioItems(
                                id="view-radio",
                                options=[
                                    {"label": " r-z plane", "value": "rz"},
                                    {"label": " x-y plane", "value": "xy"},
                                ],
                                value="rz",
                                inline=True,
                                style={"fontFamily": "var(--font-data)", "fontSize": "12px"},
                                inputStyle={"marginRight": "4px"},
                                labelStyle={
                                    "marginRight": "20px", "cursor": "pointer",
                                    "color": "var(--text-dim)",
                                },
                            ),
                        ], style={"marginBottom": "10px"}),

                        _slider_input_row("zslice", "Z-SLICE", "mm", -250, 250, 1, 0, hidden=True),
                        _slider_input_row("gridres", "GRID RES", "pts", 20, 100, 1, 60),
                    ],
                ),

                # Config columns
                html.Div(
                    style={"flex": "1", "minWidth": "320px", "display": "flex",
                           "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Span("CORE CONFIGURATIONS", className="panel-label",
                                  style={"paddingLeft": "4px"}),
                        html.Div(
                            style={"display": "flex", "gap": "12px", "flex": "1"},
                            children=[
                                _config_column("A", "air", "var(--accent)"),
                                _config_column("B", "soft-iron", "var(--data-amber)"),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ---- Compute Button ----
        html.Div(
            style={"textAlign": "center", "margin": "8px 0 24px"},
            children=[
                html.Button(
                    children=[
                        html.Span("\u25B6", style={"marginRight": "8px", "fontSize": "12px"}),
                        html.Span("COMPUTE FIELD"),
                    ],
                    id="compute-btn",
                    style={
                        "fontFamily": "var(--font-data)", "fontSize": "13px",
                        "fontWeight": "600", "letterSpacing": "0.1em",
                        "padding": "14px 48px",
                        "cursor": "pointer",
                        "background": "linear-gradient(135deg, var(--accent-dim), var(--accent))",
                        "color": "var(--bg-deep)",
                        "border": "none", "borderRadius": "8px",
                        "boxShadow": "0 0 20px var(--accent-glow), 0 4px 12px rgba(0,0,0,0.3)",
                        "transition": "all 0.2s ease",
                    },
                ),
            ],
        ),
        dcc.Loading(id="loading-spinner", type="circle", children=[
            html.Div(id="loading-target"),
        ]),

        # Hidden store
        dcc.Store(id="shared-params-store"),

        # ---- Field Maps: side by side ----
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "24px",
                   "flexWrap": "wrap"},
            children=[
                # Map A
                html.Div(
                    className="panel",
                    style={"flex": "1", "minWidth": "440px", "padding": "12px"},
                    children=[
                        html.Div([
                            html.Div(
                                "A",
                                style={
                                    "width": "22px", "height": "22px", "borderRadius": "5px",
                                    "background": "var(--accent)", "color": "var(--bg-deep)",
                                    "display": "inline-flex", "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontFamily": "var(--font-data)", "fontWeight": "600",
                                    "fontSize": "12px", "marginRight": "8px",
                                },
                            ),
                            html.Span("FIELD MAP", style={
                                "fontFamily": "var(--font-data)", "fontSize": "10px",
                                "letterSpacing": "0.1em", "color": "var(--text-muted)",
                            }),
                        ], style={"marginBottom": "8px", "display": "flex",
                                  "alignItems": "center"}),
                        dcc.Graph(
                            id="field-map-A",
                            config={"scrollZoom": False, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                            style={"height": "540px"},
                        ),
                    ],
                ),
                # Map B
                html.Div(
                    className="panel",
                    style={"flex": "1", "minWidth": "440px", "padding": "12px"},
                    children=[
                        html.Div([
                            html.Div(
                                "B",
                                style={
                                    "width": "22px", "height": "22px", "borderRadius": "5px",
                                    "background": "var(--data-amber)", "color": "var(--bg-deep)",
                                    "display": "inline-flex", "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontFamily": "var(--font-data)", "fontWeight": "600",
                                    "fontSize": "12px", "marginRight": "8px",
                                },
                            ),
                            html.Span("FIELD MAP", style={
                                "fontFamily": "var(--font-data)", "fontSize": "10px",
                                "letterSpacing": "0.1em", "color": "var(--text-muted)",
                            }),
                        ], style={"marginBottom": "8px", "display": "flex",
                                  "alignItems": "center"}),
                        dcc.Graph(
                            id="field-map-B",
                            config={"scrollZoom": False, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                            style={"height": "540px"},
                        ),
                    ],
                ),
            ],
        ),

        # ---- Probe Readout ----
        html.Div(
            id="probe-readout",
            className="panel",
            style={"marginBottom": "24px"},
            children=[
                html.Div([
                    html.Span("\u2316", style={
                        "fontSize": "18px", "color": "var(--text-muted)",
                        "marginRight": "10px",
                    }),
                    html.Span("PROBE READOUT", className="panel-label",
                              style={"marginBottom": "0"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                html.P(
                    "Click on either field map to probe exact field values at that point.",
                    style={
                        "fontFamily": "var(--font-data)", "fontSize": "12px",
                        "color": "var(--text-muted)", "margin": "0",
                    },
                ),
            ],
        ),

        # ---- Footer ----
        html.Div(
            style={
                "textAlign": "center", "paddingTop": "16px",
                "borderTop": "1px solid var(--border)",
            },
            children=[
                html.Span("emcoil v1.0", style={
                    "fontFamily": "var(--font-data)", "fontSize": "10px",
                    "letterSpacing": "0.1em", "color": "var(--text-muted)",
                }),
                html.Span(" \u00b7 ", style={"color": "var(--border-active)", "margin": "0 8px"}),
                html.Span("Biot-Savart + Magnetized Cylinder Model", style={
                    "fontFamily": "var(--font-data)", "fontSize": "10px",
                    "letterSpacing": "0.05em", "color": "var(--text-muted)",
                }),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks: slider <-> input sync
# ---------------------------------------------------------------------------

for _pair in SLIDER_INPUT_PAIRS:
    _p = _pair["prefix"]

    @callback(
        Output(f"{_p}-slider", "value"),
        Output(f"{_p}-input", "value"),
        Input(f"{_p}-slider", "value"),
        Input(f"{_p}-input", "value"),
        prevent_initial_call=True,
    )
    def _sync_slider_input(slider_val, input_val, _prefix=_p):  # noqa: E303
        trigger = ctx.triggered_id
        if trigger == f"{_prefix}-slider":
            return slider_val, slider_val
        return input_val, input_val


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
    base = {"alignItems": "center", "gap": "12px", "marginBottom": "8px", "padding": "6px 0"}
    if view == "xy":
        return {**base, "display": "flex"}
    return {**base, "display": "none"}


# ---------------------------------------------------------------------------
# Field map builders
# ---------------------------------------------------------------------------

def _build_heatmap_rz(R_m, L_m, NI, mu_r, n_grid, label):
    """Build a dark-themed Plotly figure with r-z field magnitude heatmap."""
    rmax = R_m * 8
    zmax = L_m * 3

    r_arr = np.linspace(0, rmax, n_grid)
    z_arr = np.linspace(-zmax, zmax, n_grid)

    Br, Bz, B_mag = compute_field_grid(r_arr, z_arr, R_m, L_m, NI, mu_r)

    r_mm = r_arr * 1e3
    z_mm = z_arr * 1e3

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=r_mm, y=z_mm, z=B_mag,
        colorscale=FIELD_COLORSCALE,
        colorbar=dict(
            title=dict(text="|B| (T)", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#506380", family="IBM Plex Mono, monospace", size=10),
            outlinewidth=0,
            thickness=14,
            len=0.8,
        ),
        hovertemplate=(
            "<b style='color:#00e5b0'>r</b> = %{x:.1f} mm<br>"
            "<b style='color:#00e5b0'>z</b> = %{y:.1f} mm<br>"
            "<b style='color:#f0b429'>|B|</b> = %{z:.4e} T"
            "<extra></extra>"
        ),
    ))

    # Quiver arrows
    skip = max(1, n_grid // 12)
    for i in range(0, n_grid, skip):
        for j in range(0, n_grid, skip):
            mag = np.sqrt(Br[i, j]**2 + Bz[i, j]**2)
            if mag < 1e-20:
                continue
            arrow_scale = min(rmax * 1e3 * 0.06, zmax * 1e3 * 0.06)
            dr = Br[i, j] / mag * arrow_scale
            dz = Bz[i, j] / mag * arrow_scale
            fig.add_annotation(
                x=r_mm[j] + dr, y=z_mm[i] + dz,
                ax=r_mm[j], ay=z_mm[i],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor="rgba(255,255,255,0.35)",
            )

    # Solenoid outline
    R_mm = R_m * 1e3
    L_mm = L_m * 1e3
    fig.add_shape(
        type="rect",
        x0=0, y0=-L_mm / 2, x1=R_mm, y1=L_mm / 2,
        line=dict(color="#00e5b0", width=2, dash="dash"),
    )

    fig.update_layout(
        title=dict(text=label, font=dict(size=12, color="#dfe6f0",
                   family="IBM Plex Mono, monospace")),
        xaxis_title="r (mm)",
        yaxis_title="z (mm)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1120",
        font=dict(family="IBM Plex Mono, monospace", color="#8494a7", size=11),
        xaxis=dict(gridcolor="#1a2744", zerolinecolor="#1a2744",
                   tickfont=dict(color="#506380")),
        yaxis=dict(gridcolor="#1a2744", zerolinecolor="#1a2744",
                   tickfont=dict(color="#506380")),
        margin=dict(l=60, r=20, t=40, b=50),
    )

    return fig, B_mag


def _build_heatmap_xy(R_m, L_m, NI, mu_r, n_grid, z_slice_m, label):
    """Build a dark-themed Plotly figure with x-y field magnitude heatmap."""
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

    x_mm = x_arr * 1e3
    y_mm = y_arr * 1e3

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_mm, y=y_mm, z=B_mag,
        colorscale=FIELD_COLORSCALE,
        colorbar=dict(
            title=dict(text="|B| (T)", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#506380", family="IBM Plex Mono, monospace", size=10),
            outlinewidth=0,
            thickness=14,
            len=0.8,
        ),
        hovertemplate=(
            "<b style='color:#00e5b0'>x</b> = %{x:.1f} mm<br>"
            "<b style='color:#00e5b0'>y</b> = %{y:.1f} mm<br>"
            "<b style='color:#f0b429'>|B|</b> = %{z:.4e} T"
            "<extra></extra>"
        ),
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
                arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor="rgba(255,255,255,0.35)",
            )

    # Solenoid circle
    R_mm = R_m * 1e3
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=R_mm * np.cos(theta), y=R_mm * np.sin(theta),
        mode="lines",
        line=dict(color="#00e5b0", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=f"{label} (z = {z_slice_m * 1e3:.1f} mm)",
            font=dict(size=12, color="#dfe6f0", family="IBM Plex Mono, monospace"),
        ),
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor="#1a2744",
                   zerolinecolor="#1a2744", tickfont=dict(color="#506380")),
        yaxis=dict(gridcolor="#1a2744", zerolinecolor="#1a2744",
                   tickfont=dict(color="#506380")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1120",
        font=dict(family="IBM Plex Mono, monospace", color="#8494a7", size=11),
        margin=dict(l=60, r=20, t=40, b=50),
    )

    return fig, B_mag


# ---------------------------------------------------------------------------
# Callback: Compute field maps
# ---------------------------------------------------------------------------

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

    R_m = radius_mm * 1e-3
    L_m = length_mm * 1e-3
    NI = ampturns
    n_grid = int(gridres)

    mu_r_a = get_mu_r(mat_a, None) if mat_a != "custom" else get_mu_r(None, custom_mu_a)
    mu_r_b = get_mu_r(mat_b, None) if mat_b != "custom" else get_mu_r(None, custom_mu_b)

    if view == "rz":
        fig_a, bmag_a = _build_heatmap_rz(R_m, L_m, NI, mu_r_a, n_grid,
                                           f"Config A \u00b7 {mat_a} \u00b7 \u03bc\u1d63={mu_r_a:.4g}")
        fig_b, bmag_b = _build_heatmap_rz(R_m, L_m, NI, mu_r_b, n_grid,
                                           f"Config B \u00b7 {mat_b} \u00b7 \u03bc\u1d63={mu_r_b:.4g}")
    else:
        z_slice_m = zslice_mm * 1e-3
        fig_a, bmag_a = _build_heatmap_xy(R_m, L_m, NI, mu_r_a, n_grid, z_slice_m,
                                           f"Config A \u00b7 {mat_a} \u00b7 \u03bc\u1d63={mu_r_a:.4g}")
        fig_b, bmag_b = _build_heatmap_xy(R_m, L_m, NI, mu_r_b, n_grid, z_slice_m,
                                           f"Config B \u00b7 {mat_b} \u00b7 \u03bc\u1d63={mu_r_b:.4g}")

    # Shared colorscale
    global_min = min(float(bmag_a.min()), float(bmag_b.min()))
    global_max = max(float(bmag_a.max()), float(bmag_b.max()))
    for fig in (fig_a, fig_b):
        fig.data[0].zmin = global_min
        fig.data[0].zmax = global_max

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
    """Probe field at clicked point on either map and show styled readout."""
    if params is None:
        return no_update

    click = click_a or click_b
    if click is None:
        return no_update

    point = click["points"][0]
    coord1 = point["x"]
    coord2 = point["y"]

    R_m = params["R_m"]
    L_m = params["L_m"]
    NI = params["NI"]
    mu_r_a = params["mu_r_a"]
    mu_r_b = params["mu_r_b"]
    view = params["view"]

    if view == "rz":
        r_m = coord1 * 1e-3
        z_m = coord2 * 1e-3
        res_a = compute_field(r_m, 0.0, z_m, R_m, L_m, NI, mu_r_a)
        res_b = compute_field(r_m, 0.0, z_m, R_m, L_m, NI, mu_r_b)
        loc_str = f"r = {coord1:.1f} mm  \u00b7  z = {coord2:.1f} mm"
    else:
        x_m = coord1 * 1e-3
        y_m = coord2 * 1e-3
        z_m = params["zslice_mm"] * 1e-3
        res_a = compute_field(x_m, y_m, z_m, R_m, L_m, NI, mu_r_a)
        res_b = compute_field(x_m, y_m, z_m, R_m, L_m, NI, mu_r_b)
        loc_str = f"x = {coord1:.1f}  \u00b7  y = {coord2:.1f}  \u00b7  z = {params['zslice_mm']:.1f} mm"

    fields = ["Bx", "By", "Bz", "|B|", "|B_coil|", "|B_core|"]

    header = html.Tr([
        html.Th("COMPONENT"),
        html.Th(f"CONFIG A \u00b7 {params['mat_a']}"),
        html.Th(f"CONFIG B \u00b7 {params['mat_b']}"),
    ])

    rows = []
    for f in fields:
        is_total = f == "|B|"
        rows.append(html.Tr([
            html.Td(f, className="field-label",
                     style={"fontWeight": "600" if is_total else "400"}),
            html.Td(f"{res_a[f]:.6e} T", className="val-a",
                     style={"fontWeight": "600" if is_total else "500"}),
            html.Td(f"{res_b[f]:.6e} T", className="val-b",
                     style={"fontWeight": "600" if is_total else "500"}),
        ], style={"background": "var(--accent-glow-s)" if is_total else "none"}))

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        className="probe-table",
    )

    return html.Div([
        html.Div([
            html.Span("\u2316", style={
                "fontSize": "18px", "color": "var(--accent)",
                "marginRight": "10px",
            }),
            html.Span("PROBE READOUT", className="panel-label",
                      style={"marginBottom": "0"}),
            html.Span(f"\u2014  {loc_str}", style={
                "fontFamily": "var(--font-data)", "fontSize": "12px",
                "color": "var(--text-dim)", "marginLeft": "12px",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
        table,
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
