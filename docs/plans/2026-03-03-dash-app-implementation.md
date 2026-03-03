# Dash Web Application Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive single-page Dash web app for the emcoil field solver with side-by-side config comparison, click-to-probe, and linked slider+input controls.

**Architecture:** A single `app.py` file containing the Dash layout and callbacks. The app imports the existing `emcoil.solver` and `emcoil.materials` modules directly — no physics code changes needed. Plotly heatmaps with cone/quiver overlays replace matplotlib. All state lives in Dash callbacks.

**Tech Stack:** Dash, Plotly, existing emcoil physics modules (numpy, scipy)

---

### Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add dash and plotly to requirements.txt**

```
numpy
scipy
matplotlib
pytest
dash
plotly
```

**Step 2: Install**

Run: `cd ~/Projects/emcoil && pip install dash plotly`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add dash and plotly dependencies"
```

---

### Task 2: App Layout — Control Panel

**Files:**
- Create: `app.py`

**Step 1: Write the app layout with control panel**

Create `app.py` with the Dash layout. This step builds the full control panel with all sliders, inputs, dropdowns, and the compute button. The field maps and probe readout will be empty placeholder divs for now.

```python
# app.py
"""Dash web application for emcoil electromagnetic coil field solver."""

import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback, no_update
import plotly.graph_objects as go

from emcoil.materials import MATERIAL_PRESETS, get_mu_r
from emcoil.solver import compute_field, compute_field_grid

app = Dash(__name__)

MATERIAL_OPTIONS = [{"label": name, "value": name} for name in MATERIAL_PRESETS.keys()]
MATERIAL_OPTIONS.append({"label": "custom", "value": "custom"})


def make_slider_input(id_prefix, label, min_val, max_val, step, default, unit=""):
    """Create a linked slider + numeric input pair."""
    return html.Div([
        html.Label(f"{label} ({unit})" if unit else label,
                   style={"fontWeight": "bold", "marginBottom": "4px"}),
        html.Div([
            dcc.Slider(
                id=f"{id_prefix}-slider",
                min=min_val, max=max_val, step=step, value=default,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            dcc.Input(
                id=f"{id_prefix}-input",
                type="number",
                min=min_val, max=max_val, step=step, value=default,
                style={"width": "90px", "marginLeft": "10px", "textAlign": "center"},
                debounce=True,
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"marginBottom": "12px"})


def make_config_column(config_id, default_material):
    """Create a config column (A or B) with material dropdown and custom mu_r input."""
    return html.Div([
        html.H4(f"Config {config_id}", style={"marginBottom": "8px"}),
        html.Label("Core Material", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id=f"core-{config_id.lower()}",
            options=MATERIAL_OPTIONS,
            value=default_material,
            clearable=False,
            style={"marginBottom": "8px"},
        ),
        html.Div([
            html.Label("Custom \u03bc_r", style={"fontWeight": "bold"}),
            dcc.Input(
                id=f"mu-r-{config_id.lower()}",
                type="number",
                min=0.0001, step=0.1, value=1.0,
                style={"width": "100px", "marginLeft": "10px"},
                debounce=True,
            ),
        ], id=f"custom-mu-r-container-{config_id.lower()}",
           style={"display": "none", "alignItems": "center", "marginBottom": "8px"}),
    ], style={"flex": "1", "padding": "0 16px"})


app.layout = html.Div([
    # Title
    html.H1("emcoil \u2014 Electromagnetic Coil Field Solver",
            style={"textAlign": "center", "marginBottom": "4px"}),
    html.P("Interactive field map with side-by-side comparison",
           style={"textAlign": "center", "color": "#666", "marginTop": "0"}),

    html.Hr(),

    # Control Panel
    html.Div([
        # Shared coil parameters
        html.Div([
            html.H3("Coil Parameters"),
            make_slider_input("radius", "Radius", 1, 100, 0.5, 25, "mm"),
            make_slider_input("length", "Length", 10, 500, 1, 100, "mm"),
            make_slider_input("amp-turns", "Amp-Turns", 10, 10000, 10, 500, "NI"),
        ], style={"flex": "1.5", "padding": "0 16px"}),

        # Config A
        make_config_column("A", "air"),

        # Config B
        make_config_column("B", "soft-iron"),
    ], style={"display": "flex", "marginBottom": "16px"}),

    # View controls
    html.Div([
        html.Div([
            html.Label("View", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.RadioItems(
                id="view-toggle",
                options=[
                    {"label": "r-z plane", "value": "rz"},
                    {"label": "x-y plane", "value": "xy"},
                ],
                value="rz",
                inline=True,
                style={"display": "inline-block"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(
            make_slider_input("z-slice", "Z-slice", -500, 500, 1, 150, "mm"),
            id="z-slice-container",
            style={"display": "none", "flex": "1", "marginLeft": "20px"},
        ),

        html.Div([
            make_slider_input("grid-res", "Grid Resolution", 20, 100, 5, 60, "pts"),
        ], style={"flex": "1", "marginLeft": "20px"}),

        html.Button(
            "Compute",
            id="compute-btn",
            n_clicks=0,
            style={
                "backgroundColor": "#2196F3", "color": "white",
                "border": "none", "padding": "12px 32px", "fontSize": "16px",
                "borderRadius": "4px", "cursor": "pointer", "marginLeft": "20px",
                "alignSelf": "center",
            },
        ),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "16px",
              "padding": "0 16px"}),

    html.Hr(),

    # Loading spinner wrapping field maps
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            # Side-by-side field maps
            html.Div([
                html.Div([
                    html.H3("Config A", style={"textAlign": "center"}),
                    dcc.Graph(id="field-map-a", style={"height": "600px"}),
                ], style={"flex": "1"}),
                html.Div([
                    html.H3("Config B", style={"textAlign": "center"}),
                    dcc.Graph(id="field-map-b", style={"height": "600px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex"}),
        ],
    ),

    html.Hr(),

    # Probe readout
    html.Div([
        html.H3("Probe Readout", style={"textAlign": "center"}),
        html.P("Click on either field map to probe field values at that point.",
               id="probe-instructions",
               style={"textAlign": "center", "color": "#666"}),
        html.Div(id="probe-readout", style={"padding": "0 16px"}),
    ]),

    # Hidden stores for computed grid data
    dcc.Store(id="grid-data-a"),
    dcc.Store(id="grid-data-b"),
    dcc.Store(id="shared-params"),

], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
          "fontFamily": "sans-serif"})


if __name__ == "__main__":
    app.run(debug=True)
```

**Step 2: Verify the app starts**

Run: `cd ~/Projects/emcoil && python app.py`

Open `http://localhost:8050` in a browser. Verify the control panel renders with sliders, inputs, dropdowns, and the Compute button. The field maps will be empty. Kill the server with Ctrl+C.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Dash app layout with control panel"
```

---

### Task 3: Slider-Input Sync Callbacks

**Files:**
- Modify: `app.py`

Add callbacks that bidirectionally sync each slider with its numeric input. When the slider moves, the input updates. When the input changes, the slider snaps.

**Step 1: Add sync callbacks after the layout, before `if __name__`**

Add these callbacks to `app.py` right before the `if __name__ == "__main__":` block:

```python
# --- Callbacks ---

# Slider <-> Input sync for all slider+input pairs
SLIDER_INPUT_PAIRS = ["radius", "length", "amp-turns", "z-slice", "grid-res"]


for _prefix in SLIDER_INPUT_PAIRS:
    @callback(
        Output(f"{_prefix}-input", "value"),
        Input(f"{_prefix}-slider", "value"),
        prevent_initial_call=True,
    )
    def _sync_slider_to_input(slider_val, _p=_prefix):
        return slider_val

    @callback(
        Output(f"{_prefix}-slider", "value"),
        Input(f"{_prefix}-input", "value"),
        prevent_initial_call=True,
    )
    def _sync_input_to_slider(input_val, _p=_prefix):
        return input_val


# Show/hide custom mu_r input based on dropdown selection
for _cfg in ["a", "b"]:
    @callback(
        Output(f"custom-mu-r-container-{_cfg}", "style"),
        Input(f"core-{_cfg}", "value"),
    )
    def _toggle_custom_mu_r(material, _c=_cfg):
        if material == "custom":
            return {"display": "flex", "alignItems": "center", "marginBottom": "8px"}
        return {"display": "none", "alignItems": "center", "marginBottom": "8px"}


# Show/hide z-slice control based on view toggle
@callback(
    Output("z-slice-container", "style"),
    Input("view-toggle", "value"),
)
def toggle_z_slice(view):
    if view == "xy":
        return {"display": "block", "flex": "1", "marginLeft": "20px"}
    return {"display": "none", "flex": "1", "marginLeft": "20px"}
```

**Step 2: Verify sync works**

Run: `cd ~/Projects/emcoil && python app.py`

- Drag a slider, verify the input box updates
- Type a number in the input, verify the slider moves
- Select "custom" in Config A dropdown, verify the custom mu_r input appears
- Select "x-y plane", verify z-slice slider appears
- Kill server

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add slider-input sync and toggle callbacks"
```

---

### Task 4: Compute Callback — Field Map Generation

**Files:**
- Modify: `app.py`

Add the main compute callback that runs when the Compute button is clicked. It reads all parameters, computes the field grid for both configs, and renders Plotly heatmaps with quiver arrows.

**Step 1: Add a helper function for building a Plotly field map figure**

Add this function before the callbacks section in `app.py`:

```python
def build_field_map_rz(r_arr, z_arr, Br, Bz, B_mag, R_m, L_m, mu_r, config_label):
    """Build a Plotly heatmap figure for r-z plane."""
    r_mm = r_arr * 1e3
    z_mm = z_arr * 1e3

    fig = go.Figure()

    # Heatmap of |B|
    fig.add_trace(go.Heatmap(
        x=r_mm, y=z_mm, z=B_mag,
        colorscale="Viridis",
        colorbar=dict(title="|B| (T)"),
        hovertemplate="r: %{x:.1f} mm<br>z: %{y:.1f} mm<br>|B|: %{z:.4e} T<extra></extra>",
    ))

    # Quiver arrows (subsampled)
    n_grid = len(r_arr)
    skip = max(1, n_grid // 12)
    r_sub = r_mm[::skip]
    z_sub = z_mm[::skip]
    Br_sub = Br[::skip, ::skip]
    Bz_sub = Bz[::skip, ::skip]

    # Normalize arrows for visibility
    B_sub_mag = np.sqrt(Br_sub**2 + Bz_sub**2)
    B_sub_mag = np.where(B_sub_mag == 0, 1, B_sub_mag)
    scale = (r_mm[-1] - r_mm[0]) / (len(r_sub) * 1.5)

    for i, zv in enumerate(z_sub):
        for j, rv in enumerate(r_sub):
            if rv == 0 and Br_sub[i, j] == 0 and Bz_sub[i, j] == 0:
                continue
            mag = B_sub_mag[i, j]
            dr = Br_sub[i, j] / mag * scale
            dz = Bz_sub[i, j] / mag * scale
            fig.add_annotation(
                x=rv + dr, y=zv + dz,
                ax=rv, ay=zv,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor="rgba(255,255,255,0.6)",
            )

    # Solenoid outline
    R_mm = R_m * 1e3
    L_mm = L_m * 1e3
    fig.add_shape(
        type="rect",
        x0=0, x1=R_mm, y0=-L_mm / 2, y1=L_mm / 2,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.update_layout(
        xaxis_title="r (mm)",
        yaxis_title="z (mm)",
        title=f"Config {config_label} (\u03bc_r={mu_r})",
        yaxis=dict(scaleanchor="x"),
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig


def build_field_map_xy(x_arr, y_arr, Bx_grid, By_grid, B_mag, R_m, z_slice_m, mu_r, config_label):
    """Build a Plotly heatmap figure for x-y plane."""
    x_mm = x_arr * 1e3
    y_mm = y_arr * 1e3

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_mm, y=y_mm, z=B_mag,
        colorscale="Viridis",
        colorbar=dict(title="|B| (T)"),
        hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>|B|: %{z:.4e} T<extra></extra>",
    ))

    # Quiver arrows
    n_grid = len(x_arr)
    skip = max(1, n_grid // 12)
    x_sub = x_mm[::skip]
    y_sub = y_mm[::skip]
    Bx_sub = Bx_grid[::skip, ::skip]
    By_sub = By_grid[::skip, ::skip]

    B_sub_mag = np.sqrt(Bx_sub**2 + By_sub**2)
    B_sub_mag = np.where(B_sub_mag == 0, 1, B_sub_mag)
    scale = (x_mm[-1] - x_mm[0]) / (len(x_sub) * 1.5)

    for i, yv in enumerate(y_sub):
        for j, xv in enumerate(x_sub):
            mag = B_sub_mag[i, j]
            if mag < 1e-20:
                continue
            dx = Bx_sub[i, j] / mag * scale
            dy = By_sub[i, j] / mag * scale
            fig.add_annotation(
                x=xv + dx, y=yv + dy,
                ax=xv, ay=yv,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor="rgba(255,255,255,0.6)",
            )

    # Solenoid circle
    R_mm = R_m * 1e3
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=R_mm * np.cos(theta), y=R_mm * np.sin(theta),
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        name="Solenoid",
        hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        title=f"Config {config_label} (\u03bc_r={mu_r}) at z={z_slice_m*1e3:.1f}mm",
        yaxis=dict(scaleanchor="x"),
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig
```

**Step 2: Add the compute callback**

Add this callback after the sync callbacks:

```python
@callback(
    Output("field-map-a", "figure"),
    Output("field-map-b", "figure"),
    Output("shared-params", "data"),
    Input("compute-btn", "n_clicks"),
    State("radius-input", "value"),
    State("length-input", "value"),
    State("amp-turns-input", "value"),
    State("core-a", "value"),
    State("mu-r-a", "value"),
    State("core-b", "value"),
    State("mu-r-b", "value"),
    State("view-toggle", "value"),
    State("z-slice-input", "value"),
    State("grid-res-input", "value"),
    prevent_initial_call=True,
)
def compute_field_maps(n_clicks, radius_mm, length_mm, amp_turns,
                       core_a, mu_r_a, core_b, mu_r_b,
                       view, z_slice_mm, n_grid):
    if n_clicks == 0:
        return no_update, no_update, no_update

    R = radius_mm * 1e-3
    L = length_mm * 1e-3
    NI = amp_turns
    n_grid = int(n_grid)

    # Resolve mu_r for each config
    mu_r_val_a = get_mu_r(core_a if core_a != "custom" else None,
                          mu_r_a if core_a == "custom" else None)
    mu_r_val_b = get_mu_r(core_b if core_b != "custom" else None,
                          mu_r_b if core_b == "custom" else None)

    rmax = R * 8
    zmax = L * 3

    shared = {
        "R": R, "L": L, "NI": NI,
        "mu_r_a": mu_r_val_a, "mu_r_b": mu_r_val_b,
        "view": view, "n_grid": n_grid,
    }

    if view == "rz":
        r_arr = np.linspace(0, rmax, n_grid)
        z_arr = np.linspace(-zmax, zmax, n_grid)

        Br_a, Bz_a, B_mag_a = compute_field_grid(r_arr, z_arr, R, L, NI, mu_r_val_a)
        Br_b, Bz_b, B_mag_b = compute_field_grid(r_arr, z_arr, R, L, NI, mu_r_val_b)

        # Shared colorscale
        vmax = max(np.percentile(B_mag_a, 95), np.percentile(B_mag_b, 95))

        fig_a = build_field_map_rz(r_arr, z_arr, Br_a, Bz_a, B_mag_a, R, L, mu_r_val_a, "A")
        fig_b = build_field_map_rz(r_arr, z_arr, Br_b, Bz_b, B_mag_b, R, L, mu_r_val_b, "B")

        fig_a.update_traces(zmin=0, zmax=vmax, selector=dict(type="heatmap"))
        fig_b.update_traces(zmin=0, zmax=vmax, selector=dict(type="heatmap"))

        shared["rmax"] = rmax
        shared["zmax"] = zmax

    else:  # xy
        z_slice = (z_slice_mm or 150) * 1e-3
        x_arr = np.linspace(-rmax, rmax, n_grid)
        y_arr = np.linspace(-rmax, rmax, n_grid)
        X_grid, Y_grid = np.meshgrid(x_arr, y_arr)

        B_mag_a = np.zeros_like(X_grid)
        Bx_a = np.zeros_like(X_grid)
        By_a = np.zeros_like(X_grid)
        B_mag_b = np.zeros_like(X_grid)
        Bx_b = np.zeros_like(X_grid)
        By_b = np.zeros_like(X_grid)

        for i in range(n_grid):
            for j in range(n_grid):
                res_a = compute_field(X_grid[i, j], Y_grid[i, j], z_slice, R, L, NI, mu_r_val_a)
                B_mag_a[i, j] = res_a["|B|"]
                Bx_a[i, j] = res_a["Bx"]
                By_a[i, j] = res_a["By"]

                res_b = compute_field(X_grid[i, j], Y_grid[i, j], z_slice, R, L, NI, mu_r_val_b)
                B_mag_b[i, j] = res_b["|B|"]
                Bx_b[i, j] = res_b["Bx"]
                By_b[i, j] = res_b["By"]

        vmax = max(np.percentile(B_mag_a, 95), np.percentile(B_mag_b, 95))

        fig_a = build_field_map_xy(x_arr, y_arr, Bx_a, By_a, B_mag_a, R, z_slice, mu_r_val_a, "A")
        fig_b = build_field_map_xy(x_arr, y_arr, Bx_b, By_b, B_mag_b, R, z_slice, mu_r_val_b, "B")

        fig_a.update_traces(zmin=0, zmax=vmax, selector=dict(type="heatmap"))
        fig_b.update_traces(zmin=0, zmax=vmax, selector=dict(type="heatmap"))

        shared["z_slice"] = z_slice
        shared["rmax"] = rmax

    return fig_a, fig_b, shared
```

**Step 3: Verify compute works**

Run: `cd ~/Projects/emcoil && python app.py`

- Set Config A = air, Config B = soft-iron
- Click Compute
- Two field maps should appear side by side with shared colorscale
- Soft-iron map should show much stronger field
- Kill server

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add compute callback with side-by-side field maps"
```

---

### Task 5: Click-to-Probe Callback

**Files:**
- Modify: `app.py`

Add the probe callback that fires when clicking on either field map. It computes exact field values at the clicked point for both configs and shows them in a table.

**Step 1: Add the probe callback**

Add this callback after the compute callback:

```python
@callback(
    Output("probe-readout", "children"),
    Output("probe-instructions", "style"),
    Input("field-map-a", "clickData"),
    Input("field-map-b", "clickData"),
    State("shared-params", "data"),
    prevent_initial_call=True,
)
def probe_field(click_a, click_b, shared):
    if shared is None:
        return no_update, no_update

    # Determine which map was clicked
    click = click_a or click_b
    if click is None:
        return no_update, no_update

    point = click["points"][0]
    R = shared["R"]
    L = shared["L"]
    NI = shared["NI"]
    mu_r_a = shared["mu_r_a"]
    mu_r_b = shared["mu_r_b"]
    view = shared["view"]

    if view == "rz":
        r_mm = point["x"]
        z_mm = point["y"]
        x, y, z = r_mm * 1e-3, 0.0, z_mm * 1e-3
        coord_text = f"r = {r_mm:.1f} mm, z = {z_mm:.1f} mm"
    else:
        x_mm = point["x"]
        y_mm = point["y"]
        x, y = x_mm * 1e-3, y_mm * 1e-3
        z = shared.get("z_slice", 0.15)
        coord_text = f"x = {x_mm:.1f} mm, y = {y_mm:.1f} mm, z = {z*1e3:.1f} mm"

    result_a = compute_field(x, y, z, R, L, NI, mu_r_a)
    result_b = compute_field(x, y, z, R, L, NI, mu_r_b)

    def fmt(val):
        return f"{val:.4e}"

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Component"),
            html.Th(f"Config A (\u03bc_r={mu_r_a})"),
            html.Th(f"Config B (\u03bc_r={mu_r_b})"),
        ])),
        html.Tbody([
            html.Tr([html.Td("Bx (T)"), html.Td(fmt(result_a["Bx"])), html.Td(fmt(result_b["Bx"]))]),
            html.Tr([html.Td("By (T)"), html.Td(fmt(result_a["By"])), html.Td(fmt(result_b["By"]))]),
            html.Tr([html.Td("Bz (T)"), html.Td(fmt(result_a["Bz"])), html.Td(fmt(result_b["Bz"]))]),
            html.Tr([html.Td("|B| (T)"), html.Td(fmt(result_a["|B|"])), html.Td(fmt(result_b["|B|"]))]),
            html.Tr([html.Td("|B_coil| (T)"), html.Td(fmt(result_a["|B_coil|"])), html.Td(fmt(result_b["|B_coil|"]))]),
            html.Tr([html.Td("|B_core| (T)"), html.Td(fmt(result_a["|B_core|"])), html.Td(fmt(result_b["|B_core|"]))]),
        ]),
    ], style={
        "width": "100%", "maxWidth": "600px", "margin": "0 auto",
        "borderCollapse": "collapse", "textAlign": "center",
    })

    # Add borders to cells via inline style
    header = html.Div([
        html.H4(f"Probe at: {coord_text}", style={"textAlign": "center"}),
        table,
    ])

    return header, {"display": "none"}
```

**Step 2: Verify probing works**

Run: `cd ~/Projects/emcoil && python app.py`

- Click Compute first
- Click on either field map
- The probe readout table should appear below with Bx, By, Bz, |B|, |B_coil|, |B_core| for both configs
- Kill server

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add click-to-probe with side-by-side field readout"
```

---

### Task 6: App Tests

**Files:**
- Create: `tests/test_app.py`

**Step 1: Write the tests**

```python
# tests/test_app.py
"""Tests for the Dash web application."""

import pytest
from app import app, build_field_map_rz, build_field_map_xy
import numpy as np
from emcoil.solver import compute_field_grid, compute_field


class TestBuildFieldMapRZ:
    def test_returns_plotly_figure(self):
        """build_field_map_rz should return a plotly Figure."""
        r_arr = np.linspace(0, 0.2, 5)
        z_arr = np.linspace(-0.3, 0.3, 5)
        Br, Bz, B_mag = compute_field_grid(r_arr, z_arr, R=0.025, L=0.1, NI=500, mu_r=1.0, N_loops=20)
        fig = build_field_map_rz(r_arr, z_arr, Br, Bz, B_mag, 0.025, 0.1, 1.0, "A")
        assert fig is not None
        assert hasattr(fig, "to_json")

    def test_has_heatmap_trace(self):
        """Figure should contain a heatmap trace."""
        r_arr = np.linspace(0, 0.2, 5)
        z_arr = np.linspace(-0.3, 0.3, 5)
        Br, Bz, B_mag = compute_field_grid(r_arr, z_arr, R=0.025, L=0.1, NI=500, mu_r=1.0, N_loops=20)
        fig = build_field_map_rz(r_arr, z_arr, Br, Bz, B_mag, 0.025, 0.1, 1.0, "A")
        trace_types = [t.__class__.__name__ for t in fig.data]
        assert "Heatmap" in trace_types


class TestBuildFieldMapXY:
    def test_returns_plotly_figure(self):
        """build_field_map_xy should return a plotly Figure."""
        n = 5
        x_arr = np.linspace(-0.2, 0.2, n)
        y_arr = np.linspace(-0.2, 0.2, n)
        X, Y = np.meshgrid(x_arr, y_arr)
        B_mag = np.zeros((n, n))
        Bx = np.zeros((n, n))
        By = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                res = compute_field(X[i, j], Y[i, j], 0.1, 0.025, 0.1, 500, 1.0, 20)
                B_mag[i, j] = res["|B|"]
                Bx[i, j] = res["Bx"]
                By[i, j] = res["By"]
        fig = build_field_map_xy(x_arr, y_arr, Bx, By, B_mag, 0.025, 0.1, 1.0, "A")
        assert fig is not None
        assert hasattr(fig, "to_json")


class TestAppLayout:
    def test_app_has_layout(self):
        """The Dash app should have a layout defined."""
        assert app.layout is not None

    def test_layout_has_compute_button(self):
        """Layout should contain a compute button."""
        # Check the layout tree for the compute button ID
        layout_str = str(app.layout)
        assert "compute-btn" in layout_str

    def test_layout_has_field_maps(self):
        """Layout should contain both field map graphs."""
        layout_str = str(app.layout)
        assert "field-map-a" in layout_str
        assert "field-map-b" in layout_str
```

**Step 2: Run tests**

Run: `cd ~/Projects/emcoil && python -m pytest tests/test_app.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: add Dash app tests for layout and field map builders"
```

---

### Task 7: Push to Remote

**Step 1: Run full test suite**

Run: `cd ~/Projects/emcoil && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Push**

```bash
cd ~/Projects/emcoil && git push origin main
```
