# emcoil Dash Web Application Design

## Purpose

Interactive browser-based interface for the emcoil electromagnetic coil field solver. Replaces the CLI workflow with a single-page Dash app featuring interactive field maps, click-to-probe, side-by-side config comparison, and linked slider+input controls.

## Layout

Three horizontal bands on a single page:

### Top: Control Panel

- **Shared coil parameters** — each with a slider AND a numeric input field, bidirectionally synced:
  - Radius (1–100 mm)
  - Length (10–500 mm)
  - Amp-turns (10–10000)
- **Two config columns (A and B):**
  - Core material dropdown (air, water, soft-iron, silicon-steel, ferrite, mu-metal, custom)
  - Custom μ_r numeric input (enabled when "custom" selected)
- **View controls:**
  - Toggle: r-z plane / x-y plane
  - Z-slice slider+input (visible only in x-y mode)
  - Grid resolution slider+input (20–100, default 60)
  - "Compute" button with loading spinner

### Middle: Side-by-Side Field Maps

- Two Plotly heatmaps (Config A left, Config B right)
- |B| colorscale with field direction arrows overlaid
- Solenoid outline on each
- Shared colorscale range for visual comparison
- Click on either map to place a probe marker (shown on both maps)

### Bottom: Probe Readout

- Table showing field values at clicked point for both configs:
  - Point coordinates in mm
  - Bx, By, Bz, |B|, |B_coil|, |B_core|

## Interaction Model

1. **Slider ↔ Input sync** — bidirectional; whichever changed last wins
2. **Compute button** — triggers grid computation for both configs; loading spinner while computing
3. **Click-to-probe** — clickData callback computes exact field at point for both configs, populates readout
4. **View toggle** — switching r-z / x-y clears maps and probe

## Tech Stack

- Dash + Plotly (new dependencies)
- Existing emcoil physics modules (unchanged)

## File Structure

- `app.py` — Dash application (layout + callbacks), at project root
- `tests/test_app.py` — Dash app tests
- No changes to existing modules
