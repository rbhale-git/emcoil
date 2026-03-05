# Decompose Amp-Turns into Turns + Current Controls

**Date:** 2026-03-04
**Scope:** UI-only change in `app.py`

## Problem

The app has a single AMP-TURNS slider. Users want independent control over
the number of turns (N) and the current (I), with AMP-TURNS showing the
computed product N×I.

## Design

### New controls

| Control | Range | Step | Default | Unit |
|---------|-------|------|---------|------|
| TURNS | 1–5,000 | 1 | 500 | N |
| CURRENT | 0.1–50 | 0.1 | 2.0 | A |

### AMP-TURNS behavior

- Remains visible as a read-only display (slider disabled, input read-only).
- Automatically updated to TURNS × CURRENT whenever either changes.

### Layout order (COIL PARAMETERS panel)

```
RADIUS      ████  10 mm
LENGTH      ████  50 mm
TURNS       ████  500 N
CURRENT     ████  2.0 A
AMP-TURNS   ████  1000 NI  (read-only)
```

### Callback changes

1. Add `turns` and `current` to `SLIDER_INPUT_PAIRS` (before `ampturns`).
2. New callback: TURNS or CURRENT slider/input → update ampturns-slider + ampturns-input.
3. Remove `ampturns` from the slider↔input sync loop (it's now driven by the product callback).
4. Make ampturns slider disabled and input readOnly via component props.
5. Compute callback unchanged — still reads `ampturns-input`.

### What doesn't change

- Physics engine (solver, coil, core) — still receives NI.
- CLI — keeps `--amp-turns`.
- Existing tests unaffected.
