# Optimizer Test Case Plan (Mountain Surface Convergence)

This plan defines a realistic mountain-style benchmark where optimizers search for the best point on a continuous surface.

## Goal
- Build an actual terrain surface that can be rendered as `z = f(x, y)`.
- Create one dominant global peak (true optimum) plus multiple smaller local peaks.
- Run optimizers until they converge or they reach the optimum tolerance.

## Surface Model
- Domain: continuous 2D coordinates `(x, y)` in a bounded square (for example `[0, 1] x [0, 1]`).
- Height: `z = f(x, y)`.
- Optimization objective:
  - Option A (recommended): maximize `f(x, y)`.
  - Implementation detail: convert to minimization with `L(x, y) = -f(x, y)` so all current optimizers work unchanged.

## Surface Generation (Seeded)
1. Set random seed.
2. Sample one global peak center `c_global` in interior region.
3. Add a large Gaussian hill for the global peak:
   - high amplitude, moderate width.
4. Add `N_local` smaller Gaussian hills:
   - lower amplitude than global peak,
   - random centers,
   - minimum distance from `c_global` and from each other.
5. Add low-amplitude fractal/perlin-like noise for realism.
6. Optionally add a gentle global trend toward the global peak to make convergence observable.
7. Normalize `f(x, y)` to stable range (for example `[0, 1]`).

## Configuration Schema
```json
{
  "seed": 123,
  "domain": [0.0, 1.0],
  "grid_resolution": 220,
  "max_iters": 2000,
  "surface": {
    "global_peak": {"amp": 1.0, "sigma": 0.10},
    "local_peaks": {
      "count": 6,
      "amp_range": [0.25, 0.70],
      "sigma_range": [0.04, 0.12],
      "min_dist_global": 0.20,
      "min_dist_local": 0.10
    },
    "noise": {"enabled": true, "scale": 0.03},
    "global_trend": {"enabled": true, "strength": 0.12}
  },
  "init": {
    "mode": "fixed_or_seeded_random",
    "start_points": null
  },
  "stopping": {
    "target_tol": 1e-3,
    "loss_tol": 1e-8,
    "grad_tol": 1e-5,
    "step_tol": 1e-6,
    "patience": 30
  }
}
```

## Optimizer Loop
For each optimizer (SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW):
1. Use identical start point(s) for fairness.
2. At each iteration compute gradient of `L(x, y) = -f(x, y)`.
3. Update `(x, y)` and clip to domain bounds.
4. Track metrics:
   - `loss`, `height=f(x,y)`, `grad_norm`, `step_norm`,
   - distance to true optimum point,
   - best-so-far height.

## Stop Conditions (Per Optimizer)
Stop when any condition is met:
1. Reached optimum:
   - `||x_t - x*|| <= target_tol` or `|f(x_t) - f(x*)| <= target_tol`.
2. Converged:
   - `grad_norm < grad_tol` and `step_norm < step_tol` for `patience` consecutive steps.
3. No meaningful progress:
   - `|loss_{t-k} - loss_t| < loss_tol` for `patience` window.
4. Hard cap:
   - `max_iters`.

## Visualization Plan (Actual Surface)
1. Static 3D surface plot (`plot_surface`) with:
   - global peak marker,
   - local peak markers,
   - optimizer trajectories projected on surface.
2. Top-down contour/heatmap with trajectory overlays.
3. Metric plots:
   - height vs iteration,
   - distance-to-optimum vs iteration,
   - grad norm and step norm.
4. Animation:
   - frame-by-frame trajectories moving on the actual mountain surface,
   - one panel per optimizer,
   - synchronized frame index.

## Output Artifacts
- `artifacts/surface_3d.png`
- `artifacts/surface_contour.png`
- `artifacts/metrics.png`
- `artifacts/optimizer_surface_animation.gif`
- `artifacts/results.json`

## Acceptance Criteria
- Surface clearly shows one dominant peak and several smaller peaks.
- Optimizer traces are visible directly on the rendered surface.
- Most optimizers stop by convergence or optimum criteria before `max_iters` on default config.
- Results are reproducible with fixed seed.
