# Optimizer Test Case Plan (3D Puzzle)

This document defines a concrete, seed-driven test case and visualization plan for comparing optimizers on the same 3D puzzle.

## Goals
- Provide a deterministic 3D puzzle with a smooth, differentiable objective.
- Make the puzzle solvable while optionally introducing multiple local optima.
- Compare optimizer behaviors via trajectory, loss curves, and stability metrics.

## Standard Input
```json
{
  "seed": 123,
  "grid_size": [32, 32, 32],
  "obstacle_density": 0.20,
  "start": [1, 1, 1],
  "goal": [30, 30, 30],
  "num_control_points": 8,
  "num_samples": 128,
  "weights": {
    "length": 1.0,
    "smoothness": 0.1,
    "obstacle": 10.0,
    "boundary": 5.0,
    "local_optima": 2.0
  },
  "local_optima": {
    "enabled": true,
    "count": 3,
    "strength": 1.0,
    "radius": 3.0,
    "min_distance_from_path": 4.0
  }
}
```

## Problem Representation
- **Maze**: 3D voxel grid with occupied cells as obstacles.
- **Path**: A continuous spline defined by `num_control_points` in 3D.
- **Sampling**: The spline is sampled into `num_samples` points for collision and cost checks.

## Objective Function (Differentiable)
Let `P = {p_i}` be the sampled path points.

1. **Path length**
   - Penalize total path length.
2. **Smoothness**
   - Penalize second differences of sampled points.
3. **Obstacle penalty**
   - Use a soft collision cost via signed distance field (SDF) or distance-to-nearest obstacle.
4. **Boundary penalty**
   - Penalize points outside the grid bounds.
5. **Local optima (optional)**
   - Add attractor basins to create multiple local minima.

Total loss:
- `L = w_len * L_len + w_smooth * L_smooth + w_obs * L_obs + w_bound * L_bound + w_local * L_local`

## Local Optima Injection (Optional)
Goal: introduce multiple attractive basins that are not the global optimum.

Strategy:
- Sample `count` attractor centers from free space.
- Enforce a minimum distance from the straight-line path and from each other.
- Add a negative Gaussian potential (attractor) around each center:
  - `L_local = -sum_k exp(-||p_i - c_k||^2 / (2 * radius^2))`
- Scale by `strength` and `weights.local_optima`.

This creates multiple plausible basins that can trap or slow optimizers.

## Seeded Generation
1. Initialize RNG with `seed`.
2. Generate obstacle grid via Bernoulli occupancy using `obstacle_density`.
3. Force `start` and `goal` cells to be empty.
4. Check connectivity with BFS.
   - If no valid path exists, resample with a deterministic retry count (seed + retry index).
5. Precompute SDF for the obstacle penalty.
6. If `local_optima.enabled`, generate attractor centers with constraints.
7. Generate initial control points by interpolating between start and goal with small seeded noise.

## Visualization Plan
1. **3D Scene (per optimizer)**
   - Obstacles as semi-transparent voxels.
   - Start/goal markers.
   - Current spline path and sampled points.
   - Optional display of attractor centers.
2. **Metrics Plots**
   - Loss vs iteration.
   - Gradient norm vs iteration.
   - Step norm vs iteration.
3. **Comparison Layout**
   - Small multiples with synchronized iteration.
   - Optional animation per optimizer (GIF/MP4).
4. **Snapshot Strategy**
   - Save every N iterations.
   - Always save first, best, and last states.

## Expected Outputs
- `artifacts/plots/` for metrics.
- `artifacts/renders/` for frames.
- `artifacts/animations/` for GIF/MP4.
- JSON log of configuration + metrics per optimizer.

## Implementation Steps
1. Define input schema and config loader.
2. Implement seeded maze generation + BFS validation.
3. Compute SDF for obstacle penalty.
4. Implement spline + sampling.
5. Implement objective terms and gradients.
6. Add local optima attractor field (optional).
7. Implement optimizers (SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW).
8. Build visualization and logging pipeline.
9. Validate determinism with fixed seed.
