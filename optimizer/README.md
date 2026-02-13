# Optimizer Experiment Runner

This folder contains a seeded 3D maze testbed for comparing optimizers.

## What is implemented
- Seeded 3D maze generation with connectivity validation.
- Optional local-optima attractors to create non-convex behavior.
- Path objective with length, smoothness, obstacle, boundary, and local-optima terms.
- Optimizers: SGD, SGD with momentum, AdaGrad, RMSProp, Adam, AdamW.
- End-to-end runner that outputs a JSON log of optimizer behavior.
- Optional plots (`metrics.png`, `paths.png`) when matplotlib is installed.

## Run
```bash
python3 /Users/chanfee/projects/learn-with-llm/optimizer/run_experiment.py
```

Optional custom paths:
```bash
python3 /Users/chanfee/projects/learn-with-llm/optimizer/run_experiment.py \
  --config /Users/chanfee/projects/learn-with-llm/optimizer/config/default_config.json \
  --output-dir /Users/chanfee/projects/learn-with-llm/optimizer/artifacts
```

## Key config fields
File: `/Users/chanfee/projects/learn-with-llm/optimizer/config/default_config.json`

- `seed`: reproducibility for maze + initialization.
- `grid_size`, `obstacle_density`: maze structure.
- `num_control_points`, `num_samples`: path representation and sampling.
- `weights.*`: objective term weights.
- `local_optima.enabled`: turn local minima on/off.
- `local_optima.count`, `strength`, `radius`, `min_distance_from_path`: local-optima behavior.
- `optimizers.*`: per-optimizer hyperparameters.

## Outputs
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/results.json`
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/metrics.png` (optional)
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/paths.png` (optional)
