# Optimizer Experiment Runner

This folder runs a mountain-surface benchmark where optimizers search for the global peak.

## What is implemented
- Seeded 2D mountain surface `z = f(x, y)`.
- One dominant global peak and configurable smaller local peaks.
- Real surface rendering in 3D and contour views.
- Optimizer runs with stop-on-convergence or stop-on-optimum criteria.
- Optimizers: SGD, SGD with momentum, AdaGrad, RMSProp, Adam, AdamW.
- End-to-end runner with JSON logs and animation.

## Run
```bash
python /Users/chanfee/projects/learn-with-llm/optimizer/run_experiment.py
```

## Key config
File: `/Users/chanfee/projects/learn-with-llm/optimizer/config/default_config.json`

- `surface.*`: controls global/local peaks and noise.
- `stopping.*`: convergence and optimum stop criteria.
- `animation.*`: GIF sampling and speed.
- `optimizers.*`: per-optimizer hyperparameters.

## Outputs
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/results.json`
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/surface_3d.png`
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/surface_contour.png`
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/metrics.png`
- `/Users/chanfee/projects/learn-with-llm/optimizer/artifacts/optimizer_surface_animation.gif`
