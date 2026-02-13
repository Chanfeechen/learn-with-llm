"""Experiment orchestration for optimizer comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maze import LocalOptimaConfig, MazeSpec, generate_maze
from objective import (
    LocalOptimaTermConfig,
    ObjectiveWeights,
    PathObjective,
    build_full_control_points,
    sample_polyline,
)
from optimizers_impl import build_optimizer


@dataclass(frozen=True)
class RunConfig:
  seed: int
  grid_size: tuple[int, int, int]
  obstacle_density: float
  start: tuple[int, int, int]
  goal: tuple[int, int, int]
  num_control_points: int
  num_samples: int
  steps: int
  gradient_eps: float
  init_noise_scale: float
  clip_grad_norm: float | None
  blur_passes: int
  weights: ObjectiveWeights
  local_optima: LocalOptimaConfig
  optimizers: dict[str, dict]


def _finite_difference_gradient(
    loss_fn,
    params: np.ndarray,
    eps: float,
) -> np.ndarray:
  """Computes central-difference gradients."""
  grad = np.zeros_like(params)
  for i in range(params.size):
    plus = params.copy()
    minus = params.copy()
    plus[i] += eps
    minus[i] -= eps
    grad[i] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)
  return grad


def _parse_config(raw: dict) -> RunConfig:
  weights = ObjectiveWeights(
      length=float(raw["weights"]["length"]),
      smoothness=float(raw["weights"]["smoothness"]),
      obstacle=float(raw["weights"]["obstacle"]),
      boundary=float(raw["weights"]["boundary"]),
      local_optima=float(raw["weights"]["local_optima"]),
  )
  local_optima = LocalOptimaConfig(
      enabled=bool(raw["local_optima"]["enabled"]),
      count=int(raw["local_optima"]["count"]),
      strength=float(raw["local_optima"]["strength"]),
      radius=float(raw["local_optima"]["radius"]),
      min_distance_from_path=float(raw["local_optima"]["min_distance_from_path"]),
  )
  return RunConfig(
      seed=int(raw["seed"]),
      grid_size=tuple(raw["grid_size"]),
      obstacle_density=float(raw["obstacle_density"]),
      start=tuple(raw["start"]),
      goal=tuple(raw["goal"]),
      num_control_points=int(raw["num_control_points"]),
      num_samples=int(raw["num_samples"]),
      steps=int(raw["steps"]),
      gradient_eps=float(raw["gradient_eps"]),
      init_noise_scale=float(raw["init_noise_scale"]),
      clip_grad_norm=(
          float(raw["clip_grad_norm"])
          if raw.get("clip_grad_norm") is not None
          else None
      ),
      blur_passes=int(raw.get("blur_passes", 3)),
      weights=weights,
      local_optima=local_optima,
      optimizers=dict(raw["optimizers"]),
  )


def _build_initial_params(cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
  total_points = cfg.num_control_points
  if total_points < 2:
    raise ValueError("num_control_points must be at least 2.")

  start = np.asarray(cfg.start, dtype=float)
  goal = np.asarray(cfg.goal, dtype=float)

  alphas = np.linspace(0.0, 1.0, total_points)
  points = (1.0 - alphas[:, None]) * start + alphas[:, None] * goal

  if total_points > 2:
    noise = rng.normal(
        loc=0.0,
        scale=cfg.init_noise_scale,
        size=(total_points - 2, 3),
    )
    points[1:-1] += noise

  upper = np.asarray(cfg.grid_size, dtype=float) - 1.0
  points = np.clip(points, 0.0, upper)
  return points[1:-1].reshape(-1)


def _run_one_optimizer(
    name: str,
    opt_cfg: dict,
    initial_params: np.ndarray,
    objective: PathObjective,
    start: np.ndarray,
    goal: np.ndarray,
    num_samples: int,
    steps: int,
    grad_eps: float,
    clip_grad_norm: float | None,
) -> dict:
  optimizer = build_optimizer(name, opt_cfg)
  params = initial_params.copy()
  history: list[dict] = []
  best_loss = float("inf")
  best_path = None

  def loss_fn(flat_params: np.ndarray) -> float:
    control_points = build_full_control_points(flat_params, start, goal)
    total, _ = objective.evaluate(control_points)
    return total

  for step in range(steps):
    control_points = build_full_control_points(params, start, goal)
    total, terms = objective.evaluate(control_points)

    grad = _finite_difference_gradient(loss_fn, params, grad_eps)
    grad_norm = float(np.linalg.norm(grad))

    if clip_grad_norm is not None and clip_grad_norm > 0.0 and grad_norm > clip_grad_norm:
      grad = grad * (clip_grad_norm / grad_norm)
      grad_norm = float(np.linalg.norm(grad))

    next_params = optimizer.step(params, grad)
    step_norm = float(np.linalg.norm(next_params - params))
    params = next_params

    sampled_path = sample_polyline(control_points, num_samples)
    if total < best_loss:
      best_loss = total
      best_path = sampled_path.copy()

    row = {
      "iter": step,
      "loss": float(total),
      "grad_norm": grad_norm,
      "step_norm": step_norm,
      "length": terms["length"],
      "smoothness": terms["smoothness"],
      "obstacle": terms["obstacle"],
      "boundary": terms["boundary"],
      "local_optima": terms["local_optima"],
    }
    history.append(row)

  final_control = build_full_control_points(params, start, goal)
  final_path = sample_polyline(final_control, num_samples)
  return {
    "name": name,
    "history": history,
    "final_path": final_path,
    "best_path": best_path if best_path is not None else final_path,
    "final_control_points": final_control,
  }


def run_experiment(config_path: Path, output_dir: Path) -> dict:
  """Runs optimizer comparison and writes artifacts."""
  with config_path.open("r", encoding="utf-8") as f:
    raw_cfg = json.load(f)
  cfg = _parse_config(raw_cfg)

  maze_spec = MazeSpec(
      grid_size=cfg.grid_size,
      obstacle_density=cfg.obstacle_density,
      start=cfg.start,
      goal=cfg.goal,
      blur_passes=cfg.blur_passes,
      local_optima=cfg.local_optima,
  )
  maze_data = generate_maze(maze_spec, seed=cfg.seed)

  objective = PathObjective(
      maze=maze_data,
      weights=cfg.weights,
      num_samples=cfg.num_samples,
      local_optima_cfg=LocalOptimaTermConfig(
          enabled=cfg.local_optima.enabled,
          strength=cfg.local_optima.strength,
          radius=cfg.local_optima.radius,
      ),
  )

  rng = np.random.default_rng(cfg.seed)
  initial_params = _build_initial_params(cfg, rng)

  output_dir.mkdir(parents=True, exist_ok=True)
  results: dict[str, dict] = {}

  for name, opt_cfg in cfg.optimizers.items():
    if not bool(opt_cfg.get("enabled", True)):
      continue
    result = _run_one_optimizer(
        name=name,
        opt_cfg=opt_cfg,
        initial_params=initial_params,
        objective=objective,
        start=maze_data.start,
        goal=maze_data.goal,
        num_samples=cfg.num_samples,
        steps=cfg.steps,
        grad_eps=cfg.gradient_eps,
        clip_grad_norm=cfg.clip_grad_norm,
    )
    results[name] = result

  serializable = {
    "config": raw_cfg,
    "maze": {
      "grid_size": list(maze_data.occupancy.shape),
      "obstacle_count": int(np.sum(maze_data.occupancy)),
      "attractors": maze_data.attractors.tolist(),
      "start": maze_data.start.tolist(),
      "goal": maze_data.goal.tolist(),
    },
    "results": {
      name: {
        "history": value["history"],
        "final_control_points": value["final_control_points"].tolist(),
        "final_path": value["final_path"].tolist(),
      }
      for name, value in results.items()
    },
  }

  with (output_dir / "results.json").open("w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2)

  try:
    from visualize import save_metric_plots, save_path_comparison

    histories = {k: v["history"] for k, v in results.items()}
    paths = {k: v["final_path"] for k, v in results.items()}
    save_metric_plots(histories, output_dir / "metrics.png")
    save_path_comparison(
        occupancy=maze_data.occupancy,
        start=maze_data.start,
        goal=maze_data.goal,
        attractors=maze_data.attractors,
        final_paths=paths,
        output_path=output_dir / "paths.png",
    )
  except ModuleNotFoundError:
    serializable["plots"] = {
        "generated": False,
        "reason": "matplotlib is not installed",
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
      json.dump(serializable, f, indent=2)
  return serializable
