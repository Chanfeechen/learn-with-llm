"""Experiment orchestration for mountain-surface optimizer comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maze import (
    AcceptanceConfig,
    GlobalPeakConfig,
    GlobalTrendConfig,
    LocalPeaksConfig,
    NoiseConfig,
    SurfaceConfig,
    TerrainSpec,
    generate_terrain,
)
from objective import ObjectiveWeights, SurfaceObjective
from optimizers_impl import build_optimizer


@dataclass(frozen=True)
class StoppingConfig:
  target_tol: float
  height_tol: float
  loss_tol: float
  progress_tol: float
  grad_tol: float
  step_tol: float
  patience: int


@dataclass(frozen=True)
class AnimationConfig:
  enabled: bool
  fps: int
  stride: int
  max_frames: int


@dataclass(frozen=True)
class RunConfig:
  seed: int
  domain: tuple[float, float]
  grid_resolution: int
  max_iters: int
  gradient_eps: float
  init_noise_scale: float
  clip_grad_norm: float | None
  initial_point: tuple[float, float] | None
  weights: ObjectiveWeights
  stopping: StoppingConfig
  animation: AnimationConfig
  surface: SurfaceConfig
  optimizers: dict[str, dict]


def _finite_difference_gradient(loss_fn, point: np.ndarray, eps: float) -> np.ndarray:
  """Central-difference gradient for 2D points."""
  grad = np.zeros_like(point)
  for i in range(point.size):
    plus = point.copy()
    minus = point.copy()
    plus[i] += eps
    minus[i] -= eps
    grad[i] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)
  return grad


def _parse_config(raw: dict) -> RunConfig:
  weights_raw = raw["weights"]
  weights = ObjectiveWeights(
      height=float(weights_raw.get("height", 1.0)),
      boundary=float(weights_raw.get("boundary", 2.0)),
  )

  stop_raw = raw["stopping"]
  stopping = StoppingConfig(
      target_tol=float(stop_raw.get("target_tol", 1e-3)),
      height_tol=float(stop_raw.get("height_tol", 1e-3)),
      loss_tol=float(stop_raw.get("loss_tol", 1e-8)),
      progress_tol=float(stop_raw.get("progress_tol", 1e-5)),
      grad_tol=float(stop_raw.get("grad_tol", 1e-5)),
      step_tol=float(stop_raw.get("step_tol", 1e-6)),
      patience=int(stop_raw.get("patience", 30)),
  )

  animation_raw = raw.get("animation", {})
  animation = AnimationConfig(
      enabled=bool(animation_raw.get("enabled", True)),
      fps=int(animation_raw.get("fps", 10)),
      stride=int(animation_raw.get("stride", 1)),
      max_frames=int(animation_raw.get("max_frames", 240)),
  )

  surface_raw = raw["surface"]
  global_peak_raw = surface_raw["global_peak"]
  local_peaks_raw = surface_raw["local_peaks"]
  noise_raw = surface_raw.get("noise", {})
  trend_raw = surface_raw.get("global_trend", {})
  acceptance_raw = surface_raw.get("acceptance", {})

  surface = SurfaceConfig(
      global_peak=GlobalPeakConfig(
          amp=float(global_peak_raw["amp"]),
          sigma=float(global_peak_raw["sigma"]),
      ),
      local_peaks=LocalPeaksConfig(
          count=int(local_peaks_raw["count"]),
          amp_range=(
              float(local_peaks_raw["amp_range"][0]),
              float(local_peaks_raw["amp_range"][1]),
          ),
          sigma_range=(
              float(local_peaks_raw["sigma_range"][0]),
              float(local_peaks_raw["sigma_range"][1]),
          ),
          min_dist_global=float(local_peaks_raw["min_dist_global"]),
          min_dist_local=float(local_peaks_raw["min_dist_local"]),
      ),
      noise=NoiseConfig(
          enabled=bool(noise_raw.get("enabled", True)),
          scale=float(noise_raw.get("scale", 0.03)),
          blur_passes=int(noise_raw.get("blur_passes", 2)),
      ),
      global_trend=GlobalTrendConfig(
          enabled=bool(trend_raw.get("enabled", True)),
          strength=float(trend_raw.get("strength", 0.12)),
      ),
      acceptance=AcceptanceConfig(
          max_peak_shift=float(acceptance_raw.get("max_peak_shift", 0.07)),
      ),
  )

  initial = raw.get("initial_point")
  if initial is None:
    initial_point = None
  else:
    initial_point = (float(initial[0]), float(initial[1]))

  return RunConfig(
      seed=int(raw["seed"]),
      domain=(float(raw["domain"][0]), float(raw["domain"][1])),
      grid_resolution=int(raw["grid_resolution"]),
      max_iters=int(raw["max_iters"]),
      gradient_eps=float(raw["gradient_eps"]),
      init_noise_scale=float(raw.get("init_noise_scale", 0.02)),
      clip_grad_norm=(
          float(raw["clip_grad_norm"])
          if raw.get("clip_grad_norm") is not None
          else None
      ),
      initial_point=initial_point,
      weights=weights,
      stopping=stopping,
      animation=animation,
      surface=surface,
      optimizers=dict(raw["optimizers"]),
  )


def _sample_initial_point(
    cfg: RunConfig,
    target_point: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
  low, high = cfg.domain
  if cfg.initial_point is not None:
    point = np.asarray(cfg.initial_point, dtype=float)
  else:
    domain_size = high - low
    for _ in range(1000):
      candidate = rng.uniform(low, high, size=2)
      if np.linalg.norm(candidate - target_point) >= 0.35 * domain_size:
        point = candidate
        break
    else:
      point = rng.uniform(low, high, size=2)

  point += rng.normal(loc=0.0, scale=cfg.init_noise_scale, size=2)
  return np.clip(point, low, high)


def _run_one_optimizer(
    name: str,
    optimizer_cfg: dict,
    initial_point: np.ndarray,
    objective: SurfaceObjective,
    target_point: np.ndarray,
    target_height: float,
    domain: tuple[float, float],
    max_iters: int,
    gradient_eps: float,
    clip_grad_norm: float | None,
    stopping: StoppingConfig,
    trajectory_stride: int,
) -> dict:
  optimizer = build_optimizer(name, optimizer_cfg)
  low, high = domain

  point = initial_point.copy()
  history: list[dict] = []
  trajectory: list[np.ndarray] = [point.copy()]

  stable_counter = 0
  no_progress_counter = 0
  previous_loss: float | None = None
  best_heights: list[float] = []
  stop_reason = "max_iters"

  def loss_fn(query_point: np.ndarray) -> float:
    loss, _ = objective.evaluate(query_point)
    return loss

  for iter_idx in range(1, max_iters + 1):
    loss, terms = objective.evaluate(point)
    grad = _finite_difference_gradient(loss_fn, point, gradient_eps)
    grad_norm = float(np.linalg.norm(grad))

    if clip_grad_norm is not None and clip_grad_norm > 0.0 and grad_norm > clip_grad_norm:
      grad = grad * (clip_grad_norm / grad_norm)
      grad_norm = float(np.linalg.norm(grad))

    new_point = optimizer.step(point, grad)
    new_point = np.clip(new_point, low, high)
    step_norm = float(np.linalg.norm(new_point - point))
    point = new_point

    new_loss, new_terms = objective.evaluate(point)
    distance_to_optimum = float(np.linalg.norm(point - target_point))
    height_gap = abs(target_height - new_terms["height"])

    row = {
      "iter": iter_idx,
      "loss": float(new_loss),
      "height": float(new_terms["height"]),
      "boundary": float(new_terms["boundary"]),
      "grad_norm": grad_norm,
      "step_norm": step_norm,
      "distance_to_optimum": distance_to_optimum,
      "height_gap": float(height_gap),
    }
    history.append(row)
    best_heights.append(max(row["height"], best_heights[-1] if best_heights else row["height"]))

    if iter_idx % max(1, trajectory_stride) == 0:
      trajectory.append(point.copy())

    reached_optimum = (
        distance_to_optimum <= stopping.target_tol
        or height_gap <= stopping.height_tol
    )
    if reached_optimum:
      stop_reason = "reached_optimum"
      break

    if grad_norm < stopping.grad_tol and step_norm < stopping.step_tol:
      stable_counter += 1
    else:
      stable_counter = 0
    if stable_counter >= stopping.patience:
      stop_reason = "converged"
      break

    if previous_loss is not None and abs(previous_loss - new_loss) < stopping.loss_tol:
      no_progress_counter += 1
    else:
      no_progress_counter = 0
    previous_loss = new_loss

    if no_progress_counter >= stopping.patience:
      stop_reason = "no_progress"
      break

    if len(best_heights) > stopping.patience:
      window_gain = best_heights[-1] - best_heights[-1 - stopping.patience]
      if window_gain < stopping.progress_tol:
        stop_reason = "converged"
        break

  if np.linalg.norm(trajectory[-1] - point) > 1e-12:
    trajectory.append(point.copy())

  final_loss, final_terms = objective.evaluate(point)
  return {
    "name": name,
    "history": history,
    "trajectory": np.vstack(trajectory),
    "stop_reason": stop_reason,
    "iterations": len(history),
    "final_point": point,
    "final_loss": float(final_loss),
    "final_height": float(final_terms["height"]),
    "final_distance_to_optimum": float(np.linalg.norm(point - target_point)),
    "best_height": float(max((row["height"] for row in history), default=final_terms["height"])),
  }


def run_experiment(config_path: Path, output_dir: Path) -> dict:
  """Runs optimizer benchmark and writes artifacts."""
  with config_path.open("r", encoding="utf-8") as f:
    raw_config = json.load(f)
  cfg = _parse_config(raw_config)

  terrain = generate_terrain(
      TerrainSpec(
          domain=cfg.domain,
          grid_resolution=cfg.grid_resolution,
          surface=cfg.surface,
      ),
      seed=cfg.seed,
  )
  objective = SurfaceObjective(terrain=terrain, weights=cfg.weights)

  rng = np.random.default_rng(cfg.seed)
  initial_point = _sample_initial_point(
      cfg=cfg,
      target_point=terrain.global_peak_point,
      rng=rng,
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  results: dict[str, dict] = {}

  for name, optimizer_cfg in cfg.optimizers.items():
    if not bool(optimizer_cfg.get("enabled", True)):
      continue
    results[name] = _run_one_optimizer(
        name=name,
        optimizer_cfg=optimizer_cfg,
        initial_point=initial_point,
        objective=objective,
        target_point=terrain.global_peak_point,
        target_height=terrain.global_peak_height,
        domain=cfg.domain,
        max_iters=cfg.max_iters,
        gradient_eps=cfg.gradient_eps,
        clip_grad_norm=cfg.clip_grad_norm,
        stopping=cfg.stopping,
        trajectory_stride=cfg.animation.stride,
    )

  serializable = {
    "config": raw_config,
    "terrain": {
      "domain": list(cfg.domain),
      "grid_resolution": cfg.grid_resolution,
      "global_peak_point": terrain.global_peak_point.tolist(),
      "global_peak_height": terrain.global_peak_height,
      "planted_global_peak_point": terrain.planted_global_peak_point.tolist(),
      "local_peak_points": terrain.local_peak_points.tolist(),
      "initial_point": initial_point.tolist(),
    },
    "results": {
      name: {
        "stop_reason": value["stop_reason"],
        "iterations": value["iterations"],
        "final_point": value["final_point"].tolist(),
        "final_loss": value["final_loss"],
        "final_height": value["final_height"],
        "final_distance_to_optimum": value["final_distance_to_optimum"],
        "best_height": value["best_height"],
        "history": value["history"],
      }
      for name, value in results.items()
    },
  }

  with (output_dir / "results.json").open("w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2)

  try:
    from visualize import (
      save_metrics,
      save_surface_3d,
      save_surface_animation,
      save_surface_contour,
    )

    trajectories = {name: value["trajectory"] for name, value in results.items()}
    histories = {name: value["history"] for name, value in results.items()}
    stop_reasons = {name: value["stop_reason"] for name, value in results.items()}

    save_surface_3d(
        terrain=terrain,
        trajectories=trajectories,
        output_path=output_dir / "surface_3d.png",
    )
    save_surface_contour(
        terrain=terrain,
        trajectories=trajectories,
        output_path=output_dir / "surface_contour.png",
    )
    save_metrics(histories=histories, output_path=output_dir / "metrics.png")

    if cfg.animation.enabled:
      save_surface_animation(
          terrain=terrain,
          trajectories=trajectories,
          stop_reasons=stop_reasons,
          output_path=output_dir / "optimizer_surface_animation.gif",
          fps=cfg.animation.fps,
          max_frames=cfg.animation.max_frames,
      )
  except Exception as exc:
    serializable["plots"] = {
        "generated": False,
        "reason": str(exc),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
      json.dump(serializable, f, indent=2)

  return serializable
