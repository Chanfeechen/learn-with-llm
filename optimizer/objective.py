"""Objective function definitions for path optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maze import MazeData


@dataclass(frozen=True)
class ObjectiveWeights:
  length: float
  smoothness: float
  obstacle: float
  boundary: float
  local_optima: float


@dataclass(frozen=True)
class LocalOptimaTermConfig:
  enabled: bool
  strength: float
  radius: float


def build_full_control_points(
    interior_params: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
  """Builds full control-point sequence with fixed start and goal."""
  interior = interior_params.reshape(-1, 3)
  return np.vstack([start, interior, goal])


def sample_polyline(control_points: np.ndarray, num_samples: int) -> np.ndarray:
  """Uniformly samples a polyline between control points."""
  num_segments = control_points.shape[0] - 1
  if num_segments <= 0:
    return control_points.copy()

  t = np.linspace(0.0, float(num_segments), num_samples)
  left_idx = np.floor(t).astype(int)
  left_idx = np.clip(left_idx, 0, num_segments - 1)
  alpha = t - left_idx

  left = control_points[left_idx]
  right = control_points[left_idx + 1]
  return (1.0 - alpha[:, None]) * left + alpha[:, None] * right


def _trilinear_sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
  """Trilinear interpolation for scalar 3D fields."""
  shape = np.asarray(field.shape, dtype=float)
  clipped = np.clip(points, 0.0, shape - 1.000001)

  x0 = np.floor(clipped[:, 0]).astype(int)
  y0 = np.floor(clipped[:, 1]).astype(int)
  z0 = np.floor(clipped[:, 2]).astype(int)

  x1 = np.clip(x0 + 1, 0, field.shape[0] - 1)
  y1 = np.clip(y0 + 1, 0, field.shape[1] - 1)
  z1 = np.clip(z0 + 1, 0, field.shape[2] - 1)

  xd = clipped[:, 0] - x0
  yd = clipped[:, 1] - y0
  zd = clipped[:, 2] - z0

  c000 = field[x0, y0, z0]
  c100 = field[x1, y0, z0]
  c010 = field[x0, y1, z0]
  c110 = field[x1, y1, z0]
  c001 = field[x0, y0, z1]
  c101 = field[x1, y0, z1]
  c011 = field[x0, y1, z1]
  c111 = field[x1, y1, z1]

  c00 = c000 * (1.0 - xd) + c100 * xd
  c10 = c010 * (1.0 - xd) + c110 * xd
  c01 = c001 * (1.0 - xd) + c101 * xd
  c11 = c011 * (1.0 - xd) + c111 * xd

  c0 = c00 * (1.0 - yd) + c10 * yd
  c1 = c01 * (1.0 - yd) + c11 * yd
  return c0 * (1.0 - zd) + c1 * zd


class PathObjective:
  """Computes total loss and term breakdown for a sampled path."""

  def __init__(
      self,
      maze: MazeData,
      weights: ObjectiveWeights,
      num_samples: int,
      local_optima_cfg: LocalOptimaTermConfig,
  ) -> None:
    self._maze = maze
    self._weights = weights
    self._num_samples = num_samples
    self._local_cfg = local_optima_cfg

  def evaluate(self, control_points: np.ndarray) -> tuple[float, dict[str, float]]:
    sampled = sample_polyline(control_points, self._num_samples)
    length_term = self._length_term(sampled)
    smooth_term = self._smoothness_term(sampled)
    obstacle_term = self._obstacle_term(sampled)
    boundary_term = self._boundary_term(sampled)
    local_term = self._local_optima_term(sampled)

    total = (
        self._weights.length * length_term
        + self._weights.smoothness * smooth_term
        + self._weights.obstacle * obstacle_term
        + self._weights.boundary * boundary_term
        + self._weights.local_optima * local_term
    )

    terms = {
      "length": float(length_term),
      "smoothness": float(smooth_term),
      "obstacle": float(obstacle_term),
      "boundary": float(boundary_term),
      "local_optima": float(local_term),
      "total": float(total),
    }
    return float(total), terms

  def _length_term(self, points: np.ndarray) -> float:
    diffs = points[1:] - points[:-1]
    segment_norms = np.sqrt(np.sum(diffs * diffs, axis=1) + 1e-9)
    return float(np.mean(segment_norms))

  def _smoothness_term(self, points: np.ndarray) -> float:
    if len(points) < 3:
      return 0.0
    second_diff = points[2:] - 2.0 * points[1:-1] + points[:-2]
    return float(np.mean(np.sum(second_diff * second_diff, axis=1)))

  def _obstacle_term(self, points: np.ndarray) -> float:
    values = _trilinear_sample(self._maze.obstacle_field, points)
    return float(np.mean(values * values))

  def _boundary_term(self, points: np.ndarray) -> float:
    upper = np.asarray(self._maze.occupancy.shape, dtype=float) - 1.0
    low_violation = np.clip(-points, 0.0, None)
    high_violation = np.clip(points - upper, 0.0, None)
    violation = low_violation + high_violation
    return float(np.mean(np.sum(violation * violation, axis=1)))

  def _local_optima_term(self, points: np.ndarray) -> float:
    if not self._local_cfg.enabled or len(self._maze.attractors) == 0:
      return 0.0
    radius2 = max(1e-9, self._local_cfg.radius * self._local_cfg.radius)
    deltas = points[:, None, :] - self._maze.attractors[None, :, :]
    dist2 = np.sum(deltas * deltas, axis=2)
    basins = np.exp(-dist2 / (2.0 * radius2))
    # Negative sign turns attractors into local minima.
    score = -np.mean(np.sum(basins, axis=1))
    return float(self._local_cfg.strength * score)
