"""Seeded 3D maze generation for optimizer experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalOptimaConfig:
  enabled: bool
  count: int
  strength: float
  radius: float
  min_distance_from_path: float


@dataclass(frozen=True)
class MazeSpec:
  grid_size: tuple[int, int, int]
  obstacle_density: float
  start: tuple[int, int, int]
  goal: tuple[int, int, int]
  blur_passes: int
  local_optima: LocalOptimaConfig


@dataclass(frozen=True)
class MazeData:
  occupancy: np.ndarray
  obstacle_field: np.ndarray
  field_gradient: np.ndarray
  start: np.ndarray
  goal: np.ndarray
  attractors: np.ndarray


def _neighbors(idx: tuple[int, int, int], shape: tuple[int, int, int]):
  x, y, z = idx
  max_x, max_y, max_z = shape
  candidates = (
      (x - 1, y, z),
      (x + 1, y, z),
      (x, y - 1, z),
      (x, y + 1, z),
      (x, y, z - 1),
      (x, y, z + 1),
  )
  for nx, ny, nz in candidates:
    if 0 <= nx < max_x and 0 <= ny < max_y and 0 <= nz < max_z:
      yield (nx, ny, nz)


def _is_connected(
    occupancy: np.ndarray,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> bool:
  """Checks if start and goal are connected through free cells."""
  if occupancy[start] or occupancy[goal]:
    return False

  visited = np.zeros_like(occupancy, dtype=bool)
  queue: deque[tuple[int, int, int]] = deque([start])
  visited[start] = True

  while queue:
    current = queue.popleft()
    if current == goal:
      return True
    for nxt in _neighbors(current, occupancy.shape):
      if visited[nxt] or occupancy[nxt]:
        continue
      visited[nxt] = True
      queue.append(nxt)

  return False


def _blur3d(volume: np.ndarray) -> np.ndarray:
  """Applies a 3x3x3 mean filter using only NumPy."""
  padded = np.pad(volume, 1, mode="edge")
  out = np.zeros_like(volume, dtype=float)
  for dx in range(3):
    for dy in range(3):
      for dz in range(3):
        out += padded[
            dx : dx + volume.shape[0],
            dy : dy + volume.shape[1],
            dz : dz + volume.shape[2],
        ]
  return out / 27.0


def _distance_from_line(
    points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
  """Computes point-to-segment distances for candidate attractor cells."""
  direction = goal - start
  denom = float(np.dot(direction, direction))
  if denom < 1e-12:
    return np.linalg.norm(points - start, axis=1)
  rel = points - start
  t = np.clip(np.dot(rel, direction) / denom, 0.0, 1.0)
  projection = start + t[:, None] * direction
  return np.linalg.norm(points - projection, axis=1)


def _build_obstacle_field(
    occupancy: np.ndarray,
    blur_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds a smooth obstacle potential and its gradient."""
  field = occupancy.astype(float)
  for _ in range(max(1, blur_passes)):
    field = _blur3d(field)

  max_value = float(np.max(field))
  if max_value > 0.0:
    field = field / max_value

  grad_x, grad_y, grad_z = np.gradient(field)
  gradient = np.stack([grad_x, grad_y, grad_z], axis=-1)
  return field, gradient


def _sample_attractors(
    occupancy: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cfg: LocalOptimaConfig,
    rng: np.random.Generator,
) -> np.ndarray:
  """Samples attractor centers to induce local optima."""
  if not cfg.enabled or cfg.count <= 0:
    return np.zeros((0, 3), dtype=float)

  free_cells = np.argwhere(~occupancy)
  if len(free_cells) == 0:
    return np.zeros((0, 3), dtype=float)

  free_points = free_cells.astype(float)
  line_dist = _distance_from_line(free_points, start, goal)
  candidates = free_points[line_dist >= cfg.min_distance_from_path]
  if len(candidates) == 0:
    return np.zeros((0, 3), dtype=float)

  order = rng.permutation(len(candidates))
  selected: list[np.ndarray] = []
  min_pair_dist = max(1.0, cfg.radius)
  for idx in order:
    point = candidates[idx]
    if any(np.linalg.norm(point - other) < min_pair_dist for other in selected):
      continue
    selected.append(point)
    if len(selected) >= cfg.count:
      break

  if not selected:
    return np.zeros((0, 3), dtype=float)
  return np.vstack(selected)


def generate_maze(spec: MazeSpec, seed: int, max_retries: int = 100) -> MazeData:
  """Generates a connected maze and optional local optima attractors."""
  start = np.asarray(spec.start, dtype=float)
  goal = np.asarray(spec.goal, dtype=float)
  shape = spec.grid_size

  for retry in range(max_retries):
    rng = np.random.default_rng(seed + retry)
    occupancy = rng.random(shape) < spec.obstacle_density
    occupancy[spec.start] = False
    occupancy[spec.goal] = False

    if not _is_connected(occupancy, spec.start, spec.goal):
      continue

    obstacle_field, gradient = _build_obstacle_field(occupancy, spec.blur_passes)
    attractors = _sample_attractors(
        occupancy=occupancy,
        start=start,
        goal=goal,
        cfg=spec.local_optima,
        rng=rng,
    )
    return MazeData(
      occupancy=occupancy,
      obstacle_field=obstacle_field,
      field_gradient=gradient,
      start=start,
      goal=goal,
      attractors=attractors,
    )

  raise ValueError("Failed to generate a connected maze for the given seed.")
