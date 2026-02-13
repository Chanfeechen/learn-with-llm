"""Objective definition for mountain-surface point optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maze import TerrainData


@dataclass(frozen=True)
class ObjectiveWeights:
  height: float
  boundary: float


def bilinear_sample(
    surface: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
  """Bilinear interpolation over a regular 2D grid."""
  points = np.asarray(points, dtype=float)
  if points.ndim == 1:
    points = points[None, :]

  x_low = float(x_coords[0])
  x_high = float(x_coords[-1])
  y_low = float(y_coords[0])
  y_high = float(y_coords[-1])
  nx = len(x_coords)
  ny = len(y_coords)

  x = np.clip(points[:, 0], x_low, x_high)
  y = np.clip(points[:, 1], y_low, y_high)

  tx = (x - x_low) / max(1e-12, x_high - x_low) * (nx - 1)
  ty = (y - y_low) / max(1e-12, y_high - y_low) * (ny - 1)

  x0 = np.floor(tx).astype(int)
  y0 = np.floor(ty).astype(int)
  x1 = np.clip(x0 + 1, 0, nx - 1)
  y1 = np.clip(y0 + 1, 0, ny - 1)

  xd = tx - x0
  yd = ty - y0

  f00 = surface[y0, x0]
  f10 = surface[y0, x1]
  f01 = surface[y1, x0]
  f11 = surface[y1, x1]

  fx0 = f00 * (1.0 - xd) + f10 * xd
  fx1 = f01 * (1.0 - xd) + f11 * xd
  return fx0 * (1.0 - yd) + fx1 * yd


class SurfaceObjective:
  """Objective for maximizing surface height via minimization."""

  def __init__(self, terrain: TerrainData, weights: ObjectiveWeights) -> None:
    self._terrain = terrain
    self._weights = weights
    self._x_low = float(terrain.x_coords[0])
    self._x_high = float(terrain.x_coords[-1])
    self._y_low = float(terrain.y_coords[0])
    self._y_high = float(terrain.y_coords[-1])

  def evaluate(self, point: np.ndarray) -> tuple[float, dict[str, float]]:
    point = np.asarray(point, dtype=float).reshape(1, 2)
    height = float(
        bilinear_sample(
            surface=self._terrain.surface,
            x_coords=self._terrain.x_coords,
            y_coords=self._terrain.y_coords,
            points=point,
        )[0]
    )
    boundary = self._boundary_penalty(point[0])

    # Minimize negative height to convert max-height search into minimization.
    loss = self._weights.height * (-height) + self._weights.boundary * boundary
    terms = {
      "height": height,
      "boundary": float(boundary),
      "loss": float(loss),
    }
    return float(loss), terms

  def _boundary_penalty(self, point: np.ndarray) -> float:
    x, y = float(point[0]), float(point[1])

    dx_low = max(0.0, self._x_low - x)
    dx_high = max(0.0, x - self._x_high)
    dy_low = max(0.0, self._y_low - y)
    dy_high = max(0.0, y - self._y_high)

    return dx_low * dx_low + dx_high * dx_high + dy_low * dy_low + dy_high * dy_high
