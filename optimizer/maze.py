"""Seeded mountain-surface generation for optimizer convergence tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalPeakConfig:
  amp: float
  sigma: float


@dataclass(frozen=True)
class LocalPeaksConfig:
  count: int
  amp_range: tuple[float, float]
  sigma_range: tuple[float, float]
  min_dist_global: float
  min_dist_local: float


@dataclass(frozen=True)
class NoiseConfig:
  enabled: bool
  scale: float
  blur_passes: int


@dataclass(frozen=True)
class GlobalTrendConfig:
  enabled: bool
  strength: float


@dataclass(frozen=True)
class AcceptanceConfig:
  max_peak_shift: float


@dataclass(frozen=True)
class SurfaceConfig:
  global_peak: GlobalPeakConfig
  local_peaks: LocalPeaksConfig
  noise: NoiseConfig
  global_trend: GlobalTrendConfig
  acceptance: AcceptanceConfig


@dataclass(frozen=True)
class TerrainSpec:
  domain: tuple[float, float]
  grid_resolution: int
  surface: SurfaceConfig


@dataclass(frozen=True)
class TerrainData:
  x_coords: np.ndarray
  y_coords: np.ndarray
  surface: np.ndarray
  global_peak_point: np.ndarray
  global_peak_height: float
  local_peak_points: np.ndarray
  planted_global_peak_point: np.ndarray


def _blur2d(values: np.ndarray) -> np.ndarray:
  """Applies a 3x3 mean blur using only NumPy."""
  padded = np.pad(values, ((1, 1), (1, 1)), mode="edge")
  out = np.zeros_like(values, dtype=float)
  for dx in range(3):
    for dy in range(3):
      out += padded[dx : dx + values.shape[0], dy : dy + values.shape[1]]
  return out / 9.0


def _gaussian_hill(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center: np.ndarray,
    amp: float,
    sigma: float,
) -> np.ndarray:
  """Evaluates a Gaussian hill over a 2D grid."""
  dx = x_grid - center[0]
  dy = y_grid - center[1]
  dist2 = dx * dx + dy * dy
  sigma2 = max(1e-12, sigma * sigma)
  return amp * np.exp(-dist2 / (2.0 * sigma2))


def _sample_global_peak(
    domain: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
  """Samples a global-peak center away from boundaries."""
  low, high = domain
  size = high - low
  margin = 0.15 * size
  return rng.uniform(low + margin, high - margin, size=2)


def _sample_local_peaks(
    domain: tuple[float, float],
    global_peak: np.ndarray,
    cfg: LocalPeaksConfig,
    rng: np.random.Generator,
) -> np.ndarray:
  """Samples local peak centers with spacing constraints."""
  if cfg.count <= 0:
    return np.zeros((0, 2), dtype=float)

  low, high = domain
  selected: list[np.ndarray] = []
  attempts = max(200, 200 * cfg.count)

  for _ in range(attempts):
    candidate = rng.uniform(low, high, size=2)
    if np.linalg.norm(candidate - global_peak) < cfg.min_dist_global:
      continue
    if any(np.linalg.norm(candidate - point) < cfg.min_dist_local for point in selected):
      continue
    selected.append(candidate)
    if len(selected) >= cfg.count:
      break

  if not selected:
    return np.zeros((0, 2), dtype=float)
  return np.vstack(selected)


def _normalize_surface(surface: np.ndarray) -> np.ndarray:
  shifted = surface - np.min(surface)
  max_value = float(np.max(shifted))
  if max_value > 0.0:
    shifted /= max_value
  return shifted


def _argmax_point(
    surface: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, float]:
  idx = np.unravel_index(int(np.argmax(surface)), surface.shape)
  y_idx, x_idx = idx
  point = np.asarray([x_coords[x_idx], y_coords[y_idx]], dtype=float)
  height = float(surface[y_idx, x_idx])
  return point, height


def generate_terrain(spec: TerrainSpec, seed: int, max_retries: int = 200) -> TerrainData:
  """Generates a mountain surface with one global and multiple local peaks."""
  if spec.grid_resolution < 16:
    raise ValueError("grid_resolution must be at least 16.")

  low, high = spec.domain
  if not high > low:
    raise ValueError("domain upper bound must be larger than lower bound.")

  x_coords = np.linspace(low, high, spec.grid_resolution)
  y_coords = np.linspace(low, high, spec.grid_resolution)
  x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")

  for retry in range(max_retries):
    rng = np.random.default_rng(seed + retry)
    planted_global = _sample_global_peak(spec.domain, rng)

    surface = _gaussian_hill(
        x_grid=x_grid,
        y_grid=y_grid,
        center=planted_global,
        amp=spec.surface.global_peak.amp,
        sigma=spec.surface.global_peak.sigma,
    )

    local_peaks = _sample_local_peaks(
        domain=spec.domain,
        global_peak=planted_global,
        cfg=spec.surface.local_peaks,
        rng=rng,
    )

    for point in local_peaks:
      amp = rng.uniform(*spec.surface.local_peaks.amp_range)
      sigma = rng.uniform(*spec.surface.local_peaks.sigma_range)
      surface += _gaussian_hill(
          x_grid=x_grid,
          y_grid=y_grid,
          center=point,
          amp=amp,
          sigma=sigma,
      )

    if spec.surface.noise.enabled and spec.surface.noise.scale > 0.0:
      noise = rng.normal(loc=0.0, scale=1.0, size=surface.shape)
      for _ in range(max(1, spec.surface.noise.blur_passes)):
        noise = _blur2d(noise)
      noise_std = float(np.std(noise))
      if noise_std > 0.0:
        noise = noise / noise_std
      surface += spec.surface.noise.scale * noise

    if spec.surface.global_trend.enabled and spec.surface.global_trend.strength > 0.0:
      dx = x_grid - planted_global[0]
      dy = y_grid - planted_global[1]
      dist = np.sqrt(dx * dx + dy * dy)
      max_dist = float(np.max(dist))
      if max_dist > 0.0:
        trend = 1.0 - dist / max_dist
        surface += spec.surface.global_trend.strength * trend

    surface = _normalize_surface(surface)
    actual_peak, actual_height = _argmax_point(surface, x_coords, y_coords)

    if np.linalg.norm(actual_peak - planted_global) > spec.surface.acceptance.max_peak_shift:
      continue

    return TerrainData(
        x_coords=x_coords,
        y_coords=y_coords,
        surface=surface,
        global_peak_point=actual_peak,
        global_peak_height=actual_height,
        local_peak_points=local_peaks,
        planted_global_peak_point=planted_global,
    )

  raise ValueError("Failed to generate terrain with a dominant global peak.")
