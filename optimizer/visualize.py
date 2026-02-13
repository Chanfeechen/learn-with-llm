"""Visualization utilities for mountain-surface optimizer tests."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from maze import TerrainData
from objective import bilinear_sample


def _subplot_grid(num_items: int) -> tuple[int, int]:
  cols = int(math.ceil(math.sqrt(num_items)))
  rows = int(math.ceil(num_items / cols))
  return rows, cols


def _surface_height(terrain: TerrainData, points: np.ndarray) -> np.ndarray:
  return bilinear_sample(
      surface=terrain.surface,
      x_coords=terrain.x_coords,
      y_coords=terrain.y_coords,
      points=points,
  )


def _downsample_points(points: np.ndarray, max_frames: int) -> np.ndarray:
  if len(points) <= max_frames:
    return points
  idx = np.linspace(0, len(points) - 1, max_frames).astype(int)
  return points[idx]


def save_surface_3d(
    terrain: TerrainData,
    trajectories: dict[str, np.ndarray],
    output_path: Path,
) -> None:
  """Saves a 3D mountain surface with all optimizer trajectories."""
  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(111, projection="3d")

  x_grid, y_grid = np.meshgrid(terrain.x_coords, terrain.y_coords, indexing="xy")
  stride = max(1, len(terrain.x_coords) // 60)
  ax.plot_surface(
      x_grid[::stride, ::stride],
      y_grid[::stride, ::stride],
      terrain.surface[::stride, ::stride],
      cmap="terrain",
      alpha=0.78,
      linewidth=0,
      antialiased=True,
  )

  for name, points in trajectories.items():
    heights = _surface_height(terrain, points)
    ax.plot(points[:, 0], points[:, 1], heights, linewidth=2.0, label=name, c="red")

  peak = terrain.global_peak_point.reshape(1, 2)
  peak_h = _surface_height(terrain, peak)[0]
  ax.scatter(
      peak[0, 0],
      peak[0, 1],
      peak_h,
      marker="*",
      s=160,
      c="gold",
      label="global optimum",
  )

  if len(terrain.local_peak_points) > 0:
    local_h = _surface_height(terrain, terrain.local_peak_points)
    ax.scatter(
        terrain.local_peak_points[:, 0],
        terrain.local_peak_points[:, 1],
        local_h,
        marker="x",
        s=36,
        c="orange",
        label="local peaks",
    )

  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("height")
  ax.set_title("Mountain Surface and Optimizer Trajectories")
  ax.legend(loc="upper right", fontsize=8)
  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def save_surface_contour(
    terrain: TerrainData,
    trajectories: dict[str, np.ndarray],
    output_path: Path,
) -> None:
  """Saves a contour map with optimizer trajectories."""
  fig, ax = plt.subplots(figsize=(10, 8))
  x_grid, y_grid = np.meshgrid(terrain.x_coords, terrain.y_coords, indexing="xy")

  contour = ax.contourf(x_grid, y_grid, terrain.surface, levels=40, cmap="terrain")
  fig.colorbar(contour, ax=ax, label="height")

  for name, points in trajectories.items():
    ax.plot(points[:, 0], points[:, 1], linewidth=1.8, label=name, c="red")
    ax.scatter(points[0, 0], points[0, 1], c="red", s=14)
    ax.scatter(points[-1, 0], points[-1, 1], c="red", edgecolors="black", s=18)

  peak = terrain.global_peak_point
  ax.scatter(peak[0], peak[1], c="gold", marker="*", s=150, label="global optimum")

  if len(terrain.local_peak_points) > 0:
    ax.scatter(
        terrain.local_peak_points[:, 0],
        terrain.local_peak_points[:, 1],
        c="orange",
        marker="x",
        s=36,
        label="local peaks",
    )

  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("Surface Contour With Optimizer Paths")
  ax.legend(fontsize=8, loc="upper right")
  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def save_metrics(histories: dict[str, list[dict]], output_path: Path) -> None:
  """Saves convergence metrics for all optimizers."""
  fig, axes = plt.subplots(2, 2, figsize=(14, 9))
  axes = axes.flatten()

  metric_specs = [
      ("height", "Height vs Iteration"),
      ("distance_to_optimum", "Distance To Optimum vs Iteration"),
      ("loss", "Loss vs Iteration"),
      ("grad_norm", "Gradient Norm vs Iteration"),
  ]

  for ax, (metric, title) in zip(axes, metric_specs):
    for name, history in histories.items():
      xs = [row["iter"] for row in history]
      ys = [row[metric] for row in history]
      ax.plot(xs, ys, label=name)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)

  axes[0].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def save_surface_animation(
    terrain: TerrainData,
    trajectories: dict[str, np.ndarray],
    stop_reasons: dict[str, str],
    output_path: Path,
    fps: int,
    max_frames: int,
) -> None:
  """Saves multi-panel animation of optimizer search on the surface."""
  names = list(trajectories.keys())
  rows, cols = _subplot_grid(len(names))
  fig = plt.figure(figsize=(6 * cols, 5 * rows))
  z_offset = 0.015
  optimizer_color = "red"

  x_grid, y_grid = np.meshgrid(terrain.x_coords, terrain.y_coords, indexing="xy")
  stride = max(1, len(terrain.x_coords) // 45)

  sampled = {
      name: _downsample_points(points, max_frames=max_frames)
      for name, points in trajectories.items()
  }
  sampled_heights = {name: _surface_height(terrain, points) for name, points in sampled.items()}

  axes = {}
  lines = {}
  markers = {}

  for idx, name in enumerate(names, start=1):
    ax = fig.add_subplot(rows, cols, idx, projection="3d")
    axes[name] = ax

    ax.plot_surface(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        terrain.surface[::stride, ::stride],
        cmap="terrain",
        alpha=0.55,
        linewidth=0,
        antialiased=False,
    )

    peak = terrain.global_peak_point.reshape(1, 2)
    peak_h = _surface_height(terrain, peak)[0]
    ax.scatter(peak[0, 0], peak[0, 1], peak_h, marker="*", c="gold", s=90)

    if len(terrain.local_peak_points) > 0:
      local_h = _surface_height(terrain, terrain.local_peak_points)
      ax.scatter(
          terrain.local_peak_points[:, 0],
          terrain.local_peak_points[:, 1],
          local_h,
          marker="x",
          c="orange",
          s=22,
      )

    line, = ax.plot([], [], [], linewidth=2.4, c=optimizer_color)
    marker = ax.scatter([], [], [], c=optimizer_color, s=46, edgecolors="white", linewidths=0.8)
    lines[name] = line
    markers[name] = marker

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")
    ax.set_xlim(terrain.x_coords[0], terrain.x_coords[-1])
    ax.set_ylim(terrain.y_coords[0], terrain.y_coords[-1])
    ax.set_zlim(np.min(terrain.surface), np.max(terrain.surface) + 1e-6)

  frame_count = max(len(points) for points in sampled.values())

  def _update(frame_idx: int):
    artists = []
    for name in names:
      points = sampled[name]
      heights = sampled_heights[name]
      idx = min(frame_idx, len(points) - 1)
      seg = points[: idx + 1]
      seg_h = heights[: idx + 1] + z_offset

      lines[name].set_data(seg[:, 0], seg[:, 1])
      lines[name].set_3d_properties(seg_h)

      curr = seg[-1]
      curr_h = seg_h[-1]
      markers[name]._offsets3d = ([curr[0]], [curr[1]], [curr_h])

      axes[name].set_title(f"{name} ({stop_reasons.get(name, 'n/a')})")
      artists.append(lines[name])
      artists.append(markers[name])
    return artists

  ani = animation.FuncAnimation(
      fig,
      _update,
      frames=frame_count,
      interval=max(1, int(1000 / max(1, fps))),
      blit=False,
  )
  ani.save(output_path, writer=animation.PillowWriter(fps=max(1, fps)), dpi=110)
  plt.close(fig)
