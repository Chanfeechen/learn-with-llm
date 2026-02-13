"""Visualization utilities for optimizer comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _subplot_grid(num_items: int) -> tuple[int, int]:
  cols = int(math.ceil(math.sqrt(num_items)))
  rows = int(math.ceil(num_items / cols))
  return rows, cols


def save_metric_plots(histories: dict[str, list[dict]], output_path: Path) -> None:
  """Saves loss, gradient norm, and step norm curves."""
  fig, axes = plt.subplots(1, 3, figsize=(16, 4))
  metric_names = ["loss", "grad_norm", "step_norm"]
  titles = ["Loss", "Gradient Norm", "Step Norm"]

  for ax, metric, title in zip(axes, metric_names, titles):
    for name, history in histories.items():
      xs = [row["iter"] for row in history]
      ys = [row[metric] for row in history]
      ax.plot(xs, ys, label=name)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)

  axes[0].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def save_path_comparison(
    occupancy: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    attractors: np.ndarray,
    final_paths: dict[str, np.ndarray],
    output_path: Path,
    obstacle_points_limit: int = 5000,
) -> None:
  """Saves a multi-panel 3D view of optimized paths."""
  names = list(final_paths.keys())
  rows, cols = _subplot_grid(len(names))
  fig = plt.figure(figsize=(6 * cols, 5 * rows))

  obstacles = np.argwhere(occupancy)
  if len(obstacles) > obstacle_points_limit:
    idx = np.linspace(0, len(obstacles) - 1, obstacle_points_limit).astype(int)
    obstacles = obstacles[idx]

  for i, name in enumerate(names, start=1):
    ax = fig.add_subplot(rows, cols, i, projection="3d")
    if len(obstacles) > 0:
      ax.scatter(
          obstacles[:, 0],
          obstacles[:, 1],
          obstacles[:, 2],
          s=2,
          alpha=0.08,
          c="black",
      )

    path = final_paths[name]
    ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=2, c="tab:blue")
    ax.scatter(
        [start[0], goal[0]],
        [start[1], goal[1]],
        [start[2], goal[2]],
        c=["green", "red"],
        s=40,
    )

    if len(attractors) > 0:
      ax.scatter(
          attractors[:, 0],
          attractors[:, 1],
          attractors[:, 2],
          c="orange",
          marker="x",
          s=30,
      )

    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, occupancy.shape[0] - 1)
    ax.set_ylim(0, occupancy.shape[1] - 1)
    ax.set_zlim(0, occupancy.shape[2] - 1)

  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)
