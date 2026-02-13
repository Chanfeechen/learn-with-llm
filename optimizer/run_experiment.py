"""CLI entrypoint for running the optimizer comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment import run_experiment


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--config",
      type=Path,
      default=Path(__file__).parent / "config" / "default_config.json",
      help="Path to experiment config JSON.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path(__file__).parent / "artifacts",
      help="Directory where plots and logs are written.",
  )
  return parser


def main() -> None:
  args = _build_parser().parse_args()
  run_experiment(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
  main()
