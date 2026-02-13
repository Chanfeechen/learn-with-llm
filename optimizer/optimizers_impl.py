"""NumPy optimizer implementations for fair algorithm comparisons."""

from __future__ import annotations

import numpy as np


class Optimizer:
  """Optimizer base class."""

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class SGD(Optimizer):
  def __init__(self, lr: float) -> None:
    self._lr = lr

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return params - self._lr * grad


class SGDMomentum(Optimizer):
  def __init__(self, lr: float, mu: float) -> None:
    self._lr = lr
    self._mu = mu
    self._velocity: np.ndarray | None = None

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if self._velocity is None:
      self._velocity = np.zeros_like(params)
    self._velocity = self._mu * self._velocity + grad
    return params - self._lr * self._velocity


class AdaGrad(Optimizer):
  def __init__(self, lr: float, eps: float) -> None:
    self._lr = lr
    self._eps = eps
    self._accum: np.ndarray | None = None

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if self._accum is None:
      self._accum = np.zeros_like(params)
    self._accum = self._accum + grad * grad
    adjusted = grad / (np.sqrt(self._accum) + self._eps)
    return params - self._lr * adjusted


class RMSProp(Optimizer):
  def __init__(self, lr: float, rho: float, eps: float) -> None:
    self._lr = lr
    self._rho = rho
    self._eps = eps
    self._avg_sq: np.ndarray | None = None

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if self._avg_sq is None:
      self._avg_sq = np.zeros_like(params)
    self._avg_sq = self._rho * self._avg_sq + (1.0 - self._rho) * (grad * grad)
    adjusted = grad / (np.sqrt(self._avg_sq) + self._eps)
    return params - self._lr * adjusted


class Adam(Optimizer):
  def __init__(
      self,
      lr: float,
      beta1: float,
      beta2: float,
      eps: float,
  ) -> None:
    self._lr = lr
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps
    self._m: np.ndarray | None = None
    self._v: np.ndarray | None = None
    self._t = 0

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if self._m is None:
      self._m = np.zeros_like(params)
      self._v = np.zeros_like(params)

    self._t += 1
    self._m = self._beta1 * self._m + (1.0 - self._beta1) * grad
    self._v = self._beta2 * self._v + (1.0 - self._beta2) * (grad * grad)

    m_hat = self._m / (1.0 - self._beta1**self._t)
    v_hat = self._v / (1.0 - self._beta2**self._t)
    return params - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)


class AdamW(Optimizer):
  def __init__(
      self,
      lr: float,
      beta1: float,
      beta2: float,
      eps: float,
      weight_decay: float,
  ) -> None:
    self._lr = lr
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps
    self._weight_decay = weight_decay
    self._m: np.ndarray | None = None
    self._v: np.ndarray | None = None
    self._t = 0

  def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if self._m is None:
      self._m = np.zeros_like(params)
      self._v = np.zeros_like(params)

    self._t += 1
    self._m = self._beta1 * self._m + (1.0 - self._beta1) * grad
    self._v = self._beta2 * self._v + (1.0 - self._beta2) * (grad * grad)

    m_hat = self._m / (1.0 - self._beta1**self._t)
    v_hat = self._v / (1.0 - self._beta2**self._t)
    updated = params - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)
    return updated - self._lr * self._weight_decay * params


def build_optimizer(name: str, cfg: dict) -> Optimizer:
  """Builds optimizer instances from config entries."""
  lower = name.lower()
  if lower == "sgd":
    return SGD(lr=float(cfg["lr"]))
  if lower in ("sgd_momentum", "momentum"):
    return SGDMomentum(lr=float(cfg["lr"]), mu=float(cfg["mu"]))
  if lower == "adagrad":
    return AdaGrad(lr=float(cfg["lr"]), eps=float(cfg["eps"]))
  if lower == "rmsprop":
    return RMSProp(
        lr=float(cfg["lr"]),
        rho=float(cfg["rho"]),
        eps=float(cfg["eps"]),
    )
  if lower == "adam":
    return Adam(
        lr=float(cfg["lr"]),
        beta1=float(cfg["beta1"]),
        beta2=float(cfg["beta2"]),
        eps=float(cfg["eps"]),
    )
  if lower == "adamw":
    return AdamW(
        lr=float(cfg["lr"]),
        beta1=float(cfg["beta1"]),
        beta2=float(cfg["beta2"]),
        eps=float(cfg["eps"]),
        weight_decay=float(cfg["weight_decay"]),
    )
  raise ValueError(f"Unsupported optimizer: {name}")
