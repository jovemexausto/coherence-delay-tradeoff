from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SinkhornResult:
    cost: float
    iterations: int


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def _sinkhorn_kernel_from_cost(cost: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    shifted = cost - float(np.min(cost))
    exponent = -shifted / epsilon
    exponent = np.clip(exponent, -700.0, 0.0)
    return np.exp(exponent)


def sinkhorn_cost(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 120,
    tol: float = 1e-6,
) -> SinkhornResult:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    return sinkhorn_cost_weighted(
        x,
        y,
        a,
        b,
        epsilon,
        max_iters=max_iters,
        tol=tol,
    )


def sinkhorn_cost_weighted(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 120,
    tol: float = 1e-6,
) -> SinkhornResult:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if x.shape[0] != a.shape[0] or y.shape[0] != b.shape[0]:
        raise ValueError("weight vectors must match sample sizes")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("weights must be nonnegative")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / float(np.sum(a))
    b = b / float(np.sum(b))
    cost = _pairwise_squared_distances(x, y)
    kernel = _sinkhorn_kernel_from_cost(cost, epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)
    iteration = 0

    for iteration in range(1, max_iters + 1):
        u_prev = u.copy()
        Kv = np.maximum(kernel @ v, 1e-300)
        u = a / Kv
        Ku = np.maximum(kernel.T @ u, 1e-300)
        v = b / Ku
        if np.max(np.abs(u - u_prev) / np.maximum(np.abs(u_prev), 1e-12)) < tol:
            break

    transport = (u[:, None] * kernel) * v[None, :]
    return SinkhornResult(cost=float(np.sum(transport * cost)), iterations=iteration)


def debiased_sinkhorn_divergence(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 250,
    tol: float = 1e-9,
) -> SinkhornResult:
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    return debiased_sinkhorn_divergence_weighted(
        x,
        y,
        a,
        b,
        epsilon,
        max_iters=max_iters,
        tol=tol,
    )


def debiased_sinkhorn_divergence_weighted(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 250,
    tol: float = 1e-9,
) -> SinkhornResult:
    xy = sinkhorn_cost_weighted(x, y, a, b, epsilon, max_iters=max_iters, tol=tol)
    xx = sinkhorn_cost_weighted(x, x, a, a, epsilon, max_iters=max_iters, tol=tol)
    yy = sinkhorn_cost_weighted(y, y, b, b, epsilon, max_iters=max_iters, tol=tol)
    return SinkhornResult(
        cost=xy.cost - 0.5 * xx.cost - 0.5 * yy.cost,
        iterations=max(xy.iterations, xx.iterations, yy.iterations),
    )
