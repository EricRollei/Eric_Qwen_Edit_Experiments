# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
#
# Spectrum core adapted from https://github.com/hanjq17/Spectrum (MIT License).
# Original authors: Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon
# Paper: "Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration" (CVPR 2026)
"""
Spectrum: Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration.

This is a simplified, self-contained adaptation for use with QwenImageTransformer2DModel.
Only the core Chebyshev ridge-regression predictor and the Newton-blend "Spectrum" variant
are included — the interface is intentionally minimal.

Usage:
    forecaster = SpectrumForecaster(M=4, K=20, lam=0.1, w=0.5, taylor_order=1)
    forecaster.update(step_idx, hidden_states_after_blocks)
    ...
    predicted = forecaster.predict(future_step_idx)
"""

import torch
from typing import Optional, Tuple

# Internal compute dtype for the forecaster (bf16 saves memory, sufficient accuracy)
_DTYPE = torch.bfloat16


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """Flatten a tensor to (1, N) for matrix operations, preserving original shape."""
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Restore a (1, N) tensor to its original shape."""
    return x_flat.reshape(shape)


class ChebyshevForecaster:
    """
    Chebyshev polynomial feature forecaster with ridge regression.

    Maintains a sliding window of (step_index, feature_vector) pairs, fits
    Chebyshev T-polynomial coefficients via ridge regression, and predicts
    feature vectors at future step indices.

    Args:
        M: Polynomial degree (number of Chebyshev basis functions = M+1).
        K: Maximum window size (oldest entries are dropped).
        lam: Ridge regularization strength.
        device: Torch device for buffers.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 20,
        lam: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        assert K >= M + 2, f"Window K={K} must be >= M+2={M+2}"
        self.M = M
        self.K = K
        self.lam = lam
        self.device = device

        # Buffers (lazily initialized on first update)
        self._t_buf: Optional[torch.Tensor] = None  # (n,) step indices
        self._H_buf: Optional[torch.Tensor] = None  # (n, F) flattened features
        self._shape: Optional[torch.Size] = None     # original feature shape
        self._coef: Optional[torch.Tensor] = None    # cached regression coefficients
        self._n: int = 0                              # number of stored entries

    @property
    def P(self) -> int:
        """Number of basis functions."""
        return self.M + 1

    # ------------------------------------------------------------------
    # Chebyshev basis
    # ------------------------------------------------------------------

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map step indices to tau in [-1, 1] using min/max of stored buffer."""
        if self._n < 2:
            return torch.zeros_like(t)
        t_min = self._t_buf[:self._n].min()
        t_max = self._t_buf[:self._n].max()
        rng = (t_max - t_min).clamp_min(1e-8)
        return (t - 0.5 * (t_min + t_max)) * 2.0 / rng

    def _design_matrix(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev design matrix [T0, T1, ..., TM] of shape (K', P)."""
        taus = taus.reshape(-1, 1)
        K_ = taus.shape[0]
        cols = [torch.ones(K_, 1, device=taus.device, dtype=taus.dtype)]
        if self.M >= 1:
            cols.append(taus)
        for m in range(2, self.M + 1):
            cols.append(2.0 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[: self.P], dim=1)  # (K', P)

    # ------------------------------------------------------------------
    # Update / fit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, t: float, h: torch.Tensor) -> None:
        """Record an observation (step_index, feature_vector)."""
        dev = self.device or h.device
        t_val = torch.tensor(t, dtype=_DTYPE, device=dev)
        h_flat, shape = _flatten(h.to(dev, dtype=_DTYPE))

        if self._shape is None:
            self._shape = shape
            self._t_buf = torch.empty(self.K, dtype=_DTYPE, device=dev)
            self._H_buf = torch.empty(self.K, h_flat.shape[1], dtype=_DTYPE, device=dev)

        if self._n < self.K:
            self._t_buf[self._n] = t_val
            self._H_buf[self._n] = h_flat[0]
            self._n += 1
        else:
            # Slide window: drop oldest
            self._t_buf = torch.roll(self._t_buf, -1, 0)
            self._H_buf = torch.roll(self._H_buf, -1, 0)
            self._t_buf[-1] = t_val
            self._H_buf[-1] = h_flat[0]

        # Invalidate cached coefficients
        self._coef = None

    def _fit(self) -> None:
        """Fit ridge regression coefficients (cached until next update)."""
        if self._coef is not None:
            return
        n = self._n
        taus = self._taus(self._t_buf[:n])
        X = self._design_matrix(taus).to(torch.float32)  # (n, P)
        H = self._H_buf[:n].to(torch.float32)             # (n, F)
        P = X.shape[1]

        XtX = X.T @ X + self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        try:
            L = torch.linalg.cholesky(XtX)
        except Exception:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))
        XtH = X.T @ H
        self._coef = torch.cholesky_solve(XtH, L).to(_DTYPE)  # (P, F)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        """Predict the feature vector at step index *t_star*."""
        assert self._shape is not None, "Must call update() at least once before predict()"
        dev = self._t_buf.device
        t_star_t = torch.tensor(t_star, dtype=_DTYPE, device=dev)
        self._fit()
        tau_star = self._taus(t_star_t)
        x_star = self._design_matrix(tau_star[None])  # (1, P)
        h_flat = x_star @ self._coef                   # (1, F)
        return _unflatten(h_flat, self._shape)

    def ready(self) -> bool:
        """Whether enough data has been accumulated for a meaningful prediction."""
        return self._n >= 2

    def reset(self) -> None:
        """Clear all stored data."""
        self._t_buf = None
        self._H_buf = None
        self._shape = None
        self._coef = None
        self._n = 0


class SpectrumForecaster:
    """
    Spectrum predictor — blends Chebyshev ridge regression with local Newton
    (forward difference) interpolation for improved feature forecasting.

    Args:
        M:  Chebyshev polynomial degree.
        K:  Maximum sliding-window size for the Chebyshev buffer.
        lam: Ridge regularization.
        w:  Blend weight: 0 = pure Newton, 1 = pure Chebyshev, 0.5 = equal (recommended).
        taylor_order: Newton forward-difference order (1, 2, or 3).
        device: Torch device.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 20,
        lam: float = 0.1,
        w: float = 0.5,
        taylor_order: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.cheb = ChebyshevForecaster(M=M, K=K, lam=lam, device=device)
        self.w = w
        self.taylor_order = taylor_order

    # ------------------------------------------------------------------
    # Newton forward-difference prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _newton_predict(self, t_star: float) -> torch.Tensor:
        """Predict via Newton forward differences using the last few points."""
        n = self.cheb._n
        H = self.cheb._H_buf  # (n_max, F)
        t = self.cheb._t_buf  # (n_max,)

        h_last = H[n - 1]
        if n < 2:
            return h_last.clone().reshape(self.cheb._shape)

        dt_last = (t[n - 1] - t[n - 2]).clamp_min(1e-8)
        k = ((t_star - t[n - 1].item()) / dt_last.item())
        k = torch.tensor(k, dtype=h_last.dtype, device=h_last.device)

        d1 = H[n - 1] - H[n - 2]
        out = h_last + k * d1

        if self.taylor_order >= 2 and n >= 3:
            d2 = H[n - 1] - 2.0 * H[n - 2] + H[n - 3]
            out = out + 0.5 * k * (k - 1.0) * d2

        if self.taylor_order >= 3 and n >= 4:
            d3 = H[n - 1] - 3.0 * H[n - 2] + 3.0 * H[n - 3] - H[n - 4]
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3

        return out.reshape(self.cheb._shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, t: float, h: torch.Tensor) -> None:
        """Record a (step_index, feature) observation."""
        self.cheb.update(t, h)

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        """Predict the feature vector at step *t_star* (blended Chebyshev + Newton)."""
        h_cheb = self.cheb.predict(t_star)
        h_newton = self._newton_predict(t_star)
        return (1.0 - self.w) * h_newton + self.w * h_cheb

    def ready(self) -> bool:
        return self.cheb.ready()

    def reset(self) -> None:
        self.cheb.reset()
