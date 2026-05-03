"""Epps-Pulley regularizers and factory function."""

import abc
from typing import Literal

import torch
from torch import nn


class EppsPulleyRegularizer(abc.ABC, nn.Module):
    """Implements the Epps-Pulley univariate test for an arbitrary distribution.

    Computes the integrated squared distance between the empirical
    characteristic function of `x` and a target characteristic function,
    approximated via the composite trapezoidal rule over `[0, t_max]`.
    Subclasses must implement `characteristic_fn`.

    Args:
        t_max: Upper integration bound.
        n_integration_points: Number of quadrature points.

    Notes:
        Implementation adapted from
        https://github.com/galilai-group/lejepa/blob/main/lejepa/univariate/epps_pulley.py
    """

    t: torch.Tensor
    weights: torch.Tensor

    def __init__(self, t_max: float = 3.0, n_integration_points: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, t_max, n_integration_points, dtype=torch.float32)
        dt = t_max / (n_integration_points - 1)
        weights = torch.full((n_integration_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Epps-Pulley statistic for a batch of samples.

        Args:
            x: 1-D tensor of shape `(n,)` containing i.i.d. samples.

        Returns:
            Scalar loss value.
        """
        phase = torch.outer(x, self.t)  # (n, T)

        # Decompose the empirical CF into its real (cosine) and imaginary (sine) parts.
        phi_hat_real = torch.mean(torch.cos(phase), dim=0)  # (T,)
        phi_hat_imag = torch.mean(torch.sin(phase), dim=0)  # (T,)

        # Target CF is real-valued for all supported distributions.
        phi = self.characteristic_fn(self.t)  # (T,)

        diff_sq = (phi_hat_real - phi).pow(2) + phi_hat_imag.pow(2)
        return (self.weights * diff_sq).sum()

    @abc.abstractmethod
    def characteristic_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the target characteristic function at `t`.

        Args:
            t: Real-valued tensor of evaluation points, shape `(T,)`.

        Returns:
            Real-valued tensor of shape `(T,)`.
        """
        ...


class EppsPulleyGaussian(EppsPulleyRegularizer):
    """Epps-Pulley regularizer targeting the standard normal distribution.

    The characteristic function of N(0, 1) is:

        phi(t) = exp(-t^2 / 2)
    """

    def characteristic_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the standard normal characteristic function.

        Args:
            t: Real-valued evaluation points, shape `(T,)`.

        Returns:
            Real-valued tensor of shape `(T,)`.
        """
        return torch.exp(-0.5 * t.pow(2))


class EppsPulleyLaplace(EppsPulleyRegularizer):
    """Epps-Pulley regularizer targeting the standard Laplace distribution.

    The characteristic function of Laplace(0, 1) is:

        phi(t) = 1 / (1 + t^2)
    """

    def characteristic_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the standard Laplace characteristic function.

        Args:
            t: Real-valued evaluation points, shape `(T,)`.

        Returns:
            Real-valued tensor of shape `(T,)`.
        """
        return 1.0 / (1.0 + t.pow(2))


def epps_pulley(
    distribution: Literal["gaussian", "laplace"],
    t_max: float = 3.0,
    n_integration_points: int = 17,
) -> EppsPulleyRegularizer:
    """Construct an Epps-Pulley regularizer for the given distribution.

    Args:
        distribution: Target distribution. One of ``"gaussian"`` or ``"laplace"``.
        t_max: Upper integration bound.
        n_integration_points: Number of quadrature points.

    Returns:
        An initialised :class:`EppsPulleyRegularizer`.

    Raises:
        ValueError: If `distribution` is not a supported value.
    """
    match distribution:
        case "gaussian":
            return EppsPulleyGaussian(t_max, n_integration_points)
        case "laplace":
            return EppsPulleyLaplace(t_max, n_integration_points)
        case _:
            raise ValueError(f"Unknown distribution: {distribution!r}")
