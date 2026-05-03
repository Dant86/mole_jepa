"""Regularizers for MoLeJEPA training."""

from mole_jepa.regularizers.epps_pulley import (
    EppsPulleyGaussian,
    EppsPulleyLaplace,
    EppsPulleyRegularizer,
    epps_pulley,
)

__all__ = [
    "EppsPulleyGaussian",
    "EppsPulleyLaplace",
    "EppsPulleyRegularizer",
    "epps_pulley",
]
