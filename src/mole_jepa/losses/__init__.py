"""Loss modules for MoLeJEPA."""

from mole_jepa.losses.info_nce import InfoNCELoss
from mole_jepa.losses.jepa_loss import JEPALoss, JEPALossOutput

__all__ = [
    "InfoNCELoss",
    "JEPALoss",
    "JEPALossOutput",
]
