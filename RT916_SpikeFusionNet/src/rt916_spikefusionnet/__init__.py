"""Release-safe RT916 SpikeFusionNet package."""

from .annual_loss import AnnualProtectedCappedLoss, SegmentedSMAPECappedLoss
from .annual_model import AnnualSpikeGatedTimesNet, AnnualSpikeGatedTimesNetV2
from .core import run, run_daily_asof_backtest, run_joint_da_rt_daily_backtest, train_interface

__all__ = [
    "AnnualProtectedCappedLoss",
    "AnnualSpikeGatedTimesNet",
    "AnnualSpikeGatedTimesNetV2",
    "SegmentedSMAPECappedLoss",
    "run",
    "run_daily_asof_backtest",
    "run_joint_da_rt_daily_backtest",
    "train_interface",
]

__version__ = "1.1.0-w4-da-linkage"
