"""QuenchGuard: LSTM predictive maintenance for MRI magnet quench detection.

Package layout
--------------
config       All hyper-parameters, physics thresholds and cost assumptions.
data_gen     Synthetic multi-rate sensor streams + labelled quench events.
features     Causal (leakage-free) feature extraction per the case-study table.
labeling     Horizon-based labelling with an offline embargo after events.
sequences    Time-ordered splits, scaling and LSTM sequence construction.
baselines    Non-neural baselines the LSTM must beat (LR / HistGBDT).
lstm         PyTorch LSTM detector, training loop and gradient attribution.
evaluate     Threshold policy, event recall, lead time, cost matrix, plots.
"""

from .config import Config, default_config

__version__ = "0.1.0"
__all__ = ["Config", "default_config", "__version__"]
