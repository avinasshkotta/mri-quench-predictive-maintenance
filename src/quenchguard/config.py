"""Central configuration for the QuenchGuard case study.

Every number that a reviewer might challenge lives here:

* physics thresholds   -> taken from the case-study failure-signal column
* window / horizon     -> derived from the 72 h lead-time business requirement
* split fractions      -> time-ordered, never random
* cost assumptions     -> the dollars behind the $120k quench problem
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Repository layout (absolute, so notebooks/scripts can run from any working directory)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
# Medallion architecture (Bronze -> Silver -> Gold)
RAW_DIR = DATA_DIR / "raw"          # BRONZE  : as-is simulator output, never mutated
SILVER_DIR = DATA_DIR / "silver"    # SILVER  : cleaned, typed, deduplicated, quality-gated
GOLD_DIR = DATA_DIR / "gold"        # GOLD    : feature-engineered, model-ready tables
LINEAGE_DIR = DATA_DIR / "lineage"  # per-layer manifests: sha256, rows, schema, upstream
# Backwards-compatible alias (older code/notebooks imported PROCESSED_DIR)
PROCESSED_DIR = GOLD_DIR
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class FleetConfig:
    """Synthetic fleet used for the case study (deterministic, seed-driven)."""

    n_magnets: int = 6
    days: int = 90
    start: str = "2025-01-01 00:00:00"
    seed: int = 42


@dataclass(frozen=True)
class WindowConfig:
    """Sequencing + labelling geometry.

    horizon_hours (96 h) is deliberately *longer* than min_lead_hours (72 h):
    labelling asks "will a quench happen in the next 96 h?" while the business
    only requires the first alert to land at least 72 h before the event.
    That 24 h of slack is what lets the model alert on the early trend.
    """

    step_minutes: int = 15          # one feature row = one 15-minute window
    seq_len: int = 48               # 48 * 15 min = 12 h look-back per prediction
    horizon_hours: int = 96         # label = quench inside the next 96 h
    min_lead_hours: int = 72        # acceptance target for first-alert lead time
    offline_days_after_event: int = 7  # repair window excluded from scoring

    @property
    def step_hours(self) -> float:
        return self.step_minutes / 60.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Failure signals from the case-study table, plus the degradation ramps."""

    # Helium pressure
    pressure_baseline_mpa: float = 2.35
    pressure_critical_mpa: float = 2.00       # failure signal: drift below 2.0 MPa
    pressure_ramp_hours: float = 336.0         # slow cryogen loss starts 14 days out
    pressure_end_mpa: float = 1.95
    # -> crosses 2.0 MPa roughly 79 h before the quench

    # Cold head temperature
    cold_head_baseline_k: float = 4.20
    cold_head_critical_k: float = 5.50        # failure signal: rise above 5.5 K
    cold_head_ramp_hours: float = 96.0
    cold_head_end_k: float = 6.00
    # -> crosses 5.5 K only ~27 h before the quench (too late for a static threshold)

    # Gradient coil vibration
    vibration_baseline_rms_g: float = 0.35
    vibration_ramp_hours: float = 120.0
    vibration_end_rms_g: float = 0.95
    fault_band_hz: tuple[int, int] = (100, 200)  # fault-band energy window
    resonance_hz: float = 120.0                  # 120 Hz bearing resonance peak

    # RF power output
    rf_nominal_w: float = 1000.0
    rf_dropout_p0: float = 0.002               # healthy dropout probability / reading
    rf_dropout_ramp_hours: float = 48.0
    rf_dropout_p1: float = 0.25                # terminal-stage dropout probability

    # Label / state geometry
    ramp_hours: float = 96.0                   # generic pre-failure ramp (labels)
    post_event_exclusion_hours: float = 24.0   # hysteresis after first crossing


@dataclass(frozen=True)
class SplitConfig:
    """Global time cut-points applied to every magnet (no random shuffling)."""

    train_frac: float = 0.60
    val_frac: float = 0.20                     # test = remaining 20 %


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.30
    patience: int = 5                          # early stopping on val PR-AUC
    max_pos_weight: float = 40.0               # cap for BCE pos_weight
    seed: int = 42
    device: str = "auto"                       # "auto" | "cpu" | "cuda"
    # Workaround: some Windows NVIDIA driver/torch builds crash the process
    # (STATUS_STACK_BUFFER_OVERRUN) when a cuDNN-trained LSTM is torn down at
    # interpreter exit. The native CUDA LSTM kernel is a little slower but safe.
    # Set to False if your CUDA stack shuts down cleanly.
    disable_cudnn: bool = True


@dataclass(frozen=True)
class ThresholdConfig:
    # Hard business constraint: <= 1 false page per magnet per month.
    false_alert_budget_per_magnet_month: float = 1.0
    # The validation slice is small, so we select the operating point against a
    # *tightened* budget: a safety margin that absorbs the shift to the test period.
    selection_safety_factor: float = 0.6
    # How often the scoring job runs (used to convert budget -> raw count).
    scores_per_day: float = 96.0


@dataclass(frozen=True)
class CostConfig:
    """Dollar model. Every figure is an explicit, editable assumption."""

    repair_usd: float = 120_000.0              # given in the case study
    downtime_days: int = 5                     # given in the case study
    downtime_cost_per_day: float = 15_000.0    # cancelled/rescheduled scans + labour
    planned_repair_factor: float = 0.55        # planned fix costs 55 % of emergency fix
    planned_downtime_days: int = 1             # scheduled slot, no cancelled clinics
    false_alarm_usd: float = 2_000.0           # unnecessary service visit + clinician time

    @property
    def emergency_total(self) -> float:
        return self.repair_usd + self.downtime_days * self.downtime_cost_per_day

    @property
    def planned_total(self) -> float:
        return (
            self.repair_usd * self.planned_repair_factor
            + self.planned_downtime_days * self.downtime_cost_per_day
        )

    @property
    def saving_per_detected_event(self) -> float:
        return self.emergency_total - self.planned_total


@dataclass(frozen=True)
class Config:
    fleet: FleetConfig = field(default_factory=FleetConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    # ---- path helpers -----------------------------------------------------------------
    @property
    def pressure_path(self) -> Path:
        return RAW_DIR / "pressure_1min.parquet"

    @property
    def cold_head_path(self) -> Path:
        return RAW_DIR / "cold_head_1min.parquet"

    @property
    def vibration_path(self) -> Path:
        return RAW_DIR / "vibration_window.parquet"

    @property
    def rf_path(self) -> Path:
        return RAW_DIR / "rf_window.parquet"

    @property
    def events_path(self) -> Path:
        return RAW_DIR / "events.csv"

    @property
    def features_path(self) -> Path:
        return PROCESSED_DIR / "features.parquet"

    @property
    def dataset_path(self) -> Path:
        return PROCESSED_DIR / "dataset.parquet"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / "lstm_detector.pt"

    @property
    def metrics_path(self) -> Path:
        return REPORTS_DIR / "metrics.json"

    @property
    def report_html_path(self) -> Path:
        return REPORTS_DIR / "quenchguard_report.html"


def default_config() -> Config:
    """Return the case-study configuration."""
    return Config()


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """JSON-serialisable view of the config (tuples -> lists)."""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return [_clean(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    return _clean(asdict(cfg))
