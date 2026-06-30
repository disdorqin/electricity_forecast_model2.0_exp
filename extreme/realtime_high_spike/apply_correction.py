# -*- coding: utf-8 -*-
"""
apply_correction.py — Apply spike correction to prediction packs.

Provides:
  - load_and_merge: load prediction pack + risk predictions, merge on time
  - run_correction: run full correction pipeline (lift + guardrail) over a DataFrame
  - run_correction_with_profile: version that accepts a profile config dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme.realtime_high_spike.guardrail import (
    GuardrailConfig,
    SpikeGuardrail,
)
from extreme.realtime_high_spike.residual_lift import (
    PERIOD_DEFS,
    CorrectionMode,
    ResidualLiftConfig,
    ResidualLiftCorrector,
    get_period,
)


# ── Profile representation ───────────────────────────────────────────

@dataclass
class CorrectionProfile:
    """A named tuning profile that drives both lift and guardrail configs.

    Attributes:
        name: Profile name (conservative / medium / aggressive).
        mode: CorrectionMode — RELAXED loosens thresholds for offline diagnosis.
        min_lift_floor: Minimum lift floor in RELAXED mode (price units).
        spike_prob_threshold: Minimum spike probability to apply lift.
        max_lift_ratio: Maximum lift as fraction of base_pred.
        max_absolute_lift: Maximum absolute lift in price units.
        protect_normal_hours: Extra guard in non-spike-prone periods.
        period_9_16_boost: Multiplier on 9_16 lift candidate.
    """
    name: str
    mode: CorrectionMode = CorrectionMode.NORMAL
    min_lift_floor: float = 0.0
    spike_prob_threshold: float = 0.6
    max_lift_ratio: float = 0.35
    max_absolute_lift: float = 350.0
    protect_normal_hours: bool = True
    period_9_16_boost: float = 1.15

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = CorrectionMode(self.mode)
        # In RELAXED mode, auto-set a sane min_lift_floor if not explicit
        if self.mode.is_relaxed() and self.min_lift_floor == 0.0:
            self.min_lift_floor = 30.0

    def to_lift_config(self) -> ResidualLiftConfig:
        return ResidualLiftConfig(
            spike_prob_threshold=self.spike_prob_threshold,
            max_lift_ratio=self.max_lift_ratio,
            max_absolute_lift=self.max_absolute_lift,
            protect_normal_hours=self.protect_normal_hours,
            period_9_16_boost=self.period_9_16_boost,
            lift_quantile=0.90,
            period_aware=True,
            normal_hour_prob_cap=self.spike_prob_threshold * 1.1,
            mode=self.mode,
            min_lift_floor=self.min_lift_floor,
        )

    def to_guardrail_config(self) -> GuardrailConfig:
        return GuardrailConfig(
            min_prob_for_lift=self.spike_prob_threshold,
            protect_normal_hours=self.protect_normal_hours,
            normal_hour_prob_cap=self.spike_prob_threshold * 1.1,
            max_lift_ratio_9_16=self.max_lift_ratio,
            max_absolute_lift_9_16=self.max_absolute_lift,
            max_lift_ratio_1_8=self.max_lift_ratio * 0.7,
            max_absolute_lift_1_8=self.max_absolute_lift * 0.6,
            max_lift_ratio_17_24=self.max_lift_ratio * 0.7,
            max_absolute_lift_17_24=self.max_absolute_lift * 0.6,
            mode=self.mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_used": self.name,
            "correction_mode": self.mode.value,
            "spike_prob_threshold": self.spike_prob_threshold,
            "max_lift_ratio": self.max_lift_ratio,
            "max_absolute_lift": self.max_absolute_lift,
            "protect_normal_hours": self.protect_normal_hours,
            "period_9_16_boost": self.period_9_16_boost,
            "min_lift_floor": self.min_lift_floor,
        }

    def to_dict_effective(self) -> dict[str, Any]:
        """Return the *effective* parameters after mode adjustment.

        This shows what thresholds are actually used at runtime.
        """
        lc = self.to_lift_config()
        return {
            "profile_used": self.name,
            "correction_mode": self.mode.value,
            "effective_spike_prob_threshold": (
                lc.spike_prob_threshold * 0.6 if self.mode.is_relaxed()
                else lc.spike_prob_threshold
            ),
            "lift_floor_applied": self.min_lift_floor if self.mode.is_relaxed() else 0.0,
        }


# ── Profile loader ───────────────────────────────────────────────────

def load_profile_config(path: str | Path) -> dict[str, Any]:
    """Load profile configuration from YAML or JSON file.

    Args:
        path: Path to .yaml or .json config file.

    Returns:
        Dictionary mapping profile name → profile parameters dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")

    raw = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML profile configs. "
                "Install with: pip install pyyaml"
            )
        profiles = yaml.safe_load(raw)
    elif path.suffix == ".json":
        profiles = json.loads(raw)
    else:
        raise ValueError(f"Unsupported profile config format: {path.suffix} (use .yaml or .json)")

    if not isinstance(profiles, dict):
        raise ValueError(f"Profile config must be a dict, got {type(profiles).__name__}")

    return profiles


def get_profile(
    profile_name: str,
    config_path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
    mode: Optional[CorrectionMode] = None,
) -> CorrectionProfile:
    """Resolve a CorrectionProfile by name from a config file, with optional overrides.

    Args:
        profile_name: Name of the profile (conservative / medium / aggressive / all),
                      or 'all' to indicate bulk run.
        config_path: Path to profile YAML/JSON config file.
        overrides: Optional explicit overrides (takes precedence over profile values).
        mode: CorrectionMode override (RELAXED = offline-only looser thresholds).

    Returns:
        CorrectionProfile instance.
    """
    params: dict[str, Any] = {}

    if config_path is not None:
        all_profiles = load_profile_config(config_path)
        if profile_name not in all_profiles:
            valid = list(all_profiles.keys())
            raise ValueError(
                f"Unknown profile '{profile_name}'. Available: {valid}"
            )
        base = all_profiles[profile_name]
        if isinstance(base, dict):
            params.update(base)

    # Apply overrides (explicit CLI args take precedence)
    if overrides:
        params.update(overrides)

    # Set mode if provided
    if mode is not None:
        params["mode"] = mode

    return CorrectionProfile(name=profile_name, **params)


# ── I/O helpers ──────────────────────────────────────────────────────

def load_and_merge(
    prediction_pack_path: str | Path,
    risk_predictions_path: str | Path,
) -> pd.DataFrame:
    """Load prediction pack and risk predictions, then merge on time.

    Expected columns in prediction_pack:
        - business_day, hour_business, y_true, base_fused_pred, ...

    Expected columns in risk_predictions:
        - business_day, hour_business, high_spike_prob, ...

    Returns:
        Merged DataFrame.
    """
    pp = pd.read_csv(prediction_pack_path)
    rp = pd.read_csv(risk_predictions_path)

    # Ensure string keys for merge
    for col in ("business_day",):
        if col in pp.columns:
            pp[col] = pp[col].astype(str)
        if col in rp.columns:
            rp[col] = rp[col].astype(str)

    merged = pd.merge(
        pp, rp,
        on=["business_day", "hour_business"],
        how="left",
        suffixes=("", "_risk"),
    )
    return merged


# ── Core correction runner ───────────────────────────────────────────

def run_correction(
    prediction_pack_path: str | Path,
    risk_predictions_path: str | Path,
    history_df: Optional[pd.DataFrame] = None,
    profile: Optional[CorrectionProfile] = None,
    lift_config: Optional[ResidualLiftConfig] = None,
    guardrail_config: Optional[GuardrailConfig] = None,
) -> pd.DataFrame:
    """Run the full correction pipeline over a merged prediction + risk dataset.

    Pipeline:
        1. Load and merge prediction pack + risk predictions
        2. Fit lift corrector on history (or use default candidates)
        3. Compute lift for each row
        4. Apply guardrail
        5. Return augmented DataFrame

    Args:
        prediction_pack_path: Path to prediction pack CSV.
        risk_predictions_path: Path to risk predictions CSV.
        history_df: Optional historical DataFrame for fitting lift quantiles.
        profile: Optional CorrectionProfile (overrides lift_config/guardrail_config).
        lift_config: ResidualLiftConfig (used if profile is None).
        guardrail_config: GuardrailConfig (used if profile is None).

    Returns:
        DataFrame with columns added: spike_corrected_pred, final_pred, reason_code,
        lift_applied, spike_prob, profile_used.
    """
    merged = load_and_merge(prediction_pack_path, risk_predictions_path)

    # Resolve configs from profile or direct
    if profile is not None:
        lcfg = profile.to_lift_config()
        gcfg = profile.to_guardrail_config()
        profile_meta = profile.to_dict()
    else:
        lcfg = lift_config or ResidualLiftConfig()
        gcfg = guardrail_config or GuardrailConfig()
        profile_meta = {"profile_used": "custom"}

    # Fit corrector
    corrector = ResidualLiftCorrector(lcfg)
    if history_df is not None and not history_df.empty:
        corrector.fit_from_history(history_df)
    else:
        corrector.set_lift_candidates({p: 50.0 for p in PERIOD_DEFS})
        corrector._lift_candidates["9_16"] *= lcfg.period_9_16_boost

    # Guardrail
    guardrail = SpikeGuardrail(gcfg)

    # Apply row-by-row
    spike_corrected_list: list[float] = []
    final_pred_list: list[float] = []
    reason_code_list: list[str] = []
    lift_applied_list: list[float] = []

    for _, row in merged.iterrows():
        base_pred = row.get("base_fused_pred", 0.0)
        spike_prob = row.get("high_spike_prob", 0.0)
        hour_business = row.get("hour_business", 12)

        if pd.isna(base_pred):
            base_pred = 0.0
        if pd.isna(spike_prob):
            spike_prob = 0.0

        # Step 1: compute lift
        lift_result = corrector.compute_lift(
            base_pred=float(base_pred),
            spike_prob=float(spike_prob),
            hour_business=int(hour_business),
        )
        corrected = lift_result.corrected_pred

        # Step 2: guardrail
        guard_result = guardrail.evaluate(
            base_pred=float(base_pred),
            spike_prob=float(spike_prob),
            corrected_pred=corrected,
            hour_business=int(hour_business),
        )

        spike_corrected_list.append(corrected)
        final_pred_list.append(guard_result.final_pred)
        reason_code_list.append(guard_result.reason_code)
        lift_applied_list.append(guard_result.final_pred - float(base_pred))

    merged["spike_corrected_pred"] = spike_corrected_list
    merged["final_pred"] = final_pred_list
    merged["reason_code"] = reason_code_list
    merged["lift_applied"] = lift_applied_list
    merged["spike_prob"] = merged.get("high_spike_prob", np.nan)

    # Attach profile metadata as attributes
    for key, val in profile_meta.items():
        merged.attrs[key] = val

    return merged


# ── Correction manifest writer ────────────────────────────────────────

def diagnose_zero_lift(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
    threshold_col: str = "spike_prob_threshold",
) -> None:
    """Print diagnostic information about rows where lift_applied == 0.

    Groups zero-lift rows by reason_code and shows representative samples
    so the user can understand why correction did not fire.

    Args:
        df: Corrected DataFrame (must contain lift_applied, reason_code,
            spike_prob, base_fused_pred, hour_business columns).
        top_n: Number of sample rows to show per reason.
        threshold_col: Column name for the spike-probability threshold value.
    """
    zero = df[df["lift_applied"] <= 0].copy()

    if len(zero) == 0:
        print("  [diagnose] No zero-lift rows found.")
        return

    print(f"\n  [diagnose] Total zero-lift rows: {len(zero)} / {len(df)}")

    for reason in zero["reason_code"].unique():
        subset = zero[zero["reason_code"] == reason]
        print(f"\n  reason_code='{reason}': {len(subset)} rows")

        if len(subset) > 0:
            cols = [c for c in [
                "business_day", "hour_business", "period",
                "spike_prob", "base_fused_pred", "lift_applied",
            ] if c in subset.columns]
            if not cols:
                cols = subset.columns[:5].tolist()
            samples = subset.head(top_n)[cols]
            for idx, row in samples.iterrows():
                vals = " | ".join(f"{c}={row[c]}" for c in cols)
                print(f"    row {idx}: {vals}")

    # Show reason distribution
    print(f"\n  [diagnose] Reason code distribution:")
    for reason, count in zero["reason_code"].value_counts().items():
        print(f"    {reason}: {count}")


def write_correction_manifest(
    out_dir: str | Path,
    profile: CorrectionProfile,
    metrics: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a correction_manifest.json with profile metadata and optional metrics.

    Args:
        out_dir: Output directory.
        profile: CorrectionProfile used.
        metrics: Optional metrics dict to include.
        extra: Optional extra metadata.

    Returns:
        Path to the written manifest file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "profile_used": profile.name,
        "correction_mode": profile.mode.value,
        "spike_prob_threshold": profile.spike_prob_threshold,
        "max_lift_ratio": profile.max_lift_ratio,
        "max_absolute_lift": profile.max_absolute_lift,
        "protect_normal_hours": profile.protect_normal_hours,
        "period_9_16_boost": profile.period_9_16_boost,
    }
    if profile.mode.is_relaxed():
        effective = profile.to_dict_effective()
        manifest["effective_spike_prob_threshold"] = effective.get(
            "effective_spike_prob_threshold",
        )
        manifest["lift_floor_applied"] = effective.get("lift_floor_applied", 0.0)
        manifest["note"] = (
            "RELAXED mode is for offline diagnosis only. "
            "Do NOT use in production_pipeline."
        )
    if metrics:
        manifest["metrics"] = metrics
    if extra:
        manifest.update(extra)

    manifest_path = out_dir / "correction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path
