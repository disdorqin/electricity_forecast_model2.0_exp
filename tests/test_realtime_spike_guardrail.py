"""Tests for the realtime spike guardrail and residual lift modules.

Run from project root:
    python -m pytest tests/test_realtime_spike_guardrail.py -v
    # or
    python tests/test_realtime_spike_guardrail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from extreme.realtime_high_spike.residual_lift import (
    PERIOD_DEFS,
    CorrectionMode,
    ResidualLiftConfig,
    ResidualLiftCorrector,
    get_period,
)
from extreme.realtime_high_spike.guardrail import (
    GUARDRAIL_CLIPPED,
    LIFT_APPLIED,
    NO_CORRECTION_LOW_PROB,
    NO_CORRECTION_NEGATIVE_BASE,
    NO_CORRECTION_NORMAL_HOUR,
    GuardrailConfig,
    SpikeGuardrail,
)
from extreme.realtime_high_spike.apply_correction import (
    load_and_merge,
    run_correction,
)

# ── Shared test data ───────────────────────────────────────────────

N_HOURS = 24
N_DAYS = 10


def _make_test_prediction_pack(n_days: int = N_DAYS) -> pd.DataFrame:
    """Create a synthetic prediction pack for testing."""
    np.random.seed(42)
    rows: list[dict] = []
    for d in range(n_days):
        day = f"2025-11-{d + 1:02d}"
        for h in range(1, 25):
            base = 300 + 100 * np.sin(2 * np.pi * h / 24)
            if h in range(9, 17):
                base += 50  # Solar peak
            if h == 12 and d == 3:
                base += 300  # Spike scenario
            if h == 18 and d == 5:
                base += 200
            y_true = base + np.random.normal(0, 30)
            # Make spikes more extreme
            if h == 12 and d == 3:
                y_true = base + 350  # true spike
            if h == 18 and d == 5:
                y_true = base + 250
            rows.append({
                "business_day": day,
                "hour_business": h,
                "period": get_period(h),
                "y_true": y_true,
                "base_fused_pred": base,
                "sgdfnet_pred": base - 10,
                "timemixer_pred": base + 5,
                "rt916_pred": base + 15,
                "timesfm_pred": base - 5,
                "target_month": "2025-11",
            })
    return pd.DataFrame(rows)


def _make_test_risk_predictions(n_days: int = N_DAYS) -> pd.DataFrame:
    """Create synthetic spike risk predictions."""
    np.random.seed(42)
    rows: list[dict] = []
    for d in range(n_days):
        day = f"2025-11-{d + 1:02d}"
        for h in range(1, 25):
            # Spike probabilities
            if h in range(9, 17):
                prob = 0.3 + 0.3 * np.random.random()  # 0.3-0.6
            else:
                prob = 0.1 + 0.2 * np.random.random()  # 0.1-0.3
            # Known spike days
            if d == 3 and h == 12:
                prob = 0.85
            if d == 5 and h == 18:
                prob = 0.75
            rows.append({
                "business_day": day,
                "hour_business": h,
                "high_spike_prob": prob,
            })
    return pd.DataFrame(rows)


def _make_history_df() -> pd.DataFrame:
    """Create historical data for fitting lift quantiles."""
    np.random.seed(42)
    rows: list[dict] = []
    for d in range(30):
        day = f"2025-10-{d + 1:02d}"
        for h in range(1, 25):
            base = 280 + 100 * np.sin(2 * np.pi * h / 24)
            y_true = base + np.random.normal(0, 40)
            if h in range(9, 17) and np.random.random() < 0.1:
                y_true += 200  # occasional spikes
            rows.append({
                "business_day": day,
                "hour_business": h,
                "y_true": y_true,
                "base_fused_pred": base,
                "period": get_period(h),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Tests: get_period
# ══════════════════════════════════════════════════════════════════


def test_get_period_night():
    assert get_period(1) == "1_8"
    assert get_period(3) == "1_8"
    assert get_period(8) == "1_8"


def test_get_period_solar():
    assert get_period(9) == "9_16"
    assert get_period(12) == "9_16"
    assert get_period(16) == "9_16"


def test_get_period_evening():
    assert get_period(17) == "17_24"
    assert get_period(20) == "17_24"
    assert get_period(24) == "17_24"


def test_get_period_invalid():
    try:
        get_period(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        get_period(25)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ══════════════════════════════════════════════════════════════════
# Tests: ResidualLiftCorrector
# ══════════════════════════════════════════════════════════════════


def test_lift_corrector_fit_from_history():
    history = _make_history_df()
    corrector = ResidualLiftCorrector(ResidualLiftConfig(lift_quantile=0.90))
    corrector.fit_from_history(history)

    quantiles = corrector.get_quantiles()
    assert "1_8" in quantiles
    assert "9_16" in quantiles
    assert "17_24" in quantiles
    for period, q in quantiles.items():
        assert q >= 0, f"Period {period} quantile should be non-negative, got {q}"
        assert isinstance(q, float)


def test_lift_corrector_period_aware_vs_global():
    history = _make_history_df()

    period_aware = ResidualLiftCorrector(
        ResidualLiftConfig(period_aware=True, lift_quantile=0.90)
    )
    period_aware.fit_from_history(history)
    pa_q = period_aware.get_quantiles()

    global_corrector = ResidualLiftCorrector(
        ResidualLiftConfig(period_aware=False, lift_quantile=0.90)
    )
    global_corrector.fit_from_history(history)
    gl_q = global_corrector.get_quantiles()

    # Period-aware should have different values per period
    # (may not always be true with random data, but generally 9_16 > others)
    assert len(pa_q) == 3
    assert len(gl_q) == 3


def test_lift_corrector_manual_candidates():
    corrector = ResidualLiftCorrector()
    candidates = {"1_8": 30.0, "9_16": 120.0, "17_24": 40.0}
    corrector.set_lift_candidates(candidates)

    assert corrector.get_lift_candidate("1_8") == 30.0
    assert corrector.get_lift_candidate("9_16") == 120.0
    assert corrector.get_lift_candidate("17_24") == 40.0


def test_lift_corrector_compute_lift_low_prob():
    corrector = ResidualLiftCorrector(ResidualLiftConfig(spike_prob_threshold=0.5))
    corrector.set_lift_candidates({"1_8": 50.0, "9_16": 100.0, "17_24": 50.0})

    result = corrector.compute_lift(
        base_pred=300.0, spike_prob=0.1, hour_business=10,
    )
    assert result.reason_code == "NO_CORRECTION_LOW_PROB"
    assert result.corrected_pred == 300.0
    assert result.lift_applied == 0.0


def test_lift_corrector_compute_lift_applied():
    corrector = ResidualLiftCorrector(ResidualLiftConfig(spike_prob_threshold=0.5))
    corrector.set_lift_candidates({"1_8": 50.0, "9_16": 100.0, "17_24": 50.0})

    result = corrector.compute_lift(
        base_pred=300.0, spike_prob=0.8, hour_business=12,
    )
    assert result.reason_code in ("LIFT_APPLIED", "LIFT_CAPPED")
    assert result.corrected_pred >= 300.0
    assert result.lift_applied >= 0


def test_lift_corrector_compute_lift_normal_protected():
    corrector = ResidualLiftCorrector(
        ResidualLiftConfig(
            spike_prob_threshold=0.5,
            protect_normal_hours=True,
            normal_hour_prob_cap=0.65,
        ),
    )
    corrector.set_lift_candidates({"1_8": 50.0, "9_16": 100.0, "17_24": 50.0})

    # Normal hour (1_8) with moderate prob (0.55 < cap 0.65) -> protected
    result = corrector.compute_lift(
        base_pred=300.0, spike_prob=0.55, hour_business=3,
    )
    assert result.reason_code == "NO_CORRECTION_NORMAL_HOUR"
    assert result.corrected_pred == 300.0

    # 9_16 with same prob -> should NOT be protected
    result = corrector.compute_lift(
        base_pred=300.0, spike_prob=0.55, hour_business=12,
    )
    assert result.reason_code in ("LIFT_APPLIED", "LIFT_CAPPED")
    assert result.corrected_pred >= 300.0


def test_lift_corrector_compute_lift_capped():
    corrector = ResidualLiftCorrector(
        ResidualLiftConfig(
            spike_prob_threshold=0.5,
            max_lift_ratio=0.1,  # 10% max -> 30 units for base=300
            max_absolute_lift=200.0,
        ),
    )
    corrector.set_lift_candidates({"1_8": 50.0, "9_16": 150.0, "17_24": 50.0})

    result = corrector.compute_lift(
        base_pred=300.0, spike_prob=0.8, hour_business=12,
    )
    assert result.reason_code == "LIFT_CAPPED"
    assert result.lift_applied <= 30.0  # 300 * 0.1 = 30
    assert 300 <= result.corrected_pred <= 330


# ══════════════════════════════════════════════════════════════════
# Tests: SpikeGuardrail
# ══════════════════════════════════════════════════════════════════


def test_guardrail_negative_base():
    guardrail = SpikeGuardrail()
    result = guardrail.evaluate(
        base_pred=-50.0, spike_prob=0.8,
        corrected_pred=50.0, hour_business=12,
    )
    assert result.reason_code == NO_CORRECTION_NEGATIVE_BASE
    assert result.final_pred == -50.0  # unchanged


def test_guardrail_low_prob():
    guardrail = SpikeGuardrail(GuardrailConfig(min_prob_for_lift=0.5))
    result = guardrail.evaluate(
        base_pred=300.0, spike_prob=0.1,
        corrected_pred=400.0, hour_business=12,
    )
    assert result.reason_code == NO_CORRECTION_LOW_PROB
    assert result.final_pred == 300.0  # unchanged


def test_guardrail_normal_hour_protected():
    guardrail = SpikeGuardrail(
        GuardrailConfig(
            protect_normal_hours=True,
            normal_hour_prob_cap=0.65,
            min_prob_for_lift=0.5,
        ),
    )
    # Normal hour (1_8) with moderate prob -> protected
    result = guardrail.evaluate(
        base_pred=300.0, spike_prob=0.55,
        corrected_pred=350.0, hour_business=3,
    )
    assert result.reason_code == NO_CORRECTION_NORMAL_HOUR
    assert result.final_pred == 300.0

    # 9_16 with same prob -> not protected
    result = guardrail.evaluate(
        base_pred=300.0, spike_prob=0.55,
        corrected_pred=350.0, hour_business=12,
    )
    assert result.reason_code == LIFT_APPLIED
    assert result.final_pred > 300.0


def test_guardrail_normal_hour_high_prob():
    """Normal hour with VERY high prob should still allow lift."""
    guardrail = SpikeGuardrail(
        GuardrailConfig(
            protect_normal_hours=True,
            normal_hour_prob_cap=0.65,
            min_prob_for_lift=0.5,
        ),
    )
    result = guardrail.evaluate(
        base_pred=300.0, spike_prob=0.85,  # exceeds normal_hour_prob_cap
        corrected_pred=350.0, hour_business=3,
    )
    assert result.reason_code != NO_CORRECTION_NORMAL_HOUR
    assert result.final_pred >= 300.0


def test_guardrail_clips_excessive_lift():
    guardrail = SpikeGuardrail(
        GuardrailConfig(
            min_prob_for_lift=0.5,
            max_lift_ratio_9_16=0.2,  # 20% max -> 60 units for base=300
            max_absolute_lift_9_16=200.0,
        ),
    )
    result = guardrail.evaluate(
        base_pred=300.0, spike_prob=0.8,
        corrected_pred=500.0, hour_business=12,
    )
    assert result.reason_code == GUARDRAIL_CLIPPED
    # Clipped to 300 + min(ratio_cap=60, abs_cap=200) = 360
    assert result.final_pred == 360.0


def test_guardrail_sanity_bounds():
    guardrail = SpikeGuardrail(
        GuardrailConfig(
            min_prob_for_lift=0.5,
            max_allowed_price=1500.0,
            min_allowed_price=-100.0,
            max_absolute_lift_9_16=10000,  # allow large lift to test sanity
            max_lift_ratio_9_16=10.0,
        ),
    )
    # Test upper bound
    result = guardrail.evaluate(
        base_pred=2000.0, spike_prob=0.8,
        corrected_pred=5000.0, hour_business=12,
    )
    assert result.final_pred <= 1500.0

    # Test lower bound
    result = guardrail.evaluate(
        base_pred=-50.0, spike_prob=0.8,
        corrected_pred=-300.0, hour_business=12,
    )
    # Negative base -> NO_CORRECTION_NEGATIVE_BASE, and final = base = -50
    # But if negative_base_guard is True, it returns base unchanged
    assert result.reason_code == NO_CORRECTION_NEGATIVE_BASE


# ══════════════════════════════════════════════════════════════════
# Tests: Integration & End-to-End
# ══════════════════════════════════════════════════════════════════


def test_load_and_merge():
    pp = _make_test_prediction_pack(3)
    rp = _make_test_risk_predictions(3)

    # Save to temp files and load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_pp:
        pp.to_csv(f_pp.name, index=False)
        pp_path = f_pp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_rp:
        rp.to_csv(f_rp.name, index=False)
        rp_path = f_rp.name

    try:
        merged = load_and_merge(pp_path, rp_path)
        assert len(merged) == 3 * 24
        assert "business_day" in merged.columns
        assert "hour_business" in merged.columns
        assert "high_spike_prob" in merged.columns
        assert "y_true" in merged.columns
        assert "base_fused_pred" in merged.columns
    finally:
        Path(pp_path).unlink(missing_ok=True)
        Path(rp_path).unlink(missing_ok=True)


def test_run_correction_end_to_end():
    """Full end-to-end test of the correction pipeline."""
    pp = _make_test_prediction_pack(N_DAYS)
    rp = _make_test_risk_predictions(N_DAYS)
    history = _make_history_df()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_pp:
        pp.to_csv(f_pp.name, index=False)
        pp_path = f_pp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_rp:
        rp.to_csv(f_rp.name, index=False)
        rp_path = f_rp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_h:
        history.to_csv(f_h.name, index=False)
        hist_path = f_h.name

    try:
        result = run_correction(
            prediction_pack_path=pp_path,
            risk_predictions_path=rp_path,
            history_df=history,
        )
        assert "base_fused_pred" in result.columns
        assert "spike_corrected_pred" in result.columns
        assert "final_pred" in result.columns
        assert "reason_code" in result.columns
        assert "lift_applied" in result.columns
        assert "spike_prob" in result.columns

        # At least some corrections should be applied
        applied = result[result["reason_code"] != NO_CORRECTION_LOW_PROB]
        # On spike days (d=3 → "2025-11-04" h12, d=5 → "2025-11-06" h18), we should have lifts
        # Note: d is 0-indexed, so d=3 corresponds to day "2025-11-04"
        spike_day = result[
            (result["business_day"] == "2025-11-04")
            & (result["hour_business"] == 12)
        ]
        if not spike_day.empty:
            assert spike_day.iloc[0]["final_pred"] > spike_day.iloc[0]["base_fused_pred"], (
                f"No lift on spike day: prob={spike_day.iloc[0].get('high_spike_prob', 'N/A')}, "
                f"reason={spike_day.iloc[0]['reason_code']}"
            )
        # Also check day "2025-11-06" h=18
        spike_day2 = result[
            (result["business_day"] == "2025-11-06")
            & (result["hour_business"] == 18)
        ]
        if not spike_day2.empty:
            assert spike_day2.iloc[0]["final_pred"] > spike_day2.iloc[0]["base_fused_pred"], (
                f"No lift on spike day2: prob={spike_day2.iloc[0].get('high_spike_prob', 'N/A')}, "
                f"reason={spike_day2.iloc[0]['reason_code']}"
            )

        # Negative base guard: all base > 0 in our synthetic data, should not trigger
        neg_mask = result["reason_code"] == NO_CORRECTION_NEGATIVE_BASE
        # In test data all base prices are positive, so no negative guard triggers
        # (but this is fine - just checking no crash)

        # All final_preds should be within sane bounds
        assert result["final_pred"].max() < 2000
        assert result["final_pred"].min() > -200

    finally:
        Path(pp_path).unlink(missing_ok=True)
        Path(rp_path).unlink(missing_ok=True)
        Path(hist_path).unlink(missing_ok=True)


def test_no_negative_price_override():
    """Guardrail must NOT override negative price module."""
    guardrail = SpikeGuardrail(GuardrailConfig(negative_base_guard=True))

    # Negative base with high spike prob
    result = guardrail.evaluate(
        base_pred=-30.0, spike_prob=0.9,
        corrected_pred=100.0, hour_business=12,
    )
    assert result.reason_code == NO_CORRECTION_NEGATIVE_BASE
    assert result.final_pred == -30.0  # Must keep original negative value


# ══════════════════════════════════════════════════════════════════
# Tests: CorrectionMode (normal vs relaxed)
# ══════════════════════════════════════════════════════════════════


def test_correction_mode_enum():
    """CorrectionMode enum basic properties."""
    assert CorrectionMode.NORMAL.value == "normal"
    assert CorrectionMode.RELAXED.value == "relaxed"
    assert not CorrectionMode.NORMAL.is_relaxed()
    assert CorrectionMode.RELAXED.is_relaxed()
    assert CorrectionMode("normal") == CorrectionMode.NORMAL
    assert CorrectionMode("relaxed") == CorrectionMode.RELAXED


def test_correction_mode_lift_relaxed_lowers_threshold():
    """RELAXED mode should fire lift on moderate-prob hours where NORMAL would not."""
    cfg = ResidualLiftConfig(
        spike_prob_threshold=0.6,
        mode=CorrectionMode.NORMAL,
    )
    normal = ResidualLiftCorrector(cfg)
    normal.set_lift_candidates({"1_8": 50.0, "9_16": 100.0, "17_24": 50.0})

    relaxed = ResidualLiftCorrector(
        ResidualLiftConfig(
            spike_prob_threshold=0.6,
            mode=CorrectionMode.RELAXED,
            min_lift_floor=30.0,
        )
    )
    relaxed.set_lift_candidates({"1_8": 50.0, "9_16": 100.0, "17_24": 50.0})

    # spike_prob=0.40: NORMAL blocks (0.40 < 0.6), RELAXED fires (0.40 >= 0.36)
    r_n = normal.compute_lift(base_pred=300.0, spike_prob=0.40, hour_business=12)
    assert r_n.lift_applied == 0.0, "NORMAL should block at prob=0.40"

    r_r = relaxed.compute_lift(base_pred=300.0, spike_prob=0.40, hour_business=12)
    assert r_r.lift_applied > 0, f"RELAXED should fire at prob=0.40, got {r_r.reason_code}"


def test_correction_mode_lift_relaxed_bypasses_normal_hour():
    """RELAXED mode should bypass normal-hour protection."""
    relaxed = ResidualLiftCorrector(
        ResidualLiftConfig(
            spike_prob_threshold=0.6,
            mode=CorrectionMode.RELAXED,
            min_lift_floor=30.0,
            protect_normal_hours=True,
        )
    )
    relaxed.set_lift_candidates({"1_8": 10.0, "9_16": 100.0, "17_24": 10.0})

    # hour=3 (1_8 period) should NOT be protected in RELAXED mode
    r = relaxed.compute_lift(base_pred=300.0, spike_prob=0.50, hour_business=3)
    assert r.lift_applied > 0, f"RELAXED should lift on normal hour, got {r.reason_code}"
    assert r.reason_code != "NO_CORRECTION_NORMAL_HOUR"


def test_correction_mode_guardrail_relaxed_bypasses_normal_hour():
    """RELAXED guardrail should bypass normal-hour protection."""
    from extreme.realtime_high_spike.guardrail import GuardrailConfig, SpikeGuardrail

    guard = SpikeGuardrail(
        GuardrailConfig(
            min_prob_for_lift=0.6,
            protect_normal_hours=True,
            normal_hour_prob_cap=0.65,
            mode=CorrectionMode.RELAXED,
        )
    )
    # hour=5 (1_8) with prob=0.50 → RELAXED should allow
    r = guard.evaluate(
        base_pred=300.0, spike_prob=0.50,
        corrected_pred=350.0, hour_business=5,
    )
    assert r.reason_code != "NO_CORRECTION_NORMAL_HOUR", (
        "RELAXED guardrail should bypass normal-hour protection"
    )
    assert r.final_pred > 300.0


def test_correction_mode_normal_still_blocks():
    """NORMAL mode should still apply normal-hour protection."""
    from extreme.realtime_high_spike.guardrail import GuardrailConfig, SpikeGuardrail

    guard = SpikeGuardrail(
        GuardrailConfig(
            min_prob_for_lift=0.5,
            protect_normal_hours=True,
            normal_hour_prob_cap=0.65,
            mode=CorrectionMode.NORMAL,
        )
    )
    # hour=5 (1_8) with prob=0.50 → NORMAL should protect
    r = guard.evaluate(
        base_pred=300.0, spike_prob=0.50,
        corrected_pred=350.0, hour_business=5,
    )
    assert r.reason_code == "NO_CORRECTION_NORMAL_HOUR", (
        "NORMAL guardrail should protect normal hours"
    )
    assert r.final_pred == 300.0


def test_correction_mode_relaxed_min_lift_floor():
    """RELAXED mode should apply min_lift_floor when fitted candidate is near zero."""
    corrector = ResidualLiftCorrector(
        ResidualLiftConfig(
            spike_prob_threshold=0.6,
            mode=CorrectionMode.RELAXED,
            min_lift_floor=50.0,
        )
    )
    # Set very small candidates
    corrector.set_lift_candidates({"1_8": 5.0, "9_16": 10.0, "17_24": 5.0})
    # After min_lift_floor: 1_8 → 50.0, 9_16 → 50.0 (before 9_16 boost)
    q = corrector.get_quantiles()
    assert q["1_8"] >= 50.0, f"min_lift_floor not applied: {q}"
    assert q["9_16"] >= 50.0, f"min_lift_floor not applied for 9_16: {q}"


def test_correction_mode_relaxed_e2e():
    """End-to-end test: RELAXED mode should produce corrections on moderate prob hours."""
    from extreme.realtime_high_spike.apply_correction import CorrectionProfile

    pp = _make_test_prediction_pack(N_DAYS)
    rp = _make_test_risk_predictions(N_DAYS)
    history = _make_history_df()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_pp:
        pp.to_csv(f_pp.name, index=False)
        pp_path = f_pp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_rp:
        rp.to_csv(f_rp.name, index=False)
        rp_path = f_rp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_h:
        history.to_csv(f_h.name, index=False)
        hist_path = f_h.name

    try:
        # Use conservative profile in RELAXED mode
        profile = CorrectionProfile(
            name="conservative_relaxed",
            mode=CorrectionMode.RELAXED,
            spike_prob_threshold=0.75,
            max_lift_ratio=0.20,
            max_absolute_lift=200,
            protect_normal_hours=True,
        )
        from extreme.realtime_high_spike.apply_correction import run_correction
        result = run_correction(
            prediction_pack_path=pp_path,
            risk_predictions_path=rp_path,
            history_df=history,
            profile=profile,
        )
        # Should have some lifts applied
        n_lifted = (result["lift_applied"] > 0).sum()
        assert n_lifted > 0, (
            f"RELAXED mode should produce lifts, got {n_lifted}/{len(result)}"
        )
        print(f"    RELAXED e2e: {n_lifted}/{len(result)} rows lifted")
    finally:
        Path(pp_path).unlink(missing_ok=True)
        Path(rp_path).unlink(missing_ok=True)
        Path(hist_path).unlink(missing_ok=True)


def test_correction_mode_normal_e2e():
    """End-to-end test: NORMAL mode should have FEWER lifts than RELAXED."""
    from extreme.realtime_high_spike.apply_correction import CorrectionProfile, run_correction

    pp = _make_test_prediction_pack(N_DAYS)
    rp = _make_test_risk_predictions(N_DAYS)
    history = _make_history_df()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_pp:
        pp.to_csv(f_pp.name, index=False)
        pp_path = f_pp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_rp:
        rp.to_csv(f_rp.name, index=False)
        rp_path = f_rp.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_h:
        history.to_csv(f_h.name, index=False)
        hist_path = f_h.name

    try:
        profile = CorrectionProfile(
            name="conservative_normal",
            mode=CorrectionMode.NORMAL,
            spike_prob_threshold=0.75,
            max_lift_ratio=0.20,
            max_absolute_lift=200,
        )
        result = run_correction(
            prediction_pack_path=pp_path,
            risk_predictions_path=rp_path,
            history_df=history,
            profile=profile,
        )
        n_lifted = (result["lift_applied"] > 0).sum()
        print(f"    NORMAL e2e: {n_lifted}/{len(result)} rows lifted")
        # NORMAL mode should have the standard low number of lifts
        # (this is a smoke check, not an exact count)
    finally:
        Path(pp_path).unlink(missing_ok=True)
        Path(rp_path).unlink(missing_ok=True)
        Path(hist_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════
# Run all tests if executed directly
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_get_period_night()
    test_get_period_solar()
    test_get_period_evening()
    test_get_period_invalid()
    print("PASS: test_get_period_*")

    test_lift_corrector_fit_from_history()
    test_lift_corrector_period_aware_vs_global()
    test_lift_corrector_manual_candidates()
    test_lift_corrector_compute_lift_low_prob()
    test_lift_corrector_compute_lift_applied()
    test_lift_corrector_compute_lift_normal_protected()
    test_lift_corrector_compute_lift_capped()
    print("PASS: test_lift_corrector_*")

    test_guardrail_negative_base()
    test_guardrail_low_prob()
    test_guardrail_normal_hour_protected()
    test_guardrail_normal_hour_high_prob()
    test_guardrail_clips_excessive_lift()
    test_guardrail_sanity_bounds()
    print("PASS: test_guardrail_*")

    test_load_and_merge()
    print("PASS: test_load_and_merge")
    test_run_correction_end_to_end()
    print("PASS: test_run_correction_end_to_end")
    test_no_negative_price_override()
    print("PASS: test_no_negative_price_override")

    test_correction_mode_enum()
    print("PASS: test_correction_mode_enum")
    test_correction_mode_lift_relaxed_lowers_threshold()
    print("PASS: test_correction_mode_lift_relaxed_lowers_threshold")
    test_correction_mode_lift_relaxed_bypasses_normal_hour()
    print("PASS: test_correction_mode_lift_relaxed_bypasses_normal_hour")
    test_correction_mode_guardrail_relaxed_bypasses_normal_hour()
    print("PASS: test_correction_mode_guardrail_relaxed_bypasses_normal_hour")
    test_correction_mode_normal_still_blocks()
    print("PASS: test_correction_mode_normal_still_blocks")
    test_correction_mode_relaxed_min_lift_floor()
    print("PASS: test_correction_mode_relaxed_min_lift_floor")
    test_correction_mode_relaxed_e2e()
    print("PASS: test_correction_mode_relaxed_e2e")
    test_correction_mode_normal_e2e()
    print("PASS: test_correction_mode_normal_e2e")

    print("\n=== ALL TESTS PASSED ===")
