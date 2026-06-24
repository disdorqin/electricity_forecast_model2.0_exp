from __future__ import annotations

from .repro_pipeline import (
    SEGMENTS,
    assign_period,
    business_hour,
    load_data,
)

__all__ = ["SEGMENTS", "assign_period", "business_hour", "load_data", "predict_9_16_boosted"]


def predict_9_16_boosted(
    base_pred: "np.ndarray",
    seg916_residual: "np.ndarray",
    *,
    segment_idx: int = 1,
) -> "np.ndarray":
    """对 9-16 区间(SEGMENTS 中的索引 1, 物理小时 8..15)加 segment head 残差修正.

    参数
    ----------
    base_pred: (N, 24) 数组, base model 对 24 小时的预测
    seg916_residual: (N, 8) 数组, Segment916Head 对 9-16 区间的残差输出
    segment_idx: 9-16 区间在 SEGMENTS 中的索引(默认 1, 即 9-16)

    返回
    ----
    boosted_pred: (N, 24) 数组, 9-16 区间加上残差, 其余保持不变
    """
    import numpy as np

    base_pred = np.asarray(base_pred, dtype=float)
    seg_residual = np.asarray(seg916_residual, dtype=float)
    if base_pred.ndim != 2 or base_pred.shape[1] != 24:
        raise ValueError(f"base_pred 必须是 (N, 24) 数组, 当前 shape={base_pred.shape}")
    if seg_residual.ndim != 2 or seg_residual.shape[1] != 8:
        raise ValueError(
            f"seg916_residual 必须是 (N, 8) 数组, 当前 shape={seg_residual.shape}"
        )
    if seg_residual.shape[0] != base_pred.shape[0]:
        raise ValueError(
            f"base_pred 和 seg916_residual 样本数不匹配: "
            f"{base_pred.shape[0]} vs {seg_residual.shape[0]}"
        )
    _, start, end = SEGMENTS[segment_idx]
    if (end - start) != 8:
        raise ValueError(
            f"SEGMENTS[{segment_idx}] 不是 8 小时区间, 无法激活 9-16 boost"
        )
    out = base_pred.copy()
    out[:, start:end] = out[:, start:end] + seg_residual
    return out
