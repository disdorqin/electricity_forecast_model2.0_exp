from __future__ import annotations

from .backbones import Segment916Head, TimeMixerBackbone, build_backbone

__all__ = ["Segment916Head", "TimeMixerBackbone", "build_backbone", "build_timemixer_with_916_head"]


def build_timemixer_with_916_head(
    past_dim: int,
    future_dim: int,
    pred_len: int = 24,
    hidden_dim: int = 64,
    n_blocks: int = 2,
    scales: int = 3,
    dropout: float = 0.1,
    attn_mask_916: bool = True,
) -> TimeMixerBackbone:
    """Convenience builder for v21 TimeMixer with 9-16 segment head + attention bias.

    这个便捷构造器把"9-16 分段头"与"9-16 注意力偏置"两个 v21 新结构同时打开,
    并保留与现有 TimeMixerBackbone 相同的 I/O 契约 (B, T, past_dim) / (B, 24, future_dim) -> (B, 24)。
    """
    return TimeMixerBackbone(
        past_dim=past_dim,
        future_dim=future_dim,
        pred_len=pred_len,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        scales=scales,
        dropout=dropout,
        segment_head_mode="rt_916_segment_head",
        attn_mask_916=attn_mask_916,
    )
