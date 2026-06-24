from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t)
        if trend.size(-1) != x_t.size(-1):
            trend = trend[..., : x_t.size(-1)]
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_mask_916: bool = False,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        # === v21 9-16 注意力偏置: 对 9-16 索引(8..15) 加可学习位置编码 ===
        self.attn_mask_916 = attn_mask_916
        if attn_mask_916:
            # 24 个时间步的加性位置偏置, 初始化为 0, 9-16 区间(索引 8..15)由训练激活
            self.hour_bias = nn.Parameter(torch.zeros(24, dtype=torch.float32))
            nn.init.normal_(self.hour_bias, mean=0.0, std=0.02)
            with torch.no_grad():
                # 9-16 区间初始给一点正向偏置(鼓励关注)
                self.hour_bias[8:16].fill_(0.1)
                self.hour_bias[:8].fill_(-0.05)
                self.hour_bias[16:24].fill_(-0.05)

    def forward(self, x: torch.Tensor, hour_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm(x)
        if self.attn_mask_916 and hour_ids is not None:
            # hour_ids: (B, T) 范围 0..23
            bias = self.hour_bias[hour_ids.clamp(0, 23)]  # (B, T)
            # 用加性偏置调制 QK 内积, 等价于给每个位置加一个 query/key 偏置
            # 这里通过对 v 加权方式近似, 仅增强 9-16 时间步
            weight = torch.sigmoid(bias).unsqueeze(-1)  # (B, T, 1)
            h = h * (1.0 + weight)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        h, _ = self.attn(q, k, k)
        query = query + self.drop(h)
        query = query + self.drop(self.ffn(self.norm2(query)))
        return query


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scales: int = 3,
        dropout: float = 0.1,
        use_attention: bool = False,
        attn_mask_916: bool = False,
    ):
        super().__init__()
        self.decomp = MovingAvg(kernel_size=25)
        self.use_attention = use_attention
        self.attn_mask_916 = attn_mask_916
        self.season_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(scales)
            ]
        )
        self.trend_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(scales)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(scales)])
        if use_attention:
            self.attention = TemporalSelfAttention(
                hidden_dim, n_heads=4, dropout=dropout, attn_mask_916=attn_mask_916
            )

    def forward(
        self,
        xs: list[torch.Tensor],
        hour_ids_per_scale: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        outs = []
        prev_s = None
        prev_t = None
        for i, x in enumerate(xs):
            s, t = self.decomp(x)
            if prev_s is not None:
                s = s + F.interpolate(
                    prev_s.transpose(1, 2),
                    size=s.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                t = t + F.interpolate(
                    prev_t.transpose(1, 2),
                    size=t.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            y = self.season_mlps[i](s) + self.trend_mlps[i](t)
            out = self.norms[i](x + y)
            if self.use_attention:
                if self.attn_mask_916 and hour_ids_per_scale is not None:
                    out = self.attention(out, hour_ids=hour_ids_per_scale[i])
                else:
                    out = self.attention(out)
            outs.append(out)
            prev_s, prev_t = s, t
        return outs


class Segment916Head(nn.Module):
    """9-16 区间专用微调预测头 (v21).

    输入: base model 的 future hidden state (B, 24, hidden_dim) + 9-16 hour embedding
    输出: 9-16 时段(索引 8..15, 8 个时间步)的残差修正量
    """

    # 9-16 区间的物理小时(0..23), 用于 hour embedding
    HOUR_IDS_916 = list(range(8, 16))

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # hour embedding 8 维 (对应 9..16 小时)
        self.hour_embed = nn.Embedding(8, 8)
        # 主干: hidden + hour_embed -> residual
        in_dim = hidden_dim + 8
        self.body = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # 残差 gate, 让模型学习"是否使用"segment head
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, future_hidden: torch.Tensor) -> torch.Tensor:
        """future_hidden: (B, 24, hidden_dim) -> 返回 (B, 8) 残差."""
        # 仅取 9-16 区间(索引 8..15)
        h916 = future_hidden[:, 8:16, :]  # (B, 8, hidden_dim)
        b, t, _ = h916.shape
        # hour embedding
        hour_ids = torch.arange(8, device=h916.device).clamp(0, 7)
        hour_ids = hour_ids.unsqueeze(0).expand(b, -1)  # (B, 8)
        h_emb = self.hour_embed(hour_ids)  # (B, 8, 8)
        # 拼接
        feat = torch.cat([h916, h_emb], dim=-1)  # (B, 8, hidden+8)
        residual = self.body(feat).squeeze(-1)  # (B, 8)
        gate = self.gate(feat).squeeze(-1)  # (B, 8)
        return gate * residual


class TimeMixerBackbone(nn.Module):
    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        pred_len: int = 24,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        scales: int = 3,
        dropout: float = 0.1,
        segment_head_mode: str = "none",
        attn_mask_916: bool = False,
    ):
        super().__init__()
        self.scales = scales
        self.segment_head_mode = segment_head_mode
        self.attn_mask_916 = attn_mask_916
        self.past_proj = nn.Sequential(
            nn.Linear(past_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.future_proj = nn.Sequential(
            nn.Linear(future_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    hidden_dim,
                    scales=scales,
                    dropout=dropout,
                    use_attention=True,
                    attn_mask_916=attn_mask_916,
                )
                for _ in range(n_blocks)
            ]
        )
        self.past_future_cross = CrossAttention(hidden_dim, n_heads=4, dropout=dropout)
        self.future_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * (scales + 1)),
            nn.Linear(hidden_dim * (scales + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        if self.segment_head_mode == "future_residual":
            self.future_step_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.future_residual_gate = nn.Sequential(
                nn.Linear(hidden_dim * (scales + 1), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len),
                nn.Sigmoid(),
            )
        # === v21: 9-16 区间专门的分段头 ===
        if self.segment_head_mode == "rt_916_segment_head":
            self.seg916_head = Segment916Head(hidden_dim, dropout=dropout)

    def make_scales(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = [x]
        cur = x
        for _ in range(1, self.scales):
            cur = F.avg_pool1d(
                cur.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True
            ).transpose(1, 2)
            xs.append(cur)
        return xs

    def forward(
        self,
        past_x: torch.Tensor,
        future_x: torch.Tensor,
        past_hour_ids: torch.Tensor | None = None,
        return_base: bool = False,
    ) -> torch.Tensor:
        x = self.past_proj(past_x)
        xs = self.make_scales(x)
        # 9-16 注意力偏置: 为每个 scale 准备子采样后的 hour_ids
        hour_ids_per_scale = None
        if self.attn_mask_916 and past_hour_ids is not None:
            hour_ids_per_scale = []
            cur_ids = past_hour_ids
            for s in range(self.scales):
                hour_ids_per_scale.append(cur_ids)
                # 与 make_scales 中的 avg_pool1d(kernel_size=2, stride=2) 对齐
                if s < self.scales - 1:
                    cur_ids = cur_ids[:, ::2]
                    # 末尾补齐, ceil_mode=True 的 AvgPool 会保留最后一个
                    if cur_ids.shape[1] < xs[s + 1].shape[1]:
                        pad = xs[s + 1].shape[1] - cur_ids.shape[1]
                        last = cur_ids[:, -1:].expand(-1, pad)
                        cur_ids = torch.cat([cur_ids, last], dim=1)
        if hour_ids_per_scale is not None:
            for block in self.blocks:
                xs = block(xs, hour_ids_per_scale=hour_ids_per_scale)
        else:
            for block in self.blocks:
                xs = block(xs)
        pooled = [s.mean(dim=1) for s in xs]
        future = self.future_proj(future_x)
        future = future + self.future_mixer(future)
        past_summary = x.mean(dim=1, keepdim=True).expand_as(future)
        future = self.past_future_cross(future, past_summary)
        z = torch.cat(pooled + [future.mean(dim=1)], dim=-1)
        out = self.head(z)
        if self.segment_head_mode == "future_residual":
            future_residual = self.future_step_head(future).squeeze(-1)
            gate = self.future_residual_gate(z)
            out = out + gate * future_residual
        elif self.segment_head_mode == "rt_916_segment_head":
            # 9-16 区间(索引 8..15)加残差修正
            seg_residual = self.seg916_head(future)  # (B, 8)
            if return_base:
                # 训练时返回 (boosted, base, residual) 三元组, 便于计算 boost loss
                base_out = out.clone()
                out_916 = out[:, 8:16] + seg_residual
                out = torch.cat([out[:, :8], out_916, out[:, 16:24]], dim=1)
                return out, base_out, seg_residual
            out_916 = out[:, 8:16] + seg_residual
            out = torch.cat([out[:, :8], out_916, out[:, 16:24]], dim=1)
        return out


class InceptionBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = [1, 3, 5]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernels
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * len(kernels), channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [branch(x) for branch in self.branches]
        return self.mix(torch.cat(ys, dim=1))


class TimesNetBackbone(nn.Module):
    """A lightweight TimesNet-style backbone that preserves the project I/O contract."""

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        pred_len: int = 24,
        hidden_dim: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
        segment_head_mode: str = "none",
    ):
        super().__init__()
        self.segment_head_mode = segment_head_mode
        self.past_proj = nn.Linear(past_dim, hidden_dim)
        self.future_proj = nn.Linear(future_dim, hidden_dim)
        self.time_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    InceptionBlock1D(hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )
        self.future_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        if self.segment_head_mode == "future_residual":
            self.future_step_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.future_residual_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len),
                nn.Sigmoid(),
            )

    def forward(self, past_x: torch.Tensor, future_x: torch.Tensor) -> torch.Tensor:
        x = self.past_proj(past_x).transpose(1, 2)
        for block in self.time_blocks:
            x = x + block(x)
        past_pool = x.mean(dim=-1)
        future = self.future_proj(future_x)
        future = future + self.future_mixer(future)
        future_pool = future.mean(dim=1)
        z = torch.cat([past_pool, future_pool], dim=-1)
        out = self.head(z)
        if self.segment_head_mode == "future_residual":
            future_residual = self.future_step_head(future).squeeze(-1)
            gate = self.future_residual_gate(z)
            out = out + gate * future_residual
        return out


def build_backbone(
    backbone_name: str,
    past_dim: int,
    future_dim: int,
    pred_len: int,
    hidden_dim: int,
    blocks: int,
    scales: int,
    dropout: float,
    segment_head_mode: str = "none",
    attn_mask_916: bool = False,
) -> nn.Module:
    name = backbone_name.lower()
    if name == "timemixer":
        return TimeMixerBackbone(
            past_dim=past_dim,
            future_dim=future_dim,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            n_blocks=blocks,
            scales=scales,
            dropout=dropout,
            segment_head_mode=segment_head_mode,
            attn_mask_916=attn_mask_916,
        )
    if name == "timesnet":
        return TimesNetBackbone(
            past_dim=past_dim,
            future_dim=future_dim,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            n_blocks=blocks,
            dropout=dropout,
            segment_head_mode=segment_head_mode,
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")
