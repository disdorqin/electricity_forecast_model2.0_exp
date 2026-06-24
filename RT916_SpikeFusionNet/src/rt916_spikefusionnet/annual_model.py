﻿﻿﻿from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rt916_spikefusionnet.model import (
    DataEmbedding,
    SpikeResidualBranch,
    TimesBlock,
)


class CalendarRegimeGate(nn.Module):
    def __init__(self, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 10, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
            nn.Sigmoid(),
        )

    def forward(self, base_context, spike_context, calendar_feats):
        gate_input = torch.cat([base_context, spike_context, calendar_feats], dim=-1)
        return self.gate(gate_input)


class AnnualSpikeGatedTimesNet(nn.Module):
    def __init__(
        self,
        num_variates,
        seq_len,
        pred_len,
        d_model=128,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        dropout=0.1,
        target_index=-1,
        known_target_len=None,
        delta_scale=0.12,
        editable_horizon=(9, 16),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_index = target_index
        self.known_target_len = known_target_len if known_target_len is not None else (seq_len - 2 * pred_len)
        self.delta_scale = float(delta_scale)
        self.editable_horizon = editable_horizon

        self.embedding = DataEmbedding(num_variates, d_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TimesBlock(seq_len, pred_len, d_model, top_k=top_k, num_kernels=num_kernels) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.hour_embed = nn.Embedding(25, 8)
        self.calendar_proj = nn.Sequential(
            nn.Linear(8 + 4 + 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 10),
        )
        self.spike_branch = SpikeResidualBranch(self.known_target_len, pred_len, d_model, dropout=dropout)
        self.dynamic_gate = CalendarRegimeGate(d_model, pred_len, dropout=dropout)

    def _calendar_features(self, x):
        bsz = x.size(0)
        hour_raw = torch.arange(1, self.pred_len + 1, device=x.device).unsqueeze(0).repeat(bsz, 1)
        hour_idx = hour_raw.long().clamp(1, 24)
        hour_emb = self.hour_embed(hour_idx)
        hour_sin = torch.sin(2 * math.pi * hour_raw / 24.0).unsqueeze(-1)
        hour_cos = torch.cos(2 * math.pi * hour_raw / 24.0).unsqueeze(-1)
        editable = ((hour_idx >= self.editable_horizon[0]) & (hour_idx <= self.editable_horizon[1])).float().unsqueeze(-1)
        hist_stats = torch.stack(
            [
                x.mean(dim=(1, 2)),
                x.std(dim=(1, 2)),
                x[:, -1, :].mean(dim=1),
                x[:, : self.known_target_len, self.target_index].std(dim=1),
            ],
            dim=-1,
        ).unsqueeze(1).repeat(1, self.pred_len, 1)
        feats = torch.cat([hour_emb, hist_stats, hour_sin, hour_cos, editable], dim=-1)
        return self.calendar_proj(feats).mean(dim=1), editable.squeeze(-1)

    def forward(self, x, anchor_pred=None):
        enc = self.embedding(x)
        for block in self.blocks:
            enc = self.norm(block(enc))

        future_tokens = enc[:, -self.pred_len :, :]
        base_pred = self.out_proj(future_tokens).squeeze(-1)
        base_context = future_tokens.mean(dim=1)

        target_hist = x[:, : self.known_target_len, self.target_index]
        spike_delta, spike_context = self.spike_branch(target_hist)
        spike_delta = self.delta_scale * torch.tanh(spike_delta)
        calendar_feats, editable_mask = self._calendar_features(x)
        gate = self.dynamic_gate(base_context, spike_context, calendar_feats)

        pred = base_pred + gate * spike_delta
        if anchor_pred is not None:
            anchor_pred = anchor_pred.to(pred.dtype)
            pred = editable_mask * pred + (1.0 - editable_mask) * anchor_pred
        pred = torch.clamp(pred, -0.35, 1.80)
        return pred


# ====================================================================
#  W4: AnnualSpikeGatedTimesNetV2
#  - 继承基类结构,新增 "9-16 segment-specific head":
#     * 对 9-16 时段单独一个 head,基于 base+spike_context+DA-aware hist
#       stats 做一次修正
#  - 初始化为接近恒等(zero-initialized delta),保证基线不退化
#  - 训练时配合 SMAPE-floor50 loss,在 9-16 时段直接优化业务目标
# ====================================================================
class AnnualSpikeGatedTimesNetV2(AnnualSpikeGatedTimesNet):
    def __init__(
        self,
        num_variates,
        seq_len,
        pred_len,
        d_model=128,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        dropout=0.1,
        target_index=-1,
        known_target_len=None,
        delta_scale=0.12,
        editable_horizon=(9, 16),
        segment_head_scale=0.20,
    ):
        super().__init__(
            num_variates=num_variates,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            e_layers=e_layers,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
            target_index=target_index,
            known_target_len=known_target_len,
            delta_scale=delta_scale,
            editable_horizon=editable_horizon,
        )
        self.segment_head_scale = float(segment_head_scale)
        # segment-specific head:
        #   - 全部用 Kaiming 随机初始化,保证梯度能正常流回
        #   - 训练初期 seg_delta 输出量级自然,不会立刻主导 baseline
        self.segment_head = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        for m in self.segment_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def _segment_features(self, x, base_context, spike_context, da_pred_today):
        """
        拼一个 segment 专用特征向量:
          - base_context   : d_model
          - spike_context  : d_model
          - hist stats(4)  : mean / std / min / max of da_pred_today over 24h lag
        """
        if da_pred_today is None:
            hist_stats = torch.zeros(x.size(0), 4, device=x.device, dtype=base_context.dtype)
        else:
            hist_stats = torch.stack(
                [
                    da_pred_today.mean(dim=1, keepdim=True),
                    da_pred_today.std(dim=1, keepdim=True),
                    da_pred_today.min(dim=1, keepdim=True).values,
                    da_pred_today.max(dim=1, keepdim=True).values,
                ],
                dim=1,
            ).squeeze(1)
            if hist_stats.dim() == 3:
                hist_stats = hist_stats.squeeze(-1)
        return torch.cat([base_context, spike_context, hist_stats], dim=-1)

    def forward(self, x, anchor_pred=None, da_pred_today=None):
        enc = self.embedding(x)
        for block in self.blocks:
            enc = self.norm(block(enc))

        future_tokens = enc[:, -self.pred_len :, :]
        base_pred = self.out_proj(future_tokens).squeeze(-1)
        base_context = future_tokens.mean(dim=1)

        target_hist = x[:, : self.known_target_len, self.target_index]
        spike_delta, spike_context = self.spike_branch(target_hist)
        spike_delta = self.delta_scale * torch.tanh(spike_delta)
        calendar_feats, _ = self._calendar_features(x)

        # W4: 使用 self-attention 中的 editable_horizon 对整段 mask
        # 9-16 stage 的 pred_len=8 对应小时 9..16,editable 段占整段 100%
        # 1-8 / 17-0 stage 的 pred_len=8 全部不在 9-16 区间
        h = self.pred_len
        if h == int(self.editable_horizon[1]) - int(self.editable_horizon[0]) + 1:
            editable_mask = torch.ones(1, h, 1, device=x.device, dtype=enc.dtype)
        else:
            editable_mask = torch.zeros(1, h, 1, device=x.device, dtype=enc.dtype)
            lo = max(0, int(self.editable_horizon[0]) - 1)
            hi = min(h, int(self.editable_horizon[1]))
            if hi > lo:
                editable_mask[:, lo:hi, :] = 1.0
        gate = self.dynamic_gate(base_context, spike_context, calendar_feats)

        pred = base_pred + gate * spike_delta

        # 9-16 segment head: 仅在 editable 段(9-16)上施加 tanh-bounded 修正
        seg_input = self._segment_features(x, base_context, spike_context, da_pred_today)
        # tanh 保证 seg_delta 在 [-segment_head_scale, +segment_head_scale] 范围
        # 防止极端值破坏 baseline
        seg_raw = self.segment_head(seg_input)
        seg_delta = self.segment_head_scale * torch.tanh(seg_raw)
        # editable_mask: [1, h, 1] -> broadcast over [B, h]
        pred = pred + editable_mask.squeeze(-1) * seg_delta

        if anchor_pred is not None:
            anchor_pred = anchor_pred.to(pred.dtype)
            # editable_mask for non-9-16 段全 0,正确合并 anchor
            em = editable_mask.squeeze(-1)
            pred = em * pred + (1.0 - em) * anchor_pred
        # 保持与 V1 baseline 相同的 clamp 范围
        pred = torch.clamp(pred, -0.35, 1.80)
        return pred

