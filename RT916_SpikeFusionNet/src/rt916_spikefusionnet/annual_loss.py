﻿from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnualProtectedCappedLoss(nn.Module):
    def __init__(
        self,
        low_thr,
        high_thr,
        alpha=1.0,
        diff_alpha=0.25,
        huber_beta=0.05,
        mse_gamma=0.2,
        protected_weight=1.5,
        editable_horizon=(9, 16),
    ):
        super().__init__()
        self.low_thr = float(low_thr)
        self.high_thr = float(high_thr)
        self.alpha = float(alpha)
        self.diff_alpha = float(diff_alpha)
        self.huber_beta = float(huber_beta)
        self.mse_gamma = float(mse_gamma)
        self.protected_weight = float(protected_weight)
        self.editable_horizon = editable_horizon

    def forward(self, pred, target):
        err = pred - target
        tail_mask = ((target <= self.low_thr) | (target >= self.high_thr)).float()
        tail_weight = 1.0 + self.alpha * tail_mask
        diff = torch.zeros_like(target)
        if target.size(1) > 1:
            diff[:, 1:] = torch.abs(target[:, 1:] - target[:, :-1])
        diff_scale = diff / (diff.mean(dim=1, keepdim=True) + 1e-6)
        weight = tail_weight + self.diff_alpha * diff_scale.detach()
        huber = F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_beta)
        mse = err.square()
        base = (weight * huber).mean() + self.mse_gamma * (weight * mse).mean()

        h = target.size(1)
        editable = torch.zeros_like(target)
        lo, hi = self.editable_horizon
        lo = max(1, int(lo)) - 1
        hi = min(h, int(hi))
        editable[:, lo:hi] = 1.0
        protected = 1.0 - editable
        protected_penalty = (protected * err.square()).mean()
        return base + self.protected_weight * protected_penalty


# ==============================================================
#  W4: SegmentedSMAPECappedLoss
#  - 对 9-16 segment 单独施加 SMAPE-floor50 业务目标
#  - 对非 9-16 segment 保留原 MAE(Huber) 训练目标
#  - 业务目标 / 通用目标 比例可调(smape_alpha)
#  - smape_floor 与 smape_alpha 输入均在归一化后尺度([0,1] 空间),
#    内部通过 target_max_unscaled 把"业务 floor=50 元/MWh"换算到归一化空间
# ==============================================================
class SegmentedSMAPECappedLoss(nn.Module):
    def __init__(
        self,
        low_thr,
        high_thr,
        alpha=1.0,
        diff_alpha=0.25,
        huber_beta=0.05,
        mse_gamma=0.2,
        protected_weight=1.5,
        editable_horizon=(9, 16),
        smape_alpha=1.0,
        smape_floor=50.0,
        target_max_unscaled=600.0,
        eps=1e-3,
    ):
        super().__init__()
        self.low_thr = float(low_thr)
        self.high_thr = float(high_thr)
        self.alpha = float(alpha)
        self.diff_alpha = float(diff_alpha)
        self.huber_beta = float(huber_beta)
        self.mse_gamma = float(mse_gamma)
        self.protected_weight = float(protected_weight)
        self.editable_horizon = editable_horizon
        self.smape_alpha = float(smape_alpha)
        self.smape_floor_unscaled = float(smape_floor)
        # 把业务 floor(=50 元/MWh) 换算到 [0,1] 归一化空间
        self.smape_floor = float(smape_floor) / max(float(target_max_unscaled), 1.0)
        self.eps = float(eps)

    def _smape(self, pred, target):
        # 与 calc_smape 评估函数保持一致(在归一化空间 [0,1] 上):
        #   y = max(target, smape_floor_scaled)
        #   yp = max(pred, smape_floor_scaled)
        y = torch.where(target < self.smape_floor, torch.full_like(target, self.smape_floor), target)
        yp = torch.where(pred < self.smape_floor, torch.full_like(pred, self.smape_floor), pred)
        denom = (y.abs() + yp.abs()) / 2.0 + self.eps
        return (2.0 * (y - yp).abs() / denom)

    def forward(self, pred, target):
        h = target.size(1)
        # 业务目标段(默认 9-16),因为我们是在 9-16 stage 上训练,
        # 该 stage 的 pred_len 个位置都对应 9..16 时段,所以 mask 全部为 1.
        # 兼容起见,如果 editable_horizon 长度等于 h,逐位置判定;
        # 否则认为整段 editable。
        if h == int(self.editable_horizon[1]) - int(self.editable_horizon[0]) + 1:
            editable = torch.ones_like(target)
        else:
            editable = torch.zeros_like(target)
            lo, hi = self.editable_horizon
            lo = max(1, int(lo)) - 1
            hi = min(h, int(hi))
            if hi > lo:
                editable[:, lo:hi] = 1.0
        protected = 1.0 - editable

        # segment-specific SMAPE
        smape_vec = self._smape(pred, target)
        seg_loss = (editable * smape_vec).sum() / (editable.sum() + 1e-6)

        # non-segment: 原 huber+tail+dif weight
        err = pred - target
        tail_mask = ((target <= self.low_thr) | (target >= self.high_thr)).float()
        tail_weight = 1.0 + self.alpha * tail_mask
        diff = torch.zeros_like(target)
        if target.size(1) > 1:
            diff[:, 1:] = torch.abs(target[:, 1:] - target[:, :-1])
        diff_scale = diff / (diff.mean(dim=1, keepdim=True) + 1e-6)
        weight = tail_weight + self.diff_alpha * diff_scale.detach()
        huber = F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_beta)
        mse = err.square()
        nonseg = (protected * weight * huber).sum() / (protected.sum() + 1e-6)
        nonseg_mse = (protected * weight * mse).sum() / (protected.sum() + 1e-6)

        protected_penalty = (protected * err.square()).mean()
        return (
            self.smape_alpha * seg_loss
            + nonseg
            + self.mse_gamma * nonseg_mse
            + self.protected_weight * protected_penalty
        )

