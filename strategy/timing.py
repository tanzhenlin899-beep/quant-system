"""
择时叠加层：波动率目标（volatility targeting）。

思路：
  组合波动大的时候减仓，波动小的时候满仓。
  把组合的年化波动"瞄准"一个目标值（例如 15%），
  这能显著压低最大回撤，代价是牛市里会少赚一点。

注意：这里只调整"总仓位"，不改"选哪几只"，所以它是叠加层，不是新策略。
"""

import numpy as np


class VolTargetTiming:
    def __init__(self, target_vol: float = 0.15, window: int = 20, max_lev: float = 1.0):
        self.target_vol = target_vol
        self.window = window
        self.max_lev = max_lev

    def apply(self, weights, panel):
        close = panel["close"]
        ret = close.pct_change()

        # 组合过去 window 日的已实现波动（年化）
        port_ret = (weights.shift(1) * ret).sum(axis=1, min_count=1)
        vol = port_ret.rolling(self.window).std() * np.sqrt(252)

        # 目标仓位 = 目标波动 / 当前波动，且不超过 max_lev
        scale = (self.target_vol / vol).clip(0.0, self.max_lev)
        # 今天的波动是收盘后才知道的，所以只能从明天开始用
        scale = scale.shift(1)
        # 最开始一段还没有波动估计，先按满仓处理
        scale = scale.fillna(self.max_lev)

        return weights.mul(scale, axis=0)
