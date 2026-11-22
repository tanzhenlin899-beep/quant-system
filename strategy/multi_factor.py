"""
主策略：多因子选股 + 定期调仓 + 缓冲区换手控制。

流程：
  综合打分 → 只在可交易股票里排序 → 取前 N 只 → 等权 → 按固定周期调仓

关于「缓冲区（buffer）」这个设计，是这套策略里最值得讲的一个工程细节：
  如果每期都严格按排名换股（跌出前 20 就卖），会因为排名在第 20~26 名之间
  来回抖动而频繁买卖，换手率极高，交易成本把超额收益全部吃掉。
  加了缓冲区之后：已经持有的股票只要还在前 20×(1+buffer) 名内就继续拿着，
  只有当它掉出这个更大的圈子时才换掉。
  这样换手率大幅下降，而选股质量几乎没有损失。
"""

import numpy as np
import pandas as pd

from strategy.base import Strategy


class MultiFactorTopN(Strategy):
    name = "多因子TopN"

    def __init__(
        self,
        n_hold: int = 20,
        buffer_pct: float = 0.5,
        rebalance_days: int = 10,
    ):
        self.n_hold = n_hold
        self.buffer_pct = buffer_pct      # 0 = 严格按排名换股；0.5 = 允许缓冲 50%
        self.rebalance_days = max(1, int(rebalance_days))

    def generate_weights(self, panel, score, mask):
        dates = score.index
        codes = list(score.columns)
        mask_ok = mask.reindex(index=dates, columns=codes).fillna(False)

        rebalance_dates = dates[:: self.rebalance_days]
        held: list[str] = []
        rows: dict[pd.Timestamp, dict[str, float]] = {}

        for dt in rebalance_dates:
            s = score.loc[dt].where(mask_ok.loc[dt]).dropna()
            if s.empty:
                continue
            s = s.sort_values(ascending=False)

            n = min(self.n_hold, len(s))
            n_cand = min(len(s), int(np.ceil(n * (1.0 + self.buffer_pct))))

            # 1) 老仓位里仍然排在候选区内的，继续持有（省下一笔交易）
            candidates = set(s.index[:n_cand])
            keep = [c for c in held if c in candidates][:n]

            # 2) 空出来的名额，用排名最靠前的新股票补上
            need = n - len(keep)
            if need > 0:
                chosen = set(keep)
                fresh = [c for c in s.index[:n] if c not in chosen][:need]
                hold = keep + fresh
            else:
                hold = keep

            held = hold
            if hold:
                rows[dt] = {c: 1.0 / len(hold) for c in hold}

        if not rows:
            return pd.DataFrame(0.0, index=dates, columns=codes)

        w = pd.DataFrame(rows).T.reindex(columns=codes).fillna(0.0)
        # 非调仓日沿用上一次的目标权重
        w = w.reindex(dates).ffill().fillna(0.0)
        return w
