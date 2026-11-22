"""把多个因子加权合成一个综合打分。"""

import pandas as pd

from factors.preprocess import winsorize, zscore
from factors.registry import get


def composite_score(panel, weights: dict, mask: pd.DataFrame | None = None):
    """按配置里的权重把多个因子合成综合分。

    参数
    ----
    panel   : 行情面板 {"open","close","high","low","volume"}
    weights : {"mom20": 0.35, ...}，权重不必和为 1，会自动归一
    mask    : 可交易矩阵，True 的才参与打分

    返回
    ----
    score : date × code 的综合打分（越大越好）
    detail: {因子名: 标准化后的因子矩阵}，用于单独看每个因子的 IC
    """
    parts, detail = [], {}
    total_w = 0.0

    for name, w in weights.items():
        raw = get(name)(panel)
        clean = zscore(winsorize(raw))
        if mask is not None:
            clean = clean.where(mask.reindex_like(clean).fillna(False))
        detail[name] = clean
        parts.append(w * clean)
        total_w += w

    if not parts:
        raise ValueError("factors 配置为空，至少给一个因子")

    score = sum(parts) / total_w if total_w else sum(parts)
    if mask is not None:
        score = score.where(mask.reindex_like(score).fillna(False))
    return score, detail
