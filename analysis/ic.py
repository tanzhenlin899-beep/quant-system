"""
因子有效性检验：在搭策略之前，先证明你的因子真的有预测力。

两个标准动作：
  1. RankIC = 因子排名 与 未来收益排名 的相关系数。均值越大、越稳定越好。
  2. 分层回测 = 按因子分 5 组，看每组的累计收益是否单调递增。
     如果第 1 组和第 5 组差不多，说明这个因子没用。
"""

import numpy as np
import pandas as pd


def forward_return(close: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """未来 horizon 个交易日的收益率（用来看"因子能不能预测未来"）。"""
    return close.shift(-horizon) / close - 1.0


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日 RankIC（Spearman 相关系数）。

    实现说明：Spearman 相关等价于「秩的 Pearson 相关」，因此可以整体向量化——
    先在横截面上对因子和未来收益取秩，再逐行算 Pearson 相关系数。
    早期版本逐日调用 Series.corr(method="spearman")，在 1942 个交易日 × 4 个因子
    上约需 10 秒，向量化后降到毫秒级，结果完全一致。
    """
    common_index = factor.index.intersection(fwd.index)
    common_cols = factor.columns.intersection(fwd.columns)
    f = factor.reindex(index=common_index, columns=common_cols)
    r = fwd.reindex(index=common_index, columns=common_cols)

    valid = f.notna() & r.notna()
    n = valid.sum(axis=1)
    # 样本太少的横截面不计算
    enough = n >= 10

    # 秩：只在双方都有值的格子上排名，避免缺失值干扰
    fr = f.where(valid).rank(axis=1, method="average")
    rr = r.where(valid).rank(axis=1, method="average")

    fr_mean = fr.mean(axis=1)
    rr_mean = rr.mean(axis=1)
    df_ = fr.sub(fr_mean, axis=0)
    dr_ = rr.sub(rr_mean, axis=0)

    cov = (df_ * dr_).sum(axis=1, min_count=1)
    var_f = (df_ ** 2).sum(axis=1, min_count=1)
    var_r = (dr_ ** 2).sum(axis=1, min_count=1)

    denom = (var_f * var_r) ** 0.5
    ic = cov / denom.where(denom > 0)

    ic = ic.where(enough)
    ic.name = "rank_ic"
    return ic.dropna().sort_index()


def ic_summary(ic: pd.Series) -> dict:
    """IC 的汇总统计。ICIR = IC 均值 / IC 标准差，衡量稳定性。"""
    ic = ic.dropna()
    if len(ic) == 0:
        return {}
    return {
        "IC均值": float(ic.mean()),
        "IC标准差": float(ic.std()),
        "ICIR": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
        "IC>0占比": float((ic > 0).mean()),
        "样本天数": int(len(ic)),
    }


def layered_return(
    factor: pd.DataFrame, fwd: pd.DataFrame, n_groups: int = 5
) -> pd.DataFrame:
    """分层回测：每天按因子值分 n_groups 组，算每组当天的平均未来收益。

    返回每组的累计净值曲线（起点 1.0）。
    """
    rows: dict[int, dict] = {}
    for dt in factor.index:
        a, b = factor.loc[dt], fwd.loc[dt]
        ok = a.notna() & b.notna()
        if ok.sum() < n_groups * 3:
            continue
        ranks = a[ok].rank(method="first")
        try:
            labels = pd.qcut(ranks, n_groups, labels=False)
        except ValueError:
            continue
        for g in range(n_groups):
            idx = labels[labels == g].index
            rows.setdefault(g, {})[dt] = b[idx].mean()

    df = pd.DataFrame({f"Q{g + 1}": pd.Series(v) for g, v in sorted(rows.items())})
    df = df.sort_index()
    return (1.0 + df.fillna(0.0)).cumprod()


def long_short(layered: pd.DataFrame) -> pd.Series:
    """多空组合：最高一组减最低一组，衡量因子的纯粹收益。"""
    if layered.empty:
        return pd.Series(dtype=float)
    top = layered.iloc[:, -1]
    bottom = layered.iloc[:, 0]
    return (top / top.iloc[0]) / (bottom / bottom.iloc[0])
