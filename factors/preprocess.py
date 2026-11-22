"""
因子预处理：把原始因子值变成可以横向比较的打分。

顺序不能乱：先去极值，再标准化。
  - 去极值（winsorize）：把离群的极端值截断，否则一只妖股会把全市场分布拉歪
  - 标准化（zscore）：变成均值 0、标准差 1，这样不同量纲的因子才能加权相加
"""

import numpy as np
import pandas as pd


def winsorize(df: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """MAD 去极值：超出「中位数 ± n × 1.4826 × MAD」的值被截断。

    MAD 是绝对中位差，比标准差更抗极端值。
    """
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1) * 1.4826
    upper = med + n * mad
    lower = med - n * mad
    # 某些横截面上 MAD 可能为 0（例如大量相同值），此时不截断
    upper = upper.where(mad > 0, np.inf)
    lower = lower.where(mad > 0, -np.inf)
    return df.clip(lower=lower, upper=upper, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面标准化：对每个交易日单独做（沿列方向）。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    std = std.where(std > 0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def neutralize(df: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """对给定的暴露度（如对数市值）做横截面回归，取残差。

    用途：避免因子其实只是在买"市值大小"或"某个行业"。
    没有暴露度数据时可以不调用这个函数。
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for dt in df.index:
        y = df.loc[dt]
        x = exposure.loc[dt].reindex(y.index)
        ok = y.notna() & x.notna()
        if ok.sum() < 30:
            out.loc[dt] = y
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok].values])
        beta, *_ = np.linalg.lstsq(X, y[ok].values, rcond=None)
        resid = y[ok].values - X @ beta
        out.loc[dt] = pd.Series(resid, index=y[ok].index).reindex(y.index)
    return out


def align_and_clean(score: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """把不可交易的股票打分置空，避免选中停牌股。"""
    return score.where(mask.reindex_like(score).fillna(False))
