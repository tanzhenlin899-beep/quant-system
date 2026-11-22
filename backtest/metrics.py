"""
绩效指标：把净值曲线变成几个数字。

必须报告的四个：累计收益率、年化收益率、最大回撤、夏普比率。
另外附上卡玛比率、年化波动、胜率，报告里更完整。
"""

import numpy as np
import pandas as pd


def perf(nav: pd.Series, rf: float = 0.02, freq: int = 252) -> dict:
    """输入净值序列，输出指标字典。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}

    ret = nav.pct_change().dropna()
    n = len(ret)
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (1.0 + total) ** (freq / n) - 1.0 if n > 0 else np.nan
    vol = ret.std() * np.sqrt(freq)
    sharpe = (ann - rf) / vol if vol and vol > 0 else np.nan

    dd = nav / nav.cummax() - 1.0
    mdd = dd.min()
    calmar = ann / abs(mdd) if mdd and mdd < 0 else np.nan

    return {
        "累计收益率": total,
        "年化收益率": ann,
        "年化波动率": vol,
        "最大回撤": mdd,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "日胜率": float((ret > 0).mean()),
        "交易日数": int(n),
    }


def excess_stats(nav_s: pd.Series, nav_b: pd.Series, freq: int = 252) -> dict:
    """策略相对基准的超额收益。"""
    s = nav_s / nav_s.iloc[0]
    b = nav_b.reindex(nav_s.index).ffill()
    b = b / b.iloc[0]

    diff = s.pct_change() - b.pct_change()
    diff = diff.dropna()
    ann_excess = (s.iloc[-1] / b.iloc[-1]) ** (freq / len(diff)) - 1.0 if len(diff) else np.nan
    te = diff.std() * np.sqrt(freq)          # 跟踪误差
    ir = ann_excess / te if te and te > 0 else np.nan

    return {
        "累计超额": s.iloc[-1] / b.iloc[-1] - 1.0,
        "年化超额": ann_excess,
        "跟踪误差": te,
        "信息比率": ir,
        "跑赢天数占比": float((diff > 0).mean()),
    }


def slice_period(nav: pd.Series, start: str, end: str) -> pd.Series:
    """截取一段时间，并把起点重新归一化为 1。"""
    seg = nav.loc[start:end].dropna()
    if len(seg) == 0:
        return seg
    return seg / seg.iloc[0]


def format_pct(d: dict, keys=("累计收益率", "年化收益率", "年化波动率", "最大回撤")) -> dict:
    """把小数转成百分比字符串，方便直接打印和写进报告。"""
    from typing import Any

    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in keys and isinstance(v, float) and np.isfinite(v):
            out[k] = f"{v * 100:.2f}%"
        elif isinstance(v, float) and np.isfinite(v):
            out[k] = round(v, 3)
        else:
            out[k] = v
    return out
