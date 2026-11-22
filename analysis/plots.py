"""
图表输出。

四张图分别对应报告里的四个小节：
  1. nav.png      策略 vs 基准的归一化净值曲线（含买卖点标记、回撤）
  2. ic.png       各因子的 RankIC 逐日走势
  3. layered.png  因子分层累计收益（看单调性）
  4. sensitivity.png 参数敏感性热力图（证明参数不是靠运气）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体：Windows 上一般有微软雅黑，没有就退到黑体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

STRAT_COLOR = "#c0392b"
BENCH_COLOR = "#2c3e50"


def plot_nav(
    nav_s: pd.Series,
    nav_b: pd.Series,
    trades: pd.DataFrame | None = None,
    split_date: str | None = None,
    title: str = "策略 vs 基准（归一化净值）",
    path: str | Path = "outputs/nav.png",
):
    """净值对比图：上图净值，下图回撤。"""
    s = nav_s / nav_s.iloc[0]
    b = nav_b.reindex(nav_s.index).ffill()
    b = b / b.iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(s.index, s.values, color=STRAT_COLOR, lw=1.7, label="策略净值")
    ax1.plot(b.index, b.values, color=BENCH_COLOR, lw=1.2, alpha=0.85, label="基准净值")
    ax1.axhline(1.0, color="gray", ls="--", lw=0.8)

    if split_date:
        ax1.axvline(pd.Timestamp(split_date), color="#e67e22", ls=":", lw=1.2)
        ax1.text(
            pd.Timestamp(split_date),
            ax1.get_ylim()[1],
            "  样本外 →",
            color="#e67e22",
            va="top",
            fontsize=9,
        )

    # 标注关键买卖点：每年各取一笔最大买入和最大卖出，避免所有点挤在同一年
    if trades is not None and not trades.empty and "amount" in trades:
        valid = trades[trades["date"].isin(s.index)].copy()
        if not valid.empty:
            valid["year"] = pd.to_datetime(valid["date"]).dt.year
            picks = valid.sort_values("amount").groupby(["year", "side"]).tail(1)
            for _, t in picks.iterrows():
                y = s.loc[t["date"]]
                ax1.scatter(
                    t["date"],
                    y,
                    marker="^" if t["side"] == "BUY" else "v",
                    s=50,
                    color="#27ae60" if t["side"] == "BUY" else "#8e44ad",
                    alpha=0.9,
                    zorder=5,
                )
            ax1.scatter([], [], marker="^", s=50, color="#27ae60", label="买入")
            ax1.scatter([], [], marker="v", s=50, color="#8e44ad", label="卖出")

    ax1.set_ylabel("净值（起点 = 1.0）")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    dd_s = s / s.cummax() - 1.0
    dd_b = b / b.cummax() - 1.0
    ax2.fill_between(s.index, dd_s.values, 0, color=STRAT_COLOR, alpha=0.35, label="策略回撤")
    ax2.plot(b.index, dd_b.values, color=BENCH_COLOR, lw=1.0, label="基准回撤")
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ic(ic_df: pd.DataFrame, path: str | Path = "outputs/ic.png"):
    """各因子 RankIC 的逐日柱状图 + 累计 IC 走势。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for col in ic_df.columns:
        ax1.bar(ic_df.index, ic_df[col].fillna(0), width=1.0, alpha=0.35, label=col)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("RankIC（逐日）")
    ax1.set_title("因子 RankIC：越稳定在 0 上方越好")
    ax1.legend()
    ax1.grid(alpha=0.25)

    for col in ic_df.columns:
        ax2.plot(ic_df.index, ic_df[col].fillna(0).cumsum(), lw=1.2, label=col)
    ax2.set_ylabel("累计 IC")
    ax2.set_xlabel("日期")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_layered(layered: pd.DataFrame, path: str | Path = "outputs/layered.png"):
    """分层回测曲线：组间收益应该单调分开。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("RdYlGn")
    n = len(layered.columns)
    for i, col in enumerate(layered.columns):
        ax.plot(
            layered.index,
            np.asarray(layered[col], dtype=float),
            lw=1.4,
            color=cmap(i / max(n - 1, 1)),
            label=col,
        )
    ax.set_title("因子分层累计净值（Q1 因子值最低，Q5 最高）")
    ax.set_ylabel("净值（起点 = 1.0）")
    ax.set_xlabel("日期")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity(table: pd.DataFrame, path: str | Path = "outputs/sensitivity.png"):
    """参数敏感性热力图：好的参数应该在一整片区域都好，而不是孤立的一个点。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(r) for r in table.index])
    ax.set_xlabel(str(table.columns.name or "参数 X"))
    ax.set_ylabel(str(table.index.name or "参数 Y"))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="夏普比率")
    ax.set_title("参数敏感性：夏普比率")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
