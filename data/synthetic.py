"""
离线模拟数据生成器（不需要网络）。

为什么需要它：
  第一次跑通系统时，如果卡在 akshare 下载或网络报错上，你根本看不到回测结果。
  所以这里内置一份"人造行情"，能立刻跑出完整的净值曲线、指标和图表。
  等流程通了，把 config.py 里 data.mode 改成 "akshare" 就换成真实数据。

生成逻辑（保证因子真的有效，否则回测结果没有解释意义）：
  每只股票有一个缓慢变化的"趋势漂移" mu（AR(1) 过程，半衰期约 45 个交易日），
  日收益 = mu + 市场因子 × beta + 个股噪声。
  过去的涨幅能部分预示未来的漂移，所以动量类因子有真实的预测力。

注意：这里的市场路径做了"去均值再缩放"，保证长期漂移是正的、波动率是可控的，
      否则某一次随机数抽到熊市，整个演示结果会全是负数，看不出系统到底做对了没有。
"""

import numpy as np
import pandas as pd

MKT_DAILY_DRIFT = 1.8e-4   # 市场日均漂移 ≈ 年化 4.6%
MKT_DAILY_VOL = 0.0110     # 市场日波动 ≈ 年化 17%
MU_STD = 1.2e-3            # 个股趋势漂移的横截面标准差
MU_PHI = 0.985             # 漂移的自相关系数（半衰期约 45 个交易日）
IDIO_VOL = 0.0150          # 个股特质波动


def make_panel(codes, start, end, seed=42):
    """生成面板数据。返回 (panel, benchmark)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_d, n_s = len(dates), len(codes)

    # 1) 市场因子：先标准化，再赋予目标漂移和波动（去掉随机路径的运气成分）
    mkt = rng.normal(0.0, 1.0, n_d)
    mkt = (mkt - mkt.mean()) / mkt.std() * MKT_DAILY_VOL + MKT_DAILY_DRIFT

    # 2) 个股趋势漂移 mu：AR(1)，从一个平稳分布出发
    mu = np.zeros((n_d, n_s))
    shock_sd = MU_STD * np.sqrt(1 - MU_PHI**2)
    mu[0] = rng.normal(0.0, MU_STD, n_s)
    shock = rng.normal(0.0, shock_sd, (n_d - 1, n_s))
    for t in range(1, n_d):
        mu[t] = MU_PHI * mu[t - 1] + shock[t - 1]

    # 3) 波动率聚集：真实市场的波动是成簇出现的（平静期 → 危机期 → 平静期），
    #    这里用一个 AR(1) 的对数波动过程 + 随机跳升来模拟。
    #    有了它，"波动率目标择时"才有真实的用武之地（高波动时自动减仓）。
    log_vol = np.zeros(n_d)
    vol_shock = rng.normal(0.0, 0.10, n_d)
    for t in range(1, n_d):
        log_vol[t] = 0.97 * log_vol[t - 1] + vol_shock[t]
    jumps = rng.random(n_d) < 0.012          # 1.2% 的日子进入高波动状态
    log_vol[jumps] += rng.uniform(0.6, 1.3, int(jumps.sum()))
    vol_mult = np.exp(log_vol)
    vol_mult = vol_mult / vol_mult.mean()    # 保证长期平均波动不被抬高

    # 4) 个股 beta 与特质噪声
    beta = rng.uniform(0.7, 1.3, n_s)
    idio = rng.normal(0.0, IDIO_VOL, (n_d, n_s))

    rets = mu + vol_mult[:, None] * (mkt[:, None] * beta[None, :] + idio)

    # 4) 由收益率倒推价格（相当于后复权价）
    close = 10.0 * np.cumprod(1.0 + rets, axis=0)

    # 5) 开盘价 = 昨收 × (1 + 隔夜跳空)，并夹在当日涨跌停之间
    prev_close = np.vstack([np.full((1, n_s), 10.0), close[:-1]])
    gap = rng.normal(0.0, 0.004, (n_d, n_s))
    open_ = prev_close * (1.0 + gap)
    open_ = np.clip(open_, prev_close * 0.9, prev_close * 1.1)

    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, (n_d, n_s))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, (n_d, n_s))))

    # 6) 成交量：随机为正；挑 0.4% 的日子停牌（成交量为 0）
    volume = rng.lognormal(13.0, 0.5, (n_d, n_s))
    volume[rng.random((n_d, n_s)) < 0.004] = 0.0

    panel = {
        "open": pd.DataFrame(open_, index=dates, columns=codes),
        "close": pd.DataFrame(close, index=dates, columns=codes),
        "high": pd.DataFrame(high, index=dates, columns=codes),
        "low": pd.DataFrame(low, index=dates, columns=codes),
        "volume": pd.DataFrame(volume, index=dates, columns=codes),
    }

    # 7) 基准：等权持有全部股票（相当于"买入持有"策略）
    bench = pd.Series(
        (1.0 + rets.mean(axis=1)).cumprod() * 1000.0,
        index=dates,
        name="benchmark",
    )

    return panel, bench
