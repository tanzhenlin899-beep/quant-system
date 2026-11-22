"""针对回测引擎的边界测试：模拟真实数据里"停牌当天没有行情 bar"的情况。

真实 A 股停牌时，akshare 返回的数据里那几天是**缺失行**，
拼成面板后就是 NaN。如果引擎把 NaN 当成价格 0 来估值，
持仓市值会瞬间归零，净值曲线会出现假的暴跌。
"""

import sys

sys.path.insert(0, r"C:\Users\13750\Documents\deepseek-harness\default-workspace\quant")

import numpy as np
import pandas as pd

from backtest.cost import CostModel
from backtest.engine import run_backtest

codes = ["A", "B", "C"]
dates = pd.bdate_range("2024-01-01", periods=20)

px = pd.DataFrame({
    "A": np.full(20, 10.0),
    "B": np.full(20, 20.0),
    "C": np.full(20, 30.0),
}, index=dates)

vol = pd.DataFrame(1000.0, index=dates, columns=codes)

# 关键：A 在 1/8 ~ 1/11 停牌 —— 这几天完全没有行情行（NaN）
suspend = dates[5:9]
px.loc[suspend, "A"] = np.nan
vol.loc[suspend, "A"] = 0.0

panel = {
    "open": px.copy(),
    "close": px.copy(),
    "high": px.copy(),
    "low": px.copy(),
    "volume": vol,
}

mask = px.notna() & (vol > 0)

# 权重恒定 1/3，价格不变 → 净值应该始终 = 1.0
w = pd.DataFrame(1.0 / 3.0, index=dates, columns=codes)

res = run_backtest(w, panel, mask, cost=CostModel(commission=0, stamp_duty=0,
                                                  transfer_fee=0, slippage=0,
                                                  min_commission=0),
                   init_cash=1_000_000.0, rebalance_days=1, min_trade_ratio=0.0)
nav = res["nav"]
norm = nav / nav.iloc[0]

print("每日净值（起点=1.0）:")
print(norm.round(4).to_string())
print()
print(f"最大偏离 1.0 的幅度: {np.abs(norm - 1.0).max():.6f}")
print(f"停牌期间净值: {norm.loc[suspend[0]:suspend[-1]].round(4).tolist()}")

if np.abs(norm - 1.0).max() > 0.01:
    print("\n❌ 发现问题：停牌期间净值出现非正常波动（NaN 被当成价格 0 估值）")
    raise SystemExit(1)
else:
    print("\n✅ 通过：停牌不影响净值估值")
    raise SystemExit(0)
