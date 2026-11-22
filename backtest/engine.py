"""
回测引擎（整个系统的心脏）。

采用「按份额记账」的方式，而不是简单的权重漂移，好处是：
  - 现金约束是真的（钱不够就买不了）
  - 停牌、一字板无法成交是真的（不能假装买到了）
  - 交易成本逐笔扣除

四个关键的工程细节（也是答辩必问的点）：
  1. 时序：策略在 t 日收盘后产生信号 → t+1 日开盘价成交。
     所以对权重做 shift(1)，这是防「前视偏差」的关键。
  2. 停牌估值：真实数据里停牌当天**没有行情行**（NaN）。
     如果直接把 NaN 当 0 估值，持仓市值会瞬间归零，净值出现假的暴跌。
     正确做法是沿用「最后一个已知价格」来估值。这是本节最重要的一处修复。
  3. 调仓频率：只在调仓日交易。如果每天都按目标权重微调，
     会因为股价每天变化产生大量碎单，换手率虚高上百倍，成本严重失真。
  4. 最小交易额：差额太小的调仓直接忽略（真实交易系统都有这个门槛）。
"""

import numpy as np
import pandas as pd

from backtest.cost import CostModel


def run_backtest(
    weights: pd.DataFrame,
    panel: dict,
    mask: pd.DataFrame,
    cost: CostModel | None = None,
    init_cash: float = 1_000_000.0,
    rebalance_days: int = 1,
    min_trade_ratio: float = 0.002,
):
    """跑一遍回测。

    参数
    ----
    weights        : date × code，index 是「信号产生日」
    panel          : 行情面板
    mask           : date × code 可交易矩阵
    cost           : 成本模型
    init_cash      : 初始资金
    rebalance_days : 每几个交易日调一次仓（1 = 每天都可以调）
    min_trade_ratio: 单只股票的调仓金额低于总市值的这个比例就忽略

    返回
    ----
    {"nav": Series, "trades": DataFrame}
    """
    cost = cost or CostModel()
    dates = panel["close"].index
    codes = list(panel["close"].columns)

    w = weights.reindex(index=dates, columns=codes).fillna(0.0)
    # >>> 信号 t 日产生，t+1 日执行：整体下移一行
    w_exec = w.shift(1).fillna(0.0)

    open_px = panel["open"].reindex(index=dates, columns=codes).to_numpy(dtype=float)
    close_px = panel["close"].reindex(index=dates, columns=codes).to_numpy(dtype=float)
    tradable = mask.reindex(index=dates, columns=codes).fillna(False).to_numpy(dtype=bool)
    w_arr = w_exec.to_numpy(dtype=float)

    n_days, n_stocks = open_px.shape
    shares = np.zeros(n_stocks)
    last_px = np.zeros(n_stocks)      # 每只股票「最后一个已知价格」，用于停牌估值
    cash = float(init_cash)
    nav = np.zeros(n_days)
    trades: list[dict] = []

    reb = max(1, int(rebalance_days))

    for i in range(n_days):
        o = open_px[i]
        c = close_px[i]
        o_ok = np.isfinite(o) & (o > 0)
        c_ok = np.isfinite(c) & (c > 0)

        # 1) 开盘估值：当天没有行情的股票，沿用最后已知价格（不能当 0！）
        val_open = np.where(o_ok, o, last_px)
        pv = cash + float(np.sum(shares * val_open))
        if not np.isfinite(pv) or pv <= 0:
            nav[i] = max(pv, 0.0) if np.isfinite(pv) else 0.0
            last_px = np.where(c_ok, c, np.where(o_ok, o, last_px))
            continue

        # 2) 今天是不是调仓日？（w_exec 在第 1 天、第 reb+1 天…发生变化）
        is_rebalance = (reb == 1) or (i >= 1 and (i - 1) % reb == 0)

        if is_rebalance:
            trade_px = np.where(o_ok, o, 0.0)          # 没有开盘价 = 无法成交
            target_value = w_arr[i] * pv
            with np.errstate(divide="ignore", invalid="ignore"):
                target_shares = np.where(
                    trade_px > 0, target_value / trade_px, shares
                )

            # 不可交易的股票：维持原仓位（既不能买也不能卖）
            target_shares = np.where(tradable[i], target_shares, shares)

            delta = np.round(target_shares - shares)

            # 差额太小就忽略，避免碎单
            min_value = min_trade_ratio * pv
            small = np.abs(delta) * trade_px < min_value
            delta = np.where(small, 0.0, delta)

            # 3) 先卖后买（卖了才有钱买，符合真实账户逻辑）
            for j in np.flatnonzero(delta < 0):
                price = trade_px[j]
                if price <= 0:
                    continue
                qty = min(-delta[j], shares[j])
                if qty <= 0:
                    continue
                amount = qty * price
                fee = cost.sell_cost(amount)
                cash += amount - fee
                shares[j] -= qty
                trades.append(_trade(dates[i], codes[j], "SELL", qty, price, fee, amount))

            for j in np.flatnonzero(delta > 0):
                price = trade_px[j]
                if price <= 0:
                    continue
                qty = float(delta[j])
                amount = qty * price
                fee = cost.buy_cost(amount)

                if amount + fee > cash:      # 现金不够：按可用资金缩减
                    unit = price * (
                        1.0 + cost.commission + cost.slippage + cost.transfer_fee
                    )
                    qty = float(int(cash / unit)) if unit > 0 else 0.0
                    if qty <= 0:
                        continue
                    amount = qty * price
                    fee = cost.buy_cost(amount)
                    if amount + fee > cash:
                        continue

                cash -= amount + fee
                shares[j] += qty
                trades.append(_trade(dates[i], codes[j], "BUY", qty, price, fee, amount))

        # 4) 收盘估值：同样用「最后已知价格」兜底
        val_close = np.where(c_ok, c, val_open)
        nav[i] = cash + float(np.sum(shares * val_close))

        # 5) 更新最后已知价格（收盘价优先，其次开盘价）
        last_px = np.where(c_ok, c, np.where(o_ok, o, last_px))

    nav_s = pd.Series(nav, index=dates, name="nav")
    trades_df = pd.DataFrame(trades).sort_values("date") if trades else pd.DataFrame()
    return {"nav": nav_s, "trades": trades_df}


def _trade(date, code, side, qty, price, fee, amount) -> dict:
    return {
        "date": date,
        "code": code,
        "side": side,
        "shares": qty,
        "price": price,
        "amount": amount,
        "fee": fee,
    }
