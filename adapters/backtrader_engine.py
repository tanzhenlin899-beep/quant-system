"""
用 backtrader 重跑同一套信号，用于和自研引擎交叉验证。

几个需要注意的对应关系（想清楚这几条，适配器就很好写）：

1. 成交时点 —— 这是最关键的一点
   backtrader 的 next() 在 bar 收盘时调用，此时下的单默认在**下一根 bar 开盘**成交。
   但我们自研引擎的做法是「t 日收盘出信号，用 t+1 日**开盘价**估值并成交」。
   为了严格对齐，这里打开 cheat_on_open，在 next_open() 里下单，
   这样既拿到当根 bar 的开盘价，又能在同一个开盘价成交。
   权重用 shift(1) 预处理好，语义就和自研引擎完全一致。

2. 调仓频率
   用「权重是否变化」判断调仓日，权重没变就不下单，避免每天碎单。

3. 交易成本
   A 股的印花税只在卖出收，backtrader 内置的 setcommission 做不到，
   必须自定义 CommInfoBase，在 _getcommission 里判断 size 的正负。

4. 可交易性
   停牌 / 一字板在 backtrader 里没有原生概念，
   这里用「当根 bar 的 mask」近似：不可交易就不对这一只下单（保持原仓位）。

5. 资金约束
   先卖后买，并按可用现金缩减买入数量——和自研引擎的处理方式保持一致，
   否则两边会因为「谁先拒单」产生较大差异。

已知的、无法完全消除的差异（写进报告是加分项，不是 bug）：
  - 滑点的实现方式不同：自研引擎把滑点算进费用，backtrader 直接改变成交价。
  - 最小佣金 5 元的触发时点略有差异。
"""

import backtrader as bt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 成本模型
class AShareCommission(bt.CommInfoBase):
    """A 股的真实费用结构：佣金 + 过户费 + 卖出印花税。"""

    params = (
        ("commission", 2.5e-4),   # 佣金 万 2.5
        ("min_comm", 5.0),        # 单笔最低 5 元
        ("stamp_duty", 5e-4),     # 印花税 千 0.5，仅卖出
        ("transfer_fee", 1e-5),   # 过户费
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        amount = abs(size) * price
        fee = max(amount * self.p.commission, self.p.min_comm) + amount * self.p.transfer_fee
        if size < 0:                      # size < 0 表示卖出
            fee += amount * self.p.stamp_duty
        return fee


# ---------------------------------------------------------------- 策略
class TargetWeightStrategy(bt.Strategy):
    """按外部给定的目标权重在开盘时调仓。"""

    params = (
        ("weights", None),     # DataFrame: index=信号日, columns=股票代码
        ("mask", None),        # DataFrame: 是否可交易
        ("cost", None),        # 费用参数 dict
    )

    def __init__(self):
        self.d_by_code = {d._name: d for d in self.datas}
        codes = list(self.d_by_code)

        # >>> 关键：信号 t 日产生、t+1 日开盘执行，所以整体下移一行
        w_exec = self.p.weights.reindex(columns=codes).fillna(0.0).shift(1).fillna(0.0)
        self.w_by_date = {ts.date(): row for ts, row in w_exec.iterrows()}

        m = self.p.mask.reindex(columns=codes).fillna(False)
        self.mask_by_date = {ts.date(): row for ts, row in m.iterrows()}

        c = self.p.cost or {}
        self.comm = c.get("commission", 2.5e-4)
        self.min_comm = c.get("min_commission", 5.0)
        self.stamp = c.get("stamp_duty", 5e-4)
        self.transfer = c.get("transfer_fee", 1e-5)
        self.slip = c.get("slippage", 1e-3)
        self.min_trade_ratio = 0.002

        self._last_w = None
        self.equity_curve = []

    # ---- 收盘时记录净值 ----
    def next(self):
        self.equity_curve.append(
            (bt.num2date(self.datetime[0]), float(self.broker.getvalue()))
        )

    # ---- 开盘时调仓（此时 open[0] 就是成交价）----
    def next_open(self):
        dt = self.datetime.date(0)
        row = self.w_by_date.get(dt)
        if row is None:
            return

        cur = tuple(np.round(row.values, 8))
        if cur == self._last_w:        # 非调仓日
            return
        self._last_w = cur

        tr = self.mask_by_date.get(dt, pd.Series(False, index=list(self.d_by_code)))

        # 1) 用开盘价估值
        value = float(self.broker.getvalue())
        cash = float(self.broker.getcash())
        min_value = self.min_trade_ratio * value

        targets, prices = {}, {}
        for code, data in self.d_by_code.items():
            px = float(data.open[0])
            if px <= 0 or not np.isfinite(px):
                continue
            prices[code] = px
            if not bool(tr.get(code, False)):
                targets[code] = self.getposition(data).size   # 不可交易：维持原仓位
            else:
                targets[code] = int(value * float(row.get(code, 0.0) or 0.0) / px)

        # 2) 先卖
        for code, data in self.d_by_code.items():
            if code not in targets or code not in prices:
                continue
            px = prices[code]
            delta = targets[code] - self.getposition(data).size
            if delta >= 0 or abs(delta) * px < min_value:
                continue
            qty = -delta
            gross = qty * px
            fee = max(gross * self.comm, self.min_comm) + gross * (self.transfer + self.stamp)
            cash += gross - fee
            self.sell(data=data, size=qty)

        # 3) 再买（按可用现金缩减）
        for code, data in self.d_by_code.items():
            if code not in targets or code not in prices:
                continue
            px = prices[code]
            delta = targets[code] - self.getposition(data).size
            if delta <= 0 or abs(delta) * px < min_value:
                continue
            qty = delta
            unit = px * (1.0 + self.comm + self.transfer + self.slip)
            if qty * unit > cash:                       # 钱不够就缩减
                qty = int(cash / unit)
            if qty <= 0:
                continue
            gross = qty * px
            fee = max(gross * self.comm, self.min_comm) + gross * self.transfer
            cash -= gross + fee
            self.buy(data=data, size=qty)


# ---------------------------------------------------------------- 引擎入口
def run_backtest_backtrader(
    weights: pd.DataFrame,
    panel: dict,
    mask: pd.DataFrame,
    cost_cfg: dict | None = None,
    init_cash: float = 1_000_000.0,
    codes: list[str] | None = None,
):
    """用 backtrader 跑一遍，返回和自研引擎同结构的字典。"""
    cost_cfg = cost_cfg or {}
    codes = list(codes or weights.columns)

    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    cerebro.broker.setcash(init_cash)

    # 1) 数据源：每只股票一个 feed
    for code in codes:
        df = pd.DataFrame({
            "open": panel["open"][code],
            "high": panel["high"][code],
            "low": panel["low"][code],
            "close": panel["close"][code],
            "volume": panel["volume"][code].fillna(0.0),
        }).dropna(subset=["open", "close"])
        if df.empty:
            continue
        cerebro.adddata(bt.feeds.PandasData(dataname=df, openinterest=-1), name=code)

    # 2) 成本：自定义 A 股费用 + 滑点
    cerebro.broker.addcommissioninfo(AShareCommission(
        commission=cost_cfg.get("commission", 2.5e-4),
        min_comm=cost_cfg.get("min_commission", 5.0),
        stamp_duty=cost_cfg.get("stamp_duty", 5e-4),
        transfer_fee=cost_cfg.get("transfer_fee", 1e-5),
    ))
    # 滑点交给 broker 处理：成交价 = 开盘价 × (1 + 滑点)
    cerebro.broker.set_slippage_perc(perc=cost_cfg.get("slippage", 1e-3))

    # 3) 策略
    cerebro.addstrategy(
        TargetWeightStrategy, weights=weights, mask=mask, cost=cost_cfg
    )

    # 4) 分析器（backtrader 自带的绩效模块）
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strategies = cerebro.run()
    strat = strategies[0]

    # 5) 净值曲线
    if not strat.equity_curve:
        return {"nav": pd.Series(dtype=float), "analyzers": {}}
    idx = pd.DatetimeIndex([c[0] for c in strat.equity_curve])
    nav = pd.Series([c[1] for c in strat.equity_curve], index=idx, name="nav")
    if nav.index.tz is not None:
        nav.index = nav.index.tz_localize(None)
    nav = nav[~nav.index.duplicated(keep="last")]

    return {
        "nav": nav,
        "analyzers": {
            "最大回撤": strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown"),
            "成交笔数": strat.analyzers.trades.get_analysis().get("total", {}).get("total"),
        },
    }
