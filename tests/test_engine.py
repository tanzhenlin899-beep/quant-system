"""
回测引擎的关键行为测试。

这些测试不需要 pytest 也能跑：
    python tests/test_engine.py
装了 pytest 也可以：
    pytest tests/

每个测试都对应一条「答辩会被问到的假设」，测过了才敢说它成立。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtest.cost import CostModel
from backtest.engine import run_backtest
from backtest.metrics import excess_stats, perf
from data.universe import build_mask
from factors.composite import composite_score
from strategy.multi_factor import MultiFactorTopN

ZERO_COST = CostModel(commission=0.0, stamp_duty=0.0, transfer_fee=0.0,
                      slippage=0.0, min_commission=0.0)


def _flat_panel(codes, dates, price=10.0):
    px = pd.DataFrame(price, index=dates, columns=codes)
    vol = pd.DataFrame(1000.0, index=dates, columns=codes)
    return {"open": px.copy(), "close": px.copy(), "high": px.copy(),
            "low": px.copy(), "volume": vol}


# ---------------------------------------------------------------- 1. 时序
def test_no_lookahead_bias():
    """信号在 t 日产生，必须 t+1 日开盘成交，不能用到 t 日之后的信息。

    构造：t0 收盘 10 元，t1 开盘直接跳到 20 元。
    如果代码有前视偏差（用 t0 收盘价成交），收益会翻倍；
    正确实现应该在 t1 开盘 20 元买入，没有收益。
    """
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _flat_panel(["A"], dates, price=10.0)
    panel["open"].iloc[1] = 20.0
    panel["open"].iloc[2] = 20.0
    panel["close"].iloc[1] = 20.0
    panel["close"].iloc[2] = 20.0
    mask = pd.DataFrame(True, index=dates, columns=["A"])

    w = pd.DataFrame(0.0, index=dates, columns=["A"])
    w.loc[dates[0], "A"] = 1.0        # t0 收盘决策：满仓 A

    res = run_backtest(w, panel, mask, cost=ZERO_COST, init_cash=1_000_000,
                       rebalance_days=1, min_trade_ratio=0.0)
    nav = res["nav"] / res["nav"].iloc[0]

    assert abs(nav.iloc[1] - 1.0) < 1e-9, f"疑似前视偏差，t1 净值={nav.iloc[1]:.4f}"
    assert abs(nav.iloc[2] - 1.0) < 1e-9
    return "t1 开盘 20 元买入，净值仍为 1.0 → 无前视偏差"


# ---------------------------------------------------------------- 2. 交易成本
def test_cost_calculation():
    """佣金（最低 5 元）+ 过户费 + 卖出印花税，逐笔算对。"""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _flat_panel(["A"], dates, price=10.0)
    mask = pd.DataFrame(True, index=dates, columns=["A"])

    w = pd.DataFrame(0.0, index=dates, columns=["A"])
    w.loc[dates[0], "A"] = 1.0        # 买入
    w.loc[dates[1], "A"] = 0.0        # 卖出

    cost = CostModel(commission=2.5e-4, min_commission=5.0, stamp_duty=5e-4,
                     transfer_fee=1e-5, slippage=0.0)
    res = run_backtest(w, panel, mask, cost=cost, init_cash=1_000_000,
                       rebalance_days=1, min_trade_ratio=0.0)
    tr = res["trades"]

    buy = tr[tr["side"] == "BUY"].iloc[0]
    expect_buy = max(buy["amount"] * 2.5e-4, 5.0) + buy["amount"] * 1e-5
    assert abs(buy["fee"] - expect_buy) < 1e-6, f"买入费 {buy['fee']:.2f} != {expect_buy:.2f}"

    sell = tr[tr["side"] == "SELL"].iloc[0]
    expect_sell = max(sell["amount"] * 2.5e-4, 5.0) + sell["amount"] * (1e-5 + 5e-4)
    assert abs(sell["fee"] - expect_sell) < 1e-6, f"卖出费 {sell['fee']:.2f} != {expect_sell:.2f}"
    assert sell["fee"] > buy["fee"], "卖出应该比买入多一笔印花税"
    return f"买入费 {buy['fee']:.2f} 元 / 卖出费 {sell['fee']:.2f} 元（含印花税）"


# ---------------------------------------------------------------- 3. 涨跌停
def test_limit_up_not_tradable():
    """一字涨停当天应判定为不可交易，买不进去。"""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _flat_panel(["A"], dates, price=10.0)
    panel["close"].iloc[0] = 10.0
    # t1 主板一字涨停：开=收=11.0
    panel["open"].iloc[1] = 11.0
    panel["close"].iloc[1] = 11.0
    panel["open"].iloc[2] = 11.0
    panel["close"].iloc[2] = 11.0

    cfg = {"universe": {"min_list_days": 0, "exclude_st": False, "st_codes": []}}
    mask = build_mask(panel, cfg)

    assert not bool(mask.iloc[1, 0]), "一字涨停当天被误判为可交易"
    assert bool(mask.iloc[2, 0]), "第二天不是涨停，应可交易"
    return "一字涨停被正确屏蔽"


def test_st_limit_pct():
    """ST 股涨跌停是 5%，不能按 10% 判断。"""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _flat_panel(["600001"], dates, price=10.0)
    panel["open"].iloc[1] = 10.5      # +5%，ST 的涨停
    panel["close"].iloc[1] = 10.5
    panel["open"].iloc[2] = 10.5
    panel["close"].iloc[2] = 10.5

    cfg = {"universe": {"min_list_days": 0, "exclude_st": False, "st_codes": ["600001"]}}
    mask = build_mask(panel, cfg)
    assert not bool(mask.iloc[1, 0]), "ST 股 +5% 应判为一字涨停"
    return "ST 股 5% 涨跌停被正确识别"


# ---------------------------------------------------------------- 4. 指标
def test_metrics_math():
    """用能手算的数字验证绩效指标。"""
    nav = pd.Series([1.0, 1.1, 1.05, 1.2],
                    index=pd.bdate_range("2024-01-01", periods=4))

    m = perf(nav)
    assert abs(m["累计收益率"] - 0.2) < 1e-9, m["累计收益率"]
    assert abs(m["最大回撤"] - (1.05 / 1.1 - 1)) < 1e-9, m["最大回撤"]
    assert m["交易日数"] == 3

    # 单日涨跌各 1% 的对称序列，年化波动应为 sqrt 关系
    flat = pd.Series([1.0, 1.01, 1.0, 1.01, 1.0],
                     index=pd.bdate_range("2024-01-01", periods=5))
    m2 = perf(flat)
    ret = flat.pct_change().dropna()
    assert abs(m2["年化波动率"] - ret.std() * np.sqrt(252)) < 1e-9
    return "累计收益 / 最大回撤 / 年化波动率 与手算一致"


def test_excess_stats():
    """策略与基准完全相同时，超额应为 0。"""
    dates = pd.bdate_range("2024-01-01", periods=50)
    nav = pd.Series(np.linspace(1.0, 1.2, 50), index=dates)
    ex = excess_stats(nav, nav.copy())
    assert abs(ex["累计超额"]) < 1e-9
    assert abs(ex["年化超额"]) < 1e-9
    return "策略=基准时超额收益为 0"


# ---------------------------------------------------------------- 5. 边界
def test_nhold_larger_than_universe():
    """持仓只数超过股票数时，应全部持有而不是报错。"""
    dates = pd.bdate_range("2024-01-01", periods=80)
    codes = ["A", "B", "C"]
    panel = _flat_panel(codes, dates, price=10.0)
    mask = pd.DataFrame(True, index=dates, columns=codes)
    score = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=dates)

    w = MultiFactorTopN(n_hold=20, buffer_pct=0.5, rebalance_days=5).generate_weights(
        panel, score, mask
    )
    last = w.iloc[-1]
    assert len(last[last > 0]) == 3, f"应该持有 3 只，实际 {len(last[last > 0])}"
    assert abs(last.sum() - 1.0) < 1e-9
    return "持仓 20 只 / 池子 3 只 → 持有 3 只且权重和为 1"


def test_no_tradable_stocks():
    """全市场停牌时，不应崩，净值应保持不动。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    panel = _flat_panel(["A"], dates, price=10.0)
    panel["volume"][:] = 0.0
    mask = pd.DataFrame(False, index=dates, columns=["A"])
    w = pd.DataFrame(1.0, index=dates, columns=["A"])

    res = run_backtest(w, panel, mask, cost=ZERO_COST, init_cash=1_000_000,
                       rebalance_days=1, min_trade_ratio=0.0)
    assert len(res["trades"]) == 0, "不可交易却产生了成交"
    assert np.allclose(res["nav"].values, 1_000_000.0)
    return "全市场不可交易 → 零成交、净值不变"


def test_single_day():
    """只有一个交易日的数据不应崩。"""
    dates = pd.bdate_range("2024-01-01", periods=1)
    panel = _flat_panel(["A"], dates)
    mask = pd.DataFrame(True, index=dates, columns=["A"])
    w = pd.DataFrame(1.0, index=dates, columns=["A"])
    res = run_backtest(w, panel, mask, cost=ZERO_COST, init_cash=1_000_000)
    assert len(res["nav"]) == 1
    assert perf(res["nav"]) == {}, "不足两天时 perf 应返回空字典"
    return "单日数据不崩，perf 返回空"


def test_factor_warmup_all_nan():
    """因子窗口没走满（全是 NaN）时，不应崩，权重应为 0。"""
    dates = pd.bdate_range("2024-01-01", periods=30)   # 不够 60 日动量的窗口
    codes = ["A", "B"]
    panel = _flat_panel(codes, dates)
    mask = pd.DataFrame(True, index=dates, columns=codes)

    score, _ = composite_score(panel, {"mom60": 1.0}, mask)
    assert score.isna().all().all(), "30 天数据算不出 60 日动量，应全为 NaN"

    w = MultiFactorTopN(n_hold=1, buffer_pct=0.5, rebalance_days=5).generate_weights(
        panel, score, mask
    )
    assert np.allclose(w.values, 0.0), "因子全 NaN 时不应持仓"
    return "因子预热期全 NaN → 空仓且不崩"


def test_cash_constraint():
    """资金买不起一手时，应该不买，且现金不为负。"""
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = _flat_panel(["A"], dates, price=1_000_000.0)   # 一股 100 万
    mask = pd.DataFrame(True, index=dates, columns=["A"])
    w = pd.DataFrame(1.0, index=dates, columns=["A"])

    res = run_backtest(w, panel, mask, cost=CostModel(commission=2.5e-4), init_cash=1_000.0,
                       rebalance_days=1, min_trade_ratio=0.0)
    assert len(res["trades"]) == 0, "钱不够却成交了"
    assert res["nav"].iloc[-1] == 1_000.0
    return "现金不足 → 不成交，净值不变"


# ---------------------------------------------------------------- runner
def main():
    tests = [
        ("1. 时序：无前视偏差", test_no_lookahead_bias),
        ("2. 交易成本：佣金+印花税", test_cost_calculation),
        ("3. 涨跌停：一字板不可买", test_limit_up_not_tradable),
        ("4. 涨跌停：ST 股 5%", test_st_limit_pct),
        ("5. 指标：手算校验", test_metrics_math),
        ("6. 指标：超额收益", test_excess_stats),
        ("7. 边界：持仓数>股票数", test_nhold_larger_than_universe),
        ("8. 边界：全市场停牌", test_no_tradable_stocks),
        ("9. 边界：单日数据", test_single_day),
        ("10. 边界：因子预热全 NaN", test_factor_warmup_all_nan),
        ("11. 边界：现金不足", test_cash_constraint),
    ]
    passed, failed = 0, []
    print("=" * 70)
    for name, fn in tests:
        try:
            msg = fn()
            print(f"[PASS] {name}\n        {msg}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}\n        {e}")
            failed.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}\n        {type(e).__name__}: {e}")
            failed.append(name)
    print("=" * 70)
    print(f"通过 {passed} / {len(tests)}")
    if failed:
        print("失败：" + "、".join(failed))
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
