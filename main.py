"""
一键跑通整个系统。

用法（在项目根目录 D:\\quant 下打开终端）：
    python main.py                  # 离线模拟数据，跑完整流程
    python main.py --mode akshare   # 用真实 A 股数据
    python main.py --no-timing      # 关掉波动率择时（做消融对比用）
    python main.py --quick          # 跳过参数敏感性（省时间）

输出都在 outputs/ 目录里：
    metrics.csv     样本内 / 样本外 / 全样本 的指标表
    nav.png         策略 vs 基准 净值曲线
    ic.png          因子 RankIC
    layered.png     因子分层收益
    sensitivity.png 参数敏感性热力图
    trades.csv      逐笔成交记录
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis import ic as ic_mod
from analysis import plots
from backtest.engine import run_backtest
from backtest.cost import CostModel
from backtest.metrics import excess_stats, perf
from config import CONFIG, ROOT
from data.loader import load_panel
from data.universe import apply_st_filter, build_mask
from factors.composite import composite_score
from strategy.multi_factor import MultiFactorTopN
from strategy.timing import VolTargetTiming

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


# ---------------------------------------------------------------- 小工具
def _pct(d: dict) -> dict:
    """把小数型指标转成百分比，方便阅读。"""
    keys = ("累计收益率", "年化收益率", "年化波动率", "最大回撤", "累计超额", "年化超额", "跑赢天数占比")
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in keys and isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = f"{v * 100:.2f}%"
        elif isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = round(float(v), 3)
        else:
            out[k] = v
    return out


def _slice(nav: pd.Series, start: str, end: str) -> pd.Series:
    seg = nav.loc[start:end].dropna()
    return seg / seg.iloc[0] if len(seg) else seg


def _block(nav_s, nav_b, start, end) -> dict:
    s = _slice(nav_s, start, end)
    b = _slice(nav_b, start, end)
    if len(s) < 2 or len(b) < 2:
        return {}
    row = {"策略": _pct(perf(s)), "基准": _pct(perf(b)), "超额": _pct(excess_stats(s, b))}
    return row


def _print_block(title: str, row: dict):
    if not row:
        print(f"\n【{title}】数据不足，跳过")
        return
    keys = ["累计收益率", "年化收益率", "最大回撤", "夏普比率", "年化波动率", "卡玛比率"]
    keys += [k for k in row["超额"] if k not in keys]
    table = {}
    for k in keys:
        table[k] = {
            "策略": row["策略"].get(k, ""),
            "基准": row["基准"].get(k, ""),
            "超额": row["超额"].get(k, ""),
        }
    print(f"\n【{title}】")
    print(pd.DataFrame(table).T.to_string())


def _run_one(cfg, panel, mask, score, timing_on):
    strat = MultiFactorTopN(
        n_hold=cfg["strategy"]["n_hold"],
        buffer_pct=cfg["strategy"]["buffer_pct"],
        rebalance_days=cfg["strategy"]["rebalance_days"],
    )
    w = strat.generate_weights(panel, score, mask)
    if timing_on and cfg["timing"]["enabled"]:
        t = cfg["timing"]
        w = VolTargetTiming(t["target_vol"], t["window"], t["max_lev"]).apply(w, panel)
    res = run_backtest(
        w, panel, mask,
        cost=CostModel.from_cfg(cfg["backtest"]),
        init_cash=cfg["backtest"]["init_cash"],
        rebalance_days=cfg["strategy"]["rebalance_days"],
    )
    return res, w


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["offline", "akshare", "baostock"])
    ap.add_argument("--no-timing", action="store_true", help="关掉波动率择时")
    ap.add_argument("--quick", action="store_true", help="跳过参数敏感性扫描")
    args = ap.parse_args()

    cfg = CONFIG
    if args.mode:
        cfg["data"]["mode"] = args.mode

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  量化交易系统 · 多因子选股 + 波动率择时")
    print(f"  数据模式：{cfg['data']['mode']}   区间：{cfg['data']['start']} ~ {cfg['data']['end']}")
    print("=" * 78)

    # ---- 1. 数据 ----
    print("\n[1/6] 载入数据 ...")
    panel, bench = load_panel(cfg)
    mask = build_mask(panel, cfg)
    mask = apply_st_filter(mask, cfg, is_st=panel.get("isST"))
    print(f"      股票数 {panel['close'].shape[1]}，交易日 {panel['close'].shape[0]}，"
          f"可交易比例 {mask.to_numpy().mean() * 100:.1f}%")

    # ---- 2. 因子 ----
    print("\n[2/6] 计算因子并合成综合打分 ...")
    score, detail = composite_score(panel, cfg["factors"], mask)
    print(f"      因子：{list(cfg['factors'])}")

    # ---- 3. 因子有效性检验（只在样本内做，避免用未来信息选因子）----
    print("\n[3/6] 因子有效性检验（RankIC + 分层）...")
    fwd = ic_mod.forward_return(panel["close"], horizon=20)
    split = cfg["split"]
    ic_is = {}
    for name, f in detail.items():
        # 注意用 [:, :] 显式切两维：只写 [a:b] 时类型检查器无法判断
        # 返回的是 DataFrame 还是 Series（pandas 的类型存根是这么定义的）
        s = ic_mod.rank_ic(
            f.loc[split["is_start"]: split["is_end"], :],
            fwd.loc[split["is_start"]: split["is_end"], :],
        )
        ic_is[name] = ic_mod.ic_summary(s)
    ic_table = pd.DataFrame(ic_is).T
    print(ic_table.to_string())

    layered = ic_mod.layered_return(
        score.loc[split["is_start"]: split["is_end"], :],
        fwd.loc[split["is_start"]: split["is_end"], :],
    )
    plots.plot_layered(layered, out_dir / "layered.png")

    # ---- 4. 策略与回测 ----
    print("\n[4/6] 生成信号并回测 ...")
    res, weights = _run_one(cfg, panel, mask, score, timing_on=not args.no_timing)
    nav_s = res["nav"]
    nav_b = bench.reindex(nav_s.index).ffill()

    res_raw, _ = _run_one(cfg, panel, mask, score, timing_on=False)

    n_trades = len(res["trades"])
    n_years = len(nav_s) / 252.0
    if n_trades:
        one_way = res["trades"]["amount"].sum() / 2.0      # 买卖各算一半
        avg_nav = float(nav_s.mean())
        annual_turn = one_way / avg_nav / n_years if avg_nav > 0 else 0.0
    else:
        annual_turn = 0.0
    print(f"      成交 {n_trades} 笔，年均单边换手率 {annual_turn:.1f} 倍（行业常见 5~15 倍）")
    print(f"      累计交易成本 {res['trades']['fee'].sum():,.0f} 元 "
          f"（占初始资金 {res['trades']['fee'].sum() / cfg['backtest']['init_cash'] * 100:.1f}%）")

    # ---- 5. 指标 ----
    print("\n[5/6] 计算绩效指标 ...")
    row_all = _block(nav_s, nav_b, cfg["data"]["start"], cfg["data"]["end"])
    row_is = _block(nav_s, nav_b, split["is_start"], split["is_end"])
    row_oos = _block(nav_s, nav_b, split["oos_start"], split["oos_end"])

    _print_block(f"样本内 IS（{split['is_start']} ~ {split['is_end']}）", row_is)
    _print_block(f"样本外 OOS（{split['oos_start']} ~ {split['oos_end']}）", row_oos)
    _print_block("全样本", row_all)

    # 消融：有/无择时对比
    nav_raw = res_raw["nav"]
    r_raw = _block(nav_raw, nav_b, split["oos_start"], split["oos_end"])
    if r_raw:
        print("\n【消融对比 · 样本外】")
        print(pd.DataFrame({
            "仅多因子": r_raw["策略"],
            "多因子+择时": row_oos["策略"],
        }).T.to_string())

    # ---- 6. 输出 ----
    print("\n[6/6] 保存图表与表格 ...")
    plots.plot_nav(
        nav_s, nav_b, trades=res["trades"], split_date=split["oos_start"],
        title="多因子选股 + 波动率择时 vs 基准（归一化净值）",
        path=out_dir / "nav.png",
    )
    plots.plot_ic(pd.DataFrame({k: ic_mod.rank_ic(v, fwd) for k, v in detail.items()}),
                  out_dir / "ic.png")

    # 指标表：整理成「区间 / 指标 / 策略 / 基准 / 超额」的规整格式，
    # 好处是可以直接复制进技术报告，不用再手工对齐
    main_keys = ["累计收益率", "年化收益率", "最大回撤", "夏普比率", "年化波动率", "卡玛比率"]
    excess_keys = ["累计超额", "年化超额", "跟踪误差", "信息比率", "跑赢天数占比"]
    rows = []
    for label, r in (("样本内", row_is), ("样本外", row_oos), ("全样本", row_all)):
        if not r:
            continue
        for k in main_keys:
            rows.append({
                "区间": label, "指标": k,
                "策略": r["策略"].get(k, ""), "基准": r["基准"].get(k, ""), "超额": "",
            })
        for k in excess_keys:
            rows.append({
                "区间": label, "指标": k,
                "策略": "", "基准": "", "超额": r["超额"].get(k, ""),
            })
    rows.append({"区间": "回测概况", "指标": "数据模式", "策略": cfg["data"]["mode"], "基准": "", "超额": ""})
    rows.append({"区间": "回测概况", "指标": "年均单边换手率(倍)", "策略": round(annual_turn, 2), "基准": "", "超额": ""})
    rows.append({"区间": "回测概况", "指标": "累计交易成本(元)", "策略": round(float(res["trades"]["fee"].sum()), 0) if n_trades else 0, "基准": "", "超额": ""})
    rows.append({"区间": "回测概况", "指标": "成交笔数", "策略": n_trades, "基准": "", "超额": ""})
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    if n_trades:
        res["trades"].to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    ic_table.to_csv(out_dir / "ic_summary.csv", encoding="utf-8-sig")

    # 参数敏感性（用样本内的夏普，避免样本外信息泄露）
    if not args.quick:
        print("      扫描参数敏感性（样本内）...")
        grid_n, grid_s = [10, 20, 30], [0.0, 0.5]
        table = pd.DataFrame(index=grid_n, columns=grid_s, dtype=float)
        table.index.name = "持仓只数"
        table.columns.name = "缓冲区"
        for n_hold in grid_n:
            for buf in grid_s:
                c2 = {**cfg, "strategy": {**cfg["strategy"], "n_hold": n_hold, "buffer_pct": buf}}
                r2, _ = _run_one(c2, panel, mask, score, timing_on=not args.no_timing)
                s2 = _slice(r2["nav"], split["is_start"], split["is_end"])
                m = perf(s2)
                table.loc[n_hold, buf] = float(m.get("夏普比率", np.nan))
        plots.plot_sensitivity(table, out_dir / "sensitivity.png")
        table.to_csv(out_dir / "sensitivity.csv", encoding="utf-8-sig")
        print(table.round(2).to_string())

    print(f"\n完成。所有结果在：{out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"   - {f.name}")

    # 把图另存一份到 docs/figures，方便技术报告直接引用
    # （outputs/ 不进 git，docs/figures/ 进 git，这是刻意的分工）
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("nav.png", "ic.png", "layered.png", "sensitivity.png"):
        src_f = out_dir / name
        if src_f.exists():
            shutil.copy2(src_f, fig_dir / name)
            copied.append(name)
    if copied:
        print(f"\n图表已同步到 docs/figures/：{', '.join(copied)}")


if __name__ == "__main__":
    sys.exit(main())
