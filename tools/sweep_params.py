"""
参数扫描工具：对「持仓只数」做对照实验，产出报告里能直接用的表和图。

用法（在项目根目录）：
    python tools/sweep_params.py                # 默认扫 5 / 10 / 15 / 20 / 30
    python tools/sweep_params.py 10 20 30       # 自定义

输出：
    outputs/sweep_hold.csv       各持仓数在样本内/外的指标对比
    docs/figures/sweep_hold.png  各持仓数的净值曲线对比图（进 git，可直接贴报告）

为什么要单独做这个：
    报告里「参数选择」那一节需要证据。只说"我选了 10 只"没有说服力，
    要能拿出"10 只的样本外夏普最高，且换手率落在合理区间"这样的表。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis.plots import plot_sweep
from backtest.cost import CostModel
from backtest.engine import run_backtest
from backtest.metrics import perf
from config import CONFIG
from data.loader import load_panel
from data.universe import apply_st_filter, build_mask
from factors.composite import composite_score
from strategy.multi_factor import MultiFactorTopN
from strategy.timing import VolTargetTiming


def _seg(nav: pd.Series, start: str, end: str) -> pd.Series:
    s = nav.loc[start:end].dropna()
    return s / s.iloc[0] if len(s) else s


def main():
    cfg = CONFIG
    holds = [int(x) for x in sys.argv[1:]] or [5, 10, 15, 20, 30]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据模式：{cfg['data']['mode']}，持仓数扫描：{holds}")
    panel, bench = load_panel(cfg)
    mask = apply_st_filter(build_mask(panel, cfg), cfg, is_st=panel.get("isST"))
    score, _ = composite_score(panel, cfg["factors"], mask)

    split = cfg["split"]
    cost = CostModel.from_cfg(cfg["backtest"])
    navs, rows = {}, []

    for n in holds:
        w = MultiFactorTopN(n, cfg["strategy"]["buffer_pct"],
                            cfg["strategy"]["rebalance_days"]).generate_weights(
            panel, score, mask
        )
        t = cfg["timing"]
        if t["enabled"]:
            w = VolTargetTiming(t["target_vol"], t["window"], t["max_lev"]).apply(w, panel)

        res = run_backtest(w, panel, mask, cost=cost,
                           init_cash=cfg["backtest"]["init_cash"],
                           rebalance_days=cfg["strategy"]["rebalance_days"])
        nav = res["nav"]
        navs[n] = nav

        mi = perf(_seg(nav, split["is_start"], split["is_end"]))
        mo = perf(_seg(nav, split["oos_start"], split["oos_end"]))
        avg = float(nav.mean())
        turn = ((res["trades"]["amount"].sum() / 2) / avg / (len(nav) / 252)
                if len(res["trades"]) else 0.0)
        rows.append({
            "持仓只数": n,
            "样本内年化": round(mi.get("年化收益率", float("nan")) * 100, 2),
            "样本内夏普": round(mi.get("夏普比率", float("nan")), 3),
            "样本内最大回撤": round(mi.get("最大回撤", float("nan")) * 100, 2),
            "样本外年化": round(mo.get("年化收益率", float("nan")) * 100, 2),
            "样本外夏普": round(mo.get("夏普比率", float("nan")), 3),
            "样本外最大回撤": round(mo.get("最大回撤", float("nan")) * 100, 2),
            "年均单边换手(倍)": round(turn, 1),
        })
        print(f"  持仓 {n:>2} 只  样本内年化 {rows[-1]['样本内年化']:>6.2f}%  "
              f"样本外年化 {rows[-1]['样本外年化']:>6.2f}%  "
              f"样本外夏普 {rows[-1]['样本外夏普']:>6.3f}  换手 {turn:.1f}x")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "sweep_hold.csv", index=False, encoding="utf-8-sig")

    # 净值对比图（进 docs/figures，报告直接引用）
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep(navs, bench, split_date=split["oos_start"], path=fig_dir / "sweep_hold.png")

    print(f"\n表格：{out_dir / 'sweep_hold.csv'}")
    print(f"图表：{fig_dir / 'sweep_hold.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
