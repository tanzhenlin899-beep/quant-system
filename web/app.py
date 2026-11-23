"""
Web 后端（FastAPI）。

设计原则（这条比代码本身重要）：
    **每次 /api/run 请求都真的调用 backtest.run_backtest() 重新算一遍，不读预存结果。**
    课程明确写了"禁止仅由静态页面构成"，所以这个页面必须是真的。
    你在页面上改一个参数，后端就重新载数据、重新算因子、重新撮合，
    净值曲线是这个请求现场算出来的，不是提前画好的图。

    为了演示时不至于等太久，做了两件事：
      1. 股票数可以调小（默认 40 只），缩短计算时间；
      2. 同一个参数组合的结果在内存里缓存，重复点击秒回。

启动方式：
    uvicorn web.app:app --reload --port 8000
    然后浏览器打开 http://127.0.0.1:8000
"""

import copy
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backtest.cost import CostModel
from backtest.engine import run_backtest
from backtest.metrics import excess_stats, perf
from config import CONFIG
from data.loader import load_panel
from data.universe import apply_st_filter, build_mask
from factors.composite import composite_score
from analysis import ic as ic_mod
from strategy.multi_factor import MultiFactorTopN
from strategy.timing import VolTargetTiming

app = FastAPI(title="多因子选股量化交易系统", docs_url="/api/docs")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# 结果缓存：同一组参数重复点击时秒回
_CACHE: dict[tuple, dict] = {}


# ---------------------------------------------------------------- 请求模型
class RunParams(BaseModel):
    n_hold: int = Field(20, ge=1, le=50)
    buffer_pct: float = Field(0.5, ge=0.0, le=2.0)
    rebalance_days: int = Field(10, ge=1, le=60)
    use_timing: bool = True
    target_vol: float = Field(0.18, ge=0.05, le=0.60)
    n_codes: int = Field(40, ge=10, le=100)


# ---------------------------------------------------------------- 工具
def _num(x):
    """转成 JSON 能接受的数字（NaN / inf 一律变成 None）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _period_block(nav_s, nav_b, start, end) -> dict:
    s = nav_s.loc[start:end].dropna()
    if len(s) < 2:
        return {}
    s = s / s.iloc[0]
    b = nav_b.loc[start:end].dropna()
    b = b / b.iloc[0] if len(b) else b

    ms, mb = perf(s), perf(b)
    out: dict[str, dict] = {"策略": {}, "基准": {}, "超额": {}}
    for k in ("累计收益率", "年化收益率", "最大回撤", "夏普比率", "年化波动率", "卡玛比率"):
        out["策略"][k] = _num(ms.get(k))
        out["基准"][k] = _num(mb.get(k))
    if len(b) > 1:
        ex = excess_stats(s, b)
        for k in ("累计超额", "年化超额", "信息比率", "跑赢天数占比"):
            out["超额"][k] = _num(ex.get(k))
    return out


def _compute(p: RunParams) -> dict:
    """真正干活的地方：载数据 → 算因子 → 生成信号 → 跑回测 → 算指标。"""
    t0 = time.time()
    _last = [t0]

    def _mark(name):
        """记录各阶段耗时，供前端调试面板显示。"""
        now = time.time()
        stage_t[name] = round(now - _last[0], 2)
        _last[0] = now

    stage_t: dict[str, float] = {}

    cfg = copy.deepcopy(CONFIG)
    cfg["strategy"]["n_hold"] = p.n_hold
    cfg["strategy"]["buffer_pct"] = p.buffer_pct
    cfg["strategy"]["rebalance_days"] = p.rebalance_days
    cfg["timing"]["enabled"] = p.use_timing
    cfg["timing"]["target_vol"] = p.target_vol

    panel, bench = load_panel(cfg)
    codes = list(panel["close"].columns)[: p.n_codes]
    panel = {k: v[codes] for k, v in panel.items()}
    _mark("载入数据")

    mask = build_mask(panel, cfg)
    mask = apply_st_filter(mask, cfg, is_st=panel.get("isST"))
    score, detail = composite_score(panel, cfg["factors"], mask)
    _mark("计算因子")

    w = MultiFactorTopN(p.n_hold, p.buffer_pct, p.rebalance_days).generate_weights(
        panel, score, mask
    )
    if p.use_timing:
        w = VolTargetTiming(p.target_vol, cfg["timing"]["window"],
                            cfg["timing"]["max_lev"]).apply(w, panel)
    _mark("生成信号")

    res = run_backtest(
        w, panel, mask,
        cost=CostModel.from_cfg(cfg["backtest"]),
        init_cash=cfg["backtest"]["init_cash"],
        rebalance_days=p.rebalance_days,
    )
    nav = res["nav"]
    nav_b = bench.reindex(nav.index).ffill()
    _mark("回测撮合")

    split = cfg["split"]
    periods = {
        "样本内": _period_block(nav, nav_b, split["is_start"], split["is_end"]),
        "样本外": _period_block(nav, nav_b, split["oos_start"], split["oos_end"]),
        "全样本": _period_block(nav, nav_b, split["is_start"], split["oos_end"]),
    }

    # 净值曲线（归一化到 1.0）
    ns = nav / nav.iloc[0]
    nb = nav_b / nav_b.iloc[0]
    nav_json = {
        "dates": [d.strftime("%Y-%m-%d") for d in ns.index],
        "strategy": [_num(v) for v in ns.values],
        "benchmark": [_num(v) for v in nb.values],
        "split_date": split["oos_start"],
    }

    # 因子 IC
    fwd = ic_mod.forward_return(panel["close"], horizon=20)
    ic_rows = []
    for name, f in detail.items():
        s = ic_mod.rank_ic(
            f.loc[split["is_start"]: split["is_end"], :],
            fwd.loc[split["is_start"]: split["is_end"], :],
        )
        summ = ic_mod.ic_summary(s)
        ic_rows.append({
            "因子": name,
            "IC均值": _num(summ.get("IC均值")),
            "ICIR": _num(summ.get("ICIR")),
            "IC>0占比": _num(summ.get("IC>0占比")),
        })
    ic_rows.sort(key=lambda r: abs(r["IC均值"] or 0), reverse=True)

    # 换手率与成本
    trades = res["trades"]
    n_years = len(nav) / 252.0
    if len(trades):
        one_way = float(trades["amount"].sum()) / 2.0
        avg_nav = float(nav.mean())
        turnover = one_way / avg_nav / n_years if avg_nav > 0 else 0.0
        total_fee = float(trades["fee"].sum())
    else:
        turnover, total_fee = 0.0, 0.0
    _mark("计算指标")

    # ---------------- 调试面板数据 ----------------
    last = nav.index[-1]

    # 1) 最后交易日的目标持仓
    holdings = []
    w_last = w.loc[last] if last in w.index else None
    if w_last is not None:
        for code, val in w_last.items():
            if val is not None and float(val) > 1e-6:
                holdings.append({"代码": code, "权重": round(float(val), 4)})
        holdings.sort(key=lambda x: -x["权重"])

    # 2) 最近成交（最多 60 笔）
    trade_rows = []
    if len(trades):
        for _, r in trades.tail(60).iterrows():
            trade_rows.append({
                "日期": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                "代码": str(r["code"]),
                "方向": "买入" if r["side"] == "BUY" else "卖出",
                "股数": int(r["shares"]),
                "价格": _num(r["price"]),
                "金额": _num(r["amount"]),
                "费用": _num(r["fee"]),
            })

    # 3) 最后交易日综合打分前 10 名，附各因子 z 值
    snap = []
    if last in score.index:
        top = score.loc[last].dropna().sort_values(ascending=False).head(10)
        for code, sc in top.items():
            row = {"代码": code, "综合分": _num(sc)}
            for name, f in detail.items():
                v = f.loc[last, code] if (last in f.index and code in f.columns) else None
                row[name] = _num(v)
            snap.append(row)

    return {
        "meta": {
            "n_codes": len(codes),
            "n_hold": p.n_hold,
            "buffer_pct": p.buffer_pct,
            "rebalance_days": p.rebalance_days,
            "use_timing": p.use_timing,
            "target_vol": p.target_vol,
            "n_trades": int(len(trades)),
            "turnover": _num(turnover),
            "total_fee": _num(total_fee),
            "elapsed": round(time.time() - t0, 2),
            "data_mode": cfg["data"]["mode"],
            "start": split["is_start"],
            "end": split["oos_end"],
            # 卡片标题用这两个字段，避免前端硬编码年份（改配置后标题会跟着变）
            "is_range": f"{split['is_start'][:4]}–{split['is_end'][:4]}",
            "oos_range": f"{split['oos_start'][:4]}–{split['oos_end'][:4]}",
            # 各阶段耗时，供前端调试面板显示
            "timings": stage_t,
        },
        "periods": periods,
        "holdings": holdings,
        "trades": trade_rows,
        "factors": snap,
        "nav": nav_json,
        "ic": ic_rows,
    }


# ---------------------------------------------------------------- 路由
@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/defaults")
def defaults():
    codes = CONFIG["data"]["codes"]
    return {
        "n_hold": CONFIG["strategy"]["n_hold"],
        "buffer_pct": CONFIG["strategy"]["buffer_pct"],
        "rebalance_days": CONFIG["strategy"]["rebalance_days"],
        "use_timing": CONFIG["timing"]["enabled"],
        "target_vol": CONFIG["timing"]["target_vol"],
        # 真实数据下股票池就是配置里的代码数，别让滑块显示 40 而实际只有 30
        "n_codes": min(40, len(codes)) if CONFIG["data"]["mode"] != "offline" else 40,
        "n_codes_available": 100 if CONFIG["data"]["mode"] == "offline" else len(codes),
        "factors": list(CONFIG["factors"].keys()),
        "data_mode": CONFIG["data"]["mode"],
    }


@app.post("/api/run")
def run(p: RunParams):
    key = (p.n_hold, p.buffer_pct, p.rebalance_days, p.use_timing, p.target_vol, p.n_codes)
    if key in _CACHE:
        out = copy.deepcopy(_CACHE[key])
        out["meta"]["cached"] = True
        return out
    out = _compute(p)
    out["meta"]["cached"] = False
    _CACHE[key] = copy.deepcopy(out)
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "cache_entries": len(_CACHE)}
