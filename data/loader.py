"""
数据接入模块。

两种模式（在 config.py 的 data.mode 里切换）：
  offline -> 用 data/synthetic.py 造的数据，不需要网络，用来跑通流程
  akshare -> 用真实 A 股历史数据，第一次运行会下载并缓存到 data/cache/

关于 akshare 接口的稳定性（踩过的坑，很重要）：
  1. akshare 会**改接口名**。例如 1.20.46 起 `stock_zh_a_hist` 改名为
     `stock_zh_a_hist_em`，`index_zh_a_hist` 也改名了。所以这里不写死函数名，
     而是按候选列表依次尝试，谁的签名能跑通用谁——这样升级 akshare 不至于直接崩。
  2. 东方财富的行情接口有**限流**：短时间内请求太多会被临时封 IP
     （通常 5 分钟恢复，反复触发可能要 40 分钟以上）。
     所以默认每次请求间隔 0.7 秒，并且失败后按 10s / 30s / 60s 退避重试。
     已经下载过的股票会被缓存，中断后重新运行会从断点继续，不会重复请求。

设计要点：
  - 输出统一是 date × code 的面板矩阵，后面模块不关心数据从哪来；
  - 全部用后复权价，收益率不含除权跳空。
"""

import time
from pathlib import Path

import pandas as pd

from data import synthetic

# akshare 的中文列名 -> 我们的标准列名（不同版本可能有中英文两种）
_COL_MAP = {
    "日期": "date", "date": "date", "时间": "date",
    "开盘": "open", "open": "open",
    "收盘": "close", "close": "close",
    "最高": "high", "high": "high",
    "最低": "low", "low": "low",
    "成交量": "volume", "volume": "volume", "vol": "volume",
}

# 候选接口名（新名字在前，老名字兜底）
_STOCK_HIST_FUNCS = ("stock_zh_a_hist_em", "stock_zh_a_hist")

_WANT_COLS = ["open", "close", "high", "low", "volume"]


def load_panel(cfg):
    """统一入口：返回 (panel, benchmark)。

    三种数据源：
        offline  内置模拟数据，不联网，用来跑通流程
        akshare  东方财富，速度快，但会限流
        baostock 不限流，自带停牌/ST 标记，但逐行拉取较慢
    """
    mode = cfg["data"]["mode"]
    if mode == "offline":
        return synthetic.make_panel(
            codes=_offline_codes(cfg), start=cfg["data"]["start"], end=cfg["data"]["end"]
        )
    if mode == "akshare":
        return _load_from_akshare(cfg)
    if mode == "baostock":
        from data.baostock_loader import load_panel_baostock

        return load_panel_baostock(cfg)
    raise ValueError(f"未知的 data.mode: {mode}（支持 offline / akshare / baostock）")


def _offline_codes(cfg):
    """离线模式用 100 只虚构股票（代码只是占位，不对应真实标的）。"""
    return [f"SIM{i:04d}" for i in range(100)]


# ----------------------------------------------------------------------------
# akshare 真实数据
# ----------------------------------------------------------------------------
def _load_from_akshare(cfg):
    import akshare as ak  # 延迟导入：offline 模式下不需要这个包

    start = cfg["data"]["start"].replace("-", "")
    end = cfg["data"]["end"].replace("-", "")
    cache = Path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    interval = float(cfg["data"].get("request_interval", 0.7))

    frames = {}
    total = len(cfg["data"]["codes"])
    for i, code in enumerate(cfg["data"]["codes"], 1):
        df = _fetch_one(ak, code, start, end, cache, interval)
        if df is not None and not df.empty:
            frames[code] = df
            print(f"      [{i}/{total}] {code} {len(df)} 行")
        else:
            print(f"      [{i}/{total}] {code} 跳过（无数据）")
        time.sleep(interval)

    if not frames:
        raise RuntimeError(
            "一只股票都没下载到。请检查：1) 网络；2) config 里的 codes 是否有效；"
            "3) 是否被东方财富限流（等 5 分钟后重试）"
        )

    panel = {}
    for field in ("open", "close", "high", "low", "volume"):
        panel[field] = pd.DataFrame({c: d[field] for c, d in frames.items()}).sort_index()

    bench = _fetch_index(ak, cfg["data"]["benchmark_code"], start, end, cache, interval)
    return panel, bench


def _retry(fn, *args, tries: int = 3, waits=(10, 30, 60), label: str = "", **kw):
    """带退避的重试。遇到限流就等一会儿再试，别把 IP 封死。"""
    last = None
    for attempt in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            if attempt < tries - 1:
                wait = waits[min(attempt, len(waits) - 1)]
                hint = "（可能被限流）" if ("RateLimit" in type(e).__name__ or "限流" in msg
                                           or "断开了连接" in msg) else ""
                print(f"      {label} 第 {attempt + 1} 次失败{hint}，{wait}s 后重试：{msg[:80]}")
                time.sleep(wait)
    if last is None:
        raise RuntimeError(f"{label} 重试逻辑异常：没有捕获到任何异常")
    raise last


def _stock_hist_fn(ak):
    for name in _STOCK_HIST_FUNCS:
        fn = getattr(ak, name, None)
        if callable(fn):
            return name, fn
    raise RuntimeError(
        f"akshare {ak.__version__} 里找不到可用的个股日线接口"
        f"（试过 {_STOCK_HIST_FUNCS}），请查看 akshare 更新日志"
    )


def _normalize(raw: pd.DataFrame) -> pd.DataFrame | None:
    """把 akshare 返回的原始表整理成 index=日期、列=open/close/... 的规范格式。"""
    if raw is None or raw.empty:
        return None
    df = raw.rename(columns={k: v for k, v in _COL_MAP.items() if k in raw.columns})
    if "date" not in df.columns:
        # 有些接口把日期放在 index 里
        df = df.reset_index().rename(columns=_COL_MAP)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    keep = [c for c in _WANT_COLS if c in df.columns]
    if "close" not in keep:
        return None
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    return df


def _fetch_one(ak, code, start, end, cache: Path, interval: float):
    """下载单只股票日线（后复权），并缓存成 parquet。"""
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    path = cache / f"{code}_hfq.parquet"
    if path.exists():
        # 缓存里可能是更长的区间，读出来按当前区间裁剪
        return pd.read_parquet(path).loc[s:e]

    name, fn = _stock_hist_fn(ak)
    raw = _retry(
        fn, tries=3,
        symbol=code, period="daily", start_date=start, end_date=end, adjust="hfq",
        label=f"{code}({name})",
    )
    df = _normalize(raw)
    if df is None or df.empty:
        return None
    df.to_parquet(path)
    return df.loc[s:e]


def _fetch_index(ak, code, start, end, cache: Path, interval: float):
    """下载基准指数日线。

    指数接口在不同 akshare 版本里名字差异很大，这里按候选列表逐个试，
    谁能返回带日期列的数据就用谁。
    """
    path = cache / f"index_{code}.parquet"
    if path.exists():
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        return pd.read_parquet(path).loc[s:e, "close"].rename("benchmark")

    candidates = [
        # (函数名, 参数字典)
        ("index_zh_hist_em", dict(symbol=code, period="daily",
                                  start_date=start, end_date=end)),
        ("index_zh_a_hist", dict(symbol=code, period="daily",
                                 start_date=start, end_date=end)),
        ("stock_zh_index_daily_em", dict(symbol=f"sh{code}",
                                         start_date=start, end_date=end)),
        ("stock_zh_index_daily", dict(symbol=f"sh{code}")),
        ("index_zh_daily_sina", dict(symbol=f"sh{code}")),
    ]

    errors = []
    for fname, kw in candidates:
        fn = getattr(ak, fname, None)
        if not callable(fn):
            continue
        try:
            raw = _retry(fn, tries=2, waits=(15, 30), label=f"指数({fname})", **kw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fname}: {type(e).__name__}")
            continue
        df = _normalize(raw)
        if df is not None and not df.empty and "close" in df.columns:
            print(f"      基准指数使用接口 {fname}，{len(df)} 行")
            df.to_parquet(path)
            return df["close"].rename("benchmark")
        errors.append(f"{fname}: 返回空数据")

    raise RuntimeError(
        "取不到基准指数数据，试过的接口都失败了：" + "；".join(errors)
        + "。可把 config 里的 benchmark_code 换成其他指数，或临时把基准改成等权买入持有。"
    )
