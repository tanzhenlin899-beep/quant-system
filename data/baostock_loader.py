"""
baostock 数据接入。

为什么要有这个数据源：
  1. **不限流**。akshare 用的东方财富接口按 IP 限流，被封一次要等 40 分钟；
     baostock 是 socket 协议，稳定得多。
  2. **自带停牌和 ST 标记**。返回字段里有：
        tradestatus  1=正常交易  0=停牌
        isST         1=是 ST 股  0=不是
     有了这两个字段，就不用再靠"成交量=0"去猜停牌，也不用维护 ST 名单。

两个必须注意的坑：
  * **复权基准和 akshare 不同**。同一只茅台，akshare 后复权约 9548，
    baostock 约 11952。所以缓存目录必须分开（用 data/cache/bs/），
    否则两套数据混在一起会算出完全错误的结果。
  * **成交量单位不同**。akshare 是"手"（100 股），baostock 是"股"。
    对量价类因子只是差一个常数倍数，不影响相对排序，但报告里要说明。

接口说明：
    adjustflag: "1"=后复权  "2"=前复权  "3"=不复权
    回测一律用 "1"（后复权），指数用 "3"（指数本身不需要复权）。
"""

import threading
from pathlib import Path

import pandas as pd

# baostock 用全局连接，并发调用会串数据，这里加锁保护
_LOCK = threading.Lock()

_FIELDS = "date,open,high,low,close,volume,amount,tradestatus,isST,pctChg"
_NUM_COLS = ["open", "close", "high", "low", "volume", "amount", "pctChg"]


def to_bs_code(code: str) -> str | None:
    """把 6 位代码转成 baostock 的 sh./sz. 格式。

    baostock 不支持北交所（4/8 开头），返回 None 让调用方跳过。
    """
    c = str(code).strip()
    if c.startswith(("sh.", "sz.")):
        return c
    if c.startswith("6"):
        return f"sh.{c}"
    if c.startswith(("0", "3")):
        return f"sz.{c}"
    return None


def load_panel_baostock(cfg):
    """返回 (panel, benchmark)。

    panel 里除了 open/close/high/low/volume，还额外带两列布尔面板：
        panel["tradestatus"]  True = 当天正常交易（False = 停牌）
        panel["isST"]         True = 当天是 ST 股
    这两个会被 build_mask / apply_st_filter 用上。

    优化：如果缓存已经齐全，**完全不登录 baostock**，直接读本地 parquet。
    这样答辩现场断网也能跑。
    """
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]
    cache = Path(cfg["data"]["cache_dir"]) / "bs"
    cache.mkdir(parents=True, exist_ok=True)
    bench_code = cfg["data"]["benchmark_code"]
    idx_path = cache / f"index_{bench_code}.parquet"

    # ---------- 1. 缓存齐全就直接用，不联网 ----------
    frames = _read_cache(cfg["data"]["codes"], cache, start, end)
    if frames is not None and idx_path.exists():
        print("      缓存齐全，直接读本地文件（不联网）")
        panel = _assemble_panel(frames)
        bench = pd.read_parquet(idx_path).loc[start:end, "close"].rename("benchmark")
        return panel, bench

    # ---------- 2. 缓存不全，联网补齐 ----------
    import baostock as bs

    interval = float(cfg["data"].get("request_interval_bs", 0.05))

    with _LOCK:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败：{lg.error_code} {lg.error_msg}")
        try:
            total = len(cfg["data"]["codes"])
            for i, code in enumerate(cfg["data"]["codes"], 1):
                df = _fetch_one(bs, code, start, end, cache)
                if df is None or df.empty:
                    print(f"      [{i}/{total}] {code} 跳过（无数据或 baostock 不支持）")
                else:
                    print(f"      [{i}/{total}] {code} {len(df)} 行")
                if interval:
                    import time

                    time.sleep(interval)

            frames = _read_cache(cfg["data"]["codes"], cache, start, end)
            if not frames:
                raise RuntimeError(
                    "baostock 一只股票都没取到。检查 config 里的 codes 是否是 6 位 A 股代码。"
                )

            panel = _assemble_panel(frames)
            bench = _fetch_index(bs, bench_code, start, end, cache)
        finally:
            bs.logout()

    return panel, bench


def _read_cache(codes, cache: Path, start, end):
    """把缓存里已有的股票全部读出来。有任何一个缺失就返回 None（触发联网补齐）。"""
    frames = {}
    for code in codes:
        if to_bs_code(code) is None:
            continue
        path = cache / f"{code}_hfq.parquet"
        if not path.exists():
            return None
        frames[code] = pd.read_parquet(path).loc[start:end]
    return frames or None


def _assemble_panel(frames) -> dict:
    """把 {代码: 单只股票 DataFrame} 拼成 date × code 的面板矩阵。"""
    panel = {}
    for field in ("open", "close", "high", "low", "volume"):
        panel[field] = pd.DataFrame({c: d[field] for c, d in frames.items()}).sort_index()

    index = panel["close"].index
    # 停牌 / ST 标记（缺失行按"不可交易 / 非 ST"处理）
    panel["tradestatus"] = pd.DataFrame(
        {c: d["tradestatus"] for c, d in frames.items()}
    ).reindex(index).fillna(False).astype(bool)
    panel["isST"] = pd.DataFrame(
        {c: d["isST"] for c, d in frames.items()}
    ).reindex(index).fillna(False).astype(bool)
    return panel


def _fetch_one(bs, code, start, end, cache: Path):
    """下载单只股票日线（后复权），缓存到 parquet。"""
    bs_code = to_bs_code(code)
    if bs_code is None:
        return None

    path = cache / f"{code}_hfq.parquet"
    if path.exists():
        return pd.read_parquet(path).loc[start:end]

    rs = bs.query_history_k_data_plus(
        bs_code, _FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag="1",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{code} 取数失败：{rs.error_code} {rs.error_msg}")

    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    df = _rows_to_frame(rows, rs.fields)
    if df is None or df.empty:
        return None
    df.to_parquet(path)
    return df


def _rows_to_frame(rows, fields) -> pd.DataFrame | None:
    """把 baostock 的字符串行转成规范 DataFrame（index=日期）。"""
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=fields)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 布尔化：tradestatus "1" = 正常交易；isST "1" = 是 ST
    for c in ("tradestatus", "isST"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).astype(bool)

    keep = [c for c in ("open", "close", "high", "low", "volume", "amount",
                        "pctChg", "tradestatus", "isST") if c in df.columns]
    return df[keep]


def _fetch_index(bs, code, start, end, cache: Path) -> pd.Series:
    """下载基准指数（指数用不复权）。"""
    path = cache / f"index_{code}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df.loc[start:end, "close"].rename("benchmark")

    bs_code = f"sh.{code}" if str(code).startswith("000") else f"sz.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume,amount,pctChg",
        start_date=start, end_date=end, frequency="d", adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"指数 {code} 取数失败：{rs.error_code} {rs.error_msg}")

    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    df = _rows_to_frame(rows, rs.fields)
    if df is None or df.empty or "close" not in df.columns:
        raise RuntimeError(f"指数 {code} 返回空数据，检查 benchmark_code")

    print(f"      基准指数 {bs_code}，{len(df)} 行")
    df.to_parquet(path)
    return df.loc[start:end, "close"].rename("benchmark")
