"""
数据源体检：花 3 秒判断"现在能不能用真实 A 股数据跑回测"。

用法：
    python tools/check_data.py

它会做三件事：
    1. 看本地缓存放了多少数据（缓存命中就不会再请求接口）；
    2. 真实请求一次个股接口，判断是否被限流；
    3. 看基准指数是否已缓存。

为什么要这个工具：
    东方财富的行情接口按 IP 限流，被封的时候报错信息很长但看不出重点。
    先跑这个，能立刻知道"是被限流了"还是"代码/配置有问题"。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG


def human(n: int) -> str:
    return f"{n} 个文件"


def main():
    cache = Path(CONFIG["data"]["cache_dir"])
    print("=" * 62)
    print("  数据源体检")
    print("=" * 62)

    # ---------- 1. 缓存情况 ----------
    print("\n[1/3] 本地缓存")
    if not cache.exists():
        print("      缓存目录还不存在 → 从未下载过真实数据")
        n_stock = 0
        idx_ok = False
    else:
        stocks = list(cache.glob("*_hfq.parquet"))
        idx_file = cache / f"index_{CONFIG['data']['benchmark_code']}.parquet"
        idx_ok = idx_file.exists()
        n_stock = len(stocks)
        print(f"      已缓存个股 {human(n_stock)}")
        if n_stock:
            print(f"      示例：{', '.join(p.stem for p in stocks[:5])}")
        print(f"      基准指数缓存：{'✅ 已有' if idx_ok else '❌ 没有'}")

    # ---------- 2. 个股接口连通性 ----------
    print("\n[2/3] 个股接口连通性（会真实请求一次）")
    import warnings

    warnings.filterwarnings("ignore")
    import akshare as ak

    from data.loader import _normalize, _stock_hist_fn

    name, fn = _stock_hist_fn(ak)
    print(f"      使用接口：{name}")

    # 优先用已缓存的股票来测，避免多打一次接口
    test_code = "600519"
    cached = cache / f"{test_code}_hfq.parquet" if cache.exists() else None
    if cached is not None and cached.exists():
        print(f"      {test_code} 已缓存，跳过网络请求")
        online = None
    else:
        try:
            raw = fn(symbol=test_code, period="daily",
                     start_date="20240101", end_date="20240201", adjust="hfq")
            df = _normalize(raw)
            online = df is not None and len(df) > 0
            print(f"      ✅ 接口正常，返回 {0 if df is None else len(df)} 行")
        except Exception as e:  # noqa: BLE001
            online = False
            msg = str(e)
            if "限流" in msg or "RateLimit" in type(e).__name__ or "断开" in msg:
                print("      ❌ 被东方财富限流了")
                print("         解决：等 30–40 分钟；或换网络（手机热点）立刻可用")
            else:
                print(f"      ❌ 失败：{type(e).__name__}: {msg[:120]}")

    # ---------- 3. 结论 ----------
    print("\n[3/3] 结论")
    need = len(CONFIG["data"]["codes"])
    if n_stock >= need and idx_ok:
        print(f"      ✅ 数据已齐（{n_stock}/{need} 只 + 指数已缓存），可以直接跑回测，不需要联网")
    elif online and idx_ok:
        print(f"      可以跑：接口正常，指数已有缓存，还需下载 {need - n_stock} 只个股")
    elif online is False:
        print("      ⚠️  现在跑不了：个股接口被限流")
        print("         方案 A：等 30–40 分钟后再试")
        print("         方案 B：换成手机热点（换 IP，立刻可用）")
    elif online is None and not idx_ok:
        print("      指数还没缓存，首次运行会自动下载（备用接口可能较慢）")
    else:
        print("      配置正常，可以尝试运行")

    print("\n提示：第一次跑建议用 --quick 跳过参数敏感性扫描，能快约 20 秒")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
