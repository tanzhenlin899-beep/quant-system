"""
RankIC 计算的正确性测试。

背景：早期实现逐日调用 Series.corr(method="spearman")，
在 1942 个交易日 × 4 个因子上约需 10 秒，是 Web 页面响应慢的主因。
改为向量化实现后降到毫秒级。本测试确保两种实现结果一致，
避免将来重构时悄悄改变因子统计结果。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.ic import ic_summary, rank_ic


def _reference(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """参考实现：逐日 Spearman 相关（慢，但直观）。"""
    out = {}
    for dt in factor.index:
        a, b = factor.loc[dt], fwd.loc[dt]
        ok = a.notna() & b.notna()
        if ok.sum() < 10:
            continue
        out[dt] = a[ok].corr(b[ok], method="spearman")
    return pd.Series(out).sort_index()


def test_matches_reference_with_nan():
    """含缺失值时，向量化实现应与逐日实现一致。"""
    rng = np.random.default_rng(11)
    n_d, n_s = 200, 50
    dates = pd.bdate_range("2021-01-01", periods=n_d)
    codes = [f"S{i}" for i in range(n_s)]

    f = pd.DataFrame(rng.normal(size=(n_d, n_s)), index=dates, columns=codes)
    r = f * 0.25 + pd.DataFrame(rng.normal(size=(n_d, n_s)), index=dates, columns=codes)
    f = f.mask(rng.random((n_d, n_s)) < 0.15)
    r = r.mask(rng.random((n_d, n_s)) < 0.15)

    new = rank_ic(f, r)
    old = _reference(f, r)
    common = new.index.intersection(old.index)

    assert len(common) > 100, "有效交易日太少"
    assert np.allclose(new.loc[common], old.loc[common], atol=1e-12), "与参考实现不一致"
    return f"{len(common)} 个交易日全部一致（含缺失值）"


def test_perfect_rank_correlation():
    """因子与未来收益完全同序时，RankIC 应为 1。"""
    dates = pd.bdate_range("2021-01-01", periods=30)
    codes = [f"S{i}" for i in range(20)]
    base = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (30, 1)), index=dates, columns=codes
    )
    ic = rank_ic(base, base * 2 + 1)
    assert np.allclose(ic.values, 1.0), f"完全同序时 IC 应为 1，实际 {ic.values[:3]}"

    # 完全反序应为 -1
    ic_rev = rank_ic(base, -base)
    assert np.allclose(ic_rev.values, -1.0), "完全反序时 IC 应为 -1"
    return "完全同序 IC=1，完全反序 IC=−1"


def test_insufficient_samples_skipped():
    """有效样本少于 10 只的横截面应被跳过。"""
    dates = pd.bdate_range("2021-01-01", periods=5)
    codes = [f"S{i}" for i in range(20)]
    f = pd.DataFrame(np.random.default_rng(3).normal(size=(5, 20)), index=dates, columns=codes)
    r = f.copy()
    f.iloc[:, 10:] = np.nan      # 每天只有 10 只有值
    ic = rank_ic(f, r)
    # 恰好 10 只，应保留（阈值为 >= 10）
    assert len(ic) == 5, f"恰好 10 只应保留，实际保留 {len(ic)} 天"

    f.iloc[:, 9:] = np.nan       # 只剩 9 只
    ic2 = rank_ic(f, r)
    assert len(ic2) == 0, "少于 10 只应全部跳过"
    return "样本数阈值判断正确"


def test_ic_summary_math():
    """ic_summary 的均值、ICIR 与手算一致。"""
    ic = pd.Series([0.1, 0.2, -0.1, 0.3, 0.0])
    s = ic_summary(ic)
    assert abs(s["IC均值"] - 0.1) < 1e-12
    assert abs(s["ICIR"] - ic.mean() / ic.std()) < 1e-12
    assert abs(s["IC>0占比"] - 0.6) < 1e-12
    assert s["样本天数"] == 5

    assert ic_summary(pd.Series(dtype=float)) == {}
    return "IC 均值 / ICIR / 正值占比 与手算一致"


def main():
    tests = [
        ("1. 与逐日参考实现一致（含缺失值）", test_matches_reference_with_nan),
        ("2. 完全同序与完全反序", test_perfect_rank_correlation),
        ("3. 样本数不足时跳过", test_insufficient_samples_skipped),
        ("4. IC 汇总统计", test_ic_summary_math),
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
