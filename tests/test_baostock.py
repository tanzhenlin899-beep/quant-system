"""
baostock 数据接入的离线测试（不需要联网）。

测的是纯逻辑：代码转换、字段解析、tradestatus/isST 的布尔化。
真实的网络取数能不能成功，只有在本机跑才知道了（我这边网络不通）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.baostock_loader import _rows_to_frame, to_bs_code


def test_code_conversion():
    """6 位代码 → baostock 的 sh./sz. 格式。"""
    cases = {
        "600519": "sh.600519",
        "601318": "sh.601318",
        "000001": "sz.000001",
        "300059": "sz.300059",
        "688111": "sh.688111",
        "sh.600519": "sh.600519",
        "sz.000001": "sz.000001",
        "830799": None,      # 北交所，baostock 不支持
        "430047": None,
        "SIM0001": None,     # 模拟数据代码，不该走到这里
    }
    for raw, expect in cases.items():
        got = to_bs_code(raw)
        assert got == expect, f"{raw} 期望 {expect}，实际 {got}"
    return f"{len(cases)} 个代码转换全部正确（北交所返回 None 被跳过）"


def test_row_parsing():
    """字符串行 → 数值 + 布尔，且日期成为索引。"""
    fields = ["date", "open", "high", "low", "close", "volume", "amount",
              "tradestatus", "isST", "pctChg"]
    rows = [
        ["2024-01-02", "11952.3783", "11974.6004", "11695.2007", "11743.3587",
         "3215644", "5440082548.08", "1", "0", "-2.3749"],
        ["2024-01-03", "11750.0000", "11800.0000", "11600.0000", "11700.0000",
         "2500000", "4000000000.00", "0", "1", "-0.3691"],   # 停牌 + ST
        ["2024-01-04", "", "", "", "", "", "", "0", "0", ""],  # 停牌且无价格（空字符串）
    ]
    df = _rows_to_frame(rows, fields)

    assert isinstance(df.index, pd.DatetimeIndex), "日期没有变成索引"
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
                              pd.Timestamp("2024-01-04")]

    # 数值列必须是数字类型，空字符串要变成 NaN
    assert df["close"].dtype.kind == "f"
    assert abs(df["close"].iloc[0] - 11743.3587) < 1e-3
    assert pd.isna(df["close"].iloc[2]), "空字符串应变成 NaN"

    # 布尔化：tradestatus 1=可交易，0=停牌；isST 1=是ST
    assert df["tradestatus"].dtype == bool
    assert bool(df["tradestatus"].iloc[0]) is True
    assert bool(df["tradestatus"].iloc[1]) is False
    assert bool(df["isST"].iloc[1]) is True
    return "数值解析、空字符串→NaN、tradestatus/isST 布尔化 全部正确"


def test_empty_rows():
    """空数据应返回 None，不能崩。"""
    assert _rows_to_frame([], ["date", "close"]) is None
    return "空数据返回 None"


def test_mask_uses_tradestatus():
    """build_mask 应该优先使用数据源给的停牌标记。"""
    from data.universe import apply_st_filter, build_mask

    dates = pd.bdate_range("2024-01-01", periods=4)
    px = pd.DataFrame(10.0, index=dates, columns=["A"])
    vol = pd.DataFrame(1000.0, index=dates, columns=["A"])   # 成交量正常
    panel = {"open": px.copy(), "close": px.copy(), "high": px.copy(),
             "low": px.copy(), "volume": vol,
             "tradestatus": pd.DataFrame(True, index=dates, columns=["A"]),
             "isST": pd.DataFrame(False, index=dates, columns=["A"])}

    # 第 2 天标记为停牌，但成交量仍是 1000（真实数据里可能这样）
    panel["tradestatus"].iloc[1, 0] = False
    cfg = {"universe": {"min_list_days": 0, "exclude_st": True, "st_codes": []}}

    mask = build_mask(panel, cfg)
    assert bool(mask.iloc[0, 0]) is True
    assert bool(mask.iloc[1, 0]) is False, "tradestatus=0 应判为不可交易"

    # ST 标记：第 3 天变成 ST，应被剔除
    panel["isST"].iloc[2, 0] = True
    mask2 = apply_st_filter(mask, cfg, is_st=panel["isST"])
    assert bool(mask2.iloc[2, 0]) is False, "isST=1 当天应被剔除"
    assert bool(mask2.iloc[3, 0]) is True, "摘帽后应恢复可交易"
    return "停牌标记优先于成交量；ST 逐日标记正确生效"


def main():
    tests = [
        ("1. 代码转换", test_code_conversion),
        ("2. 行解析", test_row_parsing),
        ("3. 空数据", test_empty_rows),
        ("4. mask 使用 tradestatus/isST", test_mask_uses_tradestatus),
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
