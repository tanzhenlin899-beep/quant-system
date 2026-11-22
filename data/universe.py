"""
股票池过滤：哪些股票在某个交易日"可以交易"。

这一步看起来琐碎，但它直接决定回测是否可信：
  - 停牌的股票不该被买入
  - 一字涨停的股票，现实中你买不进去
  - 一字跌停的股票，现实中你卖不出来
  - 上市不满 N 天的次新股，波动异常，通常剔除

涨跌停幅度按板块区分（这条以前偷懒统一按 10% 算，是错的）：
  主板     10%
  ST / *ST  5%（仅主板；创业板/科创板的 ST 股仍是 20%）
  创业板/科创板  20%
  北交所   30%
"""

import pandas as pd


def build_mask(panel, cfg):
    """返回 date × code 的布尔矩阵，True = 当天可以交易。"""
    close = panel["close"]
    open_ = panel["open"]
    volume = panel["volume"]

    # 1) 必须有价格、必须有成交量（成交量 0 = 停牌）
    base = close.notna() & open_.notna() & (volume.fillna(0) > 0)

    # 1b) 如果数据源直接给了停牌标记（baostock 的 tradestatus），优先用它——
    #     比"成交量=0"更可靠：有些停牌日成交量字段可能不是 0。
    tradestatus = panel.get("tradestatus")
    if isinstance(tradestatus, pd.DataFrame):
        base = base & tradestatus.reindex_like(base).fillna(False)

    # 2) 一字板屏蔽：开盘价贴着涨跌停，说明排队也成交不了
    prev_close = close.shift(1)
    limit_pct = _limit_pct(close.columns, cfg.get("universe", {}).get("st_codes") or [])
    up_price = prev_close.mul(1 + limit_pct, axis=1)
    down_price = prev_close.mul(1 - limit_pct, axis=1)
    tol = 0.005  # 允许 0.5% 的误差，避开浮点计算和最小价位跳动的影响
    at_up = (open_ >= up_price * (1 - tol)) & (open_ <= up_price * (1 + tol))
    at_down = (open_ >= down_price * (1 - tol)) & (open_ <= down_price * (1 + tol))

    # 3) 上市不足 N 个交易日的次新股剔除
    listed_days = close.notna().cumsum()
    old_enough = listed_days >= int(cfg.get("universe", {}).get("min_list_days", 0))

    return base & ~at_up & ~at_down & old_enough


def _limit_pct(columns, st_codes=()):
    """按板块/是否 ST 给出涨跌停幅度。"""
    st = {str(c) for c in st_codes}
    out = {}
    for c in columns:
        s = str(c)
        if s.startswith(("300", "301", "688", "689")):
            out[c] = 0.20          # 创业板 / 科创板（含其 ST 股）
        elif s.startswith(("8", "4", "920")):
            out[c] = 0.30          # 北交所
        elif s in st:
            out[c] = 0.05          # 主板 ST
        else:
            out[c] = 0.10          # 主板
    return pd.Series(out)


def apply_st_filter(mask, cfg, st_codes=None, is_st=None):
    """剔除 ST/*ST。

    两种方式，优先用第一种：
      1. 数据源直接给了逐日 ST 标记（baostock 的 isST 面板）—— 最准确，
         因为一只股票可能中途被 ST、后来又摘帽；
      2. config 里手写 st_codes 名单 —— 只能覆盖"当前"的 ST 股。

    两者都没有时，这个过滤不会生效（config 里的 exclude_st 等于空转）。
    """
    if not cfg.get("universe", {}).get("exclude_st"):
        return mask

    if isinstance(is_st, pd.DataFrame) and not is_st.empty:
        st_flag = is_st.reindex(index=mask.index, columns=mask.columns).fillna(False).astype(bool)
        return mask & ~st_flag

    st = set(cfg.get("universe", {}).get("st_codes") or []) | set(st_codes or [])
    if not st:
        return mask
    drop = [c for c in mask.columns if c in st]
    if not drop:
        return mask
    mask = mask.copy()
    mask[drop] = False
    return mask
