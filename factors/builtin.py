"""
内置因子。

每个函数签名统一为 fn(panel, **kwargs) -> DataFrame(date × code)
返回值都是「越大越好」。
"""

import numpy as np

from factors.registry import register


@register("mom20", desc="20 日动量：过去 20 个交易日涨幅")
def mom20(panel, **kw):
    return panel["close"].pct_change(20)


@register("mom60", desc="60 日动量：中期趋势")
def mom60(panel, **kw):
    return panel["close"].pct_change(60)


@register("rev5", desc="5 日反转：短期跌多了会反弹")
def rev5(panel, **kw):
    return -panel["close"].pct_change(5)


@register("vol20", desc="20 日低波动：波动小的股票长期表现更稳")
def vol20(panel, **kw):
    # 波动率越低越好，所以取负号
    return -panel["close"].pct_change().rolling(20).std()


@register("vol60", desc="60 日低波动")
def vol60(panel, **kw):
    return -panel["close"].pct_change().rolling(60).std()


@register("illiq20", desc="20 日流动性（成交额越小越好，流动性溢价）")
def illiq20(panel, **kw):
    amount = (panel["close"] * panel["volume"]).rolling(20).mean()
    # 成交额越小，打分越高；取对数是为了压缩极端值
    return -np.log(amount.replace(0, np.nan))


@register("ma_dev20", desc="价格相对 20 日均线的偏离度")
def ma_dev20(panel, **kw):
    close = panel["close"]
    return close / close.rolling(20).mean() - 1.0
