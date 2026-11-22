"""
因子注册表。

用装饰器把因子"登记"到一张表里，好处是：
  1. 加新因子只需要写一个函数 + 一行 @register，不用改任何其它代码；
  2. 配置文件里只写因子名字（如 "mom20"），系统自己去表里查；
  3. 答辩时可以现场加一个因子重跑，证明系统是真的。

约定：所有因子函数都返回「数值越大越好」的打分。
     需要反向的因子（比如低波动），在函数内部取负号，不要在别处打补丁。
"""

from typing import Any

FACTORS: dict[str, dict[str, Any]] = {}


def register(name, desc="", direction=1):
    """把一个函数注册成因子。"""

    def deco(fn):
        FACTORS[name] = {"fn": fn, "desc": desc, "direction": direction}
        return fn

    return deco


def get(name):
    if name not in FACTORS:
        raise KeyError(f"因子 {name!r} 没有注册，可选：{list(FACTORS)}")
    return FACTORS[name]["fn"]
