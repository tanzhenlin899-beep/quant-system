"""策略基类：约定所有策略都必须输出同样的东西，方便替换和对比。"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """所有策略的父类。

    约定（非常重要）：
      generate_weights 返回的 DataFrame，index 是「信号产生日」。
      也就是说，用 t 日收盘数据算出来的权重，代表 t 日收盘后才决定、
      t+1 日开盘才执行。回测引擎会负责把它 shift 一天。
      搞错这个约定 = 前视偏差 = 夏普比率虚高。
    """

    name = "base"

    @abstractmethod
    def generate_weights(
        self, panel: dict, score: pd.DataFrame, mask: pd.DataFrame
    ) -> pd.DataFrame:
        """返回 date × code 的目标权重矩阵。"""
        raise NotImplementedError
