"""
第三方引擎适配器。

思路（很重要）：
  不要去"替换"我们自己的回测引擎，而是把同一套信号（weights）喂给第三方框架，
  再比较两边的净值曲线是否一致。
  这叫「交叉验证」——如果两个独立实现的引擎跑出接近的结果，
  说明我们自己的引擎大概率没有算错。
  这比单纯"用了 backtrader"更有说服力，也是答辩时最好讲的一个点。

约定：
  输入完全复用主系统的 panel / weights / mask，
  所以适配器不需要知道策略是怎么算出来的，只管按权重下单。
"""

from adapters.backtrader_engine import run_backtest_backtrader

__all__ = ["run_backtest_backtrader"]
