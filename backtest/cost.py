"""交易成本模型。

A 股真实成本（2024 年后的常规水平）：
  佣金      万 2.5，买卖都收，单笔最低 5 元
  印花税    千 0.5，只有卖出收
  过户费    万 0.1，买卖都收
  滑点      千 1（下单价和成交价的差，尤其是小盘股）

不扣成本的净值曲线是假的，答辩第一个问题就会问这个。
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    commission: float = 2.5e-4
    min_commission: float = 5.0
    stamp_duty: float = 5e-4
    transfer_fee: float = 1e-5
    slippage: float = 1e-3

    @classmethod
    def from_cfg(cls, bc: dict) -> "CostModel":
        return cls(
            commission=bc.get("commission", 2.5e-4),
            min_commission=bc.get("min_commission", 5.0),
            stamp_duty=bc.get("stamp_duty", 5e-4),
            transfer_fee=bc.get("transfer_fee", 1e-5),
            slippage=bc.get("slippage", 1e-3),
        )

    def buy_cost(self, amount: float) -> float:
        return max(amount * self.commission, self.min_commission) + amount * (
            self.slippage + self.transfer_fee
        )

    def sell_cost(self, amount: float) -> float:
        return (
            max(amount * self.commission, self.min_commission)
            + amount * (self.slippage + self.transfer_fee + self.stamp_duty)
        )
