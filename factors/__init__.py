"""因子层：每个因子是一个函数，输入面板，输出 date × code 的打分矩阵。"""

from factors import builtin  # noqa: F401  触发因子注册
from factors.composite import composite_score
from factors.registry import FACTORS, register

__all__ = ["FACTORS", "register", "composite_score", "builtin"]
