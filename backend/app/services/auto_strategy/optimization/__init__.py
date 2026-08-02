"""
戦略パラメータ最適化モジュール

GAで発見された戦略構造に対して、Optunaを使用してパラメータを最適化します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy_parameter_space import StrategyParameterSpace
    from .strategy_parameter_tuner import StrategyParameterTuner

_ATTRIBUTE_EXPORTS = {
    "StrategyParameterSpace": ".strategy_parameter_space",
    "StrategyParameterTuner": ".strategy_parameter_tuner",
}

__all__ = [
    "StrategyParameterSpace",
    "StrategyParameterTuner",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
