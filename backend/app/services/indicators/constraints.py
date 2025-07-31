"""
パラメータ制約システム

インディケーターのパラメータ間の制約を定義・適用するモジュール
"""

import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ParameterConstraint(ABC):
    """パラメータ制約の基底クラス"""

    @abstractmethod
    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """制約を適用してパラメータを調整"""
        pass

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> bool:
        """パラメータが制約を満たしているかを検証"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """制約の説明を取得"""
        pass


class OrderConstraint(ParameterConstraint):
    """パラメータ間の順序制約 (例: fast_period < slow_period)"""

    def __init__(self, param1: str, param2: str, operator: str = "<", margin: int = 1):
        """
        Args:
            param1: 第一パラメータ名
            param2: 第二パラメータ名
            operator: 比較演算子 ("<", "<=", ">", ">=")
            margin: 最小差分 (デフォルト: 1)
        """
        self.param1 = param1
        self.param2 = param2
        self.operator = operator
        self.margin = margin

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """順序制約を満たすようにパラメータを調整"""
        if self.param1 not in params or self.param2 not in params:
            return params

        val1 = params[self.param1]
        val2 = params[self.param2]

        if self.operator == "<":
            if val1 >= val2:
                # val1 < val2 を保証
                params[self.param2] = val1 + self.margin
        elif self.operator == "<=":
            if val1 > val2:
                params[self.param2] = val1
        elif self.operator == ">":
            if val1 <= val2:
                params[self.param1] = val2 + self.margin
        elif self.operator == ">=":
            if val1 < val2:
                params[self.param1] = val2

        return params

    def validate(self, params: Dict[str, Any]) -> bool:
        """順序制約の検証"""
        if self.param1 not in params or self.param2 not in params:
            return True

        val1 = params[self.param1]
        val2 = params[self.param2]

        if self.operator == "<":
            return val1 < val2
        elif self.operator == "<=":
            return val1 <= val2
        elif self.operator == ">":
            return val1 > val2
        elif self.operator == ">=":
            return val1 >= val2

        return False

    def get_description(self) -> str:
        return f"{self.param1} {self.operator} {self.param2}"


class RangeConstraint(ParameterConstraint):
    """パラメータの値域制約 (例: matype は 0-8)"""

    def __init__(self, param: str, min_val: int, max_val: int):
        """
        Args:
            param: パラメータ名
            min_val: 最小値
            max_val: 最大値
        """
        self.param = param
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """値域制約を適用"""
        if self.param in params:
            # 指定範囲内でランダム生成
            params[self.param] = random.randint(self.min_val, self.max_val)
        return params

    def validate(self, params: Dict[str, Any]) -> bool:
        """値域制約の検証"""
        if self.param not in params:
            return True

        val = params[self.param]
        return self.min_val <= val <= self.max_val

    def get_description(self) -> str:
        return f"{self.param} in [{self.min_val}, {self.max_val}]"


class DependencyConstraint(ParameterConstraint):
    """パラメータ間の依存関係制約"""

    def __init__(self, source_param: str, target_param: str, dependency_func):
        """
        Args:
            source_param: 依存元パラメータ
            target_param: 依存先パラメータ
            dependency_func: 依存関係を定義する関数
        """
        self.source_param = source_param
        self.target_param = target_param
        self.dependency_func = dependency_func

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """依存関係制約を適用"""
        if self.source_param in params:
            source_val = params[self.source_param]
            target_val = self.dependency_func(source_val)
            params[self.target_param] = target_val
        return params

    def validate(self, params: Dict[str, Any]) -> bool:
        """依存関係制約の検証"""
        if self.source_param not in params or self.target_param not in params:
            return True

        source_val = params[self.source_param]
        expected_val = self.dependency_func(source_val)
        return params[self.target_param] == expected_val

    def get_description(self) -> str:
        return f"{self.target_param} depends on {self.source_param}"


class ConstraintEngine:
    """制約エンジン - パラメータ制約の管理と適用"""

    def __init__(self):
        self.constraint_registry: Dict[str, List[ParameterConstraint]] = {}
        self.logger = logging.getLogger(__name__)

    def register_constraints(
        self, indicator_name: str, constraints: List[ParameterConstraint]
    ):
        """インディケーターの制約を登録"""
        self.constraint_registry[indicator_name] = constraints
        # self.logger.debug(
        #     f"Registered {len(constraints)} constraints for {indicator_name}"
        # )

    def apply_constraints(
        self, indicator_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """制約を適用してパラメータを調整"""
        constraints = self.constraint_registry.get(indicator_name, [])

        for constraint in constraints:
            try:
                params = constraint.apply(params)
            except Exception as e:
                self.logger.warning(
                    f"制約の適用に失敗しました: {constraint.get_description()} エラー内容: {e}"
                )
                pass

        return params

    def validate_constraints(self, indicator_name: str, params: Dict[str, Any]) -> bool:
        """すべての制約が満たされているかを検証"""
        constraints = self.constraint_registry.get(indicator_name, [])

        for constraint in constraints:
            try:
                if not constraint.validate(params):
                    self.logger.warning(
                        f"制約検証に失敗しました: {constraint.get_description()}"
                    )
                    return False
            except Exception as e:
                self.logger.error(f"制約検証エラー {constraint.get_description()}: {e}")
                pass

                return False

        return True

    def get_constraints(self, indicator_name: str) -> List[ParameterConstraint]:
        """インディケーターの制約一覧を取得"""
        return self.constraint_registry.get(indicator_name, [])

    def list_indicators(self) -> List[str]:
        """制約が登録されているインディケーター一覧を取得"""
        return list(self.constraint_registry.keys())


# グローバル制約エンジンインスタンス
constraint_engine = ConstraintEngine()


def setup_default_constraints():
    """デフォルト制約の設定"""

    # MACD制約
    macd_constraints: List[ParameterConstraint] = [
        OrderConstraint("fast_period", "slow_period", "<", margin=1)
    ]
    constraint_engine.register_constraints("MACD", macd_constraints)

    # Stochastic制約
    stochastic_constraints: List[ParameterConstraint] = [
        RangeConstraint("slowk_matype", 0, 8),
        RangeConstraint("slowd_matype", 0, 8),
        RangeConstraint("fastd_matype", 0, 8),
    ]
    constraint_engine.register_constraints("STOCH", stochastic_constraints)
    constraint_engine.register_constraints("STOCHF", stochastic_constraints)
    constraint_engine.register_constraints("STOCHRSI", stochastic_constraints)

    # MACDEXT制約
    macdext_constraints: List[ParameterConstraint] = [
        OrderConstraint("fast_period", "slow_period", "<", margin=1),
        RangeConstraint("fast_ma_type", 0, 8),
        RangeConstraint("slow_ma_type", 0, 8),
        RangeConstraint("signal_ma_type", 0, 8),
    ]
    constraint_engine.register_constraints("MACDEXT", macdext_constraints)


# 初期化時に制約を設定
setup_default_constraints()
