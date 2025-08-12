"""
IndicatorNameResolver

- 文字列オペランドを Strategy インスタンスの属性に解決し、値を取得する責務を一元化
- ConditionEvaluator から切り離してテーブル化/関数化しやすくする
- 動的解決により、ハードコードされたマッピングを排除（リファクタリング改善）
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np

from app.services.indicators.config.indicator_config import indicator_registry

logger = logging.getLogger(__name__)


class IndicatorNameResolver:
    @staticmethod
    def _last_finite(x) -> float:
        try:
            # シーケンス/配列
            if hasattr(x, "__getitem__") and not isinstance(x, (str, bytes)):
                arr = np.asarray(x, dtype=float)
                for v in arr[::-1]:
                    if np.isfinite(v):
                        return float(v)
                return 0.0
            # スカラー
            val = float(x)
            return val if np.isfinite(val) else 0.0
        except Exception:
            try:
                return float(x)
            except Exception:
                return 0.0

    @classmethod
    def try_resolve_value(cls, operand: str, strategy_instance) -> Tuple[bool, float]:
        """
        文字列オペランドを解決して値を返す。
        戻り値: (resolved, value)
        """
        try:
            # 数値文字列
            if operand.replace(".", "").replace("-", "").isdigit():
                return True, float(operand)

            # 価格データ
            if operand.lower() in ["close", "high", "low", "open"]:
                if hasattr(strategy_instance, "data"):
                    price_data = getattr(strategy_instance.data, operand.capitalize())
                    return True, cls._last_finite(price_data)
                if hasattr(strategy_instance, operand.lower()):
                    return True, cls._last_finite(
                        getattr(strategy_instance, operand.lower())
                    )

            # Strategy 直下の属性（IndicatorCalculator登録済み）
            if hasattr(strategy_instance, operand):
                return True, cls._last_finite(getattr(strategy_instance, operand))

            # 動的指標名解決（リファクタリング改善）
            resolved_name = cls._resolve_indicator_name_dynamically(
                operand, strategy_instance
            )
            if resolved_name:
                return True, cls._last_finite(getattr(strategy_instance, resolved_name))

            # 未解決
            return False, 0.0
        except Exception as e:
            logger.warning(f"Error resolving operand '{operand}': {e}")
            return False, 0.0

    @classmethod
    def _resolve_indicator_name_dynamically(
        cls, operand: str, strategy_instance
    ) -> str:
        """
        動的指標名解決（リファクタリング改善）

        指標レジストリを使用してハードコードされたマッピングを排除し、
        動的に指標名を解決します。

        Args:
            operand: 解決対象の名前（例: "MACD_0", "BB_1", "RSI"）
            strategy_instance: Strategy インスタンス

        Returns:
            解決された属性名、または None
        """
        try:
            # 指標レジストリから解決を試行
            resolved_indicator = indicator_registry.resolve_indicator_name(operand)
            if resolved_indicator:
                # 出力インデックスを取得
                output_index = indicator_registry.get_output_index(operand)

                if output_index is not None:
                    # インデックス付き名前（例: "MACD_0"）
                    attr_name = f"{resolved_indicator}_{output_index}"
                    if hasattr(strategy_instance, attr_name):
                        return attr_name
                else:
                    # 単純名またはデフォルト出力
                    # まず単純名を試行
                    if hasattr(strategy_instance, resolved_indicator):
                        return resolved_indicator

                    # デフォルト出力名を試行
                    default_output = indicator_registry.get_default_output_name(
                        resolved_indicator
                    )
                    if default_output and hasattr(strategy_instance, default_output):
                        return default_output

            # 特別なケース: BB の Upper/Middle/Lower マッピング
            if operand.startswith("BB_"):
                parts = operand.split("_")
                if len(parts) >= 2:
                    base = parts[1]
                    mapping = {"Upper": "BB_0", "Middle": "BB_1", "Lower": "BB_2"}
                    mapped_name = mapping.get(base)
                    if mapped_name and hasattr(strategy_instance, mapped_name):
                        return mapped_name

            # 複数出力指標の処理
            if "_" in operand:
                base_indicator = operand.split("_")[0]

                # 指標レジストリから基本指標を解決
                resolved_base = indicator_registry.resolve_indicator_name(
                    base_indicator
                )
                if resolved_base:
                    # デフォルト出力を使用
                    default_output = indicator_registry.get_default_output_name(
                        resolved_base
                    )
                    if default_output and hasattr(strategy_instance, default_output):
                        return default_output

                    # ベース名を直接試行
                    if hasattr(strategy_instance, resolved_base):
                        return resolved_base

            return None

        except Exception as e:
            logger.warning(f"動的指標名解決エラー '{operand}': {e}")
            return None
