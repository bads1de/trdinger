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

            # 複数値指標デフォルト解決（定数に移管）
            from ..config.constants import MULTI_OUTPUT_DEFAULT_MAPPING

            if operand in MULTI_OUTPUT_DEFAULT_MAPPING:
                mapped_name = MULTI_OUTPUT_DEFAULT_MAPPING[operand]
                if hasattr(strategy_instance, mapped_name):
                    return True, cls._last_finite(
                        getattr(strategy_instance, mapped_name)
                    )

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
        動的指標名解決（簡素化版）

        戦略インスタンスの属性を直接検索して指標名を解決します。

        Args:
            operand: 解決対象の名前（例: "MACD_0", "BB_1", "RSI"）
            strategy_instance: Strategy インスタンス

        Returns:
            解決された属性名、または None
        """
        try:
            # 1. 完全一致を最初に試行
            if hasattr(strategy_instance, operand):
                return operand

            # 2. 戦略インスタンスの全属性を取得（より詳細）
            all_attrs = dir(strategy_instance)
            available_attrs = []
            for attr in all_attrs:
                if not attr.startswith("_"):
                    try:
                        attr_value = getattr(strategy_instance, attr, None)
                        # callableでない属性のみ追加
                        if not callable(attr_value):
                            available_attrs.append(attr)
                    except Exception:
                        # 属性取得でエラーが発生した場合はスキップ
                        continue

            # 3. 大文字小文字を無視した一致を試行
            operand_lower = operand.lower()
            for attr in available_attrs:
                if attr.lower() == operand_lower:
                    return attr

            # 4. 部分一致を試行（指標名の基本部分）
            if "_" in operand:
                base_name = operand.split("_")[0]
                # 基本名の完全一致
                if hasattr(strategy_instance, base_name):
                    return base_name

                # 基本名の大文字小文字を無視した一致
                base_lower = base_name.lower()
                for attr in available_attrs:
                    if attr.lower() == base_lower:
                        return attr

            # 5. 指標名を含む属性を検索
            for attr in available_attrs:
                # 指標名が属性名に含まれているかチェック
                if operand.lower() in attr.lower() or attr.lower() in operand.lower():
                    return attr

            # 6. 特別なケース: BBands の Upper/Middle/Lower マッピング
            if operand.startswith("BB_") or operand.startswith("BBANDS_"):
                bb_attrs = [
                    attr
                    for attr in available_attrs
                    if attr.startswith(("BB", "BBANDS"))
                ]
                if bb_attrs:
                    # 最初に見つかったBB関連属性を返す
                    return bb_attrs[0]

            # 7. デバッグ情報をログ出力（詳細版）
            logger.warning(
                f"指標名解決失敗: '{operand}', 利用可能属性: {available_attrs}"
            )

            # 基本的な移動平均指標の存在を確認（定数に移管）
            from ..config.constants import BASIC_MA_INDICATORS

            for ma in BASIC_MA_INDICATORS:
                if hasattr(strategy_instance, ma):
                    logger.warning(f"基本指標確認: {ma} は存在します")
                else:
                    logger.warning(f"基本指標確認: {ma} は存在しません")

            return None

        except Exception as e:
            logger.warning(f"動的指標名解決エラー '{operand}': {e}")
            return None
