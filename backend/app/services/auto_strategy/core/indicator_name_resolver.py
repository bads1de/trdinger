"""
IndicatorNameResolver

- 文字列オペランドを Strategy インスタンスの属性に解決し、値を取得する責務を一元化
- ConditionEvaluator から切り離してテーブル化/関数化しやすくする
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


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
                    return True, cls._last_finite(getattr(strategy_instance, operand.lower()))

            # Strategy 直下の属性（IndicatorCalculator登録済み）
            if hasattr(strategy_instance, operand):
                return True, cls._last_finite(getattr(strategy_instance, operand))

            # 複数出力・期間付き表記
            if "_" in operand:
                base_indicator = operand.split("_")[0]

                # STOCH 系: %Kを使用
                if base_indicator == "STOCH" and hasattr(strategy_instance, "STOCH_0"):
                    return True, cls._last_finite(getattr(strategy_instance, "STOCH_0"))
                if base_indicator == "STOCHF" and hasattr(strategy_instance, "STOCHF_0"):
                    return True, cls._last_finite(getattr(strategy_instance, "STOCHF_0"))
                if base_indicator == "STOCHRSI" and hasattr(strategy_instance, "STOCHRSI_0"):
                    return True, cls._last_finite(getattr(strategy_instance, "STOCHRSI_0"))

                # CCI, RSI, SMA, EMA: ベース名にマップ
                if base_indicator in ["CCI", "RSI", "SMA", "EMA"] and hasattr(
                    strategy_instance, base_indicator
                ):
                    return True, cls._last_finite(getattr(strategy_instance, base_indicator))

                # MACD/MACDEXT: メインライン
                if base_indicator == "MACD" and hasattr(strategy_instance, "MACD_0"):
                    return True, cls._last_finite(getattr(strategy_instance, "MACD_0"))
                if base_indicator == "MACDEXT" and hasattr(strategy_instance, "MACDEXT_0"):
                    return True, cls._last_finite(getattr(strategy_instance, "MACDEXT_0"))

                # BB: Upper/Middle/Lower -> 0/1/2
                if operand.startswith("BB_"):
                    base = operand.split("_")[1]
                    if base == "Upper" and hasattr(strategy_instance, "BB_0"):
                        return True, cls._last_finite(getattr(strategy_instance, "BB_0"))
                    if base == "Middle" and hasattr(strategy_instance, "BB_1"):
                        return True, cls._last_finite(getattr(strategy_instance, "BB_1"))
                    if base == "Lower" and hasattr(strategy_instance, "BB_2"):
                        return True, cls._last_finite(getattr(strategy_instance, "BB_2"))

            # 単純名/コンポーネント名のマッピング
            if operand in ("MACD", "MACD_0") and hasattr(strategy_instance, "MACD_0"):
                return True, cls._last_finite(getattr(strategy_instance, "MACD_0"))
            if operand == "MACD_1" and hasattr(strategy_instance, "MACD_1"):
                return True, cls._last_finite(getattr(strategy_instance, "MACD_1"))
            if operand in ("MACDEXT", "MACDEXT_0") and hasattr(
                strategy_instance, "MACDEXT_0"
            ):
                return True, cls._last_finite(getattr(strategy_instance, "MACDEXT_0"))
            if operand == "MACDEXT_1" and hasattr(strategy_instance, "MACDEXT_1"):
                return True, cls._last_finite(getattr(strategy_instance, "MACDEXT_1"))
            if operand == "BB" and hasattr(strategy_instance, "BB_1"):
                return True, cls._last_finite(getattr(strategy_instance, "BB_1"))
            # KELTNER: Middle をデフォルト
            if operand == "KELTNER" and hasattr(strategy_instance, "KELTNER_1"):
                return True, cls._last_finite(getattr(strategy_instance, "KELTNER_1"))

            if operand in ("STOCH", "STOCH_0") and hasattr(strategy_instance, "STOCH_0"):
                return True, cls._last_finite(getattr(strategy_instance, "STOCH_0"))
            if operand == "STOCH_1" and hasattr(strategy_instance, "STOCH_1"):
                return True, cls._last_finite(getattr(strategy_instance, "STOCH_1"))
            if operand in ("STOCHF", "STOCHF_0") and hasattr(strategy_instance, "STOCHF_0"):
                return True, cls._last_finite(getattr(strategy_instance, "STOCHF_0"))
            if operand == "STOCHF_1" and hasattr(strategy_instance, "STOCHF_1"):
                return True, cls._last_finite(getattr(strategy_instance, "STOCHF_1"))
            if operand in ("STOCHRSI", "STOCHRSI_0") and hasattr(
                strategy_instance, "STOCHRSI_0"
            ):
                return True, cls._last_finite(getattr(strategy_instance, "STOCHRSI_0"))
            if operand == "STOCHRSI_1" and hasattr(strategy_instance, "STOCHRSI_1"):
                return True, cls._last_finite(getattr(strategy_instance, "STOCHRSI_1"))

            # AROON（Up/Downの2本）: base名はUp(=0)
            if operand in ("AROON", "AROON_0") and hasattr(strategy_instance, "AROON_0"):
                return True, cls._last_finite(getattr(strategy_instance, "AROON_0"))
            if operand == "AROON_1" and hasattr(strategy_instance, "AROON_1"):
                return True, cls._last_finite(getattr(strategy_instance, "AROON_1"))

            # RVGI（メイン/シグナルの2本）: base=0
            if operand in ("RVGI", "RVGI_0") and hasattr(strategy_instance, "RVGI_0"):
                return True, cls._last_finite(getattr(strategy_instance, "RVGI_0"))
            if operand == "RVGI_1" and hasattr(strategy_instance, "RVGI_1"):
                return True, cls._last_finite(getattr(strategy_instance, "RVGI_1"))

            # その他ゼロセンター系（PPO/APO/TRIX/TSIなど）
            if operand in ("PPO", "APO", "TRIX", "TSI") and hasattr(
                strategy_instance, operand
            ):
                return True, cls._last_finite(getattr(strategy_instance, operand))

            # 未解決
            return False, 0.0
        except Exception:
            return False, 0.0

