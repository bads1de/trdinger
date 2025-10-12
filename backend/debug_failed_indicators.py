"""
失敗した7つのインジケーターの詳細調査
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.services.indicators import TechnicalIndicatorService


class DebugFailedIndicators:
    """失敗した7つのインジケーターのデバッグ"""

    def __init__(self):
        self.service = TechnicalIndicatorService()
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """テスト用サンプルデータの作成"""
        periods = 200
        index = pd.date_range("2022-01-01", periods=periods, freq="H")

        # テスト用のOHLCVデータ
        base = np.linspace(100, 200, periods)
        noise = np.sin(np.linspace(0, 8 * np.pi, periods)) * 5
        close = base + noise

        data = pd.DataFrame({
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.linspace(1000, 2000, periods),
        }, index=index)

        return data

    def debug_single_indicator(self, indicator_name, params=None):
        """個別インジケーターのデバッグ"""
        if params is None:
            params = {}

        print(f"\n=== {indicator_name} のデバッグ ===")
        print(f"パラメータ: {params}")

        try:
            # 指標の計算を試行
            result = self.service.calculate_indicator(
                self.sample_data,
                indicator_name,
                params
            )
            print(f"Success: result type={type(result)}, shape={getattr(result, 'shape', 'N/A')}")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def debug_all_failed_indicators(self):
        """7つの失敗したインジケーターを個別にデバッグ"""
        failed_indicators = [
            ("LINREGSLOPE", {"length": 14, "scalar": 1.0}),
            ("STOCH", {"k_length": 14, "d_length": 3, "smooth_k": 3}),
            ("AO", {"fast": 5, "slow": 34}),
            ("AROON", {"length": 14}),
            ("CHOP", {"length": 14}),
            ("BOP", {}),
            ("AD", {}),
        ]

        results = {}
        for indicator_name, params in failed_indicators:
            results[indicator_name] = self.debug_single_indicator(indicator_name, params)

        return results


if __name__ == "__main__":
    debugger = DebugFailedIndicators()
    results = debugger.debug_all_failed_indicators()

    print("\n=== デバッグ結果まとめ ===")
    for indicator, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{indicator}: {status}")