#!/usr/bin/env python3
"""
CFO/CTI実装修正テストスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np

def test_cfo_cti_fix():
    """CFO/CTI実装修正テスト"""

    try:
        # パス設定
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        # テストデータ生成 (200期間)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        np.random.seed(42)

        # 価格データ作成
        close_prices = 50000 + np.cumsum(np.random.randn(200)) * 100

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.randn(200) * 0.01),
            'high': close_prices * (1 + np.random.randn(200) * 0.02),
            'low': close_prices * (1 - np.random.randn(200) * 0.02),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, 200)
        })

        print("=== CFO/CTI実装修正テスト ===")

        # 設定確認
        from app.services.indicators.config import indicator_registry

        cfo_config = indicator_registry.get_indicator_config('CFO')
        cti_config = indicator_registry.get_indicator_config('CTI')

        print(f"CFO設定存在: {cfo_config is not None}")
        print(f"CTI設定存在: {cti_config is not None}")

        if cfo_config:
            print(f"CFOアダプタ: {cfo_config.adapter_function is not None}")
            print(f"CFOパラメータ: {cfo_config.get_parameter_ranges()}")

        if cti_config:
            print(f"CTIアダプタ: {cti_config.adapter_function is not None}")
            print(f"CTIパラメータ: {cti_config.get_parameter_ranges()}")

        # 実装テスト
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()

        print("\n--- CFO計算テスト ---")
        try:
            cfo_result = service.calculate_indicator(df.copy(), 'CFO', {})
            if cfo_result is not None and not (hasattr(cfo_result, 'empty') and cfo_result.empty):
                print(f"SUCCESS CFO calculation: shape={getattr(cfo_result, 'shape', len(cfo_result))}")
                print(f"CFO sample values: {str(cfo_result)[:100]}...")
            else:
                print("FAILED CFO calculation: None or Empty result")
        except Exception as e:
            print(f"FAILED CFO calculation: {str(e)[:100]}")

        print("\n--- CTI Calculation Test ---")
        try:
            cti_result = service.calculate_indicator(df.copy(), 'CTI', {})
            if cti_result is not None and not (hasattr(cti_result, 'empty') and cti_result.empty):
                print(f"SUCCESS CTI calculation: shape={getattr(cti_result, 'shape', len(cti_result))}")
                print(f"CTI sample values: {str(cti_result)[:100]}...")
            else:
                print("FAILED CTI calculation: None or Empty result")
        except Exception as e:
            print(f"FAILED CTI calculation: {str(e)[:100]}")

        # pandas-taバージョン情報
        try:
            import pandas_ta as ta
            print(f"\npandas-taバージョン: {ta.__version__}")
        except Exception as e:
            print(f"pandas-taインポートエラー: {e}")

        print("\n=== テスト完了 ===")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cfo_cti_fix()