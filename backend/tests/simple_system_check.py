#!/usr/bin/env python3
"""
シンプルなシステム動作確認テスト
"""

import pandas as pd
import numpy as np

def main():
    print("=== System Health Check ===")

    try:
        # 1. サービス初期化テスト
        print("1. Testing service initialization...")
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()
        print("   SUCCESS: Service initialized")

        # 2. サポートインジケータ確認
        print("2. Checking supported indicators...")
        supported = service.get_supported_indicators()
        print(f"   Found {len(supported)} supported indicators")

        # 3. STCテスト
        print("3. Testing STC indicator...")
        np.random.seed(42)
        close_prices = [100 + i*2 for i in range(50)]
        df = pd.DataFrame({'close': close_prices})

        result = service.calculate_indicator(df, 'STC', {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5})

        if result is not None:
            print("   SUCCESS: STC calculation completed")
            print(f"   Result shape: {result.shape}")
            valid_values = np.sum(~np.isnan(result))
            print(f"   Valid values: {valid_values}")
        else:
            print("   FAILED: STC calculation returned None")
            return False

        # 4. 設定レジストリ確認
        print("4. Checking indicator registry...")
        from app.services.indicators.config import indicator_registry
        configs = list(indicator_registry._configs.keys())
        print(f"   Found {len(configs)} registered configurations")

        # 5. 主要インジケータ存在確認
        print("5. Checking key indicators...")
        key_indicators = ['STC', 'RSI', 'SMA', 'EMA', 'MACD']
        missing_indicators = []

        for indicator in key_indicators:
            if indicator in supported:
                print(f"   FOUND: {indicator}")
            else:
                print(f"   MISSING: {indicator}")
                missing_indicators.append(indicator)

        if missing_indicators:
            print(f"   WARNING: {len(missing_indicators)} key indicators missing")
        else:
            print("   SUCCESS: All key indicators found")

        print("\n=== System Check Complete ===")
        print("Overall status: GOOD")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nFinal result: {'PASS' if success else 'FAIL'}")