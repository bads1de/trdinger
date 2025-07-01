#!/usr/bin/env python3
"""
指標レジストリの状態を詳細にデバッグするスクリプト

indicator_registryの初期化状況とRSI指標の登録状況を調査します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# ログレベルを詳細に設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_indicator_registry():
    """指標レジストリの詳細デバッグ"""

    print("🔍 指標レジストリ詳細デバッグ開始")
    print("=" * 60)

    try:
        # 1. 指標レジストリのインポート
        print("📦 指標レジストリインポート中...")
        from app.core.services.indicators.config.indicator_config import (
            indicator_registry,
        )

        print(f"  ✅ indicator_registry インポート成功: {type(indicator_registry)}")

        # 2. レジストリの基本状態確認
        print("\n📊 レジストリ基本状態:")
        indicators = indicator_registry.list_indicators()
        print(f"  登録済み指標数: {len(indicators)}")
        print(f"  登録済み指標: {indicators}")

        # 3. RSI指標の詳細確認
        print("\n🔍 RSI指標詳細確認:")
        rsi_supported = indicator_registry.is_indicator_supported("RSI")
        print(f"  RSI対応状況: {rsi_supported}")

        if rsi_supported:
            rsi_config = indicator_registry.get_indicator_config("RSI")
            print(f"  RSI設定: {rsi_config}")
            if rsi_config:
                print(f"    指標名: {rsi_config.indicator_name}")
                print(f"    パラメータ: {list(rsi_config.parameters.keys())}")
                print(f"    アダプター関数: {rsi_config.adapter_function}")
                print(f"    必要データ: {rsi_config.required_data}")
        else:
            print("  ❌ RSI指標が見つかりません")

        # 4. 指標解決テスト
        print("\n🔧 指標解決テスト:")
        resolved_rsi = indicator_registry.resolve_indicator_type("RSI")
        print(f"  RSI解決結果: {resolved_rsi}")

        # 5. 初期化関数の確認
        print("\n🏗️ 初期化関数確認:")
        from app.core.services.indicators.config.indicator_definitions import (
            initialize_all_indicators,
        )

        print(f"  initialize_all_indicators: {initialize_all_indicators}")

        # 6. 手動初期化テスト
        print("\n🔄 手動初期化テスト:")
        print("  → initialize_all_indicators() 実行中...")
        initialize_all_indicators()
        print("  ✅ 手動初期化完了")

        # 7. 初期化後の状態確認
        print("\n📊 初期化後の状態:")
        indicators_after = indicator_registry.list_indicators()
        print(f"  登録済み指標数: {len(indicators_after)}")
        print(f"  登録済み指標: {indicators_after}")

        rsi_supported_after = indicator_registry.is_indicator_supported("RSI")
        print(f"  RSI対応状況（初期化後）: {rsi_supported_after}")

        # 8. IndicatorInitializerのテスト
        print("\n🔧 IndicatorInitializer テスト:")
        from app.core.services.auto_strategy.factories.indicator_initializer import (
            IndicatorInitializer,
        )
        from app.core.services.indicators.parameter_manager import (
            IndicatorParameterManager,
        )
        from app.core.services.auto_strategy.factories.data_converter import (
            DataConverter,
        )

        initializer = IndicatorInitializer()

        supported_indicators = initializer.get_supported_indicators()
        print(f"  IndicatorInitializer対応指標数: {len(supported_indicators)}")
        print(f"  IndicatorInitializer対応指標: {supported_indicators}")

        # 9. 指標計算テスト
        print("\n🧮 指標計算テスト:")
        import pandas as pd

        # テストデータ作成（OHLCV形式）
        base_prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101] * 10
        test_data = pd.DataFrame(
            {
                "open": [p - 0.5 for p in base_prices],
                "high": [p + 1.0 for p in base_prices],
                "low": [p - 1.0 for p in base_prices],
                "close": base_prices,
                "volume": [1000] * len(base_prices),
            }
        )

        try:
            result, name = initializer.calculate_indicator_only(
                "RSI", {"period": 14}, test_data
            )
            print(f"  RSI計算結果: {type(result)}, 名前: {name}")
            if result is not None:
                print(f"    値の数: {len(result)}")
                print(
                    f"    最初の5値: {result.head().tolist() if hasattr(result, 'head') else result[:5]}"
                )
            else:
                print("  ❌ RSI計算失敗")
        except Exception as e:
            print(f"  ❌ RSI計算エラー: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_indicator_registry()
