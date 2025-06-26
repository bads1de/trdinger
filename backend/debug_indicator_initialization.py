#!/usr/bin/env python3
"""
指標初期化の詳細デバッグ
実際のオートストラテジー実行時に指標初期化が失敗する原因を特定
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(__file__))

# ログレベルを詳細に設定
logging.basicConfig(level=logging.DEBUG)

def create_realistic_test_data():
    """リアルなテスト用OHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    
    np.random.seed(42)
    price = 45000
    prices = []
    volumes = []
    
    for _ in range(200):
        change = np.random.normal(0, 0.015)  # 1.5%の標準偏差
        price *= (1 + change)
        price = max(price, 1000)  # 最低価格を設定
        prices.append(price)
        volumes.append(np.random.uniform(500, 2000))
    
    # バックテストライブラリのData形式をシミュレート
    class MockData:
        def __init__(self):
            self.Close = np.array(prices)
            self.High = np.array([p * (1 + np.random.uniform(0, 0.02)) for p in prices])
            self.Low = np.array([p * (1 - np.random.uniform(0, 0.02)) for p in prices])
            self.Open = np.array(prices)
            self.Volume = np.array(volumes)
    
    return MockData()

def debug_single_indicator_initialization(indicator_type, parameters):
    """単一指標の初期化を詳細にデバッグ"""
    print(f"\n🔍 {indicator_type} 指標初期化デバッグ")
    print("=" * 60)
    
    try:
        from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        
        initializer = IndicatorInitializer()
        test_data = create_realistic_test_data()
        
        # 指標遺伝子を作成
        indicator_gene = IndicatorGene(
            type=indicator_type,
            parameters=parameters,
            enabled=True
        )
        
        print(f"指標タイプ: {indicator_type}")
        print(f"パラメータ: {parameters}")
        
        # モック戦略インスタンス
        mock_strategy = Mock()
        mock_strategy.indicators = {}
        mock_strategy.I = Mock(return_value=Mock())
        
        print("\n初期化プロセス:")
        
        # Step 1: 代替指標チェック
        fallback_type = initializer._get_fallback_indicator(indicator_type)
        print(f"  1. 代替指標チェック: {indicator_type} -> {fallback_type}")
        
        if not fallback_type:
            print("  ❌ 代替指標が見つかりません")
            return False
        
        # Step 2: データ変換
        print("  2. データ変換:")
        try:
            close_data = initializer._convert_to_series(test_data.Close)
            high_data = initializer._convert_to_series(test_data.High)
            low_data = initializer._convert_to_series(test_data.Low)
            volume_data = initializer._convert_to_series(test_data.Volume)
            open_data = initializer._convert_to_series(test_data.Open)
            
            print(f"    Close: {len(close_data)} データポイント")
            print(f"    High: {len(high_data)} データポイント")
            print(f"    Low: {len(low_data)} データポイント")
            print(f"    Volume: {len(volume_data)} データポイント")
            print(f"    Open: {len(open_data)} データポイント")
        except Exception as e:
            print(f"    ❌ データ変換エラー: {e}")
            return False
        
        # Step 3: 指標計算
        print("  3. 指標計算:")
        try:
            result, indicator_name = initializer.indicator_calculator.calculate_indicator(
                fallback_type,
                parameters,
                close_data,
                high_data,
                low_data,
                volume_data,
                open_data
            )
            
            if result is not None:
                print(f"    ✅ 計算成功")
                print(f"    結果名: {indicator_name}")
                print(f"    結果タイプ: {type(result)}")
                
                if hasattr(result, 'columns'):
                    print(f"    カラム: {list(result.columns)}")
                elif isinstance(result, dict):
                    print(f"    キー: {list(result.keys())}")
                
                if hasattr(result, '__len__'):
                    print(f"    データ数: {len(result)}")
                    
                # 値の範囲を確認
                if hasattr(result, 'values'):
                    values = result.values
                    if len(values) > 0:
                        print(f"    値の範囲: {np.nanmin(values):.4f} - {np.nanmax(values):.4f}")
                elif isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, 'values') and len(value.values) > 0:
                            print(f"    {key}の範囲: {np.nanmin(value.values):.4f} - {np.nanmax(value.values):.4f}")
                
            else:
                print(f"    ❌ 計算失敗 (結果がNone)")
                return False
                
        except Exception as e:
            print(f"    ❌ 計算エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 4: 戦略への登録
        print("  4. 戦略への登録:")
        try:
            # JSON形式の指標名
            json_indicator_name = indicator_type
            
            indicator_values = (
                result.values if hasattr(result, "values") else result
            )
            
            # JSON形式で指標を登録
            mock_strategy.indicators[json_indicator_name] = mock_strategy.I(
                lambda: indicator_values, name=json_indicator_name
            )
            
            # 後方互換性のためレガシー形式でも登録
            legacy_indicator_name = initializer._get_legacy_indicator_name(
                indicator_type, parameters
            )
            if legacy_indicator_name != json_indicator_name:
                mock_strategy.indicators[legacy_indicator_name] = (
                    mock_strategy.indicators[json_indicator_name]
                )
            
            print(f"    ✅ 登録成功")
            print(f"    JSON形式: {json_indicator_name}")
            print(f"    レガシー形式: {legacy_indicator_name}")
            print(f"    登録された指標: {list(mock_strategy.indicators.keys())}")
            
        except Exception as e:
            print(f"    ❌ 登録エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: 完全な初期化テスト
        print("  5. 完全な初期化テスト:")
        try:
            result = initializer.initialize_indicator(
                indicator_gene, test_data, mock_strategy
            )
            
            if result:
                print(f"    ✅ 完全初期化成功: {result}")
                return True
            else:
                print(f"    ❌ 完全初期化失敗")
                return False
                
        except Exception as e:
            print(f"    ❌ 完全初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_multiple_indicators():
    """複数指標の初期化を一括デバッグ"""
    print("🧪 複数指標初期化デバッグ")
    print("=" * 80)
    
    # テスト対象の指標
    test_indicators = [
        ("SMA", {"period": 20}),
        ("EMA", {"period": 20}),
        ("RSI", {"period": 14}),
        ("STOCH", {"period": 14}),
        ("CCI", {"period": 14}),
        ("ADX", {"period": 14}),
        ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("ATR", {"period": 14}),
        ("BB", {"period": 20, "std_dev": 2}),
        ("OBV", {}),
    ]
    
    results = {}
    
    for indicator_type, parameters in test_indicators:
        success = debug_single_indicator_initialization(indicator_type, parameters)
        results[indicator_type] = success
    
    print("\n" + "=" * 80)
    print("📊 初期化結果サマリー")
    print("=" * 80)
    
    successful = []
    failed = []
    
    for indicator_type, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {indicator_type}")
        
        if success:
            successful.append(indicator_type)
        else:
            failed.append(indicator_type)
    
    print(f"\n成功: {len(successful)}個")
    print(f"失敗: {len(failed)}個")
    
    if failed:
        print(f"\n失敗した指標: {failed}")
        print("これらの指標が実際のオートストラテジー実行時に利用できない可能性があります")
    
    return len(failed) == 0

def main():
    """メインデバッグ実行"""
    print("🎯 指標初期化詳細デバッグ")
    print("=" * 100)
    print("目的: 実際のオートストラテジー実行時の指標初期化失敗原因を特定")
    print("=" * 100)
    
    success = debug_multiple_indicators()
    
    print("\n" + "=" * 100)
    if success:
        print("🎉 全ての指標が正常に初期化されました！")
        print("問題は他の箇所にある可能性があります")
    else:
        print("⚠️ 一部の指標で初期化に失敗しました")
        print("これが実際のSTOCHエラーの原因である可能性があります")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
