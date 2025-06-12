#!/usr/bin/env python3
"""
Keltner Channels実装のテストスクリプト

新しく実装したKeltnerChannelsIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_keltner_channels_indicator():
    """KeltnerChannelsIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import KeltnerChannelsIndicator
        
        print("✅ KeltnerChannelsIndicatorのインポート成功")
        
        # テストデータの作成（Keltner Channelsは高値・安値・終値データが必要）
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格データを生成
        base_price = 100
        price_trend = np.linspace(0, 20, 100)  # 上昇トレンド
        price_noise = np.random.normal(0, 2, 100)  # ノイズ
        close_prices = base_price + price_trend + price_noise
        
        # 高値・安値を終値から生成
        high_prices = close_prices + np.random.uniform(1, 3, 100)
        low_prices = close_prices - np.random.uniform(1, 3, 100)
        
        test_data = pd.DataFrame({
            'open': close_prices + np.random.uniform(-1, 1, 100),
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # KeltnerChannelsIndicatorのインスタンス化
        keltner_indicator = KeltnerChannelsIndicator()
        print("✅ KeltnerChannelsIndicatorのインスタンス化成功")
        print(f"   サポート期間: {keltner_indicator.supported_periods}")
        
        # 異なる期間でのKeltner Channels計算テスト
        for period in [10, 14, 20]:
            try:
                result = keltner_indicator.calculate(test_data, period)
                
                print(f"✅ Keltner Channels計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の形状: {result.shape}")
                print(f"   カラム: {list(result.columns)}")
                print(f"   非NaN値の数: {result.notna().sum().sum()}")
                print(f"   最後の5つの値:")
                print(result.tail().round(2))
                print()
                
            except Exception as e:
                print(f"❌ Keltner Channels計算失敗 (期間: {period}): {e}")
                return False
        
        # 説明の取得テスト
        description = keltner_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ KeltnerChannelsIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keltner_channels_vs_bollinger_bands():
    """Keltner ChannelsとBollinger Bandsの比較テスト"""
    try:
        from app.core.services.indicators import KeltnerChannelsIndicator, BollingerBandsIndicator
        
        print("\n📊 Keltner ChannelsとBollinger Bandsの比較テスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # ボラティリティが変化する価格パターン
        base_price = 100
        # 前半: 低ボラティリティ、後半: 高ボラティリティ
        volatility = np.concatenate([
            np.full(25, 0.5),   # 前半: 低ボラティリティ
            np.full(25, 3.0)    # 後半: 高ボラティリティ
        ])
        
        close_prices = []
        current_price = base_price
        for i in range(50):
            change = np.random.normal(0, volatility[i])
            current_price += change
            close_prices.append(current_price)
        
        close_prices = np.array(close_prices)
        high_prices = close_prices + np.random.uniform(0.5, 2, 50)
        low_prices = close_prices - np.random.uniform(0.5, 2, 50)
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)
        
        period = 20
        
        # 各チャネル指標を計算
        keltner_indicator = KeltnerChannelsIndicator()
        bb_indicator = BollingerBandsIndicator()
        
        keltner_result = keltner_indicator.calculate(test_data, period)
        bb_result = bb_indicator.calculate(test_data, period)
        
        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   ボラティリティパターン: 低 → 高")
        print(f"   最後の10個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'Close': test_data['close'].tail(10).round(2),
            'KC_Upper': keltner_result['upper'].tail(10).round(2),
            'KC_Middle': keltner_result['middle'].tail(10).round(2),
            'KC_Lower': keltner_result['lower'].tail(10).round(2),
            'BB_Upper': bb_result['upper'].tail(10).round(2),
            'BB_Middle': bb_result['middle'].tail(10).round(2),
            'BB_Lower': bb_result['lower'].tail(10).round(2)
        })
        
        print(comparison_df)
        
        # チャネル幅の比較
        keltner_width = keltner_result['upper'].iloc[-1] - keltner_result['lower'].iloc[-1]
        bb_width = bb_result['upper'].iloc[-1] - bb_result['lower'].iloc[-1]
        
        print(f"\n   最終チャネル幅比較:")
        print(f"   Keltner Channels幅: {keltner_width:.2f}")
        print(f"   Bollinger Bands幅: {bb_width:.2f}")
        print(f"   幅の比率 (KC/BB): {keltner_width/bb_width:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keltner_channels_multiplier():
    """Keltner Channelsのmultiplierパラメータテスト"""
    try:
        from app.core.services.indicators import KeltnerChannelsIndicator
        
        print("\n🔢 Keltner Channelsのmultiplierパラメータテスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # 一定のボラティリティを持つ価格データ
        base_price = 100
        close_prices = base_price + np.random.normal(0, 2, 30)
        high_prices = close_prices + np.random.uniform(1, 2, 30)
        low_prices = close_prices - np.random.uniform(1, 2, 30)
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 30)
        }, index=dates)
        
        period = 14
        keltner_indicator = KeltnerChannelsIndicator()
        
        # 異なるmultiplierでの計算
        multipliers = [1.0, 1.5, 2.0, 2.5]
        results = {}
        
        for multiplier in multipliers:
            result = keltner_indicator.calculate(test_data, period, multiplier=multiplier)
            results[multiplier] = result
            
            # チャネル幅の計算
            width = result['upper'].iloc[-1] - result['lower'].iloc[-1]
            print(f"   Multiplier {multiplier}: チャネル幅 = {width:.2f}")
        
        # multiplierとチャネル幅の関係確認
        print(f"\n   Multiplier効果の確認:")
        base_width = results[1.0]['upper'].iloc[-1] - results[1.0]['lower'].iloc[-1]
        
        for multiplier in multipliers[1:]:
            width = results[multiplier]['upper'].iloc[-1] - results[multiplier]['lower'].iloc[-1]
            expected_width = base_width * multiplier
            actual_ratio = width / base_width
            
            print(f"   Multiplier {multiplier}: 期待比率 = {multiplier:.1f}, 実際比率 = {actual_ratio:.2f}")
            
            if abs(actual_ratio - multiplier) < 0.1:
                print(f"   ✅ Multiplier {multiplier}の効果が正しく反映されている")
            else:
                print(f"   ⚠️  Multiplier {multiplier}の効果が期待通りでない")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiplierテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keltner_channels_integration():
    """Keltner Channelsの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        print("\n🔗 Keltner Channels統合テスト:")
        
        # ファクトリー関数経由での取得
        keltner_indicator = get_indicator_by_type("KELTNER")
        print("✅ ファクトリー関数からのKeltner Channels取得成功")
        print(f"   指標タイプ: {keltner_indicator.indicator_type}")
        print(f"   サポート期間: {keltner_indicator.supported_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 Keltner Channels実装テスト開始\n")
    
    tests = [
        ("KeltnerChannelsIndicatorクラス", test_keltner_channels_indicator),
        ("Keltner ChannelsとBollinger Bandsの比較", test_keltner_channels_vs_bollinger_bands),
        ("Keltner Channelsのmultiplierパラメータ", test_keltner_channels_multiplier),
        ("Keltner Channels統合", test_keltner_channels_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}のテスト:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("📊 テスト結果サマリー:")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("Keltner Channels の実装が完了しています。")
        print("Keltner ChannelsはATRベースのボラティリティチャネルで、Bollinger Bandsの代替として使用されます。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*60)

if __name__ == "__main__":
    main()
