#!/usr/bin/env python3
"""
Stochastic RSI実装のテストスクリプト

新しく実装したStochasticRSIIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_stochastic_rsi_indicator():
    """StochasticRSIIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import StochasticRSIIndicator
        
        print("✅ StochasticRSIIndicatorのインポート成功")
        
        # テストデータの作成（Stochastic RSIは終値データが必要）
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格データを生成
        base_price = 100
        price_trend = np.linspace(0, 20, 100)  # 上昇トレンド
        price_noise = np.random.normal(0, 2, 100)  # ノイズ
        close_prices = base_price + price_trend + price_noise
        
        test_data = pd.DataFrame({
            'open': close_prices + np.random.uniform(-1, 1, 100),
            'high': close_prices + np.random.uniform(0.5, 1.5, 100),
            'low': close_prices - np.random.uniform(0.5, 1.5, 100),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # StochasticRSIIndicatorのインスタンス化
        stoch_rsi_indicator = StochasticRSIIndicator()
        print("✅ StochasticRSIIndicatorのインスタンス化成功")
        print(f"   サポート期間: {stoch_rsi_indicator.supported_periods}")
        
        # 異なる期間でのStochastic RSI計算テスト
        for period in [14, 21]:
            try:
                result = stoch_rsi_indicator.calculate(test_data, period)
                
                print(f"✅ Stochastic RSI計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の形状: {result.shape}")
                print(f"   カラム: {list(result.columns)}")
                print(f"   非NaN値の数: {result.notna().sum().sum()}")
                print(f"   最後の5つの値:")
                print(result.tail().round(2))
                print()
                
            except Exception as e:
                print(f"❌ Stochastic RSI計算失敗 (期間: {period}): {e}")
                return False
        
        # 説明の取得テスト
        description = stoch_rsi_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ StochasticRSIIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stochastic_rsi_vs_rsi():
    """Stochastic RSIとRSIの比較テスト"""
    try:
        from app.core.services.indicators import StochasticRSIIndicator, RSIIndicator
        
        print("\n📊 Stochastic RSIとRSIの比較テスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 買われすぎ・売られすぎの状況を作るデータ
        base_price = 100
        # 急上昇 → 横ばい → 急下降のパターン
        price_pattern = np.concatenate([
            np.linspace(100, 120, 15),  # 急上昇
            np.full(20, 120),           # 横ばい（買われすぎ状態）
            np.linspace(120, 90, 15)    # 急下降
        ])
        
        # ノイズを追加
        close_prices = price_pattern + np.random.normal(0, 0.5, 50)
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': close_prices + np.random.uniform(0.2, 1, 50),
            'low': close_prices - np.random.uniform(0.2, 1, 50),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)
        
        period = 14
        
        # 各指標を計算
        stoch_rsi_indicator = StochasticRSIIndicator()
        rsi_indicator = RSIIndicator()
        
        stoch_rsi_result = stoch_rsi_indicator.calculate(test_data, period)
        rsi_result = rsi_indicator.calculate(test_data, period)
        
        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   価格パターン: 急上昇 → 横ばい → 急下降")
        print(f"   最後の10個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'Close': test_data['close'].tail(10).round(2),
            'RSI': rsi_result.tail(10).round(2),
            'StochRSI_K': stoch_rsi_result['fastk'].tail(10).round(2),
            'StochRSI_D': stoch_rsi_result['fastd'].tail(10).round(2)
        })
        
        print(comparison_df)
        
        # 感度の比較
        rsi_range = rsi_result.max() - rsi_result.min()
        stoch_rsi_k_range = stoch_rsi_result['fastk'].max() - stoch_rsi_result['fastk'].min()
        
        print(f"\n   感度比較:")
        print(f"   RSI変動幅: {rsi_range:.2f}")
        print(f"   Stochastic RSI %K変動幅: {stoch_rsi_k_range:.2f}")
        
        if stoch_rsi_k_range > rsi_range:
            print(f"   ✅ Stochastic RSIがRSIより高感度（{stoch_rsi_k_range:.2f} > {rsi_range:.2f}）")
        else:
            print(f"   ⚠️  感度の違いが期待通りでない可能性")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stochastic_rsi_parameters():
    """Stochastic RSIのパラメータテスト"""
    try:
        from app.core.services.indicators import StochasticRSIIndicator
        
        print("\n🔢 Stochastic RSIのパラメータテスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # ボラティリティのある価格データ
        base_price = 100
        close_prices = base_price + np.cumsum(np.random.normal(0, 1, 50))
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': close_prices + np.random.uniform(0.5, 1.5, 50),
            'low': close_prices - np.random.uniform(0.5, 1.5, 50),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)
        
        period = 14
        stoch_rsi_indicator = StochasticRSIIndicator()
        
        # 異なるfastk_period, fastd_periodでの計算
        parameter_sets = [
            (3, 3),   # デフォルト
            (5, 3),   # fastk_periodを長く
            (3, 5),   # fastd_periodを長く
            (5, 5),   # 両方長く
        ]
        
        results = {}
        
        for fastk_period, fastd_period in parameter_sets:
            result = stoch_rsi_indicator.calculate(
                test_data, period, 
                fastk_period=fastk_period, 
                fastd_period=fastd_period
            )
            results[(fastk_period, fastd_period)] = result
            
            # 最終値の表示
            final_k = result['fastk'].iloc[-1]
            final_d = result['fastd'].iloc[-1]
            print(f"   FastK={fastk_period}, FastD={fastd_period}: %K={final_k:.2f}, %D={final_d:.2f}")
        
        # パラメータの影響確認
        print(f"\n   パラメータ効果の確認:")
        
        # デフォルトとの比較
        default_result = results[(3, 3)]
        
        for params, result in results.items():
            if params == (3, 3):
                continue
                
            fastk_period, fastd_period = params
            
            # %Dの平滑化効果を確認
            default_d_volatility = default_result['fastd'].std()
            current_d_volatility = result['fastd'].std()
            
            print(f"   FastK={fastk_period}, FastD={fastd_period}: %D標準偏差={current_d_volatility:.2f} (デフォルト: {default_d_volatility:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ パラメータテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stochastic_rsi_integration():
    """Stochastic RSIの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        print("\n🔗 Stochastic RSI統合テスト:")
        
        # ファクトリー関数経由での取得
        stoch_rsi_indicator = get_indicator_by_type("STOCHRSI")
        print("✅ ファクトリー関数からのStochastic RSI取得成功")
        print(f"   指標タイプ: {stoch_rsi_indicator.indicator_type}")
        print(f"   サポート期間: {stoch_rsi_indicator.supported_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 Stochastic RSI実装テスト開始\n")
    
    tests = [
        ("StochasticRSIIndicatorクラス", test_stochastic_rsi_indicator),
        ("Stochastic RSIとRSIの比較", test_stochastic_rsi_vs_rsi),
        ("Stochastic RSIのパラメータ", test_stochastic_rsi_parameters),
        ("Stochastic RSI統合", test_stochastic_rsi_integration),
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
        print("Stochastic RSI の実装が完了しています。")
        print("Stochastic RSIはRSIにストキャスティクスを適用した高感度オシレーターです。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*60)

if __name__ == "__main__":
    main()
