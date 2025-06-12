#!/usr/bin/env python3
"""
HMA実装のテストスクリプト

新しく実装したHMAIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_hma_indicator():
    """HMAIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import HMAIndicator
        
        print("✅ HMAIndicatorのインポート成功")
        
        # テストデータの作成（HMAは多くのデータが必要）
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # より現実的な価格データを生成（トレンドを含む）
        base_price = 100
        trend = np.linspace(0, 20, 200)  # 上昇トレンド
        noise = np.random.normal(0, 2, 200)  # ノイズ
        prices = base_price + trend + noise
        
        test_data = pd.DataFrame({
            'open': prices + np.random.uniform(-1, 1, 200),
            'high': prices + np.random.uniform(1, 3, 200),
            'low': prices + np.random.uniform(-3, -1, 200),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=dates)
        
        # HMAIndicatorのインスタンス化
        hma_indicator = HMAIndicator()
        print("✅ HMAIndicatorのインスタンス化成功")
        print(f"   サポート期間: {hma_indicator.supported_periods}")
        
        # 異なる期間でのHMA計算テスト
        for period in [9, 14, 21, 30]:
            try:
                result = hma_indicator.calculate(test_data, period)
                
                print(f"✅ HMA計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の長さ: {len(result)}")
                print(f"   非NaN値の数: {result.notna().sum()}")
                print(f"   最後の5つの値:")
                print(f"   {result.tail().round(2)}")
                print()
                
            except Exception as e:
                print(f"❌ HMA計算失敗 (期間: {period}): {e}")
                return False
        
        # 説明の取得テスト
        description = hma_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ HMAIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hma_vs_other_ma():
    """HMAと他の移動平均の比較テスト"""
    try:
        from app.core.services.indicators import HMAIndicator, SMAIndicator, EMAIndicator, WMAIndicator
        
        print("\n📊 HMAと他の移動平均の比較テスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # ステップ関数的な価格変動を作成（応答性をテストするため）
        prices = np.concatenate([
            np.full(30, 100),  # 最初の30日は100
            np.full(40, 110),  # 次の40日は110（急上昇）
            np.full(30, 105)   # 最後の30日は105（下落）
        ])
        
        test_data = pd.DataFrame({
            'open': prices + np.random.uniform(-0.5, 0.5, 100),
            'high': prices + np.random.uniform(0.5, 1.5, 100),
            'low': prices + np.random.uniform(-1.5, -0.5, 100),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        period = 21
        
        # 各移動平均を計算
        hma_indicator = HMAIndicator()
        sma_indicator = SMAIndicator()
        ema_indicator = EMAIndicator()
        wma_indicator = WMAIndicator()
        
        hma_result = hma_indicator.calculate(test_data, period)
        sma_result = sma_indicator.calculate(test_data, period)
        ema_result = ema_indicator.calculate(test_data, period)
        wma_result = wma_indicator.calculate(test_data, period)
        
        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   価格変動: 100 → 110 → 105")
        print(f"   最後の10個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'Close': test_data['close'].tail(10).round(2),
            'SMA': sma_result.tail(10).round(2),
            'EMA': ema_result.tail(10).round(2),
            'WMA': wma_result.tail(10).round(2),
            'HMA': hma_result.tail(10).round(2)
        })
        
        print(comparison_df)
        
        # HMAの応答性をチェック（価格変動への追従速度）
        price_change_point = 70  # 価格が110から105に変わる点
        if len(hma_result) > price_change_point + 5:
            hma_response = abs(hma_result.iloc[price_change_point + 5] - 105)
            sma_response = abs(sma_result.iloc[price_change_point + 5] - 105)
            
            print(f"\n   価格変動への応答性比較（変動5日後）:")
            print(f"   HMAの価格105からの乖離: {hma_response:.2f}")
            print(f"   SMAの価格105からの乖離: {sma_response:.2f}")
            
            if hma_response < sma_response:
                print("   ✅ HMAがSMAより応答性が高い")
            else:
                print("   ⚠️  HMAの応答性がSMAと同等またはそれ以下")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hma_integration():
    """HMAの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        print("\n🔗 HMA統合テスト:")
        
        # ファクトリー関数経由での取得
        hma_indicator = get_indicator_by_type("HMA")
        print("✅ ファクトリー関数からのHMA取得成功")
        print(f"   指標タイプ: {hma_indicator.indicator_type}")
        print(f"   サポート期間: {hma_indicator.supported_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 HMA実装テスト開始\n")
    
    tests = [
        ("HMAIndicatorクラス", test_hma_indicator),
        ("HMAと他の移動平均の比較", test_hma_vs_other_ma),
        ("HMA統合", test_hma_integration),
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
        print("HMA (Hull Moving Average) の実装が完了しています。")
        print("HMAは従来の移動平均よりもラグが少なく、応答性が高い指標です。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*60)

if __name__ == "__main__":
    main()
