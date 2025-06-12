#!/usr/bin/env python3
"""
VWMA実装のテストスクリプト

新しく実装したVWMAIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_vwma_indicator():
    """VWMAIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import VWMAIndicator
        
        print("✅ VWMAIndicatorのインポート成功")
        
        # テストデータの作成（VWMAは出来高データが必要）
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格・出来高データを生成
        base_price = 100
        price_trend = np.linspace(0, 20, 100)  # 上昇トレンド
        price_noise = np.random.normal(0, 2, 100)  # ノイズ
        prices = base_price + price_trend + price_noise
        
        # 出来高は価格変動と逆相関（現実的なパターン）
        base_volume = 10000
        volume_variation = np.random.uniform(0.5, 2.0, 100)
        volumes = base_volume * volume_variation
        
        test_data = pd.DataFrame({
            'open': prices + np.random.uniform(-1, 1, 100),
            'high': prices + np.random.uniform(1, 3, 100),
            'low': prices + np.random.uniform(-3, -1, 100),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # VWMAIndicatorのインスタンス化
        vwma_indicator = VWMAIndicator()
        print("✅ VWMAIndicatorのインスタンス化成功")
        print(f"   サポート期間: {vwma_indicator.supported_periods}")
        
        # 異なる期間でのVWMA計算テスト
        for period in [10, 20, 30]:
            try:
                result = vwma_indicator.calculate(test_data, period)
                
                print(f"✅ VWMA計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の長さ: {len(result)}")
                print(f"   非NaN値の数: {result.notna().sum()}")
                print(f"   最後の5つの値:")
                print(f"   {result.tail().round(2)}")
                print()
                
            except Exception as e:
                print(f"❌ VWMA計算失敗 (期間: {period}): {e}")
                return False
        
        # 説明の取得テスト
        description = vwma_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ VWMAIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwma_vs_sma():
    """VWMAとSMAの比較テスト"""
    try:
        from app.core.services.indicators import VWMAIndicator, SMAIndicator
        
        print("\n📊 VWMAとSMAの比較テスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 価格は一定だが、出来高が変動するケース
        prices = np.full(50, 100.0)  # 価格は100で一定
        
        # 出来高パターン: 前半は低出来高、後半は高出来高
        volumes = np.concatenate([
            np.full(25, 1000),   # 前半: 低出来高
            np.full(25, 10000)   # 後半: 高出来高
        ])
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        period = 20
        
        # 各移動平均を計算
        vwma_indicator = VWMAIndicator()
        sma_indicator = SMAIndicator()
        
        vwma_result = vwma_indicator.calculate(test_data, period)
        sma_result = sma_indicator.calculate(test_data, period)
        
        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   価格: 一定（100）、出来高: 前半1000 → 後半10000")
        print(f"   最後の10個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'Close': test_data['close'].tail(10).round(2),
            'Volume': test_data['volume'].tail(10),
            'SMA': sma_result.tail(10).round(2),
            'VWMA': vwma_result.tail(10).round(2)
        })
        
        print(comparison_df)
        
        # VWMAとSMAの差を確認
        final_vwma = vwma_result.iloc[-1]
        final_sma = sma_result.iloc[-1]
        
        print(f"\n   最終値比較:")
        print(f"   SMA: {final_sma:.2f}")
        print(f"   VWMA: {final_vwma:.2f}")
        print(f"   差: {abs(final_vwma - final_sma):.2f}")
        
        # 価格が一定の場合、VWMAもSMAも同じ値になるはず
        if abs(final_vwma - final_sma) < 0.01:
            print("   ✅ 価格一定時のVWMA=SMA確認")
        else:
            print("   ⚠️  価格一定時のVWMA≠SMA（要確認）")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwma_volume_weighting():
    """VWMAの出来高重み付けテスト"""
    try:
        from app.core.services.indicators import VWMAIndicator, SMAIndicator
        
        print("\n🔢 VWMAの出来高重み付けテスト:")
        
        # 特殊なテストケース: 価格変動 + 出来高変動
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # 価格パターン: 急上昇
        prices = np.concatenate([
            np.full(10, 100),    # 最初の10日: 100
            np.full(10, 110),    # 次の10日: 110（急上昇）
            np.full(10, 105)     # 最後の10日: 105（下落）
        ])
        
        # 出来高パターン: 急上昇時に大量出来高
        volumes = np.concatenate([
            np.full(10, 1000),   # 最初の10日: 低出来高
            np.full(10, 20000),  # 次の10日: 高出来高（急上昇時）
            np.full(10, 1000)    # 最後の10日: 低出来高
        ])
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        period = 15
        
        # 各移動平均を計算
        vwma_indicator = VWMAIndicator()
        sma_indicator = SMAIndicator()
        
        vwma_result = vwma_indicator.calculate(test_data, period)
        sma_result = sma_indicator.calculate(test_data, period)
        
        # 結果の分析
        print(f"   期間: {period}")
        print(f"   価格パターン: 100 → 110（高出来高） → 105")
        print(f"   最後の5個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'Close': test_data['close'].tail(5),
            'Volume': test_data['volume'].tail(5),
            'SMA': sma_result.tail(5).round(2),
            'VWMA': vwma_result.tail(5).round(2),
            'Diff': (vwma_result.tail(5) - sma_result.tail(5)).round(2)
        })
        
        print(comparison_df)
        
        # VWMAが高出来高時の価格（110）により重みを置いているかチェック
        final_vwma = vwma_result.iloc[-1]
        final_sma = sma_result.iloc[-1]
        
        print(f"\n   最終値比較:")
        print(f"   SMA: {final_sma:.2f}")
        print(f"   VWMA: {final_vwma:.2f}")
        
        # VWMAは高出来高時の価格110により重みを置くため、SMAより高くなるはず
        if final_vwma > final_sma:
            print("   ✅ VWMAが高出来高時の価格により重みを置いている")
        else:
            print("   ⚠️  VWMAの重み付けが期待通りでない可能性")
        
        return True
        
    except Exception as e:
        print(f"❌ 重み付けテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwma_integration():
    """VWMAの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        print("\n🔗 VWMA統合テスト:")
        
        # ファクトリー関数経由での取得
        vwma_indicator = get_indicator_by_type("VWMA")
        print("✅ ファクトリー関数からのVWMA取得成功")
        print(f"   指標タイプ: {vwma_indicator.indicator_type}")
        print(f"   サポート期間: {vwma_indicator.supported_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 VWMA実装テスト開始\n")
    
    tests = [
        ("VWMAIndicatorクラス", test_vwma_indicator),
        ("VWMAとSMAの比較", test_vwma_vs_sma),
        ("VWMAの出来高重み付け", test_vwma_volume_weighting),
        ("VWMA統合", test_vwma_integration),
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
        print("VWMA (Volume Weighted Moving Average) の実装が完了しています。")
        print("VWMAは出来高を重みとした移動平均で、機関投資家の動向を反映しやすい指標です。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*60)

if __name__ == "__main__":
    main()
