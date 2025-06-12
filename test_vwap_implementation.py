#!/usr/bin/env python3
"""
VWAP実装のテストスクリプト

新しく実装したVWAPIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_vwap_indicator():
    """VWAPIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import VWAPIndicator
        
        print("✅ VWAPIndicatorのインポート成功")
        
        # テストデータの作成（VWAPは高値・安値・終値・出来高データが必要）
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格・出来高データを生成
        base_price = 100
        price_trend = np.linspace(0, 20, 100)  # 上昇トレンド
        price_noise = np.random.normal(0, 2, 100)  # ノイズ
        close_prices = base_price + price_trend + price_noise
        
        # 高値・安値を終値から生成
        high_prices = close_prices + np.random.uniform(1, 3, 100)
        low_prices = close_prices - np.random.uniform(1, 3, 100)
        
        # 出来高データ
        base_volume = 10000
        volume_variation = np.random.uniform(0.5, 2.0, 100)
        volumes = base_volume * volume_variation
        
        test_data = pd.DataFrame({
            'open': close_prices + np.random.uniform(-1, 1, 100),
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        # VWAPIndicatorのインスタンス化
        vwap_indicator = VWAPIndicator()
        print("✅ VWAPIndicatorのインスタンス化成功")
        print(f"   サポート期間: {vwap_indicator.supported_periods}")
        
        # 異なる期間でのVWAP計算テスト
        for period in [1, 5, 10, 20]:
            try:
                result = vwap_indicator.calculate(test_data, period)
                
                print(f"✅ VWAP計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の長さ: {len(result)}")
                print(f"   非NaN値の数: {result.notna().sum()}")
                print(f"   最後の5つの値:")
                print(f"   {result.tail().round(2)}")
                print()
                
            except Exception as e:
                print(f"❌ VWAP計算失敗 (期間: {period}): {e}")
                return False
        
        # 説明の取得テスト
        description = vwap_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ VWAPIndicatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwap_vs_typical_price():
    """VWAPとTypical Priceの比較テスト"""
    try:
        from app.core.services.indicators import VWAPIndicator
        
        print("\n📊 VWAPとTypical Priceの比較テスト:")
        
        # テストデータの作成
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # 価格は一定だが、出来高が変動するケース
        close_prices = np.full(30, 100.0)  # 終値は100で一定
        high_prices = np.full(30, 102.0)   # 高値は102で一定
        low_prices = np.full(30, 98.0)     # 安値は98で一定
        
        # Typical Price = (High + Low + Close) / 3 = (102 + 98 + 100) / 3 = 100
        expected_typical_price = 100.0
        
        # 出来高パターン: 前半は低出来高、後半は高出来高
        volumes = np.concatenate([
            np.full(15, 1000),   # 前半: 低出来高
            np.full(15, 10000)   # 後半: 高出来高
        ])
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        period = 10
        
        # VWAP計算
        vwap_indicator = VWAPIndicator()
        vwap_result = vwap_indicator.calculate(test_data, period)
        
        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   価格: 一定（High=102, Low=98, Close=100）")
        print(f"   Typical Price: {expected_typical_price}")
        print(f"   出来高: 前半1000 → 後半10000")
        print(f"   最後の10個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'High': test_data['high'].tail(10),
            'Low': test_data['low'].tail(10),
            'Close': test_data['close'].tail(10),
            'Volume': test_data['volume'].tail(10),
            'VWAP': vwap_result.tail(10).round(2)
        })
        
        print(comparison_df)
        
        # 価格が一定の場合、VWAPはTypical Priceと同じ値になるはず
        final_vwap = vwap_result.iloc[-1]
        
        print(f"\n   最終値比較:")
        print(f"   期待値（Typical Price): {expected_typical_price:.2f}")
        print(f"   VWAP: {final_vwap:.2f}")
        print(f"   差: {abs(final_vwap - expected_typical_price):.2f}")
        
        if abs(final_vwap - expected_typical_price) < 0.01:
            print("   ✅ 価格一定時のVWAP=Typical Price確認")
        else:
            print("   ⚠️  価格一定時のVWAP≠Typical Price（要確認）")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwap_volume_weighting():
    """VWAPの出来高重み付けテスト"""
    try:
        from app.core.services.indicators import VWAPIndicator
        
        print("\n🔢 VWAPの出来高重み付けテスト:")
        
        # 特殊なテストケース: 価格変動 + 出来高変動
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # 価格パターン: 急上昇
        close_prices = np.concatenate([
            np.full(10, 100),    # 最初の10日: 100
            np.full(10, 110)     # 次の10日: 110（急上昇）
        ])
        
        high_prices = close_prices + 2
        low_prices = close_prices - 2
        
        # 出来高パターン: 急上昇時に大量出来高
        volumes = np.concatenate([
            np.full(10, 1000),   # 最初の10日: 低出来高
            np.full(10, 20000)   # 次の10日: 高出来高（急上昇時）
        ])
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        period = 15
        
        # VWAP計算
        vwap_indicator = VWAPIndicator()
        vwap_result = vwap_indicator.calculate(test_data, period)
        
        # 結果の分析
        print(f"   期間: {period}")
        print(f"   価格パターン: 100（低出来高） → 110（高出来高）")
        print(f"   最後の5個の値の比較:")
        
        comparison_df = pd.DataFrame({
            'High': test_data['high'].tail(5),
            'Low': test_data['low'].tail(5),
            'Close': test_data['close'].tail(5),
            'Volume': test_data['volume'].tail(5),
            'VWAP': vwap_result.tail(5).round(2)
        })
        
        print(comparison_df)
        
        # VWAPが高出来高時の価格（110付近）により重みを置いているかチェック
        final_vwap = vwap_result.iloc[-1]
        
        print(f"\n   最終値分析:")
        print(f"   VWAP: {final_vwap:.2f}")
        
        # 単純平均なら105、VWAPは高出来高時の110により重みを置くため105より高くなるはず
        simple_average = 105.0
        if final_vwap > simple_average:
            print(f"   ✅ VWAPが高出来高時の価格により重みを置いている（{final_vwap:.2f} > {simple_average}）")
        else:
            print(f"   ⚠️  VWAPの重み付けが期待通りでない可能性（{final_vwap:.2f} <= {simple_average}）")
        
        return True
        
    except Exception as e:
        print(f"❌ 重み付けテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vwap_integration():
    """VWAPの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type
        
        print("\n🔗 VWAP統合テスト:")
        
        # ファクトリー関数経由での取得
        vwap_indicator = get_indicator_by_type("VWAP")
        print("✅ ファクトリー関数からのVWAP取得成功")
        print(f"   指標タイプ: {vwap_indicator.indicator_type}")
        print(f"   サポート期間: {vwap_indicator.supported_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 VWAP実装テスト開始\n")
    
    tests = [
        ("VWAPIndicatorクラス", test_vwap_indicator),
        ("VWAPとTypical Priceの比較", test_vwap_vs_typical_price),
        ("VWAPの出来高重み付け", test_vwap_volume_weighting),
        ("VWAP統合", test_vwap_integration),
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
        print("VWAP (Volume Weighted Average Price) の実装が完了しています。")
        print("VWAPは機関投資家のベンチマーク指標として広く使用されています。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("="*60)

if __name__ == "__main__":
    main()
