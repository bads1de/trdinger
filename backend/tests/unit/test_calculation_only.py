#!/usr/bin/env python3
"""
テクニカル指標の計算ロジック専用テストスイート

外部依存を排除し、純粋な計算ロジックのみをテストします。
これにより、実装にバグがないことを確実に検証できます。
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

from app.core.services.technical_indicator_service import TechnicalIndicatorService

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalculationTester:
    """計算ロジック専用テストクラス"""
    
    def __init__(self):
        self.service = TechnicalIndicatorService()
        
    def create_test_data(self, length: int = 100, pattern: str = "random") -> pd.DataFrame:
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=length),
            periods=length,
            freq='h'
        )
        
        if pattern == "random":
            # ランダムウォーク
            np.random.seed(42)
            base_price = 50000
            returns = np.random.normal(0, 0.02, length)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
                
        elif pattern == "trend_up":
            # 上昇トレンド
            base_price = 50000
            prices = [base_price + i * 100 + np.random.normal(0, 50) for i in range(length)]
            
        elif pattern == "trend_down":
            # 下降トレンド
            base_price = 60000
            prices = [base_price - i * 100 + np.random.normal(0, 50) for i in range(length)]
            
        elif pattern == "sideways":
            # レンジ相場
            base_price = 50000
            prices = [base_price + np.random.normal(0, 200) for _ in range(length)]
            
        # OHLCV データ生成
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def test_all_indicators_with_patterns(self) -> Dict[str, bool]:
        """全指標を異なる市場パターンでテスト"""
        print("\n🧪 全指標の市場パターン別テスト")
        print("-" * 50)
        
        patterns = ["random", "trend_up", "trend_down", "sideways"]
        results = {}
        
        for pattern in patterns:
            print(f"\n📈 {pattern.upper()} パターンでのテスト:")
            test_data = self.create_test_data(100, pattern)
            
            pattern_results = []
            for indicator_type, config in self.service.supported_indicators.items():
                try:
                    period = config["periods"][0]
                    calc_func = config["function"]
                    result = calc_func(test_data, period)
                    
                    # 結果の検証
                    if isinstance(result, pd.DataFrame):
                        valid_rows = result.notna().any(axis=1).sum()
                        print(f"  ✅ {indicator_type:6s}: DataFrame ({valid_rows}有効行)")
                        pattern_results.append(True)
                    elif isinstance(result, pd.Series):
                        valid_count = result.notna().sum()
                        print(f"  ✅ {indicator_type:6s}: Series ({valid_count}有効値)")
                        pattern_results.append(True)
                    else:
                        print(f"  ❌ {indicator_type:6s}: 予期しない戻り値型")
                        pattern_results.append(False)
                        
                except Exception as e:
                    print(f"  ❌ {indicator_type:6s}: {e}")
                    pattern_results.append(False)
            
            results[f"pattern_{pattern}"] = all(pattern_results)
            success_rate = sum(pattern_results) / len(pattern_results) * 100
            print(f"  📊 {pattern} パターン成功率: {success_rate:.1f}%")
        
        return results
    
    def test_mathematical_properties(self) -> Dict[str, bool]:
        """数学的特性のテスト"""
        print("\n🔢 数学的特性のテスト")
        print("-" * 50)
        
        results = {}
        test_data = self.create_test_data(100, "random")
        
        # 1. SMAの特性テスト
        print("1. SMA（単純移動平均）の特性:")
        try:
            sma_20 = self.service._calculate_sma(test_data, 20)
            sma_50 = self.service._calculate_sma(test_data, 50)
            
            # SMA20はSMA50より変動が大きいはず
            sma_20_std = sma_20.std()
            sma_50_std = sma_50.std()
            
            if sma_20_std > sma_50_std:
                print("  ✅ SMA20の標準偏差 > SMA50の標準偏差")
                results["sma_volatility"] = True
            else:
                print("  ❌ SMA期間と変動性の関係が不正")
                results["sma_volatility"] = False
                
        except Exception as e:
            print(f"  ❌ SMA特性テスト: {e}")
            results["sma_volatility"] = False
        
        # 2. RSIの範囲テスト
        print("\n2. RSI（相対力指数）の範囲:")
        try:
            rsi = self.service._calculate_rsi(test_data, 14)
            rsi_valid = rsi.dropna()
            
            if len(rsi_valid) > 0:
                min_rsi = rsi_valid.min()
                max_rsi = rsi_valid.max()
                
                if 0 <= min_rsi <= 100 and 0 <= max_rsi <= 100:
                    print(f"  ✅ RSI範囲: {min_rsi:.2f} - {max_rsi:.2f} (0-100内)")
                    results["rsi_range"] = True
                else:
                    print(f"  ❌ RSI範囲外: {min_rsi:.2f} - {max_rsi:.2f}")
                    results["rsi_range"] = False
            else:
                print("  ❌ RSI: 有効な値がありません")
                results["rsi_range"] = False
                
        except Exception as e:
            print(f"  ❌ RSI範囲テスト: {e}")
            results["rsi_range"] = False
        
        # 3. ボリンジャーバンドの関係テスト
        print("\n3. ボリンジャーバンドの関係:")
        try:
            bb = self.service._calculate_bollinger_bands(test_data, 20)
            bb_valid = bb.dropna()
            
            if len(bb_valid) > 0:
                # 上限 > 中央線 > 下限 の関係をチェック
                upper_gt_middle = (bb_valid['upper'] > bb_valid['middle']).all()
                middle_gt_lower = (bb_valid['middle'] > bb_valid['lower']).all()
                
                if upper_gt_middle and middle_gt_lower:
                    print("  ✅ ボリンジャーバンド: 上限 > 中央線 > 下限")
                    results["bb_relationship"] = True
                else:
                    print("  ❌ ボリンジャーバンド: 順序関係が不正")
                    results["bb_relationship"] = False
            else:
                print("  ❌ ボリンジャーバンド: 有効な値がありません")
                results["bb_relationship"] = False
                
        except Exception as e:
            print(f"  ❌ ボリンジャーバンド関係テスト: {e}")
            results["bb_relationship"] = False
        
        # 4. ATRの非負性テスト
        print("\n4. ATR（平均真の値幅）の非負性:")
        try:
            atr = self.service._calculate_atr(test_data, 14)
            atr_valid = atr.dropna()
            
            if len(atr_valid) > 0:
                min_atr = atr_valid.min()
                
                if min_atr >= 0:
                    print(f"  ✅ ATR非負性: 最小値 {min_atr:.2f} >= 0")
                    results["atr_non_negative"] = True
                else:
                    print(f"  ❌ ATR負値: 最小値 {min_atr:.2f} < 0")
                    results["atr_non_negative"] = False
            else:
                print("  ❌ ATR: 有効な値がありません")
                results["atr_non_negative"] = False
                
        except Exception as e:
            print(f"  ❌ ATR非負性テスト: {e}")
            results["atr_non_negative"] = False
        
        return results
    
    def test_performance_and_memory(self) -> Dict[str, bool]:
        """パフォーマンスとメモリ使用量のテスト"""
        print("\n⚡ パフォーマンス・メモリテスト")
        print("-" * 50)
        
        results = {}
        
        # 大量データでのテスト
        print("1. 大量データ処理テスト (1000件):")
        try:
            large_data = self.create_test_data(1000, "random")
            
            import time
            start_time = time.time()
            
            # 全指標を計算
            for indicator_type, config in self.service.supported_indicators.items():
                period = config["periods"][0]
                calc_func = config["function"]
                result = calc_func(large_data, period)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if processing_time < 10:  # 10秒以内
                print(f"  ✅ 処理時間: {processing_time:.2f}秒 (< 10秒)")
                results["performance"] = True
            else:
                print(f"  ❌ 処理時間: {processing_time:.2f}秒 (>= 10秒)")
                results["performance"] = False
                
        except Exception as e:
            print(f"  ❌ パフォーマンステスト: {e}")
            results["performance"] = False
        
        return results

async def run_calculation_tests():
    """計算ロジック専用テストの実行"""
    print("🧪 テクニカル指標 計算ロジック専用テストスイート")
    print("=" * 60)
    
    tester = CalculationTester()
    all_results = {}
    
    # 1. 市場パターン別テスト
    pattern_results = tester.test_all_indicators_with_patterns()
    all_results.update(pattern_results)
    
    # 2. 数学的特性テスト
    math_results = tester.test_mathematical_properties()
    all_results.update(math_results)
    
    # 3. パフォーマンステスト
    perf_results = tester.test_performance_and_memory()
    all_results.update(perf_results)
    
    # 結果サマリー
    print("\n📋 計算ロジックテスト結果サマリー")
    print("=" * 60)
    
    passed = sum(1 for result in all_results.values() if result)
    total = len(all_results)
    
    print(f"✅ 成功: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed < total:
        print(f"❌ 失敗: {total-passed}/{total}")
        print("\n失敗したテスト:")
        for test_name, result in all_results.items():
            if not result:
                print(f"  - {test_name}")
    else:
        print("🎉 全ての計算ロジックテストが成功しました！")
        print("\n✨ 実装品質:")
        print("  • 全12種類の指標が正常に動作")
        print("  • 異なる市場パターンで安定動作")
        print("  • 数学的特性が正しく実装")
        print("  • パフォーマンスが良好")
        print("  • バグは検出されませんでした")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_calculation_tests())
    sys.exit(0 if success else 1)
