#!/usr/bin/env python3
"""
テクニカル指標の包括的テストスイート

このスクリプトは、実装されたテクニカル指標の計算ロジック、
データベース操作、API エンドポイントを包括的にテストします。
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
from database.connection import SessionLocal, engine, Base
from database.repositories.technical_indicator_repository import TechnicalIndicatorRepository

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorTester:
    """テクニカル指標テストクラス"""
    
    def __init__(self):
        self.service = TechnicalIndicatorService()
        self.test_results = []
        
    def create_test_data(self, length: int = 100) -> pd.DataFrame:
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=length),
            periods=length,
            freq='H'
        )
        
        # シンプルなランダムウォークでテストデータ生成
        np.random.seed(42)  # 再現可能性のため
        base_price = 50000
        
        # 価格データ生成
        returns = np.random.normal(0, 0.02, length)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
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
    
    def test_calculation_methods(self) -> Dict[str, bool]:
        """各指標の計算メソッドをテスト"""
        print("\n🧪 計算メソッドのテスト")
        print("-" * 40)
        
        test_data = self.create_test_data(100)
        results = {}
        
        # 各指標をテスト
        for indicator_type, config in self.service.supported_indicators.items():
            try:
                print(f"テスト中: {indicator_type}")
                
                # 最初の期間を使用
                period = config["periods"][0]
                calc_func = config["function"]
                
                # 計算実行
                result = calc_func(test_data, period)
                
                # 結果の検証
                if isinstance(result, pd.DataFrame):
                    # 複数値指標（MACD, BB, STOCH）
                    if result.empty:
                        raise ValueError("結果が空のDataFrame")
                    if result.isna().all().all():
                        raise ValueError("全ての値がNaN")
                    print(f"  ✅ {indicator_type}: DataFrame ({result.shape})")
                elif isinstance(result, pd.Series):
                    # 単一値指標
                    if result.empty:
                        raise ValueError("結果が空のSeries")
                    if result.isna().all():
                        raise ValueError("全ての値がNaN")
                    valid_count = result.notna().sum()
                    print(f"  ✅ {indicator_type}: Series (有効値: {valid_count}/{len(result)})")
                else:
                    raise ValueError(f"予期しない戻り値の型: {type(result)}")
                
                results[indicator_type] = True
                
            except Exception as e:
                print(f"  ❌ {indicator_type}: {e}")
                results[indicator_type] = False
        
        return results
    
    def test_edge_cases(self) -> Dict[str, bool]:
        """エッジケースのテスト"""
        print("\n🔍 エッジケースのテスト")
        print("-" * 40)
        
        results = {}
        
        # 1. 少ないデータでのテスト
        print("1. 少ないデータでのテスト")
        small_data = self.create_test_data(5)
        
        for indicator_type, config in self.service.supported_indicators.items():
            try:
                period = config["periods"][0]
                calc_func = config["function"]
                result = calc_func(small_data, period)
                
                # 結果が適切に処理されているか確認
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    print(f"  ✅ {indicator_type}: 少ないデータでも動作")
                    results[f"{indicator_type}_small_data"] = True
                else:
                    results[f"{indicator_type}_small_data"] = False
                    
            except Exception as e:
                print(f"  ⚠️  {indicator_type}: {e}")
                results[f"{indicator_type}_small_data"] = False
        
        # 2. 同じ価格データでのテスト
        print("\n2. 同じ価格データでのテスト")
        flat_data = self.create_test_data(50)
        flat_data['close'] = 50000  # 全て同じ価格
        flat_data['high'] = 50000
        flat_data['low'] = 50000
        flat_data['open'] = 50000
        
        for indicator_type in ['RSI', 'ATR', 'STOCH']:
            try:
                config = self.service.supported_indicators[indicator_type]
                period = config["periods"][0]
                calc_func = config["function"]
                result = calc_func(flat_data, period)
                
                print(f"  ✅ {indicator_type}: フラットデータでも動作")
                results[f"{indicator_type}_flat_data"] = True
                
            except Exception as e:
                print(f"  ⚠️  {indicator_type}: {e}")
                results[f"{indicator_type}_flat_data"] = False
        
        return results
    
    def test_data_processing_logic(self) -> Dict[str, bool]:
        """データ処理ロジックのテスト"""
        print("\n📊 データ処理ロジックのテスト")
        print("-" * 40)
        
        results = {}
        test_data = self.create_test_data(100)
        
        # 各指標タイプの処理をテスト
        test_cases = [
            ("MACD", 12),
            ("BB", 20),
            ("STOCH", 14),
            ("SMA", 20),
            ("RSI", 14)
        ]
        
        for indicator_type, period in test_cases:
            try:
                print(f"テスト中: {indicator_type}({period})")
                
                # 計算実行
                calc_func = self.service.supported_indicators[indicator_type]["function"]
                indicator_result = calc_func(test_data, period)
                
                # 結果の処理をシミュレート
                processed_results = []
                
                if indicator_type == "MACD":
                    for timestamp in indicator_result.index:
                        macd_line = indicator_result.loc[timestamp, 'macd_line']
                        signal_line = indicator_result.loc[timestamp, 'signal_line']
                        histogram = indicator_result.loc[timestamp, 'histogram']
                        
                        if pd.notna(macd_line) and pd.notna(signal_line):
                            processed_results.append({
                                "value": float(macd_line),
                                "signal_value": float(signal_line),
                                "histogram_value": float(histogram),
                                "upper_band": None,
                                "lower_band": None,
                            })
                            
                elif indicator_type == "BB":
                    for timestamp in indicator_result.index:
                        middle = indicator_result.loc[timestamp, 'middle']
                        upper = indicator_result.loc[timestamp, 'upper']
                        lower = indicator_result.loc[timestamp, 'lower']
                        
                        if pd.notna(middle) and pd.notna(upper) and pd.notna(lower):
                            processed_results.append({
                                "value": float(middle),
                                "signal_value": None,
                                "histogram_value": None,
                                "upper_band": float(upper),
                                "lower_band": float(lower),
                            })
                            
                elif indicator_type == "STOCH":
                    for timestamp in indicator_result.index:
                        k_percent = indicator_result.loc[timestamp, 'k_percent']
                        d_percent = indicator_result.loc[timestamp, 'd_percent']
                        
                        if pd.notna(k_percent) and pd.notna(d_percent):
                            processed_results.append({
                                "value": float(k_percent),
                                "signal_value": float(d_percent),
                                "histogram_value": None,
                                "upper_band": None,
                                "lower_band": None,
                            })
                            
                else:
                    # 単一値指標
                    for timestamp, value in indicator_result.items():
                        if pd.notna(value):
                            processed_results.append({
                                "value": float(value),
                                "signal_value": None,
                                "histogram_value": None,
                                "upper_band": None,
                                "lower_band": None,
                            })
                
                print(f"  ✅ {indicator_type}: {len(processed_results)}件の有効データ")
                results[f"{indicator_type}_processing"] = True
                
            except Exception as e:
                print(f"  ❌ {indicator_type}: {e}")
                results[f"{indicator_type}_processing"] = False
        
        return results

    async def test_database_operations(self) -> Dict[str, bool]:
        """データベース操作のテスト"""
        print("\n💾 データベース操作のテスト")
        print("-" * 40)

        results = {}

        try:
            # データベーステーブルの作成
            Base.metadata.create_all(bind=engine)

            # テストデータの準備
            test_data = self.create_test_data(50)
            symbol = "TEST_BTC_USDT"
            timeframe = "1h"

            # 1. 単一値指標のテスト（SMA）
            print("1. 単一値指標の保存・取得テスト (SMA)")
            try:
                # OHLCVデータを一時的に保存（テスト用）
                # 実際のAPIは外部データソースからOHLCVを取得するため、
                # ここでは計算結果のみをテスト
                sma_result = await self.service.calculate_and_save_technical_indicator(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_type="SMA",
                    period=20,
                    limit=50
                )

                if sma_result and sma_result.get("success"):
                    calculated = sma_result.get("calculated_count", 0)
                    saved = sma_result.get("saved_count", 0)
                    print(f"  ✅ SMA: {calculated}件計算, {saved}件保存")
                    results["sma_database"] = True
                else:
                    print("  ❌ SMA: データの保存に失敗")
                    results["sma_database"] = False

            except Exception as e:
                print(f"  ❌ SMA: {e}")
                results["sma_database"] = False

            # 2. 複数値指標のテスト（MACD）
            print("\n2. 複数値指標の保存・取得テスト (MACD)")
            try:
                macd_result = await self.service.calculate_and_save_technical_indicator(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_type="MACD",
                    period=12,
                    limit=50
                )

                if macd_result and macd_result.get("success"):
                    calculated = macd_result.get("calculated_count", 0)
                    saved = macd_result.get("saved_count", 0)
                    print(f"  ✅ MACD: {calculated}件計算, {saved}件保存（signal_value, histogram_value含む）")
                    results["macd_database"] = True
                else:
                    print("  ❌ MACD: データの保存に失敗")
                    results["macd_database"] = False

            except Exception as e:
                print(f"  ❌ MACD: {e}")
                results["macd_database"] = False

            # 3. ボリンジャーバンドのテスト（新しいカラム）
            print("\n3. ボリンジャーバンドの保存・取得テスト (BB)")
            try:
                bb_result = await self.service.calculate_and_save_technical_indicator(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_type="BB",
                    period=20,
                    limit=50
                )

                if bb_result and bb_result.get("success"):
                    calculated = bb_result.get("calculated_count", 0)
                    saved = bb_result.get("saved_count", 0)
                    print(f"  ✅ BB: {calculated}件計算, {saved}件保存（upper_band, lower_band含む）")
                    results["bb_database"] = True
                else:
                    print("  ❌ BB: データの保存に失敗")
                    results["bb_database"] = False

            except Exception as e:
                print(f"  ❌ BB: {e}")
                results["bb_database"] = False

            # 4. データ取得のテスト
            print("\n4. データ取得テスト")
            try:
                with SessionLocal() as session:
                    repo = TechnicalIndicatorRepository(session)

                    # 保存されたデータを取得
                    saved_data = repo.get_technical_indicator_data(symbol=symbol, timeframe=timeframe)

                    if saved_data and len(saved_data) > 0:
                        print(f"  ✅ データ取得: {len(saved_data)}件のデータを取得")

                        # 各指標タイプが含まれているかチェック
                        indicator_types = set(item.indicator_type for item in saved_data)
                        expected_types = {"SMA", "MACD", "BB"}

                        if expected_types.issubset(indicator_types):
                            print(f"  ✅ 指標タイプ: {indicator_types}")
                            results["data_retrieval"] = True
                        else:
                            missing = expected_types - indicator_types
                            print(f"  ❌ 不足している指標タイプ: {missing}")
                            results["data_retrieval"] = False
                    else:
                        print("  ❌ データ取得: データが見つかりません")
                        results["data_retrieval"] = False

            except Exception as e:
                print(f"  ❌ データ取得: {e}")
                results["data_retrieval"] = False

        except Exception as e:
            print(f"❌ データベース操作テストでエラー: {e}")
            results["database_setup"] = False

        return results

    async def test_api_endpoints(self) -> Dict[str, bool]:
        """API エンドポイントのテスト"""
        print("\n🌐 API エンドポイントのテスト")
        print("-" * 40)

        results = {}

        try:
            # 1. サポートされている指標の取得
            print("1. /technical-indicators/supported エンドポイント")
            try:
                from app.api.technical_indicators import get_supported_indicators

                response = await get_supported_indicators()

                if response and "data" in response:
                    supported_indicators = response["data"]["supported_indicators"]
                    default_indicators = response["data"]["default_indicators"]

                    expected_count = 12  # 期待される指標数
                    if len(supported_indicators) >= expected_count:
                        print(f"  ✅ サポート指標: {len(supported_indicators)}種類")
                        print(f"  ✅ デフォルト指標: {len(default_indicators)}種類")
                        results["api_supported"] = True
                    else:
                        print(f"  ❌ 指標数不足: {len(supported_indicators)}/{expected_count}")
                        results["api_supported"] = False
                else:
                    print("  ❌ レスポンス形式が不正")
                    results["api_supported"] = False

            except Exception as e:
                print(f"  ❌ API エラー: {e}")
                results["api_supported"] = False

            # 2. 指標計算エンドポイントのテスト
            print("\n2. 指標計算エンドポイントのテスト")
            try:
                # テストデータの準備
                test_data = self.create_test_data(50)

                # 複数の指標をテスト
                test_cases = [
                    ("SMA", 20),
                    ("MACD", 12),
                    ("BB", 20),
                    ("RSI", 14)
                ]

                for indicator_type, period in test_cases:
                    try:
                        result = await self.service.calculate_and_save_technical_indicator(
                            symbol="TEST_API",
                            timeframe="1h",
                            indicator_type=indicator_type,
                            period=period,
                            limit=50
                        )

                        if result and result.get("success"):
                            calculated = result.get("calculated_count", 0)
                            print(f"  ✅ {indicator_type}({period}): {calculated}件計算")
                            results[f"api_calc_{indicator_type}"] = True
                        else:
                            print(f"  ❌ {indicator_type}({period}): 計算失敗")
                            results[f"api_calc_{indicator_type}"] = False

                    except Exception as e:
                        print(f"  ❌ {indicator_type}({period}): {e}")
                        results[f"api_calc_{indicator_type}"] = False

            except Exception as e:
                print(f"  ❌ 指標計算テストでエラー: {e}")

        except Exception as e:
            print(f"❌ API テストでエラー: {e}")

        return results

async def run_comprehensive_tests():
    """包括的テストの実行"""
    print("🧪 テクニカル指標 包括的テストスイート")
    print("=" * 60)
    
    tester = TechnicalIndicatorTester()
    all_results = {}
    
    # 1. 計算メソッドのテスト
    calc_results = tester.test_calculation_methods()
    all_results.update(calc_results)
    
    # 2. エッジケースのテスト
    edge_results = tester.test_edge_cases()
    all_results.update(edge_results)
    
    # 3. データ処理ロジックのテスト
    processing_results = tester.test_data_processing_logic()
    all_results.update(processing_results)

    # 4. データベース操作のテスト
    database_results = await tester.test_database_operations()
    all_results.update(database_results)

    # 5. API エンドポイントのテスト
    api_results = await tester.test_api_endpoints()
    all_results.update(api_results)

    # 結果サマリー
    print("\n📋 テスト結果サマリー")
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
        print("🎉 全てのテストが成功しました！")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)
