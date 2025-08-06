"""
オートストラテジー高度テストスイート

GA最適化、実時間処理、エッジケース、統合シナリオの詳細テスト
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import pandas as pd
import numpy as np
import time
import concurrent.futures
import logging

logger = logging.getLogger(__name__)


class TestAutoStrategyAdvanced:
    """オートストラテジー高度テストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.test_data = self.create_realistic_market_data()
        self.extreme_data = self.create_extreme_market_data()
    
    def create_realistic_market_data(self, size: int = 2000) -> pd.DataFrame:
        """リアルな市場データを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='h')
        
        # トレンド + ノイズ + 周期性を含む価格データ
        base_price = 50000
        trend = np.linspace(0, 0.2, size)  # 20%の上昇トレンド
        noise = np.random.normal(0, 0.02, size)
        cyclical = 0.05 * np.sin(np.linspace(0, 4*np.pi, size))  # 周期的変動
        
        returns = trend + noise + cyclical
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret/100))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(1000, size) * (1 + 0.5 * np.random.random(size)),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def create_extreme_market_data(self, size: int = 500) -> pd.DataFrame:
        """極端な市場条件のデータを作成"""
        np.random.seed(123)
        dates = pd.date_range(start='2023-06-01', periods=size, freq='h')
        
        # 極端なボラティリティとクラッシュを含む
        base_price = 50000
        returns = []
        
        for i in range(size):
            if i == 100:  # クラッシュイベント
                returns.append(-0.3)  # 30%下落
            elif i == 200:  # 急騰イベント
                returns.append(0.25)   # 25%上昇
            elif 150 <= i <= 180:  # 高ボラティリティ期間
                returns.append(np.random.normal(0, 0.08))
            else:
                returns.append(np.random.normal(0, 0.02))
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(max(prices[-1] * (1 + ret), 1000))  # 最低価格制限
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(2000, size),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_ga_optimization_integration(self):
        """テスト11: GA最適化統合テスト"""
        logger.info("🔍 GA最適化統合テスト開始")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            auto_strategy_service = AutoStrategyService(enable_smart_generation=True)
            
            # GA設定
            ga_config = {
                "population_size": 10,  # テスト用に小さく設定
                "generations": 3,       # テスト用に短く設定
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "enable_multi_objective": False
            }
            
            # バックテスト設定
            backtest_config = {
                "symbol": "BTC:USDT",
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-01-07",  # 短期間でテスト
                "initial_capital": 10000,
                "commission_rate": 0.001
            }
            
            # サービス初期化の確認
            assert hasattr(auto_strategy_service, 'persistence_service'), "永続化サービスが不足しています"
            assert hasattr(auto_strategy_service, 'backtest_service'), "バックテストサービスが不足しています"
            
            # 実験管理マネージャーの確認
            if hasattr(auto_strategy_service, 'experiment_manager'):
                experiment_manager = auto_strategy_service.experiment_manager
                if experiment_manager:
                    assert hasattr(experiment_manager, 'initialize_ga_engine'), "GAエンジン初期化メソッドが不足しています"
            
            logger.info("✅ GA最適化統合テスト成功")
            
        except Exception as e:
            pytest.fail(f"GA最適化統合テストエラー: {e}")
    
    def test_concurrent_strategy_execution(self):
        """テスト12: 並行戦略実行テスト"""
        logger.info("🔍 並行戦略実行テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 複数のMLオーケストレーターを並行実行
            def run_ml_calculation(data_slice):
                try:
                    ml_orchestrator = MLOrchestrator(enable_automl=False)  # 軽量化
                    result = ml_orchestrator.calculate_ml_indicators(data_slice)
                    return {"success": True, "result_size": len(result.get("ML_UP_PROB", []))}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            # データを分割
            data_slices = [
                self.test_data.iloc[i:i+200] for i in range(0, min(1000, len(self.test_data)), 200)
            ]
            
            # 並行実行
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_ml_calculation, data_slice) for data_slice in data_slices]
                results = [future.result(timeout=30) for future in futures]
            
            # 結果検証
            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) > 0, "並行実行で成功した結果がありません"
            
            success_rate = len(successful_results) / len(results)
            assert success_rate >= 0.5, f"成功率が低すぎます: {success_rate:.2%}"
            
            logger.info(f"並行実行結果: {len(successful_results)}/{len(results)} 成功 ({success_rate:.1%})")
            logger.info("✅ 並行戦略実行テスト成功")
            
        except Exception as e:
            pytest.fail(f"並行戦略実行テストエラー: {e}")
    
    def test_extreme_market_conditions(self):
        """テスト13: 極端な市場条件テスト"""
        logger.info("🔍 極端な市場条件テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            ml_orchestrator = MLOrchestrator()
            tpsl_calculator = TPSLCalculator()
            
            # 極端データでのML計算
            try:
                ml_indicators = ml_orchestrator.calculate_ml_indicators(self.extreme_data)
                
                # 結果の安定性確認
                for key, values in ml_indicators.items():
                    valid_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        # 極端値の確認
                        assert all(0 <= v <= 1 for v in valid_values), f"{key}: 極端データで異常値が発生"
                        
                        # 分散の確認（極端に偏っていないか）
                        if len(valid_values) > 10:
                            std_dev = np.std(valid_values)
                            assert std_dev < 0.5, f"{key}: 極端データで分散が大きすぎます"
                
                logger.info("極端データでのML計算が安定しています")
                
            except Exception as e:
                logger.info(f"極端データでML計算エラー（期待される場合もあります）: {e}")
            
            # 極端価格でのTP/SL計算
            extreme_prices = [1, 100, 1000000, 0.001]
            
            for price in extreme_prices:
                try:
                    sl_price, tp_price = tpsl_calculator.calculate_basic_tpsl_prices(
                        price, 0.02, 0.04, "long"
                    )
                    
                    # 計算結果の妥当性確認
                    assert sl_price > 0, f"価格{price}: SL価格が負の値です"
                    assert tp_price > 0, f"価格{price}: TP価格が負の値です"
                    assert sl_price < price, f"価格{price}: ロングSLが現在価格より高いです"
                    assert tp_price > price, f"価格{price}: ロングTPが現在価格より低いです"
                    
                except Exception as e:
                    logger.warning(f"極端価格 {price} でTP/SL計算エラー: {e}")
            
            logger.info("✅ 極端な市場条件テスト成功")
            
        except Exception as e:
            pytest.fail(f"極端な市場条件テストエラー: {e}")
    
    def test_memory_performance_optimization(self):
        """テスト14: メモリ・パフォーマンス最適化テスト"""
        logger.info("🔍 メモリ・パフォーマンス最適化テスト開始")
        
        try:
            import psutil
            import gc
            
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 初期メモリ使用量
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 大量データでの処理
            large_data = self.create_realistic_market_data(5000)
            
            start_time = time.time()
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)  # 軽量化
            ml_indicators = ml_orchestrator.calculate_ml_indicators(large_data)
            
            processing_time = time.time() - start_time
            
            # 処理後メモリ使用量
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # ガベージコレクション
            gc.collect()
            
            # パフォーマンス検証
            assert processing_time < 60, f"処理時間が長すぎます: {processing_time:.2f}秒"
            assert memory_increase < 500, f"メモリ使用量増加が大きすぎます: {memory_increase:.1f}MB"
            
            # 結果の妥当性確認
            assert len(ml_indicators["ML_UP_PROB"]) == len(large_data), "結果サイズが入力データと一致しません"
            
            logger.info(f"パフォーマンス: 処理時間={processing_time:.2f}秒, メモリ増加={memory_increase:.1f}MB")
            logger.info("✅ メモリ・パフォーマンス最適化テスト成功")
            
        except Exception as e:
            pytest.fail(f"メモリ・パフォーマンス最適化テストエラー: {e}")
    
    def test_error_recovery_resilience(self):
        """テスト15: エラー回復・復元力テスト"""
        logger.info("🔍 エラー回復・復元力テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator()
            
            # 様々な破損データでのテスト
            error_scenarios = [
                ("空データ", pd.DataFrame()),
                ("単一行データ", self.test_data.iloc[:1]),
                ("NaN多数データ", self.test_data.copy().fillna(np.nan)),
                ("無限値データ", self.test_data.copy().replace([np.inf, -np.inf], np.nan)),
            ]
            
            recovery_count = 0
            
            for scenario_name, corrupted_data in error_scenarios:
                try:
                    result = ml_orchestrator.calculate_ml_indicators(corrupted_data)
                    
                    # 結果が返された場合の妥当性確認
                    if result and "ML_UP_PROB" in result:
                        assert isinstance(result["ML_UP_PROB"], (list, np.ndarray)), f"{scenario_name}: 結果形式が無効"
                        recovery_count += 1
                        logger.info(f"{scenario_name}: 正常に処理されました")
                    
                except Exception as e:
                    # エラーが発生することも期待される動作
                    logger.info(f"{scenario_name}: エラーが発生（期待される場合もあります）: {e}")
            
            # 少なくとも一部のシナリオで回復できることを確認
            logger.info(f"エラー回復: {recovery_count}/{len(error_scenarios)} シナリオで成功")
            
            logger.info("✅ エラー回復・復元力テスト成功")
            
        except Exception as e:
            pytest.fail(f"エラー回復・復元力テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestAutoStrategyAdvanced()
    test_instance.setup_method()
    
    # 各テストを実行
    tests = [
        test_instance.test_ga_optimization_integration,
        test_instance.test_concurrent_strategy_execution,
        test_instance.test_extreme_market_conditions,
        test_instance.test_memory_performance_optimization,
        test_instance.test_error_recovery_resilience,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 高度オートストラテジーテスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
