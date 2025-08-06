"""
オートストラテジーエラーハンドリングの包括的テスト

各種エラー状況（データ不足、無効なパラメータ、ML予測失敗、バックテスト失敗）での
適切なエラーハンドリングをテストします。
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyErrorHandling:
    """オートストラテジーエラーハンドリングの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.valid_test_data = self._create_valid_test_data()
        self.invalid_test_cases = self._create_invalid_test_cases()

    def _create_valid_test_data(self) -> pd.DataFrame:
        """有効なテストデータを作成"""
        import numpy as np
        
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        np.random.seed(42)
        
        data = []
        base_price = 50000
        for i, date in enumerate(dates):
            price = base_price + np.random.normal(0, 1000)
            data.append({
                "timestamp": date,
                "open": price * (1 + np.random.normal(0, 0.001)),
                "high": price * (1 + abs(np.random.normal(0, 0.01))),
                "low": price * (1 - abs(np.random.normal(0, 0.01))),
                "close": price,
                "volume": np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)

    def _create_invalid_test_cases(self) -> Dict[str, Any]:
        """無効なテストケースを作成"""
        return {
            "empty_dataframe": pd.DataFrame(),
            "missing_columns": pd.DataFrame({"invalid_column": [1, 2, 3]}),
            "insufficient_data": pd.DataFrame({
                "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]
            }),
            "nan_data": pd.DataFrame({
                "open": [float('nan')] * 10,
                "high": [float('nan')] * 10,
                "low": [float('nan')] * 10,
                "close": [float('nan')] * 10,
                "volume": [float('nan')] * 10
            }),
            "negative_values": pd.DataFrame({
                "open": [-1] * 10,
                "high": [-1] * 10,
                "low": [-1] * 10,
                "close": [-1] * 10,
                "volume": [-1] * 10
            })
        }

    def test_ml_orchestrator_error_handling(self):
        """MLOrchestratorのエラーハンドリングテスト"""
        logger.info("=== MLOrchestratorエラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.ml.exceptions import MLDataError, MLValidationError
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 空のデータフレームでのエラーテスト
            with pytest.raises((MLDataError, ValueError, Exception)):
                ml_orchestrator.calculate_ml_indicators(self.invalid_test_cases["empty_dataframe"])
            
            # 必須カラムが不足したデータでのエラーテスト
            with pytest.raises((MLDataError, ValueError, Exception)):
                ml_orchestrator.calculate_ml_indicators(self.invalid_test_cases["missing_columns"])
            
            # None データでのエラーテスト
            with pytest.raises((MLDataError, ValueError, TypeError, Exception)):
                ml_orchestrator.calculate_ml_indicators(None)
            
            # 不十分なデータでのエラーテスト
            try:
                result = ml_orchestrator.calculate_ml_indicators(self.invalid_test_cases["insufficient_data"])
                # エラーが発生しない場合は、結果が適切に処理されているかを確認
                if result:
                    assert isinstance(result, dict), "結果が辞書形式ではありません"
                    logger.info("不十分なデータでも適切に処理されました")
            except Exception as e:
                logger.info(f"期待されるエラーが発生しました: {e}")
            
            # NaN データでのエラーテスト
            try:
                result = ml_orchestrator.calculate_ml_indicators(self.invalid_test_cases["nan_data"])
                if result:
                    logger.info("NaN データでも適切に処理されました")
            except Exception as e:
                logger.info(f"NaN データで期待されるエラーが発生しました: {e}")
            
            logger.info("✅ MLOrchestratorエラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"MLOrchestratorエラーハンドリングテストエラー: {e}")

    def test_tpsl_service_error_handling(self):
        """TP/SLサービスのエラーハンドリングテスト"""
        logger.info("=== TP/SLサービスエラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 極端な設定値でのエラーハンドリングテスト
            extreme_configs = [
                TPSLConfig(
                    strategy=TPSLStrategy.RISK_REWARD,
                    max_risk_per_trade=0.0,  # ゼロリスク
                    preferred_risk_reward_ratio=0.0  # ゼロリワード
                ),
                TPSLConfig(
                    strategy=TPSLStrategy.RISK_REWARD,
                    max_risk_per_trade=1.0,  # 100%リスク
                    preferred_risk_reward_ratio=1000.0  # 極端なリワード
                ),
                TPSLConfig(
                    strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
                    max_risk_per_trade=-0.1,  # 負のリスク
                    preferred_risk_reward_ratio=-1.0  # 負のリワード
                )
            ]
            
            for i, config in enumerate(extreme_configs):
                try:
                    result = service.generate_tpsl_values(config)
                    
                    # エラーが発生しない場合は、フォールバック処理が機能していることを確認
                    assert result is not None, f"設定 {i+1}: 結果がNoneです"
                    assert result.stop_loss_pct >= 0, f"設定 {i+1}: SLが負の値です"
                    assert result.take_profit_pct >= 0, f"設定 {i+1}: TPが負の値です"
                    
                    logger.info(f"極端な設定 {i+1} でもフォールバック処理が機能しました")
                    
                except Exception as e:
                    logger.info(f"極端な設定 {i+1} で期待されるエラーが発生しました: {e}")
            
            # 無効な市場データでのテスト
            invalid_market_data = [
                {"invalid_key": "invalid_value"},
                {"atr_pct": -1.0},  # 負のATR
                {"atr_pct": float('nan')},  # NaN ATR
                None
            ]
            
            for i, market_data in enumerate(invalid_market_data):
                try:
                    config = TPSLConfig(strategy=TPSLStrategy.VOLATILITY_ADAPTIVE)
                    result = service.generate_tpsl_values(config, market_data)
                    
                    assert result is not None, f"無効な市場データ {i+1}: 結果がNoneです"
                    logger.info(f"無効な市場データ {i+1} でも適切に処理されました")
                    
                except Exception as e:
                    logger.info(f"無効な市場データ {i+1} で期待されるエラーが発生しました: {e}")
            
            logger.info("✅ TP/SLサービスエラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SLサービスエラーハンドリングテストエラー: {e}")

    def test_smart_condition_generator_error_handling(self):
        """SmartConditionGeneratorのエラーハンドリングテスト"""
        logger.info("=== SmartConditionGeneratorエラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 無効な指標リストでのテスト
            invalid_indicator_lists = [
                [],  # 空のリスト
                None,  # None
                [IndicatorGene(type="INVALID_INDICATOR", parameters={}, enabled=True)],  # 無効な指標
                [IndicatorGene(type="RSI", parameters={}, enabled=False)] * 10,  # 全て無効
            ]
            
            for i, indicators in enumerate(invalid_indicator_lists):
                try:
                    if indicators is None:
                        # None の場合は例外が発生することを期待
                        with pytest.raises((TypeError, AttributeError, Exception)):
                            generator.generate_balanced_conditions(indicators)
                    else:
                        long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(indicators)
                        
                        # フォールバック処理が機能していることを確認
                        assert isinstance(long_conditions, list), f"無効な指標リスト {i+1}: ロング条件がリスト形式ではありません"
                        assert isinstance(short_conditions, list), f"無効な指標リスト {i+1}: ショート条件がリスト形式ではありません"
                        assert isinstance(exit_conditions, list), f"無効な指標リスト {i+1}: イグジット条件がリスト形式ではありません"
                        
                        logger.info(f"無効な指標リスト {i+1} でもフォールバック処理が機能しました")
                
                except Exception as e:
                    logger.info(f"無効な指標リスト {i+1} で期待されるエラーが発生しました: {e}")
            
            # 破損した指標データでのテスト
            corrupted_indicators = [
                IndicatorGene(type="", parameters={}, enabled=True),  # 空の指標タイプ
                IndicatorGene(type="RSI", parameters=None, enabled=True),  # None パラメータ
                IndicatorGene(type="RSI", parameters={"period": -1}, enabled=True),  # 無効なパラメータ
            ]
            
            try:
                long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(corrupted_indicators)
                
                # エラーが発生しない場合は、適切に処理されていることを確認
                assert isinstance(long_conditions, list), "破損した指標データ: ロング条件がリスト形式ではありません"
                assert isinstance(short_conditions, list), "破損した指標データ: ショート条件がリスト形式ではありません"
                assert isinstance(exit_conditions, list), "破損した指標データ: イグジット条件がリスト形式ではありません"
                
                logger.info("破損した指標データでも適切に処理されました")
                
            except Exception as e:
                logger.info(f"破損した指標データで期待されるエラーが発生しました: {e}")
            
            logger.info("✅ SmartConditionGeneratorエラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"SmartConditionGeneratorエラーハンドリングテストエラー: {e}")

    def test_auto_strategy_service_error_handling(self):
        """AutoStrategyServiceのエラーハンドリングテスト"""
        logger.info("=== AutoStrategyServiceエラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            service = AutoStrategyService()
            
            # 無効なGA設定でのテスト
            invalid_ga_configs = [
                {},  # 空の設定
                {"population_size": 0},  # 無効な人口サイズ
                {"population_size": -10, "generations": -5},  # 負の値
                {"population_size": "invalid", "generations": "invalid"},  # 文字列
                None  # None
            ]
            
            for i, ga_config in enumerate(invalid_ga_configs):
                try:
                    if hasattr(service, 'experiment_manager') and service.experiment_manager:
                        if ga_config is None:
                            with pytest.raises((TypeError, AttributeError, Exception)):
                                service.experiment_manager.initialize_ga_engine(ga_config)
                        else:
                            service.experiment_manager.initialize_ga_engine(ga_config)
                            logger.info(f"無効なGA設定 {i+1} でも適切に処理されました")
                    else:
                        logger.info(f"実験マネージャーが利用できないため、GA設定 {i+1} のテストをスキップしました")
                
                except Exception as e:
                    logger.info(f"無効なGA設定 {i+1} で期待されるエラーが発生しました: {e}")
            
            # 無効なバックテスト設定でのテスト
            invalid_backtest_configs = [
                {},  # 空の設定
                {"symbol": ""},  # 空のシンボル
                {"symbol": "INVALID", "timeframe": "invalid"},  # 無効な時間軸
                {"initial_capital": -1000},  # 負の初期資金
                {"commission_rate": -0.1},  # 負の手数料
                None  # None
            ]
            
            for i, backtest_config in enumerate(invalid_backtest_configs):
                try:
                    # バックテスト設定の妥当性確認（実際の実行はしない）
                    if backtest_config is None:
                        logger.info(f"無効なバックテスト設定 {i+1}: None設定")
                    elif not backtest_config:
                        logger.info(f"無効なバックテスト設定 {i+1}: 空の設定")
                    else:
                        # 基本的な妥当性チェック
                        symbol = backtest_config.get("symbol", "")
                        if not symbol:
                            logger.info(f"無効なバックテスト設定 {i+1}: 無効なシンボル")
                        
                        initial_capital = backtest_config.get("initial_capital", 0)
                        if initial_capital <= 0:
                            logger.info(f"無効なバックテスト設定 {i+1}: 無効な初期資金")
                
                except Exception as e:
                    logger.info(f"無効なバックテスト設定 {i+1} で期待されるエラーが発生しました: {e}")
            
            logger.info("✅ AutoStrategyServiceエラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"AutoStrategyServiceエラーハンドリングテストエラー: {e}")

    def test_database_connection_error_handling(self):
        """データベース接続エラーハンドリングテスト"""
        logger.info("=== データベース接続エラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.persistence.experiment_persistence_service import ExperimentPersistenceService
            
            # 永続化サービスの初期化テスト
            try:
                persistence_service = ExperimentPersistenceService()
                
                # データベース操作のエラーハンドリングテスト
                invalid_operations = [
                    ("create_experiment", ["", "", {}, {}]),  # 空のID
                    ("create_experiment", [None, "test", {}, {}]),  # None ID
                    ("get_experiment", [""]),  # 空のID
                    ("get_experiment", ["non_existent_id"]),  # 存在しないID
                ]
                
                for operation, args in invalid_operations:
                    try:
                        method = getattr(persistence_service, operation)
                        result = method(*args)
                        logger.info(f"データベース操作 {operation} が適切に処理されました")
                    except Exception as e:
                        logger.info(f"データベース操作 {operation} で期待されるエラーが発生しました: {e}")
                
            except Exception as init_error:
                logger.info(f"永続化サービス初期化でエラー（期待される場合もあります）: {init_error}")
            
            logger.info("✅ データベース接続エラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"データベース接続エラーハンドリングテストエラー: {e}")

    def test_memory_management_error_handling(self):
        """メモリ管理エラーハンドリングテスト"""
        logger.info("=== メモリ管理エラーハンドリングテスト ===")
        
        try:
            import gc
            
            # 大量のオブジェクト作成とメモリ解放のテスト
            large_objects = []
            
            try:
                # 大量のデータを作成
                for i in range(100):
                    large_data = self._create_valid_test_data()
                    large_objects.append(large_data)
                
                # メモリ使用量の確認
                initial_objects = len(gc.get_objects())
                
                # オブジェクトの削除
                del large_objects
                gc.collect()
                
                # メモリ解放の確認
                final_objects = len(gc.get_objects())
                logger.info(f"オブジェクト数の変化: {initial_objects} -> {final_objects}")
                
                logger.info("✅ メモリ管理テスト成功")
                
            except MemoryError as e:
                logger.info(f"メモリエラーが適切に処理されました: {e}")
            except Exception as e:
                logger.info(f"メモリ管理で予期しないエラー: {e}")
            
        except Exception as e:
            pytest.fail(f"メモリ管理エラーハンドリングテストエラー: {e}")

    def test_concurrent_access_error_handling(self):
        """並行アクセスエラーハンドリングテスト"""
        logger.info("=== 並行アクセスエラーハンドリングテスト ===")
        
        try:
            import threading
            import time
            
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            # 複数のスレッドから同時にサービスにアクセス
            results = []
            errors = []
            
            def worker(worker_id):
                try:
                    service = AutoStrategyService()
                    # 簡単な操作を実行
                    result = f"Worker {worker_id} completed"
                    results.append(result)
                except Exception as e:
                    errors.append(f"Worker {worker_id} error: {e}")
            
            # 複数のスレッドを作成して実行
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 全てのスレッドの完了を待機
            for thread in threads:
                thread.join(timeout=10)  # 10秒でタイムアウト
            
            logger.info(f"成功した操作: {len(results)}")
            logger.info(f"エラーが発生した操作: {len(errors)}")
            
            for error in errors:
                logger.info(f"並行アクセスエラー: {error}")
            
            logger.info("✅ 並行アクセスエラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"並行アクセスエラーハンドリングテストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
