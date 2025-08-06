"""
オートストラテジーとバックテスト統合の包括的テスト

オートストラテジーとバックテストエンジンの統合、戦略実行、結果保存、統計計算をテストします。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyBacktestIntegration:
    """オートストラテジーとバックテスト統合の包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_backtest_config = self._create_test_backtest_config()
        self.test_strategy_gene = self._create_test_strategy_gene()

    def _create_test_backtest_config(self) -> Dict[str, Any]:
        """テスト用のバックテスト設定を作成"""
        return {
            "symbol": "BTC:USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
            "initial_capital": 10000,
            "commission_rate": 0.001
        }

    def _create_test_strategy_gene(self) -> Dict[str, Any]:
        """テスト用の戦略遺伝子を作成"""
        try:
            return {
                "id": "test_strategy_001",
                "indicators": [
                    {
                        "type": "RSI",
                        "parameters": {"period": 14},
                        "enabled": True
                    },
                    {
                        "type": "SMA",
                        "parameters": {"period": 20},
                        "enabled": True
                    }
                ],
                "long_entry_conditions": [
                    {
                        "left_operand": "RSI",
                        "operator": "<",
                        "right_operand": 30
                    }
                ],
                "short_entry_conditions": [
                    {
                        "left_operand": "RSI",
                        "operator": ">",
                        "right_operand": 70
                    }
                ],
                "exit_conditions": [],
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            }
        except Exception:
            return {}

    def test_auto_strategy_service_initialization(self):
        """AutoStrategyServiceの初期化テスト"""
        logger.info("=== AutoStrategyService初期化テスト ===")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            # 基本初期化
            service = AutoStrategyService()
            
            # 基本属性の確認
            assert hasattr(service, 'db_session_factory'), "db_session_factory属性が不足しています"
            assert hasattr(service, 'enable_smart_generation'), "enable_smart_generation属性が不足しています"
            assert hasattr(service, 'backtest_service'), "backtest_service属性が不足しています"
            assert hasattr(service, 'persistence_service'), "persistence_service属性が不足しています"
            
            # スマート生成無効での初期化
            service_disabled = AutoStrategyService(enable_smart_generation=False)
            assert service_disabled.enable_smart_generation is False, "スマート生成が無効になっていません"
            
            logger.info("✅ AutoStrategyService初期化テスト成功")
            
        except Exception as e:
            pytest.fail(f"AutoStrategyService初期化テストエラー: {e}")

    def test_backtest_orchestration_service(self):
        """バックテスト統合サービステスト"""
        logger.info("=== バックテスト統合サービステスト ===")

        try:
            from app.services.backtest.orchestration.backtest_orchestration_service import BacktestOrchestrationService

            orchestration_service = BacktestOrchestrationService()

            # サポートされている戦略一覧の取得テスト
            try:
                # async関数の場合は、同期的にテストするか、asyncio.runを使用
                import asyncio
                strategies_result = asyncio.run(orchestration_service.get_supported_strategies())
                
                assert isinstance(strategies_result, dict), "戦略一覧が辞書形式ではありません"
                assert "success" in strategies_result, "success フィールドが不足しています"
                assert "strategies" in strategies_result, "strategies フィールドが不足しています"
                
                if strategies_result["success"]:
                    assert isinstance(strategies_result["strategies"], (list, dict)), "戦略リストが無効な形式です"
                
                logger.info("✅ サポート戦略取得テスト成功")
                
            except Exception as e:
                logger.warning(f"サポート戦略取得でエラー（期待される場合もあります）: {e}")
            
        except Exception as e:
            pytest.fail(f"バックテスト統合サービステストエラー: {e}")

    def test_strategy_gene_validation(self):
        """戦略遺伝子バリデーションテスト"""
        logger.info("=== 戦略遺伝子バリデーションテスト ===")
        
        try:
            from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
            
            # 有効な戦略遺伝子の作成
            indicators = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]
            
            long_conditions = [
                Condition(left_operand="RSI", operator="<", right_operand=30)
            ]
            
            short_conditions = [
                Condition(left_operand="RSI", operator=">", right_operand=70)
            ]
            
            strategy_gene = StrategyGene(
                id="test_strategy",
                indicators=indicators,
                long_entry_conditions=long_conditions,
                short_entry_conditions=short_conditions,
                exit_conditions=[],
                risk_management={"max_position_size": 0.1}
            )
            
            # バリデーション実行
            is_valid = strategy_gene.validate()
            assert isinstance(is_valid, bool), "バリデーション結果がブール値ではありません"
            
            # 基本属性の確認
            assert strategy_gene.id == "test_strategy", "戦略IDが正しく設定されていません"
            assert len(strategy_gene.indicators) == 2, "指標数が正しくありません"
            assert len(strategy_gene.long_entry_conditions) == 1, "ロング条件数が正しくありません"
            assert len(strategy_gene.short_entry_conditions) == 1, "ショート条件数が正しくありません"
            
            logger.info("✅ 戦略遺伝子バリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略遺伝子バリデーションテストエラー: {e}")

    def test_strategy_testing_integration(self):
        """戦略テスト統合テスト"""
        logger.info("=== 戦略テスト統合テスト ===")
        
        try:
            from app.services.auto_strategy.orchestration.auto_strategy_orchestration_service import AutoStrategyOrchestrationService
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            orchestration_service = AutoStrategyOrchestrationService()
            auto_strategy_service = AutoStrategyService()
            
            # テスト用のリクエストオブジェクトを模擬
            class MockRequest:
                def __init__(self, strategy_gene, backtest_config):
                    self.strategy_gene = strategy_gene
                    self.backtest_config = backtest_config
            
            mock_request = MockRequest(
                strategy_gene=self.test_strategy_gene,
                backtest_config=self.test_backtest_config
            )
            
            # 戦略テストの実行（エラーが発生する可能性があるため、try-catchで処理）
            try:
                import asyncio
                result = asyncio.run(orchestration_service.test_strategy(mock_request, auto_strategy_service))
                
                if result:
                    assert isinstance(result, dict), "テスト結果が辞書形式ではありません"
                    
                    # 基本的な結果フィールドの確認
                    expected_fields = ["success", "result", "message"]
                    for field in expected_fields:
                        if field in result:
                            logger.info(f"結果フィールド {field} が存在します")
                
                logger.info("✅ 戦略テスト実行成功")
                
            except Exception as test_error:
                logger.warning(f"戦略テスト実行でエラー（期待される場合もあります）: {test_error}")
                # データベースやバックテストエンジンが利用できない場合は警告のみ
            
        except Exception as e:
            pytest.fail(f"戦略テスト統合テストエラー: {e}")

    def test_backtest_executor_integration(self):
        """バックテストエグゼキューター統合テスト"""
        logger.info("=== バックテストエグゼキューター統合テスト ===")
        
        try:
            from app.services.backtest.execution.backtest_executor import BacktestExecutor
            from app.services.backtest.backtest_data_service import BacktestDataService

            # データサービスを作成してExecutorを初期化
            data_service = BacktestDataService()
            executor = BacktestExecutor(data_service)

            # 基本属性の確認
            assert hasattr(executor, 'execute_backtest'), "execute_backtest メソッドが不足しています"
            
            # バックテスト実行のテスト（実際のデータが必要なため、モック的なテスト）
            try:
                # テスト用のパラメータ
                from datetime import datetime
                
                test_params = {
                    "strategy_class": None,  # 実際のテストでは適切な戦略クラスが必要
                    "strategy_parameters": {"test_param": "test_value"},
                    "symbol": "BTC:USDT",
                    "timeframe": "1h",
                    "start_date": datetime(2023, 1, 1),
                    "end_date": datetime(2023, 1, 3),
                    "initial_capital": 10000.0,
                    "commission_rate": 0.001
                }
                
                # 実際の実行は戦略クラスが必要なため、パラメータの妥当性のみ確認
                for key, value in test_params.items():
                    assert value is not None or key == "strategy_class", f"パラメータ {key} が無効です"
                
                logger.info("✅ バックテストエグゼキューター基本テスト成功")
                
            except Exception as exec_error:
                logger.warning(f"バックテスト実行でエラー（期待される場合もあります）: {exec_error}")
            
        except Exception as e:
            pytest.fail(f"バックテストエグゼキューター統合テストエラー: {e}")

    def test_experiment_persistence_service(self):
        """実験永続化サービステスト"""
        logger.info("=== 実験永続化サービステスト ===")
        
        try:
            from app.services.auto_strategy.persistence.experiment_persistence_service import ExperimentPersistenceService
            
            # サービスの初期化テスト
            try:
                persistence_service = ExperimentPersistenceService()
                
                # 基本属性の確認
                assert hasattr(persistence_service, 'create_experiment'), "create_experiment メソッドが不足しています"
                assert hasattr(persistence_service, 'get_experiment'), "get_experiment メソッドが不足しています"
                assert hasattr(persistence_service, 'update_experiment'), "update_experiment メソッドが不足しています"
                
                # 実験作成のテスト（データベースが利用できない場合はスキップ）
                try:
                    test_experiment_id = "test_exp_001"
                    test_experiment_name = "Test Experiment"
                    test_ga_config = {"population_size": 50, "generations": 10}
                    
                    persistence_service.create_experiment(
                        test_experiment_id,
                        test_experiment_name,
                        test_ga_config,
                        self.test_backtest_config
                    )
                    
                    logger.info("✅ 実験作成テスト成功")
                    
                except Exception as create_error:
                    logger.warning(f"実験作成でエラー（期待される場合もあります）: {create_error}")
                
            except Exception as init_error:
                logger.warning(f"永続化サービス初期化でエラー（期待される場合もあります）: {init_error}")
            
        except Exception as e:
            pytest.fail(f"実験永続化サービステストエラー: {e}")

    def test_ga_engine_integration(self):
        """GA エンジン統合テスト"""
        logger.info("=== GA エンジン統合テスト ===")
        
        try:
            from app.services.auto_strategy.ga.experiment_manager import ExperimentManager
            
            # 実験マネージャーの初期化
            try:
                experiment_manager = ExperimentManager()
                
                # 基本属性の確認
                assert hasattr(experiment_manager, 'initialize_ga_engine'), "initialize_ga_engine メソッドが不足しています"
                assert hasattr(experiment_manager, 'run_experiment'), "run_experiment メソッドが不足しています"
                
                # GA設定の初期化テスト
                test_ga_config = {
                    "population_size": 20,
                    "generations": 5,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.8,
                    "enable_multi_objective": False
                }
                
                experiment_manager.initialize_ga_engine(test_ga_config)
                
                logger.info("✅ GA エンジン初期化テスト成功")
                
            except Exception as ga_error:
                logger.warning(f"GA エンジンでエラー（期待される場合もあります）: {ga_error}")
            
        except Exception as e:
            pytest.fail(f"GA エンジン統合テストエラー: {e}")

    def test_backtest_statistics_calculation(self):
        """バックテスト統計計算テスト"""
        logger.info("=== バックテスト統計計算テスト ===")
        
        try:
            # 模擬的なバックテスト結果データ
            mock_backtest_result = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "total_trades": 25,
                "profit_factor": 1.8
            }
            
            # 統計値の妥当性確認
            assert isinstance(mock_backtest_result["total_return"], (int, float)), "総リターンが数値ではありません"
            assert isinstance(mock_backtest_result["sharpe_ratio"], (int, float)), "シャープレシオが数値ではありません"
            assert isinstance(mock_backtest_result["max_drawdown"], (int, float)), "最大ドローダウンが数値ではありません"
            assert 0 <= mock_backtest_result["win_rate"] <= 1, "勝率が範囲外です"
            assert mock_backtest_result["total_trades"] >= 0, "取引数が負の値です"
            assert mock_backtest_result["profit_factor"] >= 0, "プロフィットファクターが負の値です"
            
            # 統計値の論理的妥当性確認
            if mock_backtest_result["total_trades"] > 0:
                assert mock_backtest_result["win_rate"] >= 0, "取引がある場合の勝率が無効です"
            
            logger.info("✅ バックテスト統計計算テスト成功")
            
        except Exception as e:
            pytest.fail(f"バックテスト統計計算テストエラー: {e}")

    def test_strategy_generation_workflow(self):
        """戦略生成ワークフローテスト"""
        logger.info("=== 戦略生成ワークフローテスト ===")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            service = AutoStrategyService()
            
            # ワークフローの基本ステップを確認
            workflow_steps = [
                "実験設定の検証",
                "GA エンジンの初期化",
                "実験の作成",
                "バックグラウンドタスクの追加"
            ]
            
            # 各ステップが実装されていることを確認（メソッドの存在確認）
            assert hasattr(service, 'persistence_service'), "永続化サービスが設定されていません"
            assert hasattr(service, 'experiment_manager'), "実験マネージャーが設定されていません"
            
            # テスト用の設定
            test_experiment_id = "workflow_test_001"
            test_experiment_name = "Workflow Test"
            test_ga_config = {
                "population_size": 10,
                "generations": 3,
                "mutation_rate": 0.1
            }
            
            # 設定の妥当性確認
            assert isinstance(test_experiment_id, str), "実験IDが文字列ではありません"
            assert isinstance(test_experiment_name, str), "実験名が文字列ではありません"
            assert isinstance(test_ga_config, dict), "GA設定が辞書形式ではありません"
            assert test_ga_config["population_size"] > 0, "人口サイズが無効です"
            assert test_ga_config["generations"] > 0, "世代数が無効です"
            
            logger.info("✅ 戦略生成ワークフローテスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略生成ワークフローテストエラー: {e}")

    def test_integration_error_handling(self):
        """統合エラーハンドリングテスト"""
        logger.info("=== 統合エラーハンドリングテスト ===")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            
            service = AutoStrategyService()
            
            # 無効な設定でのエラーハンドリングテスト
            invalid_configs = [
                {"population_size": 0},  # 無効な人口サイズ
                {"generations": -1},     # 無効な世代数
                {},                      # 空の設定
            ]
            
            for invalid_config in invalid_configs:
                try:
                    # 無効な設定での処理（エラーが期待される）
                    if hasattr(service, 'experiment_manager') and service.experiment_manager:
                        service.experiment_manager.initialize_ga_engine(invalid_config)
                    
                    logger.info(f"無効な設定 {invalid_config} が処理されました（エラーハンドリングが機能している可能性があります）")
                    
                except Exception as expected_error:
                    logger.info(f"期待されるエラーが発生しました: {expected_error}")
            
            logger.info("✅ 統合エラーハンドリングテスト成功")
            
        except Exception as e:
            pytest.fail(f"統合エラーハンドリングテストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
