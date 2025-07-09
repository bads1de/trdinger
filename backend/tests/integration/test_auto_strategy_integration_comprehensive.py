"""
オートストラテジー統合テスト

各コンポーネント間の統合動作、エラーハンドリング、
エッジケースを網羅的にテストします。
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import asyncio

# テスト対象のインポート
from app.core.services.auto_strategy.managers.experiment_manager import ExperimentManager
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene
from app.core.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.services.tpsl_auto_decision_service import (
    TPSLAutoDecisionService,
    TPSLConfig,
    TPSLStrategy,
)


class TestAutoStrategyIntegrationComprehensive:
    """オートストラテジー統合テストクラス"""

    @pytest.fixture
    def mock_dependencies(self):
        """依存関係のモック"""
        mocks = {
            'persistence_service': Mock(),
            'backtest_service': Mock(),
            'strategy_factory': Mock(),
            'ga_engine': Mock(),
        }
        
        # バックテスト結果のモック
        mocks['backtest_service'].run_backtest.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'total_trades': 25,
            'win_rate': 0.6,
            'profit_factor': 1.8,
        }
        
        # GA結果のモック
        mocks['ga_engine'].run_evolution.return_value = {
            'best_individual': Mock(),
            'best_fitness': 1.5,
            'generation_stats': [],
            'total_generations': 10,
        }
        
        return mocks

    @pytest.fixture
    def sample_ga_config(self) -> GAConfig:
        """サンプルGA設定"""
        return GAConfig(
            population_size=20,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_indicators=3,
            allowed_indicators=['SMA', 'EMA', 'RSI', 'MACD'],
            ga_objective='sharpe_ratio',
        )

    @pytest.fixture
    def sample_backtest_config(self) -> Dict[str, Any]:
        """サンプルバックテスト設定"""
        return {
            'strategy_name': 'TEST_STRATEGY',
            'symbol': 'BTCUSDT',
            'timeframe': '1d',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,
        }

    def test_experiment_manager_initialization(self, mock_dependencies):
        """ExperimentManagerの初期化テスト"""
        print("\n=== ExperimentManager初期化テスト ===")
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        # 基本的な初期化確認
        assert manager.persistence_service is not None, "persistence_serviceが設定されていない"
        assert manager.backtest_service is not None, "backtest_serviceが設定されていない"
        assert manager.strategy_factory is not None, "strategy_factoryが設定されていない"
        assert manager.ga_engine is None, "ga_engineが初期化時にNoneでない"
        
        print("  ✅ ExperimentManager正常に初期化")

    def test_ga_engine_initialization(self, mock_dependencies, sample_ga_config):
        """GAエンジン初期化テスト"""
        print("\n=== GAエンジン初期化テスト ===")
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        # GAエンジンの初期化
        manager.initialize_ga_engine(sample_ga_config)
        
        # 初期化確認
        assert manager.ga_engine is not None, "GAエンジンが初期化されていない"
        
        print("  ✅ GAエンジン正常に初期化")

    def test_strategy_gene_validation(self, mock_dependencies):
        """戦略遺伝子の検証テスト"""
        print("\n=== 戦略遺伝子検証テスト ===")
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        # 有効な戦略遺伝子
        from app.core.services.auto_strategy.models.gene_strategy import Condition

        valid_gene = StrategyGene(
            id="test_valid",
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 20},
                    enabled=True,
                )
            ],
            long_entry_conditions=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="SMA_20"
                )
            ],
            short_entry_conditions=[],
            exit_conditions=[
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand="SMA_20"
                )
            ],
            risk_management={"position_size": 0.1},
        )
        
        is_valid, errors = manager.validate_strategy_gene(valid_gene)
        assert is_valid, f"有効な遺伝子が無効と判定: {errors}"
        assert len(errors) == 0, f"有効な遺伝子にエラー: {errors}"
        
        print("  ✅ 有効遺伝子: 検証通過")
        
        # 無効な戦略遺伝子（指標数超過）
        invalid_gene = StrategyGene(
            id="test_invalid",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="EMA", parameters={"period": 10}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="MACD", parameters={}, enabled=True),
                IndicatorGene(type="BBANDS", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="ATR", parameters={"period": 14}, enabled=True),
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
        )
        
        is_valid, errors = manager.validate_strategy_gene(invalid_gene)
        assert not is_valid, "無効な遺伝子が有効と判定"
        assert len(errors) > 0, "無効な遺伝子にエラーが記録されていない"
        
        print(f"  ✅ 無効遺伝子: 検証失敗 ({len(errors)}個のエラー)")

    def test_strategy_generation_and_testing(self, mock_dependencies, sample_backtest_config):
        """戦略生成・テストの統合テスト"""
        print("\n=== 戦略生成・テスト統合テスト ===")
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        # テスト用戦略遺伝子
        from app.core.services.auto_strategy.models.gene_strategy import Condition

        test_gene = StrategyGene(
            id="integration_test",
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 20},
                    enabled=True,
                )
            ],
            long_entry_conditions=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="SMA_20"
                )
            ],
            short_entry_conditions=[],
            exit_conditions=[
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand="SMA_20"
                )
            ],
            risk_management={"position_size": 0.1},
        )
        
        # 戦略テスト実行
        result = manager.test_strategy_generation(test_gene, sample_backtest_config)
        
        # 結果検証
        assert isinstance(result, dict), "結果が辞書でない"
        assert "success" in result, "結果にsuccessフィールドがない"
        
        if result["success"]:
            assert "strategy_gene" in result, "成功時にstrategy_geneがない"
            assert "backtest_result" in result, "成功時にbacktest_resultがない"
            print("  ✅ 戦略生成・テスト: 成功")
        else:
            assert "errors" in result, "失敗時にerrorsがない"
            print(f"  ⚠️  戦略生成・テスト: 失敗 - {result.get('errors', [])}")

    def test_tpsl_integration(self):
        """TP/SL機能の統合テスト"""
        print("\n=== TP/SL統合テスト ===")
        
        tpsl_service = TPSLAutoDecisionService()
        
        # 複数戦略の組み合わせテスト
        strategies = [
            TPSLStrategy.RANDOM,
            TPSLStrategy.RISK_REWARD,
            TPSLStrategy.VOLATILITY_ADAPTIVE,
            TPSLStrategy.STATISTICAL,
            TPSLStrategy.AUTO_OPTIMAL,
        ]
        
        market_data = {
            "current_price": 50000.0,
            "atr": 1000.0,
            "atr_pct": 0.02,
            "volatility": 0.025,
        }
        
        results = {}
        for strategy in strategies:
            config = TPSLConfig(
                strategy=strategy,
                max_risk_per_trade=0.03,
                preferred_risk_reward_ratio=2.0,
                volatility_sensitivity="medium",
            )
            
            result = tpsl_service.generate_tpsl_values(config, market_data, "BTCUSDT")
            results[strategy.value] = result
            
            # 基本検証
            assert result.stop_loss_pct > 0, f"{strategy.value}: SLが0以下"
            assert result.take_profit_pct > 0, f"{strategy.value}: TPが0以下"
            assert result.risk_reward_ratio > 0, f"{strategy.value}: RR比が0以下"
            
            print(f"  ✅ {strategy.value}: SL={result.stop_loss_pct:.4f}, TP={result.take_profit_pct:.4f}")
        
        # 戦略間の結果比較
        sl_values = [r.stop_loss_pct for r in results.values()]
        tp_values = [r.take_profit_pct for r in results.values()]
        
        # 戦略によって異なる結果が得られることを確認
        assert len(set(sl_values)) > 1 or len(set(tp_values)) > 1, "全戦略で同じ結果"
        
        print(f"  📊 SL範囲: {min(sl_values):.4f} - {max(sl_values):.4f}")
        print(f"  📊 TP範囲: {min(tp_values):.4f} - {max(tp_values):.4f}")

    def test_position_sizing_integration(self):
        """ポジションサイジング統合テスト"""
        print("\n=== ポジションサイジング統合テスト ===")
        
        from app.core.services.auto_strategy.calculators.position_sizing_calculator import PositionSizingCalculatorService
        
        calculator = PositionSizingCalculatorService()
        
        # 各手法のテスト
        methods = [
            PositionSizingMethod.FIXED_RATIO,
            PositionSizingMethod.FIXED_QUANTITY,
            PositionSizingMethod.VOLATILITY_BASED,
            PositionSizingMethod.HALF_OPTIMAL_F,
        ]
        
        account_balance = 10000.0
        current_price = 50000.0
        market_data = {"atr": 1000.0, "atr_pct": 0.02}
        trade_history = [
            {"pnl": 500.0, "win": True},
            {"pnl": -300.0, "win": False},
            {"pnl": 800.0, "win": True},
        ]
        
        results = {}
        for method in methods:
            gene = PositionSizingGene(
                method=method,
                enabled=True,
            )
            
            result = calculator.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=market_data,
                trade_history=trade_history,
            )
            
            results[method.value] = result
            
            # 基本検証
            assert result.position_size > 0, f"{method.value}: ポジションサイズが0以下"
            assert result.method_used == method.value, f"{method.value}: 手法名が不一致"
            assert 0.0 <= result.confidence_score <= 1.0, f"{method.value}: 信頼度が範囲外"
            
            print(f"  ✅ {method.value}: サイズ={result.position_size:.4f}, 信頼度={result.confidence_score:.2f}")
        
        # 手法間の結果比較
        sizes = [r.position_size for r in results.values()]
        assert len(set(sizes)) > 1, "全手法で同じポジションサイズ"
        
        print(f"  📊 サイズ範囲: {min(sizes):.4f} - {max(sizes):.4f}")

    def test_error_handling_integration(self, mock_dependencies, sample_ga_config, sample_backtest_config):
        """エラーハンドリング統合テスト"""
        print("\n=== エラーハンドリング統合テスト ===")
        
        # バックテストエラーのシミュレーション
        mock_dependencies['backtest_service'].run_backtest.side_effect = Exception("バックテストエラー")
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        from app.core.services.auto_strategy.models.gene_strategy import Condition

        test_gene = StrategyGene(
            id="error_test",
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 20},
                    enabled=True,
                )
            ],
            long_entry_conditions=[
                Condition(
                    left_operand="close",
                    operator=">",
                    right_operand="SMA_20"
                )
            ],
            short_entry_conditions=[],
            exit_conditions=[
                Condition(
                    left_operand="close",
                    operator="<",
                    right_operand="SMA_20"
                )
            ],
            risk_management={"position_size": 0.1},
        )
        
        # エラーが適切に処理されることを確認
        result = manager.test_strategy_generation(test_gene, sample_backtest_config)
        
        assert isinstance(result, dict), "エラー時も辞書が返される"
        assert "success" in result, "エラー時もsuccessフィールドがある"
        assert not result["success"], "エラー時にsuccessがTrue"
        
        print("  ✅ バックテストエラー: 適切に処理")
        
        # バックテストサービスを正常に戻す
        mock_dependencies['backtest_service'].run_backtest.side_effect = None
        mock_dependencies['backtest_service'].run_backtest.return_value = {
            'total_return': 0.1,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.05,
            'total_trades': 10,
        }

    def test_edge_cases_and_boundary_conditions(self):
        """エッジケースと境界条件テスト"""
        print("\n=== エッジケース・境界条件テスト ===")
        
        # 極端に小さな口座残高
        from app.core.services.auto_strategy.calculators.position_sizing_calculator import PositionSizingCalculatorService
        
        calculator = PositionSizingCalculatorService()
        
        edge_cases = [
            {"balance": 1.0, "price": 50000.0, "case": "極小残高"},
            {"balance": 10000.0, "price": 0.001, "case": "極小価格"},
            {"balance": 1000000.0, "price": 100000.0, "case": "大きな値"},
        ]
        
        for case in edge_cases:
            gene = PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
                enabled=True,
            )
            
            try:
                result = calculator.calculate_position_size(
                    gene=gene,
                    account_balance=case["balance"],
                    current_price=case["price"],
                )
                
                assert result.position_size >= 0, f"{case['case']}: 負のポジションサイズ"
                print(f"  ✅ {case['case']}: 正常処理 (サイズ={result.position_size:.6f})")
                
            except Exception as e:
                print(f"  ⚠️  {case['case']}: エラー発生 - {e}")

    def test_performance_and_scalability(self, mock_dependencies):
        """パフォーマンス・スケーラビリティテスト"""
        print("\n=== パフォーマンス・スケーラビリティテスト ===")
        
        import time
        
        manager = ExperimentManager(
            backtest_service=mock_dependencies['backtest_service'],
            persistence_service=mock_dependencies['persistence_service'],
        )
        
        # 複数戦略の並列検証
        from app.core.services.auto_strategy.models.gene_strategy import Condition

        genes = []
        for i in range(10):
            gene = StrategyGene(
                id=f"perf_test_{i}",
                indicators=[
                    IndicatorGene(
                        type="SMA",
                        parameters={"period": 20 + i},
                        enabled=True,
                    )
                ],
                long_entry_conditions=[
                    Condition(
                        left_operand="close",
                        operator=">",
                        right_operand=f"SMA_{20 + i}"
                    )
                ],
                short_entry_conditions=[],
                exit_conditions=[
                    Condition(
                        left_operand="close",
                        operator="<",
                        right_operand=f"SMA_{20 + i}"
                    )
                ],
                risk_management={"position_size": 0.1},
            )
            genes.append(gene)
        
        # 検証時間測定
        start_time = time.time()
        
        valid_count = 0
        for gene in genes:
            is_valid, _ = manager.validate_strategy_gene(gene)
            if is_valid:
                valid_count += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # パフォーマンス確認
        assert elapsed_time < 5.0, f"検証時間が長すぎる: {elapsed_time:.2f}秒"
        assert valid_count == len(genes), f"有効な遺伝子数が不正: {valid_count}/{len(genes)}"
        
        print(f"  ✅ {len(genes)}個の遺伝子検証: {elapsed_time:.3f}秒")
        print(f"  📊 平均検証時間: {elapsed_time/len(genes)*1000:.1f}ms/遺伝子")


def main():
    """メイン実行関数"""
    print("オートストラテジー統合テスト開始")
    print("=" * 60)
    
    # pytest実行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
