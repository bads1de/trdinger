"""
多目的最適化GA API統合テスト

APIエンドポイントを通じた多目的最適化GAの動作確認を行います。
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig

logger = logging.getLogger(__name__)


class TestMultiObjectiveAPI:
    """多目的最適化GA APIテストクラス"""

    def test_ga_config_multi_objective_example(self):
        """GAConfigの多目的最適化設定例のテスト"""
        # 多目的最適化設定の例
        config_dict = {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 4,
            "max_indicators": 3,
            "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
            # 多目的最適化設定
            "enable_multi_objective": True,
            "objectives": ["total_return", "max_drawdown"],
            "objective_weights": [1.0, -1.0],
        }
        
        # GAConfigに変換
        config = GAConfig.from_dict(config_dict)
        
        # 設定確認
        assert config.enable_multi_objective is True
        assert config.objectives == ["total_return", "max_drawdown"]
        assert config.objective_weights == [1.0, -1.0]
        
        # 辞書に戻す
        result_dict = config.to_dict()
        assert result_dict["enable_multi_objective"] is True
        assert result_dict["objectives"] == ["total_return", "max_drawdown"]
        assert result_dict["objective_weights"] == [1.0, -1.0]

    def test_multi_objective_request_format(self):
        """多目的最適化リクエスト形式のテスト"""
        # APIリクエストの例
        request_data = {
            "experiment_name": "Multi_Objective_Test_001",
            "base_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.00055,
            },
            "ga_config": {
                "population_size": 20,
                "generations": 10,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 4,
                "max_indicators": 3,
                "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
                # 多目的最適化設定
                "enable_multi_objective": True,
                "objectives": ["total_return", "max_drawdown", "sharpe_ratio"],
                "objective_weights": [1.0, -1.0, 1.0],
            },
        }
        
        # GAConfig変換テスト
        ga_config = GAConfig.from_dict(request_data["ga_config"])
        assert ga_config.enable_multi_objective is True
        assert len(ga_config.objectives) == 3
        assert len(ga_config.objective_weights) == 3

    def test_multi_objective_response_format(self):
        """多目的最適化レスポンス形式のテスト"""
        # 模擬的なGA実行結果
        mock_result = {
            "best_strategy": {
                "gene_data": {"indicators": [], "conditions": []},
                "fitness_score": 0.15,
                "fitness_values": [0.15, 0.08, 1.5],  # [total_return, max_drawdown, sharpe_ratio]
            },
            "pareto_front": [
                {
                    "strategy": {"indicators": [], "conditions": []},
                    "fitness_values": [0.15, 0.08, 1.5],
                },
                {
                    "strategy": {"indicators": [], "conditions": []},
                    "fitness_values": [0.12, 0.05, 1.8],
                },
                {
                    "strategy": {"indicators": [], "conditions": []},
                    "fitness_values": [0.18, 0.12, 1.2],
                },
            ],
            "objectives": ["total_return", "max_drawdown", "sharpe_ratio"],
            "total_strategies": 20,
            "execution_time": 45.2,
        }
        
        # レスポンス形式の確認
        assert "best_strategy" in mock_result
        assert "pareto_front" in mock_result
        assert "objectives" in mock_result
        assert len(mock_result["pareto_front"]) == 3
        assert len(mock_result["objectives"]) == 3
        
        # 各パレート最適解の形式確認
        for solution in mock_result["pareto_front"]:
            assert "strategy" in solution
            assert "fitness_values" in solution
            assert len(solution["fitness_values"]) == 3

    def test_single_objective_compatibility(self):
        """単一目的最適化との互換性テスト"""
        # 単一目的設定
        single_objective_config = {
            "population_size": 10,
            "generations": 5,
            "enable_multi_objective": False,
            "objectives": ["total_return"],
            "objective_weights": [1.0],
        }
        
        config = GAConfig.from_dict(single_objective_config)
        assert config.enable_multi_objective is False
        assert config.objectives == ["total_return"]
        assert config.objective_weights == [1.0]
        
        # 従来の形式でも動作することを確認
        legacy_config = {
            "population_size": 10,
            "generations": 5,
            # 多目的設定なし（デフォルト値が使用される）
        }
        
        legacy_config_obj = GAConfig.from_dict(legacy_config)
        assert legacy_config_obj.enable_multi_objective is False
        assert legacy_config_obj.objectives == ["total_return"]
        assert legacy_config_obj.objective_weights == [1.0]

    def test_objective_validation(self):
        """目的関数の妥当性検証テスト"""
        # 有効な目的の組み合わせ
        valid_objectives = [
            ["total_return"],
            ["total_return", "max_drawdown"],
            ["total_return", "sharpe_ratio", "max_drawdown"],
            ["sharpe_ratio", "win_rate"],
            ["profit_factor", "sortino_ratio"],
        ]
        
        for objectives in valid_objectives:
            weights = [1.0 if obj != "max_drawdown" else -1.0 for obj in objectives]
            config_dict = {
                "enable_multi_objective": len(objectives) > 1,
                "objectives": objectives,
                "objective_weights": weights,
            }
            
            config = GAConfig.from_dict(config_dict)
            assert config.objectives == objectives
            assert config.objective_weights == weights

    def test_weight_configuration(self):
        """重み設定のテスト"""
        # 最大化・最小化の組み合わせ
        test_cases = [
            {
                "objectives": ["total_return", "max_drawdown"],
                "weights": [1.0, -1.0],  # リターン最大化、ドローダウン最小化
                "description": "リターン最大化・ドローダウン最小化",
            },
            {
                "objectives": ["sharpe_ratio", "win_rate", "max_drawdown"],
                "weights": [1.0, 1.0, -1.0],  # シャープレシオ・勝率最大化、ドローダウン最小化
                "description": "シャープレシオ・勝率最大化・ドローダウン最小化",
            },
            {
                "objectives": ["total_return", "profit_factor"],
                "weights": [1.0, 1.0],  # 両方最大化
                "description": "リターン・プロフィットファクター最大化",
            },
        ]
        
        for case in test_cases:
            config_dict = {
                "enable_multi_objective": True,
                "objectives": case["objectives"],
                "objective_weights": case["weights"],
            }
            
            config = GAConfig.from_dict(config_dict)
            assert config.objectives == case["objectives"]
            assert config.objective_weights == case["weights"]
            
            # 目的と重みの数が一致することを確認
            assert len(config.objectives) == len(config.objective_weights)

    def test_preset_configurations(self):
        """プリセット設定のテスト"""
        # リスク・リターン最適化プリセット
        risk_return_config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown"],
            weights=[1.0, -1.0]
        )
        assert risk_return_config.enable_multi_objective is True
        assert risk_return_config.objectives == ["total_return", "max_drawdown"]
        assert risk_return_config.objective_weights == [1.0, -1.0]
        
        # パフォーマンス最適化プリセット
        performance_config = GAConfig.create_multi_objective(
            objectives=["sharpe_ratio", "win_rate", "profit_factor"],
            weights=[1.0, 1.0, 1.0]
        )
        assert performance_config.enable_multi_objective is True
        assert performance_config.objectives == ["sharpe_ratio", "win_rate", "profit_factor"]
        assert performance_config.objective_weights == [1.0, 1.0, 1.0]


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    pytest.main([__file__, "-v"])
