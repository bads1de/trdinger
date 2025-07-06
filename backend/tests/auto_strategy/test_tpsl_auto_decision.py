"""
TP/SL自動決定機能の統合テスト

新しいTP/SL自動決定機能の包括的なテストを実装し、
各決定方式の動作確認、既存システムとの互換性、
エラーハンドリングの検証を行います。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

# テスト対象のインポート
from app.core.services.auto_strategy.services.tpsl_auto_decision_service import (
    TPSLAutoDecisionService,
    TPSLConfig,
    TPSLStrategy,
    TPSLResult
)
from app.core.services.auto_strategy.calculators.risk_reward_calculator import (
    RiskRewardCalculator,
    RiskRewardConfig,
    RiskRewardProfile
)
from app.core.services.auto_strategy.generators.volatility_based_generator import (
    VolatilityBasedGenerator,
    VolatilityConfig
)
from app.core.services.auto_strategy.generators.statistical_tpsl_generator import (
    StatisticalTPSLGenerator,
    StatisticalConfig,
    OptimizationObjective
)


class TestTPSLAutoDecisionService:
    """TP/SL自動決定サービスのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.service = TPSLAutoDecisionService()
        self.default_config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium"
        )
    
    def test_random_strategy_generation(self):
        """ランダム戦略のテスト"""
        config = TPSLConfig(strategy=TPSLStrategy.RANDOM)
        result = self.service.generate_tpsl_values(config)
        
        assert isinstance(result, TPSLResult)
        assert result.strategy_used == "random"
        assert 0.005 <= result.stop_loss_pct <= 0.1
        assert 0.01 <= result.take_profit_pct <= 0.2
        assert result.risk_reward_ratio > 1.0
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_risk_reward_strategy_generation(self):
        """リスクリワード戦略のテスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.04,
            preferred_risk_reward_ratio=2.5
        )
        result = self.service.generate_tpsl_values(config)
        
        assert result.strategy_used == "risk_reward"
        assert result.stop_loss_pct == 0.04
        assert result.take_profit_pct == 0.04 * 2.5
        assert abs(result.risk_reward_ratio - 2.5) < 0.1
        assert result.confidence_score >= 0.7
    
    def test_volatility_adaptive_strategy(self):
        """ボラティリティ適応戦略のテスト"""
        config = TPSLConfig(
            strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
            volatility_sensitivity="high"
        )
        
        # ATRデータを含む市場データ
        market_data = {
            "atr_pct": 0.025,
            "trend_strength": 0.8,
            "volume_ratio": 1.2
        }
        
        result = self.service.generate_tpsl_values(config, market_data)
        
        assert result.strategy_used == "volatility_adaptive"
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.confidence_score >= 0.5
    
    def test_statistical_strategy(self):
        """統計的戦略のテスト"""
        config = TPSLConfig(strategy=TPSLStrategy.STATISTICAL)
        result = self.service.generate_tpsl_values(config, symbol="BTC/USDT")
        
        assert result.strategy_used == "statistical"
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.confidence_score >= 0.5
    
    def test_auto_optimal_strategy(self):
        """自動最適化戦略のテスト"""
        config = TPSLConfig(strategy=TPSLStrategy.AUTO_OPTIMAL)
        result = self.service.generate_tpsl_values(config)
        
        assert result.strategy_used == "auto_optimal"
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.metadata is not None
        assert "selected_from" in result.metadata
    
    def test_validation_success(self):
        """バリデーション成功のテスト"""
        result = TPSLResult(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            strategy_used="test",
            confidence_score=0.8
        )
        
        is_valid = self.service.validate_tpsl_values(result, self.default_config)
        assert is_valid is True
    
    def test_validation_failure(self):
        """バリデーション失敗のテスト"""
        # SLが範囲外
        result = TPSLResult(
            stop_loss_pct=0.15,  # 上限超過
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            strategy_used="test",
            confidence_score=0.8
        )
        
        is_valid = self.service.validate_tpsl_values(result, self.default_config)
        assert is_valid is False


class TestRiskRewardCalculator:
    """リスクリワード計算機のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.calculator = RiskRewardCalculator()
        self.default_config = RiskRewardConfig(
            target_ratio=2.0,
            profile=RiskRewardProfile.BALANCED
        )
    
    def test_basic_calculation(self):
        """基本的なTP計算のテスト"""
        result = self.calculator.calculate_take_profit(0.03, self.default_config)
        
        assert result.take_profit_pct == 0.06  # 3% * 2.0
        assert result.actual_risk_reward_ratio == 2.0
        assert result.is_ratio_achieved is True
    
    def test_tp_limit_adjustment(self):
        """TP上限調整のテスト"""
        config = RiskRewardConfig(target_ratio=10.0, max_tp_limit=0.15)
        result = self.calculator.calculate_take_profit(0.03, config)
        
        assert result.take_profit_pct == 0.15  # 上限に調整
        assert result.actual_risk_reward_ratio == 5.0  # 0.15 / 0.03
        assert result.is_ratio_achieved is False
        assert "TP上限制限適用" in result.adjustment_reason
    
    def test_optimal_ratio_calculation(self):
        """最適比率計算のテスト"""
        market_conditions = {
            "volatility": "high",
            "trend_strength": 0.8
        }
        
        optimal_ratio = self.calculator.calculate_optimal_ratio(
            0.03, market_conditions
        )
        
        assert 1.0 <= optimal_ratio <= 5.0
    
    def test_multiple_tp_levels(self):
        """複数TPレベル計算のテスト"""
        ratios = [1.5, 2.0, 3.0]
        tp_levels = self.calculator.calculate_multiple_tp_levels(0.02, ratios)
        
        assert len(tp_levels) == 3
        assert tp_levels[0] == (0.03, 1.5)  # 2% * 1.5
        assert tp_levels[1] == (0.04, 2.0)  # 2% * 2.0
        assert tp_levels[2] == (0.06, 3.0)  # 2% * 3.0


class TestVolatilityBasedGenerator:
    """ボラティリティベース生成器のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.generator = VolatilityBasedGenerator()
        self.default_config = VolatilityConfig()
    
    def test_with_atr_data(self):
        """ATRデータありのテスト"""
        market_data = {
            "atr": 50.0,
            "high": np.array([100, 105, 102, 108]),
            "low": np.array([95, 98, 97, 103]),
            "close": np.array([98, 103, 100, 106])
        }
        
        result = self.generator.generate_volatility_based_tpsl(
            market_data, self.default_config, 1000.0
        )
        
        assert result.atr_value == 50.0
        assert result.atr_pct == 0.05  # 50 / 1000
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
    
    def test_without_atr_data(self):
        """ATRデータなしのテスト"""
        market_data = {}
        
        result = self.generator.generate_volatility_based_tpsl(
            market_data, self.default_config, 1000.0
        )
        
        assert result.atr_pct == 0.02  # デフォルト値
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.confidence_score < 1.0  # データ不足による信頼度低下


class TestStatisticalTPSLGenerator:
    """統計的TP/SL生成器のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.generator = StatisticalTPSLGenerator()
        self.default_config = StatisticalConfig(
            optimization_objective=OptimizationObjective.SHARPE_RATIO
        )
    
    def test_statistical_generation(self):
        """統計的生成のテスト"""
        result = self.generator.generate_statistical_tpsl(
            self.default_config,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.sample_size > 0
        assert result.expected_performance is not None
        assert "sharpe_ratio" in result.expected_performance
    
    def test_market_regime_filtering(self):
        """市場レジームフィルタリングのテスト"""
        market_conditions = {
            "trend": "up",
            "volatility": "low"
        }
        
        result = self.generator.generate_statistical_tpsl(
            self.default_config,
            market_conditions=market_conditions
        )
        
        assert result.metadata["market_regime"] == "bull_market"


class TestIntegrationWithRandomGeneGenerator:
    """RandomGeneGeneratorとの統合テスト"""
    
    @patch('app.core.services.auto_strategy.generators.random_gene_generator.TPSLAutoDecisionService')
    def test_advanced_risk_management_integration(self, mock_service_class):
        """高度なリスク管理機能の統合テスト"""
        # モックサービスの設定
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_result = TPSLResult(
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
            risk_reward_ratio=2.0,
            strategy_used="risk_reward",
            confidence_score=0.85
        )
        mock_service.generate_tpsl_values.return_value = mock_result
        mock_service.validate_tpsl_values.return_value = True
        
        # テスト用のGA設定
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        ga_config = GAConfig(
            tpsl_strategy="risk_reward",
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium"
        )
        
        generator = RandomGeneGenerator(ga_config)
        risk_management = generator._generate_risk_management()
        
        # 結果の検証
        assert risk_management["stop_loss"] == 0.025
        assert risk_management["take_profit"] == 0.05
        assert risk_management["_tpsl_strategy"] == "risk_reward"
        assert risk_management["_risk_reward_ratio"] == 2.0
        assert risk_management["_confidence_score"] == 0.85


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_service_fallback_on_error(self):
        """サービスエラー時のフォールバック"""
        service = TPSLAutoDecisionService()
        
        # 無効な戦略でエラーを発生させる
        with patch.object(service, '_generate_random_tpsl', side_effect=Exception("Test error")):
            config = TPSLConfig(strategy=TPSLStrategy.RANDOM)
            result = service.generate_tpsl_values(config)
            
            # フォールバック結果が返されることを確認
            assert result.strategy_used == "fallback"
            assert result.confidence_score == 0.3
    
    def test_calculator_error_handling(self):
        """計算機エラーハンドリング"""
        calculator = RiskRewardCalculator()
        
        # 無効な入力でエラーを発生させる
        config = RiskRewardConfig(target_ratio=-1.0)  # 無効な比率
        result = calculator.calculate_take_profit(0.03, config)
        
        # エラーが適切に処理されることを確認
        assert result is not None
        assert result.take_profit_pct > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
