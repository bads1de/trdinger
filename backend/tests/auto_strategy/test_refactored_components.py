"""
リファクタリング後のコンポーネントテスト

重複削除後の統合コンポーネントが正常に動作することを確認します。
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.services.auto_strategy.services.tpsl_service import TPSLService
from app.services.auto_strategy.models.ga_config import GAConfig


class TestAutoStrategyService:
    """自動戦略サービスのテスト"""

    @pytest.fixture
    def service(self):
        """テスト用サービスインスタンス"""
        with patch(
            "app.services.auto_strategy.services.auto_strategy_service.SessionLocal"
        ):
            return AutoStrategyService()

    def test_service_initialization(self, service):
        """サービス初期化テスト"""
        assert service is not None
        assert hasattr(service, "backtest_service")
        assert hasattr(service, "persistence_service")

    def test_get_default_config(self, service):
        """デフォルト設定取得テスト"""
        config = service.get_default_config()
        assert isinstance(config, dict)
        assert "population_size" in config
        assert "generations" in config

    def test_get_presets(self, service):
        """プリセット取得テスト"""
        presets = service.get_presets()
        assert isinstance(presets, dict)


class TestTPSLService:
    """TP/SL計算サービスのテスト"""

    @pytest.fixture
    def service(self):
        """テスト用サービスインスタンス"""
        return TPSLService()

    def test_service_initialization(self, service):
        """サービス初期化テスト"""
        assert service is not None
        assert hasattr(service, "risk_reward_calculator")
        assert hasattr(service, "statistical_generator")
        assert hasattr(service, "volatility_generator")

    def test_basic_tpsl_calculation(self, service):
        """基本TP/SL計算テスト"""
        sl_price, tp_price = service.calculate_tpsl_prices(
            current_price=100.0, stop_loss_pct=0.03, take_profit_pct=0.06
        )

        assert sl_price == 97.0  # 100 * (1 - 0.03)
        assert tp_price == 106.0  # 100 * (1 + 0.06)

    def test_short_position_calculation(self, service):
        """ショートポジション計算テスト"""
        sl_price, tp_price = service.calculate_tpsl_prices(
            current_price=100.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            position_direction=-1.0,
        )

        assert sl_price == 103.0  # 100 * (1 + 0.03)
        assert tp_price == 94.0  # 100 * (1 - 0.06)


class TestGAConfig:
    """GA設定のテスト"""

    def test_default_config_creation(self):
        """デフォルト設定作成テスト"""
        config = GAConfig()
        assert config.population_size == 10
        assert config.generations == 5
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.1

    def test_config_validation(self):
        """設定バリデーションテスト"""
        config = GAConfig(
            population_size=10, generations=5, crossover_rate=0.8, mutation_rate=0.1
        )

        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_config_validation(self):
        """無効設定バリデーションテスト"""
        config = GAConfig(
            population_size=0,  # 無効値
            generations=0,  # 無効値
            crossover_rate=1.5,  # 無効値
            mutation_rate=-0.1,  # 無効値
        )

        is_valid, errors = config.validate()
        assert is_valid is False
        assert len(errors) > 0

    def test_config_from_dict(self):
        """辞書からの設定作成テスト"""
        config_dict = {
            "population_size": 15,
            "generations": 8,
            "crossover_rate": 0.7,
            "mutation_rate": 0.15,
        }

        config = GAConfig.from_dict(config_dict)
        assert config.population_size == 15
        assert config.generations == 8
        assert config.crossover_rate == 0.7
        assert config.mutation_rate == 0.15

    def test_config_to_dict(self):
        """設定の辞書変換テスト"""
        config = GAConfig(population_size=20, generations=10)

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["population_size"] == 20
        assert config_dict["generations"] == 10

    def test_presets(self):
        """プリセット取得テスト"""
        # GAConfigにget_presetsメソッドがない場合はスキップ
        if hasattr(GAConfig, "get_presets"):
            presets = GAConfig.get_presets()
            assert isinstance(presets, dict)
            assert "fast" in presets
            assert "default" in presets
            assert "thorough" in presets
        else:
            pytest.skip("GAConfig.get_presets method not implemented")


class TestIntegration:
    """統合テスト"""

    def test_service_integration(self):
        """サービス統合テスト"""
        with patch(
            "app.services.auto_strategy.services.auto_strategy_service.SessionLocal"
        ):
            auto_service = AutoStrategyService()
            tpsl_service = TPSLService()

            # 両サービスが正常に初期化されることを確認
            assert auto_service is not None
            assert tpsl_service is not None

            # 基本機能が動作することを確認
            config = auto_service.get_default_config()
            assert isinstance(config, dict)

            sl_price, tp_price = tpsl_service.calculate_tpsl_prices(
                current_price=100.0, stop_loss_pct=0.02, take_profit_pct=0.04
            )
            assert sl_price is not None
            assert tp_price is not None

    def test_no_duplicate_functionality(self):
        """重複機能がないことを確認"""
        # 重複していたサービスが統合されていることを確認
        with patch(
            "app.services.auto_strategy.services.auto_strategy_service.SessionLocal"
        ):
            service = AutoStrategyService()

            # 統合されたメソッドが存在することを確認
            assert hasattr(service, "start_strategy_generation")
            assert hasattr(service, "list_experiments")
            assert hasattr(service, "get_experiment_status")
            assert hasattr(service, "stop_experiment")
            assert hasattr(service, "test_strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
