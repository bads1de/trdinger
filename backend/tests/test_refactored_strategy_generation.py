"""
リファクタリング後の戦略生成テスト

リファクタリング後のシステムで実際に戦略を生成し、動作確認を行います。
"""

import pytest
import uuid
from unittest.mock import Mock, patch

from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.utils.error_handling import AutoStrategyErrorHandler
from app.services.auto_strategy.utils.auto_strategy_utils import AutoStrategyUtils
from app.services.auto_strategy.config.constants import (
    validate_symbol,
    validate_timeframe,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)


class TestRefactoredStrategyGeneration:
    """リファクタリング後の戦略生成テスト"""

    def test_ga_config_creation_and_validation(self):
        """GAConfig作成と検証のテスト"""
        # デフォルト設定
        config = GAConfig.create_fast()
        assert config.population_size == 10
        assert config.generations == 5

        # 検証テスト
        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(errors) == 0

        # BaseConfigの機能テスト
        config_dict = config.to_dict()
        assert "population_size" in config_dict
        assert config_dict["population_size"] == 10

    def test_shared_constants_validation(self):
        """共通定数の検証テスト"""
        # サポートされているシンボル
        assert validate_symbol("BTC/USDT:USDT") is True
        assert validate_symbol("ETH/USDT:USDT") is False  # サポート外

        # サポートされている時間軸
        assert validate_timeframe("1h") is True
        assert validate_timeframe("5m") is False  # サポート外

    def test_auto_strategy_utils_integration(self):
        """AutoStrategyUtilsの統合テスト"""
        # データ変換
        assert AutoStrategyUtils.safe_convert_to_float("123.45") == 123.45
        assert AutoStrategyUtils.safe_convert_to_int("42") == 42

        # シンボル正規化
        assert AutoStrategyUtils.normalize_symbol("BTC") == "BTC:USDT"

        # 検証
        assert AutoStrategyUtils.validate_range(5, 1, 10) is True
        assert AutoStrategyUtils.validate_range(15, 1, 10) is False

    @patch("app.services.auto_strategy.services.auto_strategy_service.BacktestService")
    @patch(
        "app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService"
    )
    @patch(
        "app.services.auto_strategy.services.auto_strategy_service.ExperimentManager"
    )
    def test_auto_strategy_service_initialization(
        self, mock_manager, mock_persistence, mock_backtest
    ):
        """AutoStrategyServiceの初期化テスト"""
        # モックの設定
        mock_backtest_instance = Mock()
        mock_persistence_instance = Mock()
        mock_manager_instance = Mock()

        mock_backtest.return_value = mock_backtest_instance
        mock_persistence.return_value = mock_persistence_instance
        mock_manager.return_value = mock_manager_instance

        # サービス初期化
        service = AutoStrategyService(enable_smart_generation=True)

        # 初期化確認
        assert service.enable_smart_generation is True
        assert hasattr(service, "db_session_factory")

    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        # GA エラー
        error = ValueError("テストGA エラー")
        result = AutoStrategyErrorHandler.handle_ga_error(error, "テストコンテキスト")

        assert result["error_code"] == "GA_ERROR"
        assert "テストGA エラー" in result["message"]

        # 戦略生成エラー
        strategy_data = {"indicators": ["RSI", "SMA"]}
        result = AutoStrategyErrorHandler.handle_strategy_generation_error(
            error, strategy_data, "戦略生成テスト"
        )

        assert result["success"] is False
        assert result["strategy"] is None
        assert result["strategy_data"] == strategy_data

    def test_config_merge_functionality(self):
        """設定マージ機能のテスト"""
        base_config = {
            "population_size": 10,
            "generations": 5,
            "advanced": {"crossover_rate": 0.8, "mutation_rate": 0.1},
        }

        override_config = {
            "population_size": 20,
            "advanced": {"mutation_rate": 0.2, "elite_size": 2},
        }

        merged = AutoStrategyUtils.merge_configs(base_config, override_config)

        assert merged["population_size"] == 20  # オーバーライド
        assert merged["generations"] == 5  # 保持
        assert merged["advanced"]["crossover_rate"] == 0.8  # 保持
        assert merged["advanced"]["mutation_rate"] == 0.2  # オーバーライド
        assert merged["advanced"]["elite_size"] == 2  # 新規追加

    @patch("app.services.indicators.TechnicalIndicatorService")
    def test_indicator_id_generation(self, mock_service):
        """指標ID生成のテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.get_supported_indicators.return_value = {
            "SMA": {"description": "Simple Moving Average"},
            "RSI": {"description": "Relative Strength Index"},
            "MACD": {"description": "MACD"},
        }
        mock_service.return_value = mock_instance

        # 指標ID取得
        indicator_ids = AutoStrategyUtils.get_all_indicator_ids()

        # 検証
        assert "" in indicator_ids  # 空文字は0
        assert indicator_ids[""] == 0
        assert "SMA" in indicator_ids
        assert "RSI" in indicator_ids
        assert "MACD" in indicator_ids
        assert "ML_UP_PROB" in indicator_ids  # ML指標も含まれる
        assert "ML_DOWN_PROB" in indicator_ids
        assert "ML_RANGE_PROB" in indicator_ids

    def test_encoding_info_generation(self):
        """エンコーディング情報生成のテスト"""
        # テスト用の指標ID
        test_indicator_ids = {
            "": 0,
            "SMA": 1,
            "RSI": 2,
            "MACD": 3,
            "ML_UP_PROB": 4,
            "ML_DOWN_PROB": 5,
        }

        encoding_info = AutoStrategyUtils.get_encoding_info(test_indicator_ids)

        assert encoding_info["indicator_count"] == 5  # 空文字除く
        assert encoding_info["max_indicators"] == 5
        assert encoding_info["encoding_length"] == 32
        assert "SMA" in encoding_info["supported_indicators"]
        assert "ML_UP_PROB" in encoding_info["supported_indicators"]

    def test_strategy_generation_config_validation(self):
        """戦略生成設定の検証テスト"""
        # 有効な設定
        valid_ga_config = {
            "population_size": 10,
            "generations": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 2,
        }

        valid_backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
        }

        # GAConfig作成
        ga_config = GAConfig.from_dict(valid_ga_config)
        is_valid, errors = ga_config.validate()
        assert is_valid is True

        # バックテスト設定検証
        assert validate_symbol(valid_backtest_config["symbol"]) is True
        assert validate_timeframe(valid_backtest_config["timeframe"]) is True

    def test_default_strategy_gene_creation(self):
        """デフォルト戦略遺伝子作成のテスト"""
        strategy_gene = AutoStrategyUtils.create_default_strategy_gene()

        assert strategy_gene is not None
        assert len(strategy_gene.indicators) == 2  # SMA, RSI
        assert len(strategy_gene.entry_conditions) == 1
        assert len(strategy_gene.exit_conditions) == 0  # TP/SL使用時は空
        assert strategy_gene.metadata["generated_by"] == "AutoStrategyUtils"

    def test_performance_measurement(self):
        """パフォーマンス測定のテスト"""

        def test_function(x, y):
            return x + y

        result, execution_time = AutoStrategyUtils.time_function(test_function, 5, 10)

        assert result == 15
        assert execution_time >= 0
        assert isinstance(execution_time, float)

    def test_logger_setup(self):
        """ロガー設定のテスト"""
        logger = AutoStrategyUtils.setup_auto_strategy_logger(
            "test_auto_strategy", "INFO"
        )

        assert logger.name == "test_auto_strategy"
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
