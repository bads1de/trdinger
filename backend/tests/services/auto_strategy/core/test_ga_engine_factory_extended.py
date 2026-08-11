"""
GeneticAlgorithmEngineFactoryの拡張テスト

既存の ``core/test_ga_engine_factory.py`` でカバーされていないエッジケース
（ログレベル、例外処理、セットアップヘルパー）を検証します。
"""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from app.services.auto_strategy.config.ga.ga_config import GAConfig
from app.services.auto_strategy.core.engine.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.engine.ga_engine_factory import (
    GeneticAlgorithmEngineFactory,
)
from app.services.backtest.services.backtest_service import BacktestService


class TestFactoryLogLevel:
    """``create_engine`` でのログレベル設定テスト"""

    def _capture_log_level(self, log_level_str: str) -> int:
        """指定の log_level で create_engine を呼び、logger のレベルを返す"""
        target_logger = logging.getLogger("app.services.auto_strategy")
        original_level = target_logger.level

        try:
            with patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
            ):
                mock_backtest_service = Mock(spec=BacktestService)
                ga_config = Mock(spec=GAConfig)
                ga_config.log_level = log_level_str
                ga_config.population_size = 5

                GeneticAlgorithmEngineFactory.create_engine(
                    backtest_service=mock_backtest_service, ga_config=ga_config
                )
                return target_logger.level
        finally:
            target_logger.setLevel(original_level)

    def test_log_level_debug(self) -> None:
        level = self._capture_log_level("DEBUG")
        assert level == logging.DEBUG

    def test_log_level_info(self) -> None:
        level = self._capture_log_level("INFO")
        assert level == logging.INFO

    def test_log_level_warning(self) -> None:
        level = self._capture_log_level("WARNING")
        assert level == logging.WARNING

    def test_log_level_error(self) -> None:
        level = self._capture_log_level("ERROR")
        assert level == logging.ERROR

    def test_log_level_critical(self) -> None:
        level = self._capture_log_level("CRITICAL")
        assert level == logging.CRITICAL

    def test_invalid_log_level_falls_back_to_info(self) -> None:
        """未知の log_level 文字列は logging.INFO にフォールバック"""
        level = self._capture_log_level("INVALID_LEVEL_NAME")
        assert level == logging.INFO

    def test_log_level_lowercase_normalized(self) -> None:
        """getattr(logging, "info".upper()) で大文字化して評価される"""
        level = self._capture_log_level("info")
        assert level == logging.INFO


class TestFactoryReturns:
    """``create_engine`` の戻り値テスト"""

    def test_returns_genetic_algorithm_engine(self) -> None:
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
        ):
            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = Mock(spec=GAConfig)
            ga_config.log_level = "INFO"
            ga_config.population_size = 5

            engine = GeneticAlgorithmEngineFactory.create_engine(
                backtest_service=mock_backtest_service, ga_config=ga_config
            )
            assert isinstance(engine, GeneticAlgorithmEngine)

    def test_uses_real_ga_config_object(self) -> None:
        """実際の GAConfig() インスタンスも受け入れる"""
        from app.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = GAConfig(population_size=7)

        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )
        assert isinstance(engine, GeneticAlgorithmEngine)
        # Default GAConfig uses RandomGeneGenerator (not mocked)
        assert engine.gene_generator is not None
        assert isinstance(engine.gene_generator, RandomGeneGenerator)
        assert engine.gene_generator.config.population_size == 7
