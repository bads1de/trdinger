"""
GeneticAlgorithmEngineFactoryの拡張テスト

既存の ``core/test_ga_engine_factory.py`` と ``test_ga_engine.py`` でカバー
されていないエッジケース（ログレベル、ハイブリッドモデルタイプ、例外処理、
セットアップヘルパー）を検証します。
"""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from app.services.auto_strategy.config.ga.ga_config import GAConfig, HybridConfig
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
                ga_config.hybrid_config = Mock()
                ga_config.hybrid_config.mode = False
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


class TestFactoryHybridMode:
    """``create_engine`` のハイブリッドモード分岐テスト"""

    def test_standard_mode_when_hybrid_mode_false(self) -> None:
        """hybrid_config.mode = False のとき standard evaluator を使用"""
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
        ):
            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = Mock(spec=GAConfig)
            ga_config.log_level = "INFO"
            ga_config.hybrid_config = Mock()
            ga_config.hybrid_config.mode = False
            ga_config.population_size = 5

            engine = GeneticAlgorithmEngineFactory.create_engine(
                backtest_service=mock_backtest_service, ga_config=ga_config
            )

            assert engine.hybrid_mode is False
            # Standard mode uses IndividualEvaluator (not HybridIndividualEvaluator)
            assert (
                engine.individual_evaluator.__class__.__name__ == "IndividualEvaluator"
            )

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_hybrid_with_single_model_type(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
    ) -> None:
        """単一モデルタイプのハイブリッドセットアップ"""
        predictor = Mock()
        predictor.load_latest_models.return_value = True
        mock_predictor_cls.return_value = predictor

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = Mock(spec=GAConfig)
        ga_config.log_level = "INFO"
        ga_config.hybrid_config = Mock()
        ga_config.hybrid_config.mode = True
        ga_config.hybrid_config.model_types = None
        ga_config.hybrid_config.model_type = "lightgbm"
        ga_config.population_size = 5

        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )

        # HybridPredictor is constructed with single mode
        mock_predictor_cls.assert_called_once_with(
            trainer_type="single", model_type="lightgbm"
        )
        assert engine.hybrid_mode is True
        # The engine uses HybridIndividualEvaluator
        assert (
            engine.individual_evaluator.__class__.__name__
            == "HybridIndividualEvaluator"
        )

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_hybrid_with_model_types_ensemble(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
    ) -> None:
        """複数モデルタイプ（アンサンブル）のハイブリッドセットアップ"""
        predictor = Mock()
        predictor.load_latest_models.return_value = True
        mock_predictor_cls.return_value = predictor

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = Mock(spec=GAConfig)
        ga_config.log_level = "INFO"
        ga_config.hybrid_config = Mock()
        ga_config.hybrid_config.mode = True
        ga_config.hybrid_config.model_types = ["lightgbm", "xgboost", "catboost"]
        ga_config.hybrid_config.model_type = None
        ga_config.population_size = 5

        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )

        # HybridPredictor is constructed with model_types list
        mock_predictor_cls.assert_called_once_with(
            trainer_type="single", model_types=["lightgbm", "xgboost", "catboost"]
        )
        assert engine.hybrid_mode is True

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_hybrid_default_model_type_when_both_none(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
    ) -> None:
        """model_types も model_type も無い場合はデフォルト 'lightgbm'"""
        predictor = Mock()
        predictor.load_latest_models.return_value = False
        mock_predictor_cls.return_value = predictor

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = Mock(spec=GAConfig)
        ga_config.log_level = "INFO"
        ga_config.hybrid_config = Mock()
        ga_config.hybrid_config.mode = True
        ga_config.hybrid_config.model_types = None
        ga_config.hybrid_config.model_type = None
        ga_config.population_size = 5

        GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )

        # Falls back to "lightgbm"
        args, kwargs = mock_predictor_cls.call_args
        assert kwargs.get("model_type") == "lightgbm"

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_hybrid_handles_load_model_exception(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
    ) -> None:
        """load_latest_models が例外を投げてもクラッシュしない"""
        predictor = Mock()
        predictor.load_latest_models.side_effect = RuntimeError("Model load failed")
        mock_predictor_cls.return_value = predictor

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = Mock(spec=GAConfig)
        ga_config.log_level = "INFO"
        ga_config.hybrid_config = Mock()
        ga_config.hybrid_config.mode = True
        ga_config.hybrid_config.model_types = None
        ga_config.hybrid_config.model_type = "lightgbm"
        ga_config.population_size = 5

        # Should not raise
        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )

        # Engine is still created with hybrid mode
        assert engine.hybrid_mode is True
        assert (
            engine.individual_evaluator.__class__.__name__
            == "HybridIndividualEvaluator"
        )

    @patch(
        "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
    )
    @patch(
        "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    @patch("app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor")
    def test_hybrid_feature_adapter_is_constructed(
        self,
        mock_predictor_cls,
        mock_adapter_cls,
        mock_gene_generator_cls,
    ) -> None:
        """HybridFeatureAdapter は引数なしで構築される"""
        predictor = Mock()
        predictor.load_latest_models.return_value = True
        mock_predictor_cls.return_value = predictor
        mock_adapter_cls.return_value = Mock()

        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = Mock(spec=GAConfig)
        ga_config.log_level = "INFO"
        ga_config.hybrid_config = Mock()
        ga_config.hybrid_config.mode = True
        ga_config.hybrid_config.model_types = None
        ga_config.hybrid_config.model_type = "xgboost"
        ga_config.population_size = 5

        GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service, ga_config=ga_config
        )

        # HybridFeatureAdapter is constructed without arguments
        mock_adapter_cls.assert_called_once_with()


class TestFactoryReturns:
    """``create_engine`` の戻り値テスト"""

    def test_returns_genetic_algorithm_engine(self) -> None:
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
        ):
            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = Mock(spec=GAConfig)
            ga_config.log_level = "INFO"
            ga_config.hybrid_config = Mock()
            ga_config.hybrid_config.mode = False
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


class TestFactoryWithHybridConfig:
    """HybridConfig インスタンスを使った統合テスト"""

    def test_real_hybrid_config_single_model(self) -> None:
        with (
            patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
            ),
            patch(
                "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
            ) as mock_adapter,
            patch(
                "app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor"
            ) as mock_predictor_cls,
        ):
            predictor = Mock()
            predictor.load_latest_models.return_value = True
            mock_predictor_cls.return_value = predictor
            mock_adapter.return_value = Mock()

            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = GAConfig(
                log_level="DEBUG",
                hybrid_config=HybridConfig(
                    mode=True,
                    model_type="lightgbm",
                    model_types=None,
                ),
            )

            engine = GeneticAlgorithmEngineFactory.create_engine(
                backtest_service=mock_backtest_service, ga_config=ga_config
            )

            assert engine.hybrid_mode is True
            assert (
                engine.individual_evaluator.__class__.__name__
                == "HybridIndividualEvaluator"
            )
            mock_predictor_cls.assert_called_once_with(
                trainer_type="single", model_type="lightgbm"
            )

    def test_real_hybrid_config_ensemble(self) -> None:
        with (
            patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
            ),
            patch(
                "app.services.auto_strategy.core.hybrid.hybrid_feature_adapter.HybridFeatureAdapter"
            ),
            patch(
                "app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor"
            ) as mock_predictor_cls,
        ):
            predictor = Mock()
            predictor.load_latest_models.return_value = False
            mock_predictor_cls.return_value = predictor

            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = GAConfig(
                log_level="INFO",
                hybrid_config=HybridConfig(
                    mode=True,
                    model_type=None,
                    model_types=["lightgbm", "xgboost"],
                ),
            )

            engine = GeneticAlgorithmEngineFactory.create_engine(
                backtest_service=mock_backtest_service, ga_config=ga_config
            )

            assert engine.hybrid_mode is True
            mock_predictor_cls.assert_called_once_with(
                trainer_type="single", model_types=["lightgbm", "xgboost"]
            )

    def test_real_hybrid_config_disabled(self) -> None:
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.RandomGeneGenerator"
        ):
            mock_backtest_service = Mock(spec=BacktestService)
            ga_config = GAConfig(
                log_level="WARNING",
                hybrid_config=HybridConfig(
                    mode=False,
                    model_type="lightgbm",
                ),
            )

            engine = GeneticAlgorithmEngineFactory.create_engine(
                backtest_service=mock_backtest_service, ga_config=ga_config
            )

            assert engine.hybrid_mode is False
            assert (
                engine.individual_evaluator.__class__.__name__ == "IndividualEvaluator"
            )
