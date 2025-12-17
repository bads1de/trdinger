"""
PositionSizingGeneratorのテスト

ポジションサイジング遺伝子生成ロジックのテスト
"""

import pytest
from unittest.mock import Mock, patch


class TestPositionSizingGeneratorInit:
    """初期化のテスト"""

    def test_init_stores_config(self):
        """設定が保存される"""
        from app.services.auto_strategy.generators.component_generators import (
            PositionSizingGenerator,
        )

        config = Mock()
        generator = PositionSizingGenerator(config)

        assert generator.config == config


class TestGeneratePositionSizingGene:
    """generate_position_sizing_geneのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.component_generators import (
            PositionSizingGenerator,
        )

        config = Mock()
        return PositionSizingGenerator(config)

    def test_returns_random_gene_on_success(self, generator):
        """成功時にランダム遺伝子を返す"""
        from app.services.auto_strategy.genes import (
            PositionSizingGene,
            PositionSizingMethod,
        )

        mock_gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            fixed_ratio=0.05,
            max_position_size=10.0,
            enabled=True,
        )

        with patch(
            "app.services.auto_strategy.generators.component_generators.create_random_position_sizing_gene",
            return_value=mock_gene,
        ):
            result = generator.generate_position_sizing_gene()

        assert result.method == PositionSizingMethod.VOLATILITY_BASED
        assert result.fixed_ratio == 0.05
        assert result.max_position_size == 10.0
        assert result.enabled is True

    def test_returns_fallback_gene_on_error(self, generator):
        """エラー時にフォールバック遺伝子を返す"""
        from app.services.auto_strategy.genes import (
            PositionSizingMethod,
        )

        with patch(
            "app.services.auto_strategy.generators.component_generators.create_random_position_sizing_gene",
            side_effect=ValueError("Random generation failed"),
        ):
            result = generator.generate_position_sizing_gene()

        # フォールバック値を確認
        assert result.method == PositionSizingMethod.FIXED_RATIO
        assert result.fixed_ratio == 0.1
        assert result.max_position_size == 20.0
        assert result.enabled is True

    def test_logs_error_on_failure(self, generator):
        """失敗時にエラーをログ"""
        with patch(
            "app.services.auto_strategy.generators.component_generators.create_random_position_sizing_gene",
            side_effect=ValueError("Test error"),
        ):
            with patch(
                "app.services.auto_strategy.generators.component_generators.logger"
            ) as mock_logger:
                generator.generate_position_sizing_gene()

        mock_logger.error.assert_called_once()
        assert "Test error" in str(mock_logger.error.call_args)

    def test_passes_config_to_create_function(self, generator):
        """設定が生成関数に渡される"""
        from app.services.auto_strategy.genes import (
            PositionSizingGene,
            PositionSizingMethod,
        )

        mock_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,
            max_position_size=10.0,
            enabled=True,
        )

        with patch(
            "app.services.auto_strategy.generators.component_generators.create_random_position_sizing_gene",
            return_value=mock_gene,
        ) as mock_create:
            generator.generate_position_sizing_gene()

        mock_create.assert_called_once_with(generator.config)


class TestPositionSizingGeneIntegration:
    """統合テスト"""

    def test_generated_gene_has_required_attributes(self):
        """生成された遺伝子に必須属性がある"""
        from app.services.auto_strategy.generators.component_generators import (
            PositionSizingGenerator,
        )
        from app.services.auto_strategy.genes import (
            PositionSizingGene,
            PositionSizingMethod,
        )

        config = Mock()
        generator = PositionSizingGenerator(config)

        # フォールバック遺伝子を使用するようにエラーを発生させる
        with patch(
            "app.services.auto_strategy.generators.component_generators.create_random_position_sizing_gene",
            side_effect=ValueError("Force fallback"),
        ):
            result = generator.generate_position_sizing_gene()

        # 必須属性の存在を確認
        assert hasattr(result, "method")
        assert hasattr(result, "fixed_ratio")
        assert hasattr(result, "max_position_size")
        assert hasattr(result, "enabled")

        # 型の確認
        assert isinstance(result.method, PositionSizingMethod)
        assert isinstance(result.fixed_ratio, float)
        assert isinstance(result.max_position_size, float)
        assert isinstance(result.enabled, bool)
