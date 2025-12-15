"""
パラメータ依存関係制約のテスト

Issue: GeneValidator で個々のパラメータの範囲チェックは行っているが、
パラメータ間の論理的整合性（例: MACD の fast < slow）をチェックしていない問題。

解決策: IndicatorConfig に parameter_constraints を追加し、
GeneValidator でパラメータ依存関係を検証する。
"""

from unittest.mock import MagicMock, patch

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
)


class TestParameterConstraints:
    """ParameterConfig 制約機能のテスト"""

    def test_indicator_config_has_parameter_constraints_field(self):
        """IndicatorConfig に parameter_constraints フィールドが存在することをテスト"""
        config = IndicatorConfig(
            indicator_name="MACD",
        )

        # parameter_constraints フィールドが存在することを確認
        assert hasattr(
            config, "parameter_constraints"
        ), "IndicatorConfig に parameter_constraints フィールドがありません"

    def test_indicator_config_with_less_than_constraint(self):
        """< 制約（fast < slow）を定義できることをテスト"""
        config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        assert config.parameter_constraints is not None
        assert len(config.parameter_constraints) == 1
        assert config.parameter_constraints[0]["type"] == "less_than"
        assert config.parameter_constraints[0]["param1"] == "fast"
        assert config.parameter_constraints[0]["param2"] == "slow"

    def test_validate_constraints_valid_macd_params(self):
        """有効な MACD パラメータ（fast=12, slow=26）が検証をパス"""
        config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        params = {"fast": 12, "slow": 26, "signal": 9}

        # validate_constraints メソッドが存在することを確認
        assert hasattr(config, "validate_constraints")

        # 有効なパラメータはパスする
        is_valid, errors = config.validate_constraints(params)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_constraints_invalid_macd_params(self):
        """無効な MACD パラメータ（fast=50, slow=10）が検証に失敗"""
        config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        params = {"fast": 50, "slow": 10, "signal": 9}

        is_valid, errors = config.validate_constraints(params)
        assert is_valid is False
        assert len(errors) > 0
        assert "fast" in errors[0] or "slow" in errors[0]

    def test_validate_constraints_equal_values_fails(self):
        """同じ値（fast=20, slow=20）も検証に失敗"""
        config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        params = {"fast": 20, "slow": 20, "signal": 9}

        is_valid, errors = config.validate_constraints(params)
        assert is_valid is False

    def test_validate_constraints_missing_param_skipped(self):
        """制約のパラメータが存在しない場合はスキップ"""
        config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        # slow が存在しない
        params = {"fast": 12, "signal": 9}

        is_valid, errors = config.validate_constraints(params)
        # 制約チェックはスキップされるので True
        assert is_valid is True

    def test_validate_constraints_no_constraints(self):
        """制約なしの場合は常に True"""
        config = IndicatorConfig(
            indicator_name="RSI",
        )

        params = {"length": 14}

        is_valid, errors = config.validate_constraints(params)
        assert is_valid is True
        assert len(errors) == 0


class TestGeneValidatorWithConstraints:
    """GeneValidator がパラメータ制約を検証するテスト"""

    def test_validator_checks_parameter_constraints(self):
        """GeneValidator が IndicatorConfig の制約を使用してパラメータを検証"""
        from app.services.auto_strategy.models.validator import GeneValidator

        validator = GeneValidator()

        # 無効な MACD パラメータを持つ指標遺伝子をモック
        indicator_gene = MagicMock()
        indicator_gene.type = "MACD"
        indicator_gene.parameters = {"fast": 50, "slow": 10, "signal": 9}  # 無効
        indicator_gene.enabled = True
        indicator_gene.timeframe = None

        # indicator_registry をモックして制約付き IndicatorConfig を返す
        mock_config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        with patch(
            "app.services.indicators.config.indicator_registry"
        ) as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            # 無効なパラメータ制約により検証が失敗することを確認
            is_valid = validator.validate_indicator_gene(indicator_gene)

            # 制約違反のため False を返すはず
            assert is_valid is False

    def test_validator_passes_valid_constraints(self):
        """GeneValidator が有効なパラメータ制約をパスさせる"""
        from app.services.auto_strategy.models.validator import GeneValidator

        validator = GeneValidator()

        # 有効な MACD パラメータを持つ指標遺伝子をモック
        indicator_gene = MagicMock()
        indicator_gene.type = "MACD"
        indicator_gene.parameters = {"fast": 12, "slow": 26, "signal": 9}  # 有効
        indicator_gene.enabled = True
        indicator_gene.timeframe = None

        mock_config = IndicatorConfig(
            indicator_name="MACD",
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "slow"},
            ],
        )

        with patch(
            "app.services.indicators.config.indicator_registry"
        ) as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            is_valid = validator.validate_indicator_gene(indicator_gene)

            # 有効なパラメータなので True を返すはず
            assert is_valid is True

    def test_validator_passes_without_constraints(self):
        """制約なしの指標は従来通り検証される"""
        from app.services.auto_strategy.models.validator import GeneValidator

        validator = GeneValidator()

        # RSI（制約なし）の指標遺伝子をモック
        indicator_gene = MagicMock()
        indicator_gene.type = "RSI"
        indicator_gene.parameters = {"length": 14}
        indicator_gene.enabled = True
        indicator_gene.timeframe = None

        mock_config = IndicatorConfig(
            indicator_name="RSI",
            # parameter_constraints なし
        )

        with patch(
            "app.services.indicators.config.indicator_registry"
        ) as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            is_valid = validator.validate_indicator_gene(indicator_gene)

            assert is_valid is True


class TestGreaterThanConstraint:
    """greater_than 制約のテスト"""

    def test_validate_greater_than_constraint(self):
        """> 制約（upper > lower）が正しく検証される"""
        config = IndicatorConfig(
            indicator_name="BB",
            parameter_constraints=[
                {
                    "type": "greater_than",
                    "param1": "upper_band",
                    "param2": "lower_band",
                },
            ],
        )

        # 有効なパラメータ
        valid_params = {"upper_band": 2.0, "lower_band": 1.0, "period": 20}
        is_valid, errors = config.validate_constraints(valid_params)
        assert is_valid is True

        # 無効なパラメータ
        invalid_params = {"upper_band": 1.0, "lower_band": 2.0, "period": 20}
        is_valid, errors = config.validate_constraints(invalid_params)
        assert is_valid is False


class TestMinDifferenceConstraint:
    """min_difference 制約のテスト（パラメータ間の最小差を保証）"""

    def test_validate_min_difference_constraint(self):
        """min_difference 制約が正しく検証される"""
        config = IndicatorConfig(
            indicator_name="EMA_CROSS",
            parameter_constraints=[
                {
                    "type": "min_difference",
                    "param1": "slow",
                    "param2": "fast",
                    "min_diff": 5,
                },
            ],
        )

        # 有効なパラメータ（差が 5 以上）
        valid_params = {"fast": 10, "slow": 20}  # diff = 10
        is_valid, errors = config.validate_constraints(valid_params)
        assert is_valid is True

        # 無効なパラメータ（差が 5 未満）
        invalid_params = {"fast": 10, "slow": 12}  # diff = 2
        is_valid, errors = config.validate_constraints(invalid_params)
        assert is_valid is False


