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
    ParameterConfig,
)


class TestParameterConstraints:
    """ParameterConfig 制約機能のテスト"""

    def test_indicator_config_has_parameter_constraints_field(self):
        """IndicatorConfig に parameter_constraints フィールドが存在することをテスト"""
        config = IndicatorConfig(
            indicator_name="MACD",
        )

        # parameter_constraints フィールドが存在することを確認
        assert hasattr(config, "parameter_constraints"), (
            "IndicatorConfig に parameter_constraints フィールドがありません"
        )

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

        is_valid, _ = config.validate_constraints(params)
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

        is_valid, _ = config.validate_constraints(params)
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
        from app.services.auto_strategy.genes.validator import GeneValidator

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
        from app.services.auto_strategy.genes.validator import GeneValidator

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
        from app.services.auto_strategy.genes.validator import GeneValidator

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
        is_valid, _ = config.validate_constraints(valid_params)
        assert is_valid is True

        # 無効なパラメータ
        invalid_params = {"upper_band": 1.0, "lower_band": 2.0, "period": 20}
        is_valid, _ = config.validate_constraints(invalid_params)
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
        is_valid, _ = config.validate_constraints(valid_params)
        assert is_valid is True

        # 無効なパラメータ（差が 5 未満）
        invalid_params = {"fast": 10, "slow": 12}  # diff = 2
        is_valid, _ = config.validate_constraints(invalid_params)
        assert is_valid is False


def _macd_like_config(
    parameter_constraints: list[dict] | None = None,
    *,
    fast_max: int | float = 60,
    slow_max: int | float = 130,
) -> IndicatorConfig:
    """制約テスト用の MACD 風 IndicatorConfig を構築する。"""
    if parameter_constraints is None:
        parameter_constraints = [
            {"type": "less_than", "param1": "fast", "param2": "slow"},
        ]
    return IndicatorConfig(
        indicator_name="MACD",
        parameters={
            "fast": ParameterConfig(
                name="fast", default_value=12, min_value=2, max_value=fast_max
            ),
            "slow": ParameterConfig(
                name="slow", default_value=26, min_value=5, max_value=slow_max
            ),
            "signal": ParameterConfig(
                name="signal", default_value=9, min_value=2, max_value=50
            ),
        },
        parameter_constraints=parameter_constraints,
    )


class TestRepairParameters:
    """repair_parameters（GA生成・変異・交叉後の制約修復）のテスト"""

    def test_repair_swaps_violating_pair(self):
        """fast >= slow の違反ペアをスワップで修復し遺伝子材料を温存する"""
        config = _macd_like_config()
        params = {"fast": 50, "slow": 10, "signal": 9}

        changed = config.repair_parameters(params)

        assert params["fast"] == 10
        assert params["slow"] == 50
        assert changed == 2
        is_valid, _ = config.validate_constraints(params)
        assert is_valid is True

    def test_repair_noop_on_valid_params(self):
        """有効なパラメータは変更されない"""
        config = _macd_like_config()
        params = {"fast": 12, "slow": 26, "signal": 9}

        changed = config.repair_parameters(params)

        assert params == {"fast": 12, "slow": 26, "signal": 9}
        assert changed == 0

    def test_repair_equal_values_nudges(self):
        """同値（スワップで解消不可）は近傍値へ調整される"""
        config = _macd_like_config()
        params = {"fast": 30, "slow": 30, "signal": 9}

        changed = config.repair_parameters(params)

        assert params["fast"] < params["slow"]
        assert params["slow"] == 31
        assert changed >= 1

    def test_repair_coerces_fractional_integer_params(self):
        """デフォルトが整数のパラメータは小数から整数へ丸められる"""
        config = _macd_like_config()
        params = {"fast": 12.7, "slow": 26.2, "signal": 9.1}

        config.repair_parameters(params)

        assert params == {"fast": 12, "slow": 26, "signal": 9}
        assert all(isinstance(v, int) for v in params.values())

    def test_repair_keeps_float_params_float(self):
        """デフォルトが実数のパラメータ（scalar 等）は丸められない"""
        config = IndicatorConfig(
            indicator_name="TEST_FLOAT",
            parameters={
                "scalar": ParameterConfig(
                    name="scalar", default_value=100.0, min_value=50.0, max_value=200.0
                ),
            },
        )

        params = {"scalar": 123.456}
        changed = config.repair_parameters(params)

        assert params["scalar"] == 123.456
        assert changed == 0

    def test_repair_clamps_swapped_values_to_own_ranges(self):
        """スワップ後の値が相手側の探索範囲を超える場合はクランプされる"""
        # fast [2,70] / slow [2,50]（slow の上限が fast より小さい COPPOCK 風）
        config = _macd_like_config(fast_max=70, slow_max=50)
        params = {"fast": 70, "slow": 40, "signal": 9}

        config.repair_parameters(params)

        # スワップで slow=70 → slow の上限 50 へクランプ
        assert params == {"fast": 40, "slow": 50, "signal": 9}

    def test_repair_chain_constraints_multi_pass(self):
        """連鎖制約（fast<medium<slow）は複数パスで収束する"""
        config = IndicatorConfig(
            indicator_name="UO",
            parameters={
                "fast": ParameterConfig(
                    name="fast", default_value=7, min_value=2, max_value=50
                ),
                "medium": ParameterConfig(
                    name="medium", default_value=14, min_value=2, max_value=70
                ),
                "slow": ParameterConfig(
                    name="slow", default_value=28, min_value=5, max_value=140
                ),
            },
            parameter_constraints=[
                {"type": "less_than", "param1": "fast", "param2": "medium"},
                {"type": "less_than", "param1": "medium", "param2": "slow"},
            ],
        )

        params = {"fast": 60, "medium": 10, "slow": 5}
        config.repair_parameters(params)

        assert params["fast"] < params["medium"] < params["slow"]
        is_valid, _ = config.validate_constraints(params)
        assert is_valid is True

    def test_repair_skips_non_numeric_values(self):
        """文字列・bool・None のパラメータは対象外"""
        config = _macd_like_config()
        params = {"fast": "many", "slow": None, "signal": True}

        changed = config.repair_parameters(params)

        assert params == {"fast": "many", "slow": None, "signal": True}
        assert changed == 0

    def test_repair_greater_than_swaps(self):
        """greater_than 制約もスワップで修復される"""
        config = IndicatorConfig(
            indicator_name="BB",
            parameters={
                "upper_band": ParameterConfig(
                    name="upper_band", default_value=2.0, min_value=0.5, max_value=4.0
                ),
                "lower_band": ParameterConfig(
                    name="lower_band", default_value=1.0, min_value=0.1, max_value=2.0
                ),
            },
            parameter_constraints=[
                {
                    "type": "greater_than",
                    "param1": "upper_band",
                    "param2": "lower_band",
                },
            ],
        )

        params = {"upper_band": 1.0, "lower_band": 2.0}
        config.repair_parameters(params)

        assert params["upper_band"] == 2.0
        assert params["lower_band"] == 1.0

    def test_repair_min_difference_nudges(self):
        """min_difference 違反は param1 を範囲内で引き上げる"""
        config = IndicatorConfig(
            indicator_name="EMA_CROSS",
            parameters={
                "fast": ParameterConfig(
                    name="fast", default_value=10, min_value=2, max_value=60
                ),
                "slow": ParameterConfig(
                    name="slow", default_value=26, min_value=5, max_value=130
                ),
            },
            parameter_constraints=[
                {
                    "type": "min_difference",
                    "param1": "slow",
                    "param2": "fast",
                    "min_diff": 5,
                },
            ],
        )

        params = {"fast": 10, "slow": 12}  # diff = 2 → 不足 3
        config.repair_parameters(params)

        assert params["slow"] - params["fast"] >= 5
        assert params == {"fast": 10, "slow": 15}

    def test_repair_clamp_ranges_option(self):
        """clamp_ranges=True の場合のみ制約対象外のパラメータも範囲へクランプ"""
        config = _macd_like_config()
        params = {"fast": 12, "slow": 26, "signal": 200}  # signal 上限 50

        unchanged = dict(params)
        changed_default = config.repair_parameters(unchanged)
        assert unchanged == params
        assert changed_default == 0

        repaired = dict(params)
        changed_clamp = config.repair_parameters(repaired, clamp_ranges=True)
        assert repaired == {"fast": 12, "slow": 26, "signal": 50}
        assert changed_clamp == 1
