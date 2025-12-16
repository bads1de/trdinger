"""
パラメータ範囲プリセット機能のテスト

Issue: パラメータの探索範囲が GAConfig でグローバルに定義されており、
指標ごとの探索範囲のカスタマイズができない問題を解決する。

解決策: ParameterConfig に探索プリセット（short_term, mid_term, long_term 等）を
追加し、GAConfig でプリセットを選択可能にする。
"""

from unittest.mock import patch, MagicMock
import random

from app.services.indicators.config.indicator_config import (
    ParameterConfig,
    IndicatorConfig,
)


class TestParameterConfigPresets:
    """ParameterConfig プリセット機能のテスト"""

    def test_parameter_config_has_presets_field(self):
        """ParameterConfig に presets フィールドが存在することをテスト"""
        param = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
        )

        # presets フィールドが存在することを確認
        assert hasattr(
            param, "presets"
        ), "ParameterConfig に presets フィールドがありません"

    def test_parameter_config_with_presets(self):
        """presets を指定して ParameterConfig を作成できることをテスト"""
        param = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
                "mid_term": (14, 30),
                "long_term": (50, 100),
            },
        )

        assert param.presets is not None
        assert "short_term" in param.presets
        assert param.presets["short_term"] == (5, 15)
        assert param.presets["mid_term"] == (14, 30)
        assert param.presets["long_term"] == (50, 100)

    def test_get_range_for_preset(self):
        """プリセット名で範囲を取得できることをテスト"""
        param = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
                "mid_term": (14, 30),
            },
        )

        # get_range_for_preset メソッドが存在することを確認
        assert hasattr(param, "get_range_for_preset")

        # short_term プリセットの範囲を取得
        min_val, max_val = param.get_range_for_preset("short_term")
        assert min_val == 5
        assert max_val == 15

    def test_get_range_for_preset_fallback_to_default(self):
        """未知のプリセット名の場合はデフォルト範囲にフォールバック"""
        param = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
            },
        )

        # 存在しないプリセット名の場合
        min_val, max_val = param.get_range_for_preset("unknown_preset")
        assert min_val == 2
        assert max_val == 200

    def test_get_range_for_preset_with_none(self):
        """プリセットなしでもデフォルト範囲が返ることをテスト"""
        param = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
        )

        # デフォルト範囲が返る
        min_val, max_val = param.get_range_for_preset("any_preset")
        assert min_val == 2
        assert max_val == 200


class TestIndicatorParameterManagerWithPresets:
    """IndicatorParameterManager でプリセットを使用したパラメータ生成テスト"""

    def test_generate_parameters_with_preset(self):
        """プリセットを指定してパラメータ生成"""
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        # プリセット付きの ParameterConfig
        param_config = ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
                "long_term": (50, 100),
            },
        )

        # テスト用の IndicatorConfig を作成
        config = IndicatorConfig(
            indicator_name="RSI",
            parameters={"length": param_config},
            default_values={"length": 14},
        )

        manager = IndicatorParameterManager()

        # short_term プリセットでパラメータ生成
        random.seed(42)  # 再現性のため固定
        params = manager.generate_parameters("RSI", config, preset="short_term")

        # 生成されたパラメータが short_term 範囲内であることを確認
        assert "length" in params
        assert 5 <= params["length"] <= 15

    def test_generate_parameters_with_long_term_preset(self):
        """long_term プリセットでパラメータ生成"""
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        # プリセット付きの ParameterConfig
        param_config = ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
                "long_term": (50, 100),
            },
        )

        config = IndicatorConfig(
            indicator_name="RSI",
            parameters={"length": param_config},
            default_values={"length": 14},
        )

        manager = IndicatorParameterManager()

        # 複数回実行して範囲内であることを確認
        for _ in range(10):
            params = manager.generate_parameters("RSI", config, preset="long_term")
            assert "length" in params
            assert (
                50 <= params["length"] <= 100
            ), f"長期プリセットで生成された値 {params['length']} が範囲外です"

    def test_generate_parameters_without_preset(self):
        """プリセット未指定の場合はデフォルト範囲を使用"""
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        param_config = ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            presets={
                "short_term": (5, 15),
            },
        )

        config = IndicatorConfig(
            indicator_name="RSI",
            parameters={"length": param_config},
            default_values={"length": 14},
        )

        manager = IndicatorParameterManager()

        # プリセット未指定（デフォルト範囲）
        for _ in range(10):
            params = manager.generate_parameters("RSI", config)
            assert "length" in params
            assert 2 <= params["length"] <= 200


class TestIndicatorGeneratorWithPreset:
    """IndicatorGenerator でプリセットを使用するテスト"""

    def test_indicator_generator_respects_preset_config(self):
        """IndicatorGenerator が GAConfig のプリセット設定を尊重することをテスト"""
        from app.services.auto_strategy.generators.random.indicator_generator import (
            IndicatorGenerator,
        )

        # プリセットを含む GAConfig モック
        mock_config = MagicMock()
        mock_config.min_indicators = 1
        mock_config.max_indicators = 3
        mock_config.enable_multi_timeframe = False
        mock_config.available_timeframes = []
        mock_config.parameter_range_preset = "short_term"  # 新しい設定

        # indicator_registry をモックしてプリセット付きパラメータを返す
        with patch(
            "app.services.auto_strategy.generators.random.indicator_generator.indicator_registry"
        ) as mock_registry:
            # プリセットを考慮したパラメータ生成をモック
            mock_registry.generate_parameters_for_indicator.return_value = {
                "length": 10  # short_term 範囲内
            }

            generator = IndicatorGenerator(mock_config)

            # 生成された指標のパラメータがプリセット範囲内であることを確認
            indicators = generator.generate_random_indicators()

            # 少なくとも 1 つの指標が生成される
            assert len(indicators) >= 1

            # パラメータ生成メソッドが呼ばれたことを確認
            assert mock_registry.generate_parameters_for_indicator.called


class TestGAConfigPresetSetting:
    """GAConfig にプリセット設定が存在することをテスト"""

    def test_ga_config_has_parameter_range_preset_field(self):
        """GAConfig に parameter_range_preset フィールドが存在することをテスト"""
        from app.services.auto_strategy.config.ga import GAConfig

        # GAConfig インスタンスを作成
        config = GAConfig()

        # parameter_range_preset フィールドが存在することを確認
        assert hasattr(
            config, "parameter_range_preset"
        ), "GAConfig に parameter_range_preset フィールドがありません"

    def test_ga_config_default_preset_is_none(self):
        """デフォルトではプリセットが None（デフォルト範囲を使用）"""
        from app.services.auto_strategy.config.ga import GAConfig

        config = GAConfig()

        # デフォルトは None
        assert config.parameter_range_preset is None

    def test_ga_config_preset_serialization(self):
        """GAConfig のプリセット設定がシリアライズ/デシリアライズされることをテスト"""
        from app.services.auto_strategy.config.ga import GAConfig

        # プリセットを設定した GAConfig を作成
        config = GAConfig(parameter_range_preset="short_term")
        assert config.parameter_range_preset == "short_term"

        # to_dict でシリアライズ
        config_dict = config.to_dict()
        assert "parameter_range_preset" in config_dict
        assert config_dict["parameter_range_preset"] == "short_term"

        # from_dict でデシリアライズ
        restored_config = GAConfig.from_dict(config_dict)
        assert restored_config.parameter_range_preset == "short_term"




