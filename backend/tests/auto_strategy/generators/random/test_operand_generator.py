"""
OperandGeneratorのテスト

ランダム戦略のオペランド選択ロジックのテスト
"""

import pytest
import random as std_random
from unittest.mock import Mock, patch


class TestOperandGeneratorInit:
    """初期化のテスト"""

    def test_init_with_default_weights(self):
        """デフォルトの重みで初期化"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        # spec=[]で属性を持たないモックを作成
        # getattrがデフォルト値を返すようになる
        config = Mock(spec=[])

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            generator = OperandGenerator(config)

        assert generator.config == config
        assert generator.price_data_weight == 5  # デフォルト
        assert generator.volume_data_weight == 2  # デフォルト
        assert generator.oi_fr_data_weight == 1  # デフォルト

    def test_init_with_custom_weights(self):
        """カスタム重みで初期化"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 10
        config.volume_data_weight = 5
        config.oi_fr_data_weight = 3

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            generator = OperandGenerator(config)

        assert generator.price_data_weight == 10
        assert generator.volume_data_weight == 5
        assert generator.oi_fr_data_weight == 3


class TestChooseOperand:
    """choose_operandのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 2
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1

        with patch.object(
            OperandGenerator,
            "_initialize_valid_indicators",
            return_value={"RSI", "MACD", "SMA", "EMA"},
        ):
            return OperandGenerator(config)

    def test_returns_operand_from_choices(self, generator):
        """選択肢からオペランドを返す"""
        # インジケーター付きの指標リスト
        indicator1 = Mock()
        indicator1.type = "RSI"
        indicator2 = Mock()
        indicator2.type = "SMA"

        # ランダム選択をモック
        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.choice",
            return_value="RSI",
        ):
            result = generator.choose_operand([indicator1, indicator2])

        assert result == "RSI"

    def test_returns_close_when_no_choices(self, generator):
        """選択肢がない場合はcloseを返す"""
        # 空のリスト
        with patch.object(generator, "_valid_indicator_names", set()):
            generator.price_data_weight = 0
            generator.volume_data_weight = 0
            generator.oi_fr_data_weight = 0
            result = generator.choose_operand([])

        assert result == "close"

    def test_includes_basic_data_sources(self, generator):
        """基本データソースが含まれる"""
        # price_data_weight回だけ基本データソースが追加される
        indicator = Mock()
        indicator.type = "RSI"

        # 選択肢を収集するためにモック
        original_choice = std_random.choice

        collected_choices = []

        def mock_choice(seq):
            collected_choices.extend(seq)
            return original_choice(seq)

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.choice",
            side_effect=mock_choice,
        ):
            generator.choose_operand([indicator])

        # 基本データソースが含まれることを確認
        assert "close" in collected_choices
        assert "open" in collected_choices
        assert "high" in collected_choices
        assert "low" in collected_choices

    def test_includes_volume_data(self, generator):
        """出来高データが含まれる"""
        collected_choices = []

        def mock_choice(seq):
            collected_choices.extend(seq)
            return seq[0]

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.choice",
            side_effect=mock_choice,
        ):
            generator.choose_operand([])

        assert "volume" in collected_choices

    def test_includes_oi_fr_data(self, generator):
        """OI/FRデータソースが含まれる"""
        collected_choices = []

        def mock_choice(seq):
            collected_choices.extend(seq)
            return seq[0]

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.choice",
            side_effect=mock_choice,
        ):
            generator.choose_operand([])

        assert "OpenInterest" in collected_choices
        assert "FundingRate" in collected_choices


class TestChooseRightOperand:
    """choose_right_operandのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 1
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1
        config.numeric_threshold_probability = 0.8
        config.min_compatibility_score = 0.5
        config.strict_compatibility_score = 0.8

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            return OperandGenerator(config)

    def test_returns_numeric_threshold_with_high_probability(self, generator):
        """高い確率で数値閾値を返す"""
        # random.random() が 0.5 を返すとき（< 0.8）、数値を返す
        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.random",
            return_value=0.5,
        ):
            with patch.object(
                generator, "generate_threshold_value", return_value=50.0
            ) as mock_gen:
                result = generator.choose_right_operand("RSI", [], "gt")

        mock_gen.assert_called_once_with("RSI", "gt")
        assert result == 50.0

    def test_returns_compatible_operand_with_low_probability(self, generator):
        """低い確率で互換性のあるオペランドを返す"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        generator.config.numeric_threshold_probability = 0.2

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.random",
            return_value=0.5,
        ):  # > 0.2
            with patch.object(
                generator, "choose_compatible_operand", return_value="SMA"
            ):
                with patch.object(
                    operand_grouping_system,
                    "get_compatibility_score",
                    return_value=0.9,
                ):
                    result = generator.choose_right_operand("RSI", [], "gt")

        assert result == "SMA"

    def test_fallback_to_numeric_on_low_compatibility(self, generator):
        """互換性が低い場合は数値にフォールバック"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        generator.config.numeric_threshold_probability = 0.2
        generator.config.min_compatibility_score = 0.8

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.random",
            return_value=0.5,
        ):  # > 0.2
            with patch.object(
                generator, "choose_compatible_operand", return_value="volume"
            ):
                with patch.object(
                    operand_grouping_system,
                    "get_compatibility_score",
                    return_value=0.3,  # < 0.8
                ):
                    with patch.object(
                        generator, "generate_threshold_value", return_value=30.0
                    ):
                        result = generator.choose_right_operand("RSI", [], "gt")

        assert result == 30.0


class TestChooseCompatibleOperand:
    """choose_compatible_operandのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 1
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1
        config.strict_compatibility_score = 0.9
        config.min_compatibility_score = 0.6

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            return OperandGenerator(config)

    def test_returns_strictly_compatible_operand(self, generator):
        """厳密に互換性のあるオペランドを返す"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        indicator = Mock()
        indicator.type = "RSI"

        with patch.object(
            operand_grouping_system, "get_compatible_operands", return_value=["SMA"]
        ):
            with patch(
                "app.services.auto_strategy.generators.random.operand_generator.random.choice",
                return_value="SMA",
            ):
                result = generator.choose_compatible_operand("RSI", [indicator])

        assert result == "SMA"

    def test_returns_high_compatible_when_no_strict(self, generator):
        """厳密な互換性がない場合は高い互換性から選択"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        indicator = Mock()
        indicator.type = "RSI"

        # 最初の呼び出し（厳密）は空、2回目（高い互換性）はEMAを返す
        with patch.object(
            operand_grouping_system,
            "get_compatible_operands",
            side_effect=[[], ["EMA"]],
        ):
            with patch(
                "app.services.auto_strategy.generators.random.operand_generator.random.choice",
                return_value="EMA",
            ):
                result = generator.choose_compatible_operand("RSI", [indicator])

        assert result == "EMA"

    def test_fallback_to_any_operand(self, generator):
        """互換性のあるオペランドがない場合は任意のオペランドにフォールバック"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        indicator = Mock()
        indicator.type = "RSI"

        with patch.object(
            operand_grouping_system, "get_compatible_operands", side_effect=[[], []]
        ):
            with patch(
                "app.services.auto_strategy.generators.random.operand_generator.random.choice",
                return_value="close",
            ):
                result = generator.choose_compatible_operand("RSI", [indicator])

        assert result == "close"

    def test_final_fallback_returns_close(self, generator):
        """全てのフォールバックがない場合はcloseを返す"""
        from app.services.auto_strategy.core.operand_grouping import (
            operand_grouping_system,
        )

        with patch.object(
            operand_grouping_system, "get_compatible_operands", side_effect=[[], []]
        ):
            # 空のインジケーターリストで、left_operandも"close"の場合
            # フォールバックリストからleft_operand("close")が除外されるため
            # 基本データソース（open, high, low, volume等）または最終フォールバック"close"が返される
            result = generator.choose_compatible_operand("close", [])

        # 結果は有効なオペランドである（フォールバックロジックによる）
        valid_fallbacks = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "OpenInterest",
            "FundingRate",
        ]
        assert result in valid_fallbacks


class TestGenerateThresholdValue:
    """generate_threshold_valueのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 1
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1
        config.threshold_ranges = {}

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            return OperandGenerator(config)

    def test_funding_rate_threshold(self, generator):
        """FundingRateの閾値生成"""
        with patch.object(
            generator, "_get_safe_threshold", return_value=0.0005
        ) as mock_thresh:
            result = generator.generate_threshold_value("FundingRate", "gt")

        mock_thresh.assert_called_once_with(
            "funding_rate", [0.0001, 0.001], allow_choice=True
        )
        assert result == 0.0005

    def test_open_interest_threshold(self, generator):
        """OpenInterestの閾値生成"""
        with patch.object(
            generator, "_get_safe_threshold", return_value=10000000
        ) as mock_thresh:
            result = generator.generate_threshold_value("OpenInterest", "gt")

        mock_thresh.assert_called_once_with(
            "open_interest", [1000000, 50000000], allow_choice=True
        )
        assert result == 10000000

    def test_volume_threshold(self, generator):
        """volumeの閾値生成"""
        with patch.object(
            generator, "_get_safe_threshold", return_value=50000
        ) as mock_thresh:
            result = generator.generate_threshold_value("volume", "gt")

        mock_thresh.assert_called_once_with("volume", [1000, 100000])
        assert result == 50000

    def test_oscillator_0_100_threshold(self, generator):
        """0-100オシレーターの閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.OSCILLATOR_0_100

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=70.0
            ) as mock_thresh:
                result = generator.generate_threshold_value("RSI", "gt")

        mock_thresh.assert_called_once_with("oscillator_0_100", [20, 80])
        assert result == 70.0

    def test_oscillator_plus_minus_100_threshold(self, generator):
        """±100オシレーターの閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=50.0
            ) as mock_thresh:
                result = generator.generate_threshold_value("CCI", "gt")

        mock_thresh.assert_called_once_with("oscillator_plus_minus_100", [-100, 100])
        assert result == 50.0

    def test_momentum_zero_centered_threshold(self, generator):
        """モメンタム（ゼロ中心）の閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.MOMENTUM_ZERO_CENTERED

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=0.1
            ) as mock_thresh:
                result = generator.generate_threshold_value("MOMENTUM", "gt")

        mock_thresh.assert_called_once_with("momentum_zero_centered", [-0.5, 0.5])
        assert result == 0.1

    def test_price_ratio_threshold(self, generator):
        """価格比率の閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.PRICE_RATIO

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=1.02
            ) as mock_thresh:
                result = generator.generate_threshold_value("RATIO", "gt")

        mock_thresh.assert_called_once_with("price_ratio", [0.95, 1.05])
        assert result == 1.02

    def test_price_absolute_threshold(self, generator):
        """価格絶対値の閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.PRICE_ABSOLUTE

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=0.98
            ) as mock_thresh:
                result = generator.generate_threshold_value("PRICE", "gt")

        mock_thresh.assert_called_once_with("price_ratio", [0.95, 1.05])
        assert result == 0.98

    def test_volume_scale_type_threshold(self, generator):
        """ボリュームスケールタイプの閾値生成"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.config.indicator_config import IndicatorScaleType

        mock_indicator_config = Mock()
        mock_indicator_config.scale_type = IndicatorScaleType.VOLUME

        with patch.object(
            indicator_registry,
            "get_indicator_config",
            return_value=mock_indicator_config,
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=25000
            ) as mock_thresh:
                result = generator.generate_threshold_value("OBV", "gt")

        mock_thresh.assert_called_once_with("volume", [1000, 100000])
        assert result == 25000

    def test_fallback_threshold(self, generator):
        """フォールバック閾値生成"""
        from app.services.indicators.config import indicator_registry

        # インジケーター設定が見つからない
        with patch.object(
            indicator_registry, "get_indicator_config", return_value=None
        ):
            with patch.object(
                generator, "_get_safe_threshold", return_value=1.0
            ) as mock_thresh:
                result = generator.generate_threshold_value("UNKNOWN", "gt")

        mock_thresh.assert_called_once_with("price_ratio", [0.95, 1.05])
        assert result == 1.0


class TestGetSafeThreshold:
    """_get_safe_thresholdのテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        from app.services.auto_strategy.generators.random.operand_generator import (
            OperandGenerator,
        )

        config = Mock()
        config.price_data_weight = 1
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1
        config.threshold_ranges = {}

        with patch.object(
            OperandGenerator, "_initialize_valid_indicators", return_value=set()
        ):
            return OperandGenerator(config)

    def test_uses_config_range(self, generator):
        """設定からの範囲を使用"""
        generator.config.threshold_ranges = {"test_key": [10, 20]}

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.uniform",
            return_value=15.0,
        ):
            result = generator._get_safe_threshold("test_key", [0, 100])

        assert result == 15.0

    def test_uses_default_range_when_config_missing(self, generator):
        """設定がない場合はデフォルト範囲を使用"""
        generator.config.threshold_ranges = {}

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.uniform",
            return_value=50.0,
        ):
            result = generator._get_safe_threshold("missing_key", [0, 100])

        assert result == 50.0

    def test_discrete_choice_with_allow_choice(self, generator):
        """allow_choiceがTrueで離散値リストから選択"""
        generator.config.threshold_ranges = {"discrete_key": [1, 2, 3, 4, 5]}

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.choice",
            return_value=3,
        ):
            result = generator._get_safe_threshold(
                "discrete_key", [0, 100], allow_choice=True
            )

        assert result == 3.0

    def test_range_selection_even_with_allow_choice(self, generator):
        """allow_choiceでも2要素リストは範囲として扱う"""
        generator.config.threshold_ranges = {"range_key": [10, 50]}

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.uniform",
            return_value=30.0,
        ):
            result = generator._get_safe_threshold(
                "range_key", [0, 100], allow_choice=True
            )

        assert result == 30.0

    def test_fallback_on_invalid_range(self, generator):
        """無効な範囲の場合はデフォルトにフォールバック"""
        generator.config.threshold_ranges = {"invalid_key": "not_a_list"}

        with patch(
            "app.services.auto_strategy.generators.random.operand_generator.random.uniform",
            return_value=50.0,
        ):
            result = generator._get_safe_threshold("invalid_key", [0, 100])

        assert result == 50.0


