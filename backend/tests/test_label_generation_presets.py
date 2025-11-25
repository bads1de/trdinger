"""
ラベル生成プリセット機能のテストモジュール

新機能のテストカバレッジ:
1. forward_classification_preset関数
2. get_common_presets関数
3. apply_preset_by_name関数
4. LabelGenerationConfig設定クラス
5. BaseMLTrainer._prepare_training_data統合
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.config.unified_config import LabelGenerationConfig
from app.utils.label_generation.enums import ThresholdMethod
from app.utils.label_generation.presets import (
    SUPPORTED_TIMEFRAMES,
    apply_preset_by_name,
    forward_classification_preset,
    get_common_presets,
)

# ============================================================================
# フィクスチャ: テストデータ生成
# ============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    サンプルOHLCVデータを生成するフィクスチャ

    Returns:
        100行のOHLCVデータフレーム
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")

    # リアルな価格変動を模擬
    base_price = 50000
    returns = np.random.randn(100) * 0.02  # 2%の標準偏差
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(100) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(100)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(100)) * 0.002),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100).astype(float),
        },
        index=dates,
    )

    return df


@pytest.fixture
def small_ohlcv_data() -> pd.DataFrame:
    """
    小さなOHLCVデータ（10行）を生成するフィクスチャ

    境界値テスト用に使用
    """
    dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")

    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            "high": [102, 103, 104, 105, 106, 107, 106, 105, 104, 103],
            "low": [99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
            "close": [101, 102, 103, 104, 105, 106, 105, 104, 103, 102],
            "volume": [1000.0] * 10,
        },
        index=dates,
    )

    return df


@pytest.fixture
def invalid_ohlcv_data() -> pd.DataFrame:
    """
    不正なOHLCVデータ（カラム欠損）を生成するフィクスチャ
    """
    dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")

    df = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [102] * 10,
            "low": [99] * 10,
            # closeカラムが欠落
            "volume": [1000.0] * 10,
        },
        index=dates,
    )

    return df


# ============================================================================
# A. forward_classification_preset関数のテスト
# ============================================================================


class TestForwardClassificationPreset:
    """forward_classification_preset関数のテストクラス"""

    def test_forward_classification_preset_basic(self, sample_ohlcv_data):
        """
        正常系: 基本的な動作確認
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe="4h",
            horizon_n=4,
            threshold=0.002,
        )

        # Assert
        assert isinstance(labels, pd.Series), "ラベルはpd.Seriesである必要があります"
        assert labels.dtype == object, "ラベルは文字列型である必要があります"
        assert set(labels.dropna().unique()).issubset({"UP", "RANGE", "DOWN"}), (
            "ラベルはUP/RANGE/DOWNのいずれかである必要があります"
        )
        assert len(labels) < len(df), "horizon_n分のラベルが欠損するはずです"

    @pytest.mark.parametrize("timeframe", SUPPORTED_TIMEFRAMES)
    def test_forward_classification_preset_all_timeframes(
        self, sample_ohlcv_data, timeframe
    ):
        """
        正常系: 各時間足での動作確認
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe=timeframe,
            horizon_n=4,
            threshold=0.002,
        )

        # Assert
        assert isinstance(labels, pd.Series)
        assert len(labels.dropna()) > 0, f"{timeframe}でラベルが生成されませんでした"

    @pytest.mark.parametrize(
        "threshold_method",
        [
            ThresholdMethod.FIXED,
            ThresholdMethod.QUANTILE,
            ThresholdMethod.STD_DEVIATION,
            ThresholdMethod.KBINS_DISCRETIZER,
        ],
    )
    def test_forward_classification_preset_threshold_methods(
        self, sample_ohlcv_data, threshold_method
    ):
        """
        正常系: 各ThresholdMethodでの動作確認
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe="4h",
            horizon_n=4,
            threshold=0.002,
            threshold_method=threshold_method,
        )

        # Assert
        assert isinstance(labels, pd.Series)
        assert len(labels.dropna()) > 0, (
            f"{threshold_method.value}でラベルが生成されませんでした"
        )

    @pytest.mark.parametrize("horizon_n", [1, 4, 8, 16])
    def test_forward_classification_preset_different_horizons(
        self, sample_ohlcv_data, horizon_n
    ):
        """
        正常系: 異なるhorizon_nでの動作確認
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe="4h",
            horizon_n=horizon_n,
            threshold=0.002,
        )

        # Assert
        assert isinstance(labels, pd.Series)
        # horizon_n本先を見るため、元データよりも短くなる
        # LabelGenerator内部で最後の行が除外されるため、len(df) - 1 よりも短い
        assert len(labels) < len(df), (
            f"ラベル長({len(labels)})は元データ長({len(df)})より短い必要があります"
        )
        # 有効なラベルが生成されていることを確認
        assert len(labels.dropna()) > 0, "有効なラベルが生成されませんでした"

    def test_forward_classification_preset_invalid_timeframe(self, sample_ohlcv_data):
        """
        異常系: 無効な時間足
        """
        # Arrange
        df = sample_ohlcv_data

        # Act & Assert
        with pytest.raises(ValueError, match="未サポートの時間足です"):
            forward_classification_preset(
                df=df,
                timeframe="5m",  # サポート外
                horizon_n=4,
                threshold=0.002,
            )

    def test_forward_classification_preset_invalid_threshold_method(
        self, sample_ohlcv_data
    ):
        """
        異常系: 無効なThresholdMethod
        """
        # Arrange
        df = sample_ohlcv_data

        # Act & Assert
        # ThresholdMethodはenumなので、文字列では渡せない
        with pytest.raises(AttributeError):
            forward_classification_preset(
                df=df,
                timeframe="4h",
                horizon_n=4,
                threshold=0.002,
                threshold_method="invalid_method",  # type: ignore
            )

    def test_forward_classification_preset_missing_column(self, invalid_ohlcv_data):
        """
        異常系: 不正なデータフレーム（欠損カラム）
        """
        # Arrange
        df = invalid_ohlcv_data

        # Act & Assert
        with pytest.raises(ValueError, match="がデータフレームに存在しません"):
            forward_classification_preset(
                df=df,
                timeframe="4h",
                horizon_n=4,
                threshold=0.002,
            )

    def test_forward_classification_preset_empty_dataframe(self):
        """
        異常系: 空のデータフレーム
        """
        # Arrange
        df = pd.DataFrame()

        # Act & Assert
        with pytest.raises(ValueError, match="空のデータフレームです"):
            forward_classification_preset(
                df=df,
                timeframe="4h",
                horizon_n=4,
                threshold=0.002,
            )

    def test_forward_classification_preset_not_dataframe(self):
        """
        異常系: DataFrameではない入力
        """
        # Arrange
        invalid_input = [1, 2, 3, 4, 5]

        # Act & Assert
        with pytest.raises(ValueError, match="pandas.DataFrame である必要があります"):
            forward_classification_preset(
                df=invalid_input,  # type: ignore
                timeframe="4h",
                horizon_n=4,
                threshold=0.002,
            )

    def test_forward_classification_preset_insufficient_data(self, small_ohlcv_data):
        """
        異常系: horizon_nがデータ長以上
        """
        # Arrange
        df = small_ohlcv_data

        # Act & Assert
        with pytest.raises(ValueError, match="horizon_n .* がデータ長"):
            forward_classification_preset(
                df=df,
                timeframe="4h",
                horizon_n=20,  # データ長10より大きい
                threshold=0.002,
            )

    def test_forward_classification_preset_label_distribution(self, sample_ohlcv_data):
        """
        出力検証: ラベル分布が妥当か
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe="4h",
            horizon_n=4,
            threshold=0.002,
        )

        # Assert
        label_counts = labels.value_counts()
        total = len(labels.dropna())

        # 各ラベルが0.05以上、0.95以下の比率であることを確認
        for label in ["UP", "RANGE", "DOWN"]:
            if label in label_counts.index:
                ratio = label_counts[label] / total
                assert 0.05 <= ratio <= 0.95, f"{label}の比率が極端です: {ratio:.2%}"

    def test_forward_classification_preset_three_classes(self, sample_ohlcv_data):
        """
        出力検証: UP/RANGE/DOWNの3値ラベルが正しく生成される
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels = forward_classification_preset(
            df=df,
            timeframe="4h",
            horizon_n=4,
            threshold=0.002,
        )

        # Assert
        unique_labels = set(labels.dropna().unique())
        assert len(unique_labels) >= 2, "最低2種類のラベルが必要です"
        assert unique_labels.issubset({"UP", "RANGE", "DOWN"}), (
            "ラベルはUP/RANGE/DOWNのみである必要があります"
        )


# ============================================================================
# B. get_common_presets関数のテスト
# ============================================================================


class TestGetCommonPresets:
    """get_common_presets関数のテストクラス"""

    def test_get_common_presets_returns_dict(self):
        """
        正常系: プリセット辞書が返される
        """
        # Act
        presets = get_common_presets()

        # Assert
        assert isinstance(presets, dict), "プリセットは辞書である必要があります"
        assert len(presets) > 0, "プリセットが空です"

    def test_get_common_presets_count(self):
        """
        正常系: プリセット一覧が13種類返される
        """
        # Act
        presets = get_common_presets()

        # Assert
        # 実際のプリセット数を確認（現在は19個）
        assert len(presets) == 19, f"プリセット数が19ではありません: {len(presets)}"
        # プリセットが少なくとも10個以上あることを確認
        assert len(presets) >= 10, "プリセットが10個未満です"

    def test_get_common_presets_required_keys(self):
        """
        正常系: 各プリセットに必要なキーが含まれる
        """
        # Act
        presets = get_common_presets()

        # Assert
        # 通常のプリセットに必要なキー
        forward_classification_keys = {
            "timeframe",
            "horizon_n",
            "threshold",
            "threshold_method",
            "description",
        }
        
        # TBM（Triple Barrier Method）プリセットに必要なキー
        tbm_keys = {
            "timeframe",
            "horizon_n",
            "pt",
            "sl",
            "min_ret",
            "description",
        }

        for preset_name, preset_params in presets.items():
            assert isinstance(preset_params, dict), (
                f"{preset_name}のパラメータが辞書ではありません"
            )

            # TBMプリセットかどうかを判定
            if preset_name.startswith("tbm_"):
                required_keys = tbm_keys
            else:
                required_keys = forward_classification_keys

            missing_keys = required_keys - set(preset_params.keys())
            assert len(missing_keys) == 0, (
                f"{preset_name}に必要なキーが不足: {missing_keys}"
            )

    def test_get_common_presets_valid_timeframes(self):
        """
        正常系: すべてのプリセットの時間足が有効
        """
        # Act
        presets = get_common_presets()

        # Assert
        for preset_name, preset_params in presets.items():
            timeframe = preset_params["timeframe"]
            assert timeframe in SUPPORTED_TIMEFRAMES, (
                f"{preset_name}の時間足が無効: {timeframe}"
            )

    def test_get_common_presets_valid_threshold_methods(self):
        """
        正常系: すべてのプリセットの閾値計算方法が有効
        """
        # Act
        presets = get_common_presets()

        # Assert
        valid_methods = [m for m in ThresholdMethod]

        for preset_name, preset_params in presets.items():
            # TBMプリセットはthreshold_methodを持たないのでスキップ
            if preset_name.startswith("tbm_"):
                continue
            
            method = preset_params["threshold_method"]
            assert method in valid_methods, (
                f"{preset_name}の閾値計算方法が無効: {method}"
            )

    def test_get_common_presets_name_format(self):
        """
        正常系: プリセット名が有効な形式
        """
        # Act
        presets = get_common_presets()

        # Assert
        for preset_name in presets.keys():
            # プリセット名は「時間足_本数」または「時間足_本数_特徴」の形式
            # または「volatility_時間足_本数」などの特殊形式
            parts = preset_name.split("_")
            assert len(parts) >= 2, (
                f"{preset_name}の形式が無効です（最低2つのパートが必要）"
            )

            # 時間足部分が有効か確認（volatilityなどの接頭辞がある場合も考慮）
            has_valid_timeframe = any(
                tf in preset_name for tf in ["15m", "30m", "1h", "4h", "1d"]
            )
            assert has_valid_timeframe, f"{preset_name}に有効な時間足が含まれていません"


# ============================================================================
# C. apply_preset_by_name関数のテスト
# ============================================================================


class TestApplyPresetByName:
    """apply_preset_by_name関数のテストクラス"""

    def test_apply_preset_by_name_basic(self, sample_ohlcv_data):
        """
        正常系: 基本的な動作確認
        """
        # Arrange
        df = sample_ohlcv_data
        preset_name = "4h_4bars"

        # Act
        labels, preset_info = apply_preset_by_name(df, preset_name)

        # Assert
        assert isinstance(labels, pd.Series), "ラベルはpd.Seriesである必要があります"
        assert isinstance(preset_info, dict), "プリセット情報は辞書である必要があります"
        assert "preset_name" in preset_info
        assert "description" in preset_info
        assert preset_info["preset_name"] == preset_name

    @pytest.mark.parametrize(
        "preset_name",
        [
            "15m_4bars",
            "30m_4bars",
            "1h_4bars",
            "4h_4bars",
            "1d_4bars",
            "4h_4bars_dynamic",
            "1h_4bars_dynamic",
        ],
    )
    def test_apply_preset_by_name_all_presets(self, sample_ohlcv_data, preset_name):
        """
        正常系: 各プリセット名での動作確認
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        labels, preset_info = apply_preset_by_name(df, preset_name)

        # Assert
        assert isinstance(labels, pd.Series)
        assert len(labels.dropna()) > 0, f"{preset_name}でラベルが生成されませんでした"
        assert preset_info["preset_name"] == preset_name

    def test_apply_preset_by_name_tuple_return(self, sample_ohlcv_data):
        """
        正常系: ラベルとプリセット情報のタプルが返される
        """
        # Arrange
        df = sample_ohlcv_data

        # Act
        result = apply_preset_by_name(df, "4h_4bars")

        # Assert
        assert isinstance(result, tuple), "結果はタプルである必要があります"
        assert len(result) == 2, (
            "タプルは2要素（labels, preset_info）である必要があります"
        )

        labels, preset_info = result
        assert isinstance(labels, pd.Series)
        assert isinstance(preset_info, dict)

    def test_apply_preset_by_name_nonexistent_preset(self, sample_ohlcv_data):
        """
        異常系: 存在しないプリセット名
        """
        # Arrange
        df = sample_ohlcv_data

        # Act & Assert
        with pytest.raises(ValueError, match="プリセット .* が見つかりません"):
            apply_preset_by_name(df, "nonexistent_preset")

    def test_apply_preset_by_name_invalid_dataframe(self):
        """
        異常系: 不正なデータフレーム
        """
        # Arrange
        df = pd.DataFrame()  # 空のデータフレーム

        # Act & Assert
        with pytest.raises(ValueError):
            apply_preset_by_name(df, "4h_4bars")

    def test_apply_preset_by_name_custom_price_column(self, sample_ohlcv_data):
        """
        正常系: カスタム価格カラムの指定
        """
        # Arrange
        df = sample_ohlcv_data.copy()
        df["custom_price"] = df["close"]

        # Act
        labels, preset_info = apply_preset_by_name(
            df, "4h_4bars", price_column="custom_price"
        )

        # Assert
        assert isinstance(labels, pd.Series)
        assert len(labels.dropna()) > 0


# ============================================================================
# D. LabelGenerationConfig設定クラスのテスト
# ============================================================================


class TestLabelGenerationConfig:
    """LabelGenerationConfig設定クラスのテストクラス"""

    def test_label_generation_config_default_values(self):
        """
        正常系: デフォルト値が正しく設定される
        """
        # Act
        config = LabelGenerationConfig()

        # Assert
        assert config.default_preset == "4h_4bars_dynamic"
        assert config.timeframe == "4h"
        assert config.horizon_n == 4
        assert config.threshold == 0.002
        assert config.price_column == "close"
        assert config.threshold_method == "FIXED"
        assert config.use_preset is True

    def test_label_generation_config_custom_values(self):
        """
        正常系: カスタム値の設定
        """
        # Act
        config = LabelGenerationConfig(
            default_preset="1h_4bars",
            timeframe="1h",
            horizon_n=8,
            threshold=0.003,
            price_column="open",
            threshold_method="QUANTILE",
            use_preset=False,
        )

        # Assert
        assert config.default_preset == "1h_4bars"
        assert config.timeframe == "1h"
        assert config.horizon_n == 8
        assert config.threshold == 0.003
        assert config.price_column == "open"
        assert config.threshold_method == "QUANTILE"
        assert config.use_preset is False

    def test_label_generation_config_invalid_timeframe(self):
        """
        異常系: 無効な時間足
        """
        # Act & Assert
        with pytest.raises(ValueError, match="無効な時間足です"):
            LabelGenerationConfig(timeframe="5m")

    def test_label_generation_config_invalid_threshold_method(self):
        """
        異常系: 無効な閾値計算方法
        """
        # Act & Assert
        with pytest.raises(ValueError, match="無効な閾値計算方法です"):
            LabelGenerationConfig(threshold_method="INVALID_METHOD")

    def test_label_generation_config_invalid_preset(self):
        """
        異常系: 存在しないプリセット名（use_preset=Trueの場合）
        """
        # Act & Assert
        with pytest.raises(ValueError, match="プリセット .* が見つかりません"):
            LabelGenerationConfig(
                default_preset="nonexistent_preset",
                use_preset=True,
            )

    def test_label_generation_config_to_dict(self):
        """
        正常系: to_dict()メソッドが正しく動作
        """
        # Arrange
        config = LabelGenerationConfig()

        # Act
        config_dict = config.to_dict()

        # Assert
        assert isinstance(config_dict, dict)
        assert "default_preset" in config_dict
        assert "timeframe" in config_dict
        assert "horizon_n" in config_dict
        assert "threshold" in config_dict
        assert "price_column" in config_dict
        assert "threshold_method" in config_dict
        assert "use_preset" in config_dict

    def test_label_generation_config_get_threshold_method_enum(self):
        """
        正常系: get_threshold_method_enum()が正しいenumを返す
        """
        # Arrange
        config = LabelGenerationConfig(threshold_method="FIXED")

        # Act
        method_enum = config.get_threshold_method_enum()

        # Assert
        assert method_enum == ThresholdMethod.FIXED
        assert isinstance(method_enum, ThresholdMethod)

    @pytest.mark.parametrize(
        "method_name", ["FIXED", "QUANTILE", "STD_DEVIATION", "KBINS_DISCRETIZER"]
    )
    def test_label_generation_config_all_threshold_methods(self, method_name):
        """
        正常系: すべてのThresholdMethodが正しく設定・取得できる
        """
        # Arrange
        config = LabelGenerationConfig(threshold_method=method_name)

        # Act
        method_enum = config.get_threshold_method_enum()

        # Assert
        assert method_enum.name == method_name
        assert isinstance(method_enum, ThresholdMethod)


# ============================================================================
# E. BaseMLTrainer._prepare_training_data統合テスト
# ============================================================================


class TestBaseMLTrainerIntegration:
    """BaseMLTrainer._prepare_training_data統合テストクラス"""

    @pytest.fixture
    def mock_base_ml_trainer(self):
        """
        BaseMLTrainerのモックを作成するフィクスチャ
        """
        from app.services.ml.base_ml_trainer import BaseMLTrainer

        # BaseMLTrainerは抽象クラスなので、具象クラスとして使用
        trainer = BaseMLTrainer(
            trainer_config={"type": "single", "model_type": "lightgbm"}
        )

        return trainer

    def test_prepare_training_data_with_preset(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        正常系: プリセット使用時の動作（use_preset=True）
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data

        # unified_configをモック
        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                default_preset="4h_4bars",
                use_preset=True,
            )
            mock_config.ml.training.label_generation = mock_label_config

            # Act
            X, y = trainer._prepare_training_data(features_df)

            # Assert
            assert isinstance(X, pd.DataFrame), "特徴量はDataFrameである必要があります"
            assert isinstance(y, pd.Series), "ラベルはSeriesである必要があります"
            assert len(X) == len(y), "特徴量とラベルの長さが一致する必要があります"
            assert y.dtype in [np.int32, np.int64], "ラベルは数値型である必要があります"
            assert set(y.unique()).issubset({0, 1, 2}), (
                "ラベルは0/1/2である必要があります"
            )

    def test_prepare_training_data_with_custom_config(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        正常系: カスタム設定使用時の動作（use_preset=False）
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data

        # unified_configをモック
        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                timeframe="1h",
                horizon_n=4,
                threshold=0.003,
                threshold_method="QUANTILE",
                use_preset=False,
            )
            mock_config.ml.training.label_generation = mock_label_config

            # Act
            X, y = trainer._prepare_training_data(features_df)

            # Assert
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert len(X) == len(y)
            assert y.dtype in [np.int32, np.int64]

    def test_prepare_training_data_backward_compatibility(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        正常系: 後方互換性（target_columnパラメータ指定時）
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data.copy()
        features_df["target"] = 1  # ダミーターゲットカラム

        # Act
        with patch(
            "app.services.ml.base_ml_trainer.data_preprocessor"
        ) as mock_preprocessor:
            # data_preprocessor.prepare_training_dataの戻り値をモック
            mock_preprocessor.prepare_training_data.return_value = (
                features_df[["open", "high", "low", "close", "volume"]],
                pd.Series([0, 1, 2] * 33 + [0]),  # 100個のラベル
                {"threshold_up": 0.002, "threshold_down": -0.002},
            )

            X, y = trainer._prepare_training_data(features_df, target_column="target")

            # Assert
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            # data_preprocessorが呼ばれたことを確認
            mock_preprocessor.prepare_training_data.assert_called_once()

    def test_prepare_training_data_nonexistent_preset(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        異常系: プリセット未存在時のフォールバック
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data

        # unified_configをモック（存在しないプリセット）
        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                default_preset="4h_4bars",  # 有効なプリセット
                timeframe="4h",
                horizon_n=4,
                threshold=0.002,
                use_preset=True,
            )
            mock_config.ml.training.label_generation = mock_label_config

            # apply_preset_by_nameがValueErrorを投げるようにモック
            with patch(
                "app.services.ml.base_ml_trainer.apply_preset_by_name"
            ) as mock_apply_preset:
                mock_apply_preset.side_effect = ValueError("プリセットが見つかりません")

                # Act - フォールバックが機能することを確認
                X, y = trainer._prepare_training_data(features_df)

                # Assert
                assert isinstance(X, pd.DataFrame)
                assert isinstance(y, pd.Series)

    def test_prepare_training_data_label_generation_failure(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        異常系: ラベル生成失敗時のエラーハンドリング
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data

        # unified_configをモック
        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                default_preset="4h_4bars",
                use_preset=True,
            )
            mock_config.ml.training.label_generation = mock_label_config

            # forward_classification_presetがエラーを投げるようにモック
            with patch(
                "app.services.ml.base_ml_trainer.forward_classification_preset"
            ) as mock_forward:
                mock_forward.side_effect = Exception("ラベル生成エラー")

                # フォールバックのdata_preprocessorもモック
                with patch(
                    "app.services.ml.base_ml_trainer.data_preprocessor"
                ) as mock_preprocessor:
                    mock_preprocessor.prepare_training_data.return_value = (
                        features_df[["open", "high", "low", "close", "volume"]],
                        pd.Series([0, 1, 2] * 33 + [0]),
                        {"threshold_up": 0.002, "threshold_down": -0.002},
                    )

                    # Act - フォールバックが機能することを確認
                    X, y = trainer._prepare_training_data(features_df)

                    # Assert
                    assert isinstance(X, pd.DataFrame)
                    assert isinstance(y, pd.Series)

    def test_prepare_training_data_feature_columns_saved(
        self, mock_base_ml_trainer, sample_ohlcv_data
    ):
        """
        正常系: 特徴量カラムが保存される
        """
        # Arrange
        trainer = mock_base_ml_trainer
        features_df = sample_ohlcv_data

        # unified_configをモック
        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                default_preset="4h_4bars",
                use_preset=True,
            )
            mock_config.ml.training.label_generation = mock_label_config

            # Act
            X, y = trainer._prepare_training_data(features_df)

            # Assert
            assert trainer.feature_columns is not None, (
                "特徴量カラムが保存されていません"
            )
            assert len(trainer.feature_columns) > 0, "特徴量カラムが空です"
            assert all(col in X.columns for col in trainer.feature_columns), (
                "特徴量カラムとXのカラムが一致しません"
            )


# ============================================================================
# 統合テスト: エンドツーエンド
# ============================================================================


class TestEndToEndIntegration:
    """エンドツーエンド統合テストクラス"""

    def test_full_workflow_preset_to_trainer(self, sample_ohlcv_data):
        """
        統合テスト: プリセット→ラベル生成→トレーナー準備の全体フロー
        """
        # Arrange
        df = sample_ohlcv_data

        # Step 1: プリセット一覧取得
        presets = get_common_presets()
        assert len(presets) > 0

        # Step 2: プリセットでラベル生成
        labels, preset_info = apply_preset_by_name(df, "4h_4bars")
        assert isinstance(labels, pd.Series)
        assert len(labels.dropna()) > 0

        # Step 3: トレーナーでの使用をシミュレート
        from app.services.ml.base_ml_trainer import BaseMLTrainer

        trainer = BaseMLTrainer(
            trainer_config={"type": "single", "model_type": "lightgbm"}
        )

        with patch("app.services.ml.base_ml_trainer.unified_config") as mock_config:
            mock_label_config = LabelGenerationConfig(
                default_preset="4h_4bars",
                use_preset=True,
            )
            mock_config.ml.training.label_generation = mock_label_config

            X, y = trainer._prepare_training_data(df)

            # Assert
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert len(X) == len(y)
            assert set(y.unique()).issubset({0, 1, 2})

    def test_config_initialization_and_usage(self):
        """
        統合テスト: 設定初期化と使用
        """
        # Step 1: 設定作成
        config = LabelGenerationConfig(
            default_preset="1h_4bars",
            timeframe="1h",
            horizon_n=4,
            threshold=0.002,
            threshold_method="KBINS_DISCRETIZER",
            use_preset=True,
        )

        # Step 2: 設定を辞書に変換
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

        # Step 3: ThresholdMethod enumを取得
        method_enum = config.get_threshold_method_enum()
        assert method_enum == ThresholdMethod.KBINS_DISCRETIZER

        # Step 4: プリセット適用
        presets = get_common_presets()
        assert config.default_preset in presets
