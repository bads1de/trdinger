import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.label_generation.label_generation_service import (
    LabelGenerationService,
)
from app.config.unified_config import unified_config


class TestLabelGenerationService:
    @pytest.fixture
    def service(self):
        return LabelGenerationService()

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        df = pd.DataFrame(
            {
                "open": np.random.rand(100),
                "high": np.random.rand(100),
                "low": np.random.rand(100),
                "close": np.random.rand(100),
                "volume": np.random.rand(100),
                "feature1": np.random.rand(100),
            },
            index=dates,
        )
        return df

    def test_prepare_labels_with_target_column(self, service, sample_data):
        """target_column指定時の後方互換性テスト"""
        # ターゲットカラムを追加
        sample_data["target"] = np.random.randint(0, 3, size=len(sample_data))

        # モックの設定
        with patch(
            "app.services.ml.label_generation.label_generation_service.data_preprocessor"
        ) as mock_preprocessor:
            # data_preprocessor.prepare_training_data の戻り値を設定
            # features, labels, threshold_info
            mock_preprocessor.prepare_training_data.return_value = (
                sample_data.drop(columns=["target"]),
                sample_data["target"],
                {},
            )

            features, labels = service.prepare_labels(
                sample_data, target_column="target"
            )

            assert features.shape[1] == sample_data.shape[1] - 1  # target以外
            assert len(labels) == len(sample_data)
            mock_preprocessor.prepare_training_data.assert_called_once()

    def test_prepare_labels_with_preset(self, service, sample_data):
        """プリセット使用時のテスト"""
        # 設定のモック
        with patch(
            "app.services.ml.label_generation.label_generation_service.unified_config"
        ) as mock_config:
            mock_config.ml.training.label_generation.use_preset = True
            mock_config.ml.training.label_generation.default_preset = (
                "standard_classification"
            )

            with patch(
                "app.services.ml.label_generation.label_generation_service.apply_preset_by_name"
            ) as mock_apply:
                # apply_preset_by_name の戻り値 (labels, info)
                mock_labels = pd.Series(
                    ["UP", "DOWN", "RANGE"] * 33 + ["UP"], index=sample_data.index
                )
                mock_apply.return_value = (mock_labels, {})

                features, labels = service.prepare_labels(sample_data)

                assert len(features) == len(sample_data)
                assert len(labels) == len(sample_data)
                # 文字列ラベルが数値に変換されていること
                assert (
                    labels.dtype == int
                    or labels.dtype == np.int32
                    or labels.dtype == np.int64
                )
                assert set(labels.unique()).issubset({0, 1, 2})

                mock_apply.assert_called_once()

    def test_prepare_labels_fallback_custom(self, service, sample_data):
        """プリセット失敗時のフォールバックテスト"""
        with patch(
            "app.services.ml.label_generation.label_generation_service.unified_config"
        ) as mock_config:
            mock_config.ml.training.label_generation.use_preset = True
            mock_config.ml.training.label_generation.default_preset = "unknown_preset"

            # apply_preset_by_name が ValueError を投げる
            with patch(
                "app.services.ml.label_generation.label_generation_service.apply_preset_by_name",
                side_effect=ValueError,
            ):
                with patch(
                    "app.services.ml.label_generation.label_generation_service.forward_classification_preset"
                ) as mock_forward:
                    mock_labels = pd.Series(["UP"] * 100, index=sample_data.index)
                    mock_forward.return_value = mock_labels

                    features, labels = service.prepare_labels(sample_data)

                    assert len(labels) == 100
                    mock_forward.assert_called_once()

    def test_prepare_labels_cleaning(self, service, sample_data):
        """NaN除去とラベルマッピングのテスト"""
        with patch(
            "app.services.ml.label_generation.label_generation_service.unified_config"
        ) as mock_config:
            mock_config.ml.training.label_generation.use_preset = False

            with patch(
                "app.services.ml.label_generation.label_generation_service.forward_classification_preset"
            ) as mock_forward:
                # NaNを含むラベルを返す
                labels_with_nan = pd.Series(
                    [np.nan, "UP", "DOWN", "RANGE", np.nan], index=sample_data.index[:5]
                )
                # 残りは埋める
                labels_full = pd.concat(
                    [
                        labels_with_nan,
                        pd.Series(["UP"] * 95, index=sample_data.index[5:]),
                    ]
                )

                mock_forward.return_value = labels_full

                features, labels = service.prepare_labels(sample_data)

                # NaNの行が削除されているはず (2行削除)
                assert len(features) == 98
                assert len(labels) == 98
                assert not labels.isna().any()

                # マッピング確認
                # UP->2, DOWN->0, RANGE->1
                assert labels.iloc[0] == 2  # UP
                assert labels.iloc[1] == 0  # DOWN
