import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
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
        # OHLCVデータと特徴量データの両方を兼ねるサンプル
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
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

    @patch("app.services.ml.label_generation.label_generation_service.LabelCache")
    def test_prepare_labels_tbm(self, mock_label_cache_cls, service, sample_data):
        """TBMを使用したラベル生成のテスト"""
        # モックの設定
        mock_label_cache = mock_label_cache_cls.return_value
        
        # ラベルデータの作成 (0と1のランダム)
        labels = pd.Series(
            np.random.randint(0, 2, size=len(sample_data)),
            index=sample_data.index
        )
        # 一部をNaNにしてフィルタリングを確認
        labels.iloc[-5:] = np.nan
        
        mock_label_cache.get_labels.return_value = labels

        # テスト実行
        features, result_labels = service.prepare_labels(
            features_df=sample_data,
            ohlcv_df=sample_data,
            pt_factor=2.0,
            sl_factor=1.0
        )

        # 検証
        # 1. LabelCacheが初期化されたか
        mock_label_cache_cls.assert_called_once_with(sample_data)

        # 2. get_labelsが正しいパラメータで呼ばれたか
        mock_label_cache.get_labels.assert_called_once()
        call_kwargs = mock_label_cache.get_labels.call_args.kwargs
        assert call_kwargs["threshold_method"] == "TRIPLE_BARRIER"
        assert call_kwargs["pt_factor"] == 2.0
        assert call_kwargs["sl_factor"] == 1.0
        assert call_kwargs["binary_label"] is True

        # 3. NaNが除去されているか
        assert len(features) == 95
        assert len(result_labels) == 95
        assert not result_labels.isna().any()
        
        # 4. インデックスが一致しているか
        pd.testing.assert_index_equal(features.index, result_labels.index)

    @patch("app.services.ml.label_generation.label_generation_service.LabelCache")
    def test_prepare_labels_error_handling(self, mock_label_cache_cls, service, sample_data):
        """エラーハンドリングのテスト"""
        mock_label_cache = mock_label_cache_cls.return_value
        mock_label_cache.get_labels.side_effect = Exception("Test Error")

        from app.utils.error_handler import DataError
        
        with pytest.raises(DataError, match="ラベル生成に失敗しました"):
            service.prepare_labels(
                features_df=sample_data,
                ohlcv_df=sample_data
            )
