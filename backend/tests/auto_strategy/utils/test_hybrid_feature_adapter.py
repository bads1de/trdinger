"""
HybridFeatureAdapter Tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from app.services.auto_strategy.utils.hybrid_feature_adapter import (
    HybridFeatureAdapter,
    MLFeatureError,
)
from app.services.auto_strategy.models.strategy_gene import StrategyGene


class TestHybridFeatureAdapter:
    """HybridFeatureAdapterのテスト"""

    @pytest.fixture
    def adapter(self):
        return HybridFeatureAdapter(wavelet_config={"enabled": False})

    @pytest.fixture
    def sample_ohlcv(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1H")
        data = {
            "open": np.random.rand(100) * 100,
            "high": np.random.rand(100) * 105,
            "low": np.random.rand(100) * 95,
            "close": np.random.rand(100) * 100,
            "volume": np.random.rand(100) * 1000,
        }
        return pd.DataFrame(data, index=idx)

    @pytest.fixture
    def mock_gene(self):
        gene = Mock(spec=StrategyGene)
        gene.indicators = [
            Mock(type="RSI", parameters={"period": 14}),
            Mock(type="SMA", parameters={"period": 20}),
        ]
        gene.get_effective_long_conditions.return_value = ["cond1"]
        gene.get_effective_short_conditions.return_value = ["cond2"]
        gene.has_long_short_separation.return_value = True

        # Nested mocks for tpsl/position sizing
        gene.tpsl_gene = Mock(take_profit_pct=0.05, stop_loss_pct=0.02)
        gene.position_sizing_gene = Mock(method="fixed")

        return gene

    def test_extract_gene_features(self, adapter, mock_gene):
        """Gene特徴量の抽出テスト"""
        features = adapter._extract_gene_features(mock_gene)

        assert features["indicator_count"] == 2
        assert features["has_rsi"] == 1
        assert features["has_sma"] == 1
        assert features["has_macd"] == 0
        assert features["avg_indicator_period"] == 17.0  # (14+20)/2
        assert features["has_tpsl"] == 1
        assert features["has_position_sizing"] == 1

    def test_gene_to_features_basic(self, adapter, mock_gene, sample_ohlcv):
        """基本的な特徴量変換"""
        df = adapter.gene_to_features(mock_gene, sample_ohlcv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        # Check gene static features propagated as columns
        assert "indicator_count" in df.columns
        assert df["indicator_count"].iloc[0] == 2

        # Check derived features
        assert "close_return_1" in df.columns

    def test_gene_to_features_validation(self, adapter):
        """入力検証エラー"""
        with pytest.raises(MLFeatureError, match="Geneがnull"):
            adapter.gene_to_features(None, pd.DataFrame())

        with pytest.raises(MLFeatureError, match="OHLCVデータが空"):
            adapter.gene_to_features(Mock(), pd.DataFrame())

    def test_augment_derived_features(self, adapter, sample_ohlcv):
        """派生特徴量生成"""
        df = adapter._augment_with_derived_features(sample_ohlcv)

        assert "close_return_1" in df.columns
        assert "close_rolling_mean_5" in df.columns
        assert "volume_rolling_mean_5" in df.columns
        assert "hl_spread" in df.columns

    def test_batch_processing(self, adapter, mock_gene, sample_ohlcv):
        """バッチ処理"""
        genes = [mock_gene, mock_gene]
        results = adapter.genes_to_features_batch(genes, sample_ohlcv)

        assert len(results) == 2
        assert isinstance(results[0], pd.DataFrame)
        assert len(results[0]) == 100

    def test_wavelet_transformer_init(self):
        """WaveletTransformerの初期化"""
        adapter = HybridFeatureAdapter(
            wavelet_config={"enabled": True, "base_wavelet": "haar"}
        )
        assert adapter._wavelet_transformer is not None

    def test_preprocessing_fallback(self, adapter, sample_ohlcv):
        """前処理フォールバック"""
        # Inject infinity to test clean up
        sample_ohlcv.iloc[0, 0] = np.inf

        processed = adapter._fallback_preprocess(sample_ohlcv)
        assert not np.isinf(processed.iloc[0, 0])
        assert not processed.isnull().values.any()

    @patch("app.services.ml.base_ml_trainer.BaseMLTrainer")
    def test_apply_preprocessing_with_trainer(self, MockTrainer, adapter, sample_ohlcv):
        """BaseMLTrainerを使用した前処理"""
        mock_trainer = MockTrainer.return_value
        # Mock _preprocess_data to return transformed df
        mock_trainer._preprocess_data.return_value = (sample_ohlcv, sample_ohlcv)

        processed = adapter._apply_preprocessing(sample_ohlcv)
        # Verify adapter instantiates BaseMLTrainer lazily
        assert adapter._preprocess_trainer is not None
        assert isinstance(processed, pd.DataFrame)
