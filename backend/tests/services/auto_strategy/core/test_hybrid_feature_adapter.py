"""
HybridFeatureAdapter Tests
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
    HybridFeatureAdapter,
)
from app.services.ml.common.exceptions import MLFeatureError
from app.services.auto_strategy.genes.strategy import StrategyGene


class TestHybridFeatureAdapter:
    """HybridFeatureAdapterのテスト"""

    @pytest.fixture
    def adapter(self):
        return HybridFeatureAdapter(wavelet_config={"enabled": False})

    @pytest.fixture
    def sample_ohlcv(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
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
            Mock(type="RSI", parameters={"period": 14}, enabled=True),
            Mock(type="SMA", parameters={"period": 20}, enabled=True),
        ]
        gene.long_entry_conditions = ["cond1"]
        gene.short_entry_conditions = ["cond2"]
        gene.has_long_short_separation.return_value = True

        # Nested mocks for tpsl/position sizing
        gene.tpsl_gene = Mock(take_profit_pct=0.05, stop_loss_pct=0.02, enabled=True)
        gene.position_sizing_gene = Mock(method="fixed", enabled=True)

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
            with pytest.raises(MLFeatureError, match="変換失敗"):
                adapter.gene_to_features(None, pd.DataFrame())

            with pytest.raises(MLFeatureError, match="変換失敗"):
                adapter.gene_to_features(Mock(), pd.DataFrame())

    def test_augment_derived_features(self, adapter, sample_ohlcv):
        """派生特徴量生成"""
        df = adapter._augment_with_derived_features(sample_ohlcv)

        assert "close_return_1" in df.columns
        assert "close_rolling_mean_5" in df.columns
        assert "volume_rolling_mean_5" in df.columns
        assert "hl_spread" in df.columns

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

    def test_fallback_preprocess_does_not_backfill_future_values(self, adapter):
        """フォールバック前処理が未来値で埋めないことを確認"""
        df = pd.DataFrame(
            {
                "a": [np.nan, 1.0],
                "b": [np.nan, 2.0],
            }
        )

        processed = adapter._fallback_preprocess(df)

        assert processed.iloc[0]["a"] == 0
        assert processed.iloc[0]["b"] == 0

    def test_augment_derived_features_does_not_backfill_lagged_values(self, adapter):
        """派生特徴量のラグ列が未来値で埋まらないことを確認"""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [10.0, 11.0, 12.0],
            }
        )

        result = adapter._augment_with_derived_features(df)

        assert result["close_lag1"].iloc[0] == 0
        assert result["volume_lag1"].iloc[0] == 0

    def test_cached_derived_features_does_not_store_dataframe_hash(
        self, adapter, sample_ohlcv
    ):
        """DataFrame 自体にキャッシュ用属性を載せないことを確認"""
        features = pd.DataFrame(
            {"derived": np.arange(len(sample_ohlcv), dtype=float)},
            index=sample_ohlcv.index,
        )

        adapter._cache_derived_features(sample_ohlcv, features)

        assert not hasattr(sample_ohlcv, "_cached_hash")
        cached = adapter._get_cached_derived_features(sample_ohlcv)
        assert cached is not None
        pd.testing.assert_frame_equal(cached, features)

    def test_cached_derived_features_reflects_dataframe_mutation(
        self, adapter, sample_ohlcv
    ):
        """同じ DataFrame を in-place 更新したら古い派生特徴量を返さない"""
        features = pd.DataFrame(
            {"derived": np.arange(len(sample_ohlcv), dtype=float)},
            index=sample_ohlcv.index,
        )

        adapter._cache_derived_features(sample_ohlcv, features)

        sample_ohlcv.loc[sample_ohlcv.index[0], "close"] *= 10

        cached = adapter._get_cached_derived_features(sample_ohlcv)
        assert cached is None

    def test_cached_derived_features_are_isolated_from_mutation(
        self, adapter, sample_ohlcv
    ):
        """キャッシュ済み派生特徴量が外部変更で壊れないことを確認"""
        features = pd.DataFrame(
            {"derived": np.arange(len(sample_ohlcv), dtype=float)},
            index=sample_ohlcv.index,
        )

        adapter._cache_derived_features(sample_ohlcv, features)

        # 元の DataFrame を変更しても、キャッシュは影響を受けないこと
        features.iloc[0, 0] = -123.0

        cached_first = adapter._get_cached_derived_features(sample_ohlcv)
        assert cached_first is not None
        original_value = cached_first.iloc[0, 0]
        assert original_value != -123.0

        # 返却されたキャッシュを変更しても、次回取得結果は壊れないこと
        cached_first.iloc[0, 0] = -456.0
        cached_second = adapter._get_cached_derived_features(sample_ohlcv)
        assert cached_second is not None
        assert cached_second.iloc[0, 0] == original_value

    def test_get_preprocess_callable_without_handler_returns_none(self, adapter):
        """前処理ハンドラ未指定時は暗黙依存を持たない"""
        assert adapter._get_preprocess_callable() is None

    def test_apply_preprocessing_with_injected_handler(self, sample_ohlcv):
        """明示注入された前処理ハンドラを使用する"""

        handler = Mock(return_value=(sample_ohlcv, sample_ohlcv))
        adapter = HybridFeatureAdapter(
            wavelet_config={"enabled": False},
            preprocess_handler=handler,
        )

        processed = adapter._apply_preprocessing(sample_ohlcv)

        handler.assert_called_once()
        pd.testing.assert_frame_equal(processed, sample_ohlcv)
