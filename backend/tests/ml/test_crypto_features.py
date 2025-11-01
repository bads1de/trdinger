"""
CryptoFeaturesのテスト

ユーザーが作成した暗号通貨特化特徴量エンジニアリングのテストです。
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.ml.feature_engineering.crypto_features import CryptoFeatures


class TestCryptoFeatures:
    """CryptoFeaturesのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = {
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_crypto_features_initialization(self):
        """初期化テスト"""
        crypto_features = CryptoFeatures()
        assert crypto_features is not None
        assert crypto_features.feature_groups is not None

    def test_create_crypto_features_basic(self, sample_ohlcv_data):
        """基本特徴量生成テスト"""
        crypto_features = CryptoFeatures()
        result = crypto_features.create_crypto_features(sample_ohlcv_data)

        # 基本列が存在することを確認
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

        # 新しい特徴量が追加されていることを確認
        assert len(result.columns) > len(sample_ohlcv_data.columns)

    def test_create_crypto_features_with_fr_and_oi(self, sample_ohlcv_data):
        """ファンディングレートと建玉データを含むテスト"""
        crypto_features = CryptoFeatures()

        # ファンディングレートデータを作成
        fr_data = pd.DataFrame({
            'funding_rate': np.random.randn(100) * 0.001
        }, index=sample_ohlcv_data.index)

        # 建玉データを作成
        oi_data = pd.DataFrame({
            'open_interest': np.random.randint(100000, 1000000, 100)
        }, index=sample_ohlcv_data.index)

        result = crypto_features.create_crypto_features(
            sample_ohlcv_data,
            funding_rate_data=fr_data,
            open_interest_data=oi_data
        )

        # FR関連特徴量が追加されていることを確認
        fr_columns = [col for col in result.columns if col.startswith('fr_')]
        assert len(fr_columns) > 0

        # OI関連特徴量が追加されていることを確認
        oi_columns = [col for col in result.columns if col.startswith('oi_')]
        assert len(oi_columns) > 0

    def test_create_crypto_features_data_quality(self, sample_ohlcv_data):
        """データ品質テスト"""
        crypto_features = CryptoFeatures()

        # NaN値を注入
        sample_ohlcv_data.loc[sample_ohlcv_data.index[10], 'close'] = np.nan

        result = crypto_features.create_crypto_features(sample_ohlcv_data)

        # NaN値が補完されていることを確認
        assert not result['close'].isna().any()

    def test_create_crypto_features_no_infinity(self, sample_ohlcv_data):
        """無限大値チェックテスト"""
        crypto_features = CryptoFeatures()

        # 無限大値を注入
        sample_ohlcv_data.loc[sample_ohlcv_data.index[5], 'close'] = np.inf

        result = crypto_features.create_crypto_features(sample_ohlcv_data)

        # 無限大値が置換されていることを確認
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_create_crypto_features_temporal_features(self, sample_ohlcv_data):
        """時間特徴量テスト"""
        crypto_features = CryptoFeatures()
        result = crypto_features.create_crypto_features(sample_ohlcv_data)

        # 時間関連特徴量を確認
        temporal_columns = [
            'hour', 'day_of_week', 'is_weekend',
            'asia_hours', 'europe_hours', 'us_hours'
        ]
        for col in temporal_columns:
            assert col in result.columns

    def test_create_crypto_features_technical_indicators(self, sample_ohlcv_data):
        """テクニカル指標テスト"""
        crypto_features = CryptoFeatures()
        result = crypto_features.create_crypto_features(sample_ohlcv_data)

        # RSIとボリンジャーバンドの特徴量を確認
        rsi_columns = [col for col in result.columns if col.startswith('rsi_')]
        bb_columns = [col for col in result.columns if col.startswith('bb_')]

        assert len(rsi_columns) > 0
        assert len(bb_columns) > 0


if __name__ == '__main__':
    pytest.main([__file__])
