"""
FeatureEngineeringService のテスト

特徴量計算のパフォーマンスと正確性を検証します。
"""

import pytest
import pandas as pd
import numpy as np
import time

from app.services.ml.common.exceptions import MLFeatureError


class TestFeatureEngineeringServicePerformance:
    """FeatureEngineeringService のパフォーマンステスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用OHLCVデータ"""
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        close = 10000 + np.cumsum(np.random.randn(n_samples) * 10)
        high = close + np.abs(np.random.randn(n_samples) * 5)
        low = close - np.abs(np.random.randn(n_samples) * 5)
        open_ = close + np.random.randn(n_samples) * 3
        volume = np.abs(np.random.randn(n_samples) * 1000) + 100

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).set_index("timestamp")

        return df

    def test_calculate_advanced_features_returns_dataframe(self, sample_ohlcv_data):
        """calculate_advanced_features がDataFrameを返すことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        # 元のカラム + 追加の特徴量カラム
        assert len(result.columns) > len(sample_ohlcv_data.columns)

    def test_calculate_advanced_features_no_duplicate_columns(self, sample_ohlcv_data):
        """calculate_advanced_features が重複カラムを持たないことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        # 重複カラムがないか確認
        assert len(result.columns) == len(set(result.columns))

    def test_calculate_advanced_features_no_inf_values(self, sample_ohlcv_data):
        """calculate_advanced_features が無限大値を含まないことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        # 数値列に無限大がないか確認
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.any(
                np.isinf(result[col].values)
            ), f"Column {col} contains inf values"

    def test_calculate_advanced_features_performance(self, sample_ohlcv_data):
        """calculate_advanced_features のパフォーマンステスト"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()

        # ウォームアップ（JITコンパイル等）
        _ = service.calculate_advanced_features(sample_ohlcv_data.iloc[:50])

        # キャッシュをクリア
        service.feature_cache.clear()

        start_time = time.time()
        result = service.calculate_advanced_features(sample_ohlcv_data)
        duration = time.time() - start_time

        print(f"\nProcessing {len(sample_ohlcv_data)} rows took {duration:.4f} seconds")
        print(f"Generated {len(result.columns)} features")

        # 500行で30秒以内（余裕を持たせた値）
        assert duration < 30.0, f"Processing took too long: {duration:.2f}s"

    def test_calculate_advanced_features_requires_datetime_index_or_timestamp(self):
        """DatetimeIndexもtimestamp列も無い入力は例外にする"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9],
                "close": [1.05, 2.05, 3.05],
                "volume": [100, 110, 120],
            }
        )

        service = FeatureEngineeringService()

        with pytest.raises(MLFeatureError, match="DatetimeIndexまたはtimestamp"):
            service.calculate_advanced_features(df)

    def test_calculate_advanced_features_cache_isolated_from_mutation(
        self, sample_ohlcv_data
    ):
        """キャッシュ済み特徴量が外部の in-place 変更で壊れないことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()

        first_result = service.calculate_advanced_features(sample_ohlcv_data)
        first_index = first_result.index[0]
        original_close = first_result.loc[first_index, "close"]

        # 返却された DataFrame を変更しても、キャッシュには影響しないこと
        first_result.loc[first_index, "close"] = original_close + 999.0

        second_result = service.calculate_advanced_features(sample_ohlcv_data)
        assert second_result.loc[first_index, "close"] == original_close

        # キャッシュから返ってきた DataFrame を変更しても、次回取得結果は壊れないこと
        second_result.loc[first_index, "close"] = original_close - 999.0
        third_result = service.calculate_advanced_features(sample_ohlcv_data)
        assert third_result.loc[first_index, "close"] == original_close

        cached_entry = next(iter(service.feature_cache.values()))
        cached_df = cached_entry["data"]
        assert cached_df.loc[first_index, "close"] == original_close


class TestFeatureEngineeringServiceQuality:
    """特徴量の品質（シグナル捕捉能力）を検証するテスト"""

    @pytest.fixture
    def sample_market_data(self):
        """テスト用の市場データ（トレンドを含む）を生成"""
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        returns[200:300] += 0.02
        returns[600:700] -= 0.02

        price = 100 * np.exp(np.cumsum(returns))
        volume = np.random.randint(100, 1000, 1000)
        volume[200:300] += 500
        volume[600:700] += 500

        return pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": volume,
            },
            index=dates,
        )

    def test_feature_correlation_with_trend(self, sample_market_data):
        """特徴量がトレンドと相関しているか検証"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        features_df = service.calculate_advanced_features(
            ohlcv_data=sample_market_data,
            funding_rate_data=None,
            open_interest_data=None,
        )

        uptrend_rsi = features_df["RSI"].iloc[220:280].mean()
        normal_rsi = features_df["RSI"].iloc[0:100].mean()
        downtrend_rsi = features_df["RSI"].iloc[620:680].mean()

        print(f"\nUptrend RSI: {uptrend_rsi:.2f}")
        print(f"Normal RSI: {normal_rsi:.2f}")
        print(f"Downtrend RSI: {downtrend_rsi:.2f}")

        assert uptrend_rsi > 60, "上昇トレンド中のRSIが低すぎます"
        assert downtrend_rsi < 40, "下降トレンド中のRSIが高すぎます"
        assert uptrend_rsi > normal_rsi, "上昇トレンドのRSIが通常時より高くありません"

        trend_adx = features_df["ADX"].iloc[250:300].mean()
        range_adx = features_df["ADX"].iloc[0:100].mean()

        print(f"Trend ADX: {trend_adx:.2f}")
        print(f"Range ADX: {range_adx:.2f}")

        assert trend_adx > range_adx, "トレンド区間のADXがレンジ区間より高くありません"
        assert trend_adx > 25, "トレンド区間のADXが低すぎます (通常25以上がトレンド)"

        trend_vol_ma = features_df["Volume_MA_20"].iloc[250:300].mean()
        range_vol_ma = features_df["Volume_MA_20"].iloc[0:100].mean()

        print(f"Trend Volume MA: {trend_vol_ma:.2f}")
        print(f"Range Volume MA: {range_vol_ma:.2f}")

        assert trend_vol_ma > range_vol_ma, "トレンド区間の出来高移動平均が増加していません"

    def test_fakeout_detection_features_existence(self, sample_market_data):
        """ダマシ検知用特徴量が正しく計算されているか検証"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        fe_service = FeatureEngineeringService()
        features = fe_service.calculate_advanced_features(sample_market_data)

        expected_patterns = [
            "Volume_Divergence",
            "Void_Oscillator",
            "Fractal_Dim",
            "VPIN",
        ]

        found_count = 0
        for pattern in expected_patterns:
            if any(pattern in col for col in features.columns):
                found_count += 1

        assert found_count >= 2, (
            f"ダマシ検知系特徴量が見つかりません。Columns: {features.columns.tolist()}"
        )




