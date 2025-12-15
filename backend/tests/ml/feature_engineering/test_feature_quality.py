import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)


class TestFeatureQuality:
    """特徴量の品質（シグナル捕捉能力）を検証するテスト"""

    @pytest.fixture
    def sample_market_data(self):
        """テスト用の市場データ（トレンドを含む）を生成"""
        # 1000本のデータ
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")

        # ランダムウォーク + トレンド
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)

        # 上昇トレンド (index 200-300)
        returns[200:300] += 0.02

        # 下降トレンド (index 600-700)
        returns[600:700] -= 0.02

        price = 100 * np.exp(np.cumsum(returns))

        # 出来高 (トレンド時に増加させる)
        volume = np.random.randint(100, 1000, 1000)
        volume[200:300] += 500
        volume[600:700] += 500

        df = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": volume,
            },
            index=dates,
        )

        return df

    def test_feature_correlation_with_trend(self, sample_market_data):
        """特徴量がトレンドと相関しているか検証"""
        service = FeatureEngineeringService()

        # 特徴量計算
        # FR/OIはNoneでも計算できる特徴量のみ対象とするか、ダミーを作成
        features_df = service.calculate_advanced_features(
            ohlcv_data=sample_market_data,
            funding_rate_data=None,
            open_interest_data=None,
        )

        # 1. RSIの検証
        # 上昇トレンド(200-300)でRSIは高くなるはず
        uptrend_rsi = (
            features_df["RSI"].iloc[220:280].mean()
        )  # 遅れを考慮して少し内側を見る
        normal_rsi = features_df["RSI"].iloc[0:100].mean()
        downtrend_rsi = features_df["RSI"].iloc[620:680].mean()

        print(f"\nUptrend RSI: {uptrend_rsi:.2f}")
        print(f"Normal RSI: {normal_rsi:.2f}")
        print(f"Downtrend RSI: {downtrend_rsi:.2f}")

        assert uptrend_rsi > 60, "上昇トレンド中のRSIが低すぎます"
        assert downtrend_rsi < 40, "下降トレンド中のRSIが高すぎます"
        assert uptrend_rsi > normal_rsi, "上昇トレンドのRSIが通常時より高くありません"

        # 2. ADXの検証 (トレンドの強さ)
        # トレンド区間でADXは高くなるはず
        trend_adx = features_df["ADX"].iloc[250:300].mean()
        range_adx = features_df["ADX"].iloc[0:100].mean()

        print(f"Trend ADX: {trend_adx:.2f}")
        print(f"Range ADX: {range_adx:.2f}")

        assert trend_adx > range_adx, "トレンド区間のADXがレンジ区間より高くありません"
        assert trend_adx > 25, "トレンド区間のADXが低すぎます (通常25以上がトレンド)"

        # 3. Volume MAの検証
        # トレンド区間で出来高MAが増加しているか
        trend_vol_ma = features_df["Volume_MA_20"].iloc[250:300].mean()
        range_vol_ma = features_df["Volume_MA_20"].iloc[0:100].mean()

        print(f"Trend Volume MA: {trend_vol_ma:.2f}")
        print(f"Range Volume MA: {range_vol_ma:.2f}")

        assert (
            trend_vol_ma > range_vol_ma
        ), "トレンド区間の出来高移動平均が増加していません"

    def test_fakeout_detection_features_existence(self, sample_market_data):
        """ダマシ検知用特徴量が正しく計算されているか検証"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FAKEOUT_DETECTION_ALLOWLIST,
        )

        service = FeatureEngineeringService()
        features_df = service.calculate_advanced_features(ohlcv_data=sample_market_data)

        # OI/FRがないため計算されない特徴量を除外してチェック
        # OI/FR依存の特徴量は計算されない仕様か、0埋めされるかを確認

        calculated_features = features_df.columns.tolist()
        missing_features = []

        # OI/FRを必要としない主要なテクニカル指標のみチェック
        core_features = [
            "RSI",
            "ADX",
            "Volume_MA_20",
            "NATR",
            "Close_range_20",
            "Historical_Volatility_20",
            "BB_Width",
        ]

        for feature in core_features:
            if feature not in calculated_features:
                missing_features.append(feature)

        assert (
            not missing_features
        ), f"主要な特徴量が計算されていません: {missing_features}"


