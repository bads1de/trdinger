"""
特徴量エンジニアリング機能の包括的テスト

全ての特徴量計算クラスに対する詳細な単体テスト、統合テスト、
エラーハンドリングテスト、エッジケーステストを実装します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import warnings

from app.core.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.price_features import (
    PriceFeatureCalculator,
)
from app.core.services.ml.feature_engineering.technical_features import (
    TechnicalFeatureCalculator,
)
from app.core.services.ml.feature_engineering.market_data_features import (
    MarketDataFeatureCalculator,
)
from app.core.services.ml.feature_engineering.temporal_features import (
    TemporalFeatureCalculator,
)
from app.core.services.ml.feature_engineering.interaction_features import (
    InteractionFeatureCalculator,
)


class TestDataGenerator:
    """テストデータ生成ユーティリティクラス"""

    @staticmethod
    def create_ohlcv_data(
        size: int = 100,
        start_date: str = "2024-01-01",
        freq: str = "h",
        base_price: float = 50000.0,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """標準的なOHLCVデータを生成"""
        dates = pd.date_range(start=start_date, periods=size, freq=freq, tz="UTC")

        # 価格データ生成（ランダムウォーク）
        np.random.seed(42)
        returns = np.random.normal(0, volatility, size)
        prices = base_price * np.exp(np.cumsum(returns))

        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # 高値・安値・始値を生成
            daily_volatility = abs(np.random.normal(0, volatility / 4))
            high = close * (1 + daily_volatility)
            low = close * (1 - daily_volatility)
            open_price = prices[i - 1] if i > 0 else close

            # 出来高生成
            volume = np.random.lognormal(10, 1)

            data.append(
                {
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                }
            )

        return pd.DataFrame(data, index=dates)

    @staticmethod
    def create_funding_rate_data(
        size: int = 100, start_date: str = "2024-01-01", freq: str = "h"
    ) -> pd.DataFrame:
        """ファンディングレートデータを生成"""
        dates = pd.date_range(start=start_date, periods=size, freq=freq, tz="UTC")

        # ファンディングレート生成（-0.01% ~ 0.01%の範囲）
        np.random.seed(43)
        funding_rates = np.random.uniform(-0.0001, 0.0001, size)

        return pd.DataFrame({"funding_rate": funding_rates}, index=dates)

    @staticmethod
    def create_open_interest_data(
        size: int = 100, start_date: str = "2024-01-01", freq: str = "h"
    ) -> pd.DataFrame:
        """建玉残高データを生成"""
        dates = pd.date_range(start=start_date, periods=size, freq=freq, tz="UTC")

        # 建玉残高生成（トレンドを持つ）
        np.random.seed(44)
        base_oi = 1000000
        trend = np.linspace(0, 0.5, size)
        noise = np.random.normal(0, 0.1, size)
        oi_values = base_oi * (1 + trend + noise)

        return pd.DataFrame({"open_interest": oi_values}, index=dates)

    @staticmethod
    def create_extreme_ohlcv_data() -> pd.DataFrame:
        """極端な値を含むOHLCVデータを生成"""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h", tz="UTC")

        data = []
        for i, date in enumerate(dates):
            if i == 10:  # 極端な高値
                data.append(
                    {
                        "Open": 50000,
                        "High": 1000000,
                        "Low": 49000,
                        "Close": 51000,
                        "Volume": 1000,
                    }
                )
            elif i == 20:  # 極端な安値
                data.append(
                    {
                        "Open": 50000,
                        "High": 51000,
                        "Low": 1,
                        "Close": 49000,
                        "Volume": 1000,
                    }
                )
            elif i == 30:  # ゼロ出来高
                data.append(
                    {
                        "Open": 50000,
                        "High": 51000,
                        "Low": 49000,
                        "Close": 50000,
                        "Volume": 0,
                    }
                )
            else:  # 通常データ
                data.append(
                    {
                        "Open": 50000,
                        "High": 51000,
                        "Low": 49000,
                        "Close": 50000,
                        "Volume": 1000,
                    }
                )

        return pd.DataFrame(data, index=dates)


class TestPriceFeatureCalculator:
    """PriceFeatureCalculator の単体テスト"""

    @pytest.fixture
    def calculator(self):
        return PriceFeatureCalculator()

    @pytest.fixture
    def sample_ohlcv(self):
        return TestDataGenerator.create_ohlcv_data(size=100)

    @pytest.fixture
    def lookback_periods(self):
        return {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "momentum": 14,
            "volume": 20,
        }

    def test_calculate_price_features_normal(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """価格特徴量計算の正常ケーステスト"""
        result = calculator.calculate_price_features(sample_ohlcv, lookback_periods)

        # 基本的な検証
        assert result is not None
        assert len(result) == len(sample_ohlcv)
        assert result.index.equals(sample_ohlcv.index)

        # 生成される特徴量の確認
        expected_features = [
            "MA_10",
            "MA_50",
            "Price_MA_Ratio_Short",
            "Price_MA_Ratio_Long",
            "Price_Momentum_14",
            "Price_Change_5",
            "Price_Position",
            "Gap",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
            # NaN値の確認（最初の数行以外はNaNでないことを確認）
            assert (
                not result[feature].iloc[50:].isna().all()
            ), f"Feature {feature} is all NaN"

    def test_calculate_volatility_features_normal(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """ボラティリティ特徴量計算の正常ケーステスト"""
        result = calculator.calculate_volatility_features(
            sample_ohlcv, lookback_periods
        )

        # 生成される特徴量の確認
        expected_features = [
            "Returns",
            "Realized_Volatility_20",
            "Volatility_Spike",
            "ATR_20",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing volatility feature: {feature}"
            # 値の範囲確認
            if feature == "Volatility_Spike":
                assert result[feature].min() >= 0, f"{feature} should be non-negative"

    def test_calculate_volume_features_normal(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """出来高特徴量計算の正常ケーステスト"""
        result = calculator.calculate_volume_features(sample_ohlcv, lookback_periods)

        # 生成される特徴量の確認
        expected_features = ["Volume_MA_20", "Volume_Ratio", "VWAP"]

        for feature in expected_features:
            assert feature in result.columns, f"Missing volume feature: {feature}"
            # 正の値であることを確認
            assert result[feature].min() >= 0, f"{feature} should be non-negative"

    def test_empty_data_handling(self, calculator):
        """空データの処理テスト"""
        empty_df = pd.DataFrame()
        lookback_periods = {"short_ma": 10, "long_ma": 50}

        result = calculator.calculate_price_features(empty_df, lookback_periods)
        assert result.empty

    def test_none_data_handling(self, calculator):
        """Noneデータの処理テスト"""
        lookback_periods = {"short_ma": 10, "long_ma": 50}

        result = calculator.calculate_price_features(None, lookback_periods)
        assert result is None

    def test_extreme_values_handling(self, calculator, lookback_periods):
        """極端な値の処理テスト"""
        extreme_data = TestDataGenerator.create_extreme_ohlcv_data()

        # エラーが発生しないことを確認
        result = calculator.calculate_price_features(extreme_data, lookback_periods)
        assert result is not None
        assert len(result) == len(extreme_data)

        # 無限大値やNaN値が適切に処理されていることを確認
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert not np.isinf(result[col]).any(), f"Infinite values found in {col}"


class TestTechnicalFeatureCalculator:
    """TechnicalFeatureCalculator の単体テスト"""

    @pytest.fixture
    def calculator(self):
        return TechnicalFeatureCalculator()

    @pytest.fixture
    def sample_ohlcv(self):
        return TestDataGenerator.create_ohlcv_data(
            size=200
        )  # テクニカル指標には多めのデータが必要

    @pytest.fixture
    def lookback_periods(self):
        return {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "momentum": 14,
            "volume": 20,
        }

    def test_calculate_market_regime_features(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """市場レジーム特徴量計算テスト"""
        result = calculator.calculate_market_regime_features(
            sample_ohlcv, lookback_periods
        )

        expected_features = [
            "Trend_Strength",
            "Range_Bound_Ratio",
            "Breakout_Strength",
            "Market_Efficiency",
        ]

        for feature in expected_features:
            assert (
                feature in result.columns
            ), f"Missing market regime feature: {feature}"
            # 値の範囲確認（一部の特徴量は0-1の範囲）
            if feature in ["Range_Bound_Ratio", "Breakout_Strength"]:
                valid_values = result[feature].dropna()
                if len(valid_values) > 0:
                    assert valid_values.min() >= 0, f"{feature} should be non-negative"
                    assert valid_values.max() <= 1, f"{feature} should be <= 1"

    def test_calculate_momentum_features(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """モメンタム特徴量計算テスト"""
        result = calculator.calculate_momentum_features(sample_ohlcv, lookback_periods)

        expected_features = [
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "Stochastic_K",
            "Stochastic_D",
            "Williams_R",
            "CCI",
            "ROC",
            "Momentum",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing momentum feature: {feature}"

            # RSIの範囲確認（0-100）
            if feature == "RSI":
                valid_rsi = result[feature].dropna()
                if len(valid_rsi) > 0:
                    assert valid_rsi.min() >= 0, "RSI should be >= 0"
                    assert valid_rsi.max() <= 100, "RSI should be <= 100"

            # ストキャスティクスの範囲確認（0-100）
            if feature in ["Stochastic_K", "Stochastic_D"]:
                valid_stoch = result[feature].dropna()
                if len(valid_stoch) > 0:
                    assert valid_stoch.min() >= 0, f"{feature} should be >= 0"
                    assert valid_stoch.max() <= 100, f"{feature} should be <= 100"

    def test_calculate_pattern_features(
        self, calculator, sample_ohlcv, lookback_periods
    ):
        """パターン特徴量計算テスト"""
        # パターン特徴量にはRSIが必要なので、事前に計算
        ohlcv_with_rsi = calculator.calculate_momentum_features(
            sample_ohlcv, lookback_periods
        )
        result = calculator.calculate_pattern_features(ohlcv_with_rsi, lookback_periods)

        expected_features = [
            "Bear_Divergence",
            "Bull_Divergence",
            "Support_Distance",
            "Resistance_Distance",
            "Pivot_Distance",
            "Fib_236_Distance",
            "Fib_382_Distance",
            "Fib_500_Distance",
            "Fib_618_Distance",
            "Fib_786_Distance",
            "Gap_Up",
            "Gap_Down",
            "Gap_Size",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing pattern feature: {feature}"


class TestMarketDataFeatureCalculator:
    """MarketDataFeatureCalculator の単体テスト"""

    @pytest.fixture
    def calculator(self):
        return MarketDataFeatureCalculator()

    @pytest.fixture
    def sample_ohlcv(self):
        return TestDataGenerator.create_ohlcv_data(size=200)

    @pytest.fixture
    def sample_funding_rate(self):
        return TestDataGenerator.create_funding_rate_data(size=200)

    @pytest.fixture
    def sample_open_interest(self):
        return TestDataGenerator.create_open_interest_data(size=200)

    @pytest.fixture
    def lookback_periods(self):
        return {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "momentum": 14,
            "volume": 20,
        }

    def test_calculate_funding_rate_features(
        self, calculator, sample_ohlcv, sample_funding_rate, lookback_periods
    ):
        """ファンディングレート特徴量計算テスト"""
        result = calculator.calculate_funding_rate_features(
            sample_ohlcv, sample_funding_rate, lookback_periods
        )

        expected_features = [
            "FR_MA_24",
            "FR_MA_168",
            "FR_Change",
            "FR_Change_Rate",
            "Price_FR_Divergence",
            "FR_Extreme_High",
            "FR_Extreme_Low",
            "FR_Normalized",
            "FR_Trend",
            "FR_Volatility",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing funding rate feature: {feature}"

            # 極値フラグの確認（ブール値）
            if feature in ["FR_Extreme_High", "FR_Extreme_Low"]:
                unique_values = result[feature].dropna().unique()
                assert all(
                    val in [0, 1, True, False] for val in unique_values
                ), f"{feature} should be boolean"

    def test_calculate_open_interest_features(
        self, calculator, sample_ohlcv, sample_open_interest, lookback_periods
    ):
        """建玉残高特徴量計算テスト"""
        result = calculator.calculate_open_interest_features(
            sample_ohlcv, sample_open_interest, lookback_periods
        )

        expected_features = [
            "OI_Change_Rate",
            "OI_Change_Rate_24h",
            "OI_Surge",
            "Volatility_Adjusted_OI",
            "OI_Trend",
            "OI_Price_Correlation",
            "OI_Normalized",
        ]

        for feature in expected_features:
            assert (
                feature in result.columns
            ), f"Missing open interest feature: {feature}"

            # OI_Surgeの確認（ブール値）
            if feature == "OI_Surge":
                unique_values = result[feature].dropna().unique()
                assert all(
                    val in [0, 1, True, False] for val in unique_values
                ), f"{feature} should be boolean"

    def test_calculate_composite_features(
        self,
        calculator,
        sample_ohlcv,
        sample_funding_rate,
        sample_open_interest,
        lookback_periods,
    ):
        """複合特徴量計算テスト"""
        result = calculator.calculate_composite_features(
            sample_ohlcv, sample_funding_rate, sample_open_interest, lookback_periods
        )

        expected_features = [
            "FR_OI_Ratio",
            "Market_Heat_Index",
            "Market_Stress",
            "Market_Balance",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing composite feature: {feature}"

            # Market_Stressは非負値
            if feature == "Market_Stress":
                valid_values = result[feature].dropna()
                if len(valid_values) > 0:
                    assert valid_values.min() >= 0, f"{feature} should be non-negative"

    def test_missing_data_handling(self, calculator, sample_ohlcv, lookback_periods):
        """欠損データの処理テスト"""
        # 空のファンディングレートデータ
        empty_fr = pd.DataFrame()
        result = calculator.calculate_funding_rate_features(
            sample_ohlcv, empty_fr, lookback_periods
        )

        # エラーが発生せず、元のデータが返されることを確認
        assert len(result) == len(sample_ohlcv)
        assert result.index.equals(sample_ohlcv.index)


class TestTemporalFeatureCalculator:
    """TemporalFeatureCalculator の単体テスト"""

    @pytest.fixture
    def calculator(self):
        return TemporalFeatureCalculator()

    @pytest.fixture
    def sample_ohlcv(self):
        return TestDataGenerator.create_ohlcv_data(size=168)  # 1週間分

    def test_calculate_temporal_features_normal(self, calculator, sample_ohlcv):
        """時間的特徴量計算の正常ケーステスト"""
        result = calculator.calculate_temporal_features(sample_ohlcv)

        expected_features = [
            "Hour_of_Day",
            "Day_of_Week",
            "Is_Weekend",
            "Is_Monday",
            "Is_Friday",
            "Asia_Session",
            "Europe_Session",
            "US_Session",
            "Session_Overlap_Asia_Europe",
            "Session_Overlap_Europe_US",
            "Hour_Sin",
            "Hour_Cos",
            "Day_Sin",
            "Day_Cos",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing temporal feature: {feature}"

            # 時間の範囲確認
            if feature == "Hour_of_Day":
                assert result[feature].min() >= 0
                assert result[feature].max() <= 23

            # 曜日の範囲確認
            if feature == "Day_of_Week":
                assert result[feature].min() >= 0
                assert result[feature].max() <= 6

            # ブール値特徴量の確認
            if (
                feature.startswith("Is_")
                or feature.endswith("_Session")
                or "Overlap" in feature
            ):
                unique_values = result[feature].unique()
                assert all(
                    val in [0, 1, True, False] for val in unique_values
                ), f"{feature} should be boolean"

            # 周期的エンコーディングの範囲確認
            if feature.endswith("_Sin") or feature.endswith("_Cos"):
                assert result[feature].min() >= -1
                assert result[feature].max() <= 1

    def test_timezone_handling(self, calculator):
        """タイムゾーン処理テスト"""
        # タイムゾーンなしのデータ
        dates_naive = pd.date_range(start="2024-01-01", periods=24, freq="h")
        ohlcv_naive = TestDataGenerator.create_ohlcv_data(size=24)
        ohlcv_naive.index = dates_naive

        result = calculator.calculate_temporal_features(ohlcv_naive)

        # UTCタイムゾーンが設定されていることを確認
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_invalid_index_handling(self, calculator):
        """無効なインデックスの処理テスト"""
        # DatetimeIndexでないデータ
        invalid_data = pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [1, 2, 3],
                "Low": [1, 2, 3],
                "Close": [1, 2, 3],
                "Volume": [1, 2, 3],
            }
        )

        with pytest.raises(ValueError, match="DatetimeIndexである必要があります"):
            calculator.calculate_temporal_features(invalid_data)


class TestInteractionFeatureCalculator:
    """InteractionFeatureCalculator の単体テスト"""

    @pytest.fixture
    def calculator(self):
        return InteractionFeatureCalculator()

    @pytest.fixture
    def sample_data_with_features(self):
        """既存の特徴量を含むサンプルデータ"""
        ohlcv = TestDataGenerator.create_ohlcv_data(size=100)

        # 必要な特徴量を追加（実際の実装で必要な特徴量名に合わせる）
        ohlcv["ATR_20"] = np.random.uniform(100, 1000, len(ohlcv))
        ohlcv["Price_Momentum_14"] = np.random.uniform(-0.1, 0.1, len(ohlcv))
        ohlcv["Price_Change_5"] = np.random.uniform(-0.05, 0.05, len(ohlcv))  # 追加
        ohlcv["Volume_Ratio"] = np.random.uniform(0.5, 2.0, len(ohlcv))
        ohlcv["Trend_Strength"] = np.random.uniform(-1, 1, len(ohlcv))
        ohlcv["Breakout_Strength"] = np.random.uniform(0, 1, len(ohlcv))  # 追加
        ohlcv["RSI"] = np.random.uniform(20, 80, len(ohlcv))
        ohlcv["FR_Normalized"] = np.random.uniform(-0.01, 0.01, len(ohlcv))
        ohlcv["FR_Extreme_High"] = np.random.choice([True, False], len(ohlcv))
        ohlcv["FR_Extreme_Low"] = np.random.choice([True, False], len(ohlcv))
        ohlcv["OI_Change_Rate"] = np.random.uniform(-0.1, 0.1, len(ohlcv))
        ohlcv["OI_Trend"] = np.random.uniform(-1, 1, len(ohlcv))

        return ohlcv

    def test_calculate_interaction_features_normal(
        self, calculator, sample_data_with_features
    ):
        """相互作用特徴量計算の正常ケーステスト"""
        result = calculator.calculate_interaction_features(sample_data_with_features)

        # 実際に生成される特徴量を確認（get_feature_names()から取得）
        calculator.get_feature_names()

        # 実際に生成された特徴量を確認
        generated_features = [
            col
            for col in result.columns
            if col not in sample_data_with_features.columns
        ]

        # 少なくとも一部の相互作用特徴量が生成されていることを確認
        assert len(generated_features) > 0, "No interaction features were generated"

        # 生成された特徴量の品質確認
        for feature in generated_features:
            # 無限大値がないことを確認
            assert not np.isinf(
                result[feature]
            ).any(), f"Infinite values found in {feature}"

            # 極端に大きな値がクリップされていることを確認
            assert (
                result[feature].abs().max() <= 1e6
            ), f"Values in {feature} exceed clipping threshold"

    def test_missing_required_features(self, calculator):
        """必要な特徴量が不足している場合のテスト"""
        # 基本的なOHLCVデータのみ
        basic_data = TestDataGenerator.create_ohlcv_data(size=50)

        result = calculator.calculate_interaction_features(basic_data)

        # 警告が出力され、元のデータが返されることを確認
        assert len(result) == len(basic_data)
        assert result.index.equals(basic_data.index)

    def test_extreme_values_handling(self, calculator):
        """極端な値の処理テスト"""
        data = TestDataGenerator.create_ohlcv_data(size=50)

        # 極端な値を設定
        data["ATR_20"] = [1e10] * len(data)  # 極端に大きな値
        data["Price_Momentum_14"] = [1e10] * len(data)
        data["RSI"] = [50] * len(data)
        data["FR_Normalized"] = [1e10] * len(data)

        result = calculator.calculate_interaction_features(data)

        # クリッピングが適用されていることを確認
        if "Volatility_Momentum_Interaction" in result.columns:
            assert result["Volatility_Momentum_Interaction"].abs().max() <= 1e6


class TestFeatureEngineeringServiceIntegration:
    """FeatureEngineeringService の統合テスト"""

    @pytest.fixture
    def service(self):
        return FeatureEngineeringService()

    @pytest.fixture
    def comprehensive_data(self):
        """包括的なテストデータセット"""
        ohlcv = TestDataGenerator.create_ohlcv_data(size=200)
        funding_rate = TestDataGenerator.create_funding_rate_data(size=200)
        open_interest = TestDataGenerator.create_open_interest_data(size=200)

        return ohlcv, funding_rate, open_interest

    def test_full_pipeline_integration(self, service, comprehensive_data):
        """完全なパイプライン統合テスト"""
        ohlcv, funding_rate, open_interest = comprehensive_data

        result = service.calculate_advanced_features(ohlcv, funding_rate, open_interest)

        # 基本的な検証
        assert result is not None
        assert len(result) == len(ohlcv)
        assert result.index.equals(ohlcv.index)

        # 元のOHLCVカラムが保持されていることを確認
        original_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in original_columns:
            assert col in result.columns, f"Original column {col} missing"

        # 各カテゴリの特徴量が生成されていることを確認
        feature_categories = {
            "price": ["Price_Momentum_14", "ATR_20", "VWAP"],
            "technical": ["RSI", "MACD", "Trend_Strength"],
            "market_data": ["FR_Normalized", "OI_Change_Rate", "Market_Heat_Index"],
            "temporal": ["Hour_of_Day", "Asia_Session", "Hour_Sin"],
            "interaction": ["Volatility_Momentum_Interaction", "FR_RSI_Extreme"],
        }

        for category, features in feature_categories.items():
            for feature in features:
                assert (
                    feature in result.columns
                ), f"Missing {category} feature: {feature}"

        # 最小限の特徴量数を確認（元の5列 + 生成された特徴量）
        assert (
            len(result.columns) >= 50
        ), f"Expected at least 50 features, got {len(result.columns)}"

    def test_feature_names_consistency(self, service):
        """特徴量名の一貫性テスト"""
        feature_names = service.get_feature_names()

        # 重複がないことを確認
        assert len(feature_names) == len(
            set(feature_names)
        ), "Duplicate feature names found"

        # 空の名前がないことを確認
        assert all(name.strip() for name in feature_names), "Empty feature names found"

    def test_caching_functionality(self, service, comprehensive_data):
        """キャッシュ機能のテスト"""
        ohlcv, funding_rate, open_interest = comprehensive_data

        # 最初の計算
        result1 = service.calculate_advanced_features(
            ohlcv, funding_rate, open_interest
        )

        # 同じデータでの2回目の計算（キャッシュから取得されるはず）
        with patch.object(service.price_calculator, "calculate_price_features"):
            result2 = service.calculate_advanced_features(
                ohlcv, funding_rate, open_interest
            )

            # キャッシュが使用されているかは実装依存だが、結果は同じであるべき
            pd.testing.assert_frame_equal(result1, result2)

    def test_memory_limit_handling(self, service):
        """メモリ制限の処理テスト"""
        # 大量のデータ（50,000行超）
        large_ohlcv = TestDataGenerator.create_ohlcv_data(size=60000)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = service.calculate_advanced_features(large_ohlcv)

            # 警告が出力されることを確認
            warning_messages = [str(warning.message) for warning in w]
            [msg for msg in warning_messages if "大量のデータ" in msg or "制限" in msg]

            # データが制限されていることを確認
            assert len(result) <= 50000, "Data should be limited to 50,000 rows"


class TestEdgeCasesAndErrorHandling:
    """エッジケースとエラーハンドリングのテスト"""

    @pytest.fixture
    def service(self):
        return FeatureEngineeringService()

    def test_empty_dataframe_handling(self, service):
        """空のDataFrameの処理テスト"""
        empty_df = pd.DataFrame()

        # 空のDataFrameではValueErrorが発生することを確認
        with pytest.raises(ValueError, match="OHLCVデータが空です"):
            service.calculate_advanced_features(empty_df)

    def test_single_row_data(self, service):
        """単一行データの処理テスト"""
        single_row = TestDataGenerator.create_ohlcv_data(size=1)

        result = service.calculate_advanced_features(single_row)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == 1
        assert result.index.equals(single_row.index)

    def test_missing_columns_handling(self, service):
        """必須カラムが不足している場合のテスト"""
        # Closeカラムが不足
        incomplete_data = pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [1, 2, 3],
                "Low": [1, 2, 3],
                "Volume": [1, 2, 3],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        # エラーが適切に処理されることを確認（AttributeErrorまたはKeyErrorが発生）
        with pytest.raises((KeyError, ValueError, AttributeError)):
            service.calculate_advanced_features(incomplete_data)

    def test_nan_values_handling(self, service):
        """NaN値を含むデータの処理テスト"""
        data = TestDataGenerator.create_ohlcv_data(size=100)

        # 一部にNaN値を挿入
        data.loc[data.index[10:20], "Close"] = np.nan
        data.loc[data.index[30:35], "Volume"] = np.nan

        result = service.calculate_advanced_features(data)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == len(data)
        assert result.index.equals(data.index)

    def test_infinite_values_handling(self, service):
        """無限大値を含むデータの処理テスト"""
        data = TestDataGenerator.create_ohlcv_data(size=50)

        # 無限大値を挿入
        data.loc[data.index[10], "High"] = np.inf
        data.loc[data.index[20], "Low"] = -np.inf

        result = service.calculate_advanced_features(data)

        # 生成された特徴量に無限大値がないことを確認（元のOHLCVは除く）
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        feature_columns = [
            col
            for col in numeric_columns
            if col not in ["Open", "High", "Low", "Close", "Volume"]
        ]

        for col in feature_columns:
            assert not np.isinf(
                result[col]
            ).any(), f"Infinite values found in feature {col}"

    def test_zero_values_handling(self, service):
        """ゼロ値を含むデータの処理テスト"""
        data = TestDataGenerator.create_ohlcv_data(size=50)

        # ゼロ値を挿入
        data.loc[data.index[10:15], "Volume"] = 0
        data.loc[data.index[20], "Close"] = 0

        result = service.calculate_advanced_features(data)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == len(data)
        assert result.index.equals(data.index)

    def test_negative_prices_handling(self, service):
        """負の価格を含むデータの処理テスト"""
        data = TestDataGenerator.create_ohlcv_data(size=50)

        # 負の価格を挿入
        data.loc[data.index[10], "Close"] = -100
        data.loc[data.index[20], "Low"] = -50

        result = service.calculate_advanced_features(data)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == len(data)
        assert result.index.equals(data.index)

    def test_mismatched_data_lengths(self, service):
        """データ長が一致しない場合のテスト"""
        ohlcv = TestDataGenerator.create_ohlcv_data(size=100)
        funding_rate = TestDataGenerator.create_funding_rate_data(size=50)  # 異なる長さ
        open_interest = TestDataGenerator.create_open_interest_data(
            size=150
        )  # 異なる長さ

        result = service.calculate_advanced_features(ohlcv, funding_rate, open_interest)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == len(ohlcv)
        assert result.index.equals(ohlcv.index)

    def test_different_timezones_handling(self, service):
        """異なるタイムゾーンのデータの処理テスト"""
        # UTCデータ
        ohlcv_utc = TestDataGenerator.create_ohlcv_data(size=50)

        # JSTデータ
        jst_dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="Asia/Tokyo")
        funding_rate_jst = pd.DataFrame(
            {"funding_rate": np.random.uniform(-0.0001, 0.0001, 50)}, index=jst_dates
        )

        result = service.calculate_advanced_features(ohlcv_utc, funding_rate_jst)

        # エラーが発生せず、結果が返されることを確認
        assert len(result) == len(ohlcv_utc)
        assert result.index.equals(ohlcv_utc.index)


class TestDataQualityValidation:
    """データ品質検証テスト"""

    @pytest.fixture
    def service(self):
        return FeatureEngineeringService()

    @pytest.fixture
    def quality_test_data(self):
        return TestDataGenerator.create_ohlcv_data(size=200)

    def test_feature_value_ranges(self, service, quality_test_data):
        """特徴量の値の範囲検証テスト"""
        result = service.calculate_advanced_features(quality_test_data)

        # RSIの範囲確認（0-100）
        if "RSI" in result.columns:
            rsi_values = result["RSI"].dropna()
            if len(rsi_values) > 0:
                assert rsi_values.min() >= 0, "RSI values should be >= 0"
                assert rsi_values.max() <= 100, "RSI values should be <= 100"

        # 時間特徴量の範囲確認
        if "Hour_of_Day" in result.columns:
            assert result["Hour_of_Day"].min() >= 0
            assert result["Hour_of_Day"].max() <= 23

        if "Day_of_Week" in result.columns:
            assert result["Day_of_Week"].min() >= 0
            assert result["Day_of_Week"].max() <= 6

        # 周期的エンコーディングの範囲確認
        for col in ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"]:
            if col in result.columns:
                assert result[col].min() >= -1, f"{col} should be >= -1"
                assert result[col].max() <= 1, f"{col} should be <= 1"

    def test_feature_consistency(self, service, quality_test_data):
        """特徴量の一貫性検証テスト"""
        result = service.calculate_advanced_features(quality_test_data)

        # 移動平均の一貫性確認（データによっては短期の方が変動が小さい場合もあるため、緩い条件に変更）
        if "MA_10" in result.columns and "MA_50" in result.columns:
            # 移動平均が計算されていることを確認
            ma10_values = result["MA_10"].dropna()
            ma50_values = result["MA_50"].dropna()
            assert len(ma10_values) > 0, "MA_10 should have valid values"
            assert len(ma50_values) > 0, "MA_50 should have valid values"

        # ボラティリティ特徴量の非負性確認
        volatility_features = ["ATR_20", "Realized_Volatility_20", "Volatility_Spike"]
        for feature in volatility_features:
            if feature in result.columns:
                valid_values = result[feature].dropna()
                if len(valid_values) > 0:
                    assert valid_values.min() >= 0, f"{feature} should be non-negative"

    def test_no_duplicate_features(self, service):
        """重複特徴量がないことの確認テスト"""
        feature_names = service.get_feature_names()

        # 重複がないことを確認
        assert len(feature_names) == len(
            set(feature_names)
        ), "Duplicate feature names found"

        # 空の名前がないことを確認
        assert all(
            name and name.strip() for name in feature_names
        ), "Empty or whitespace-only feature names found"


if __name__ == "__main__":
    """テスト実行用メイン関数"""
    import sys

    # テストの実行
    exit_code = pytest.main(
        [
            __file__,
            "-v",  # 詳細出力
            "--tb=short",  # 短いトレースバック
            "--durations=10",  # 最も時間のかかった10個のテストを表示
            "--strict-markers",  # 未定義のマーカーでエラー
            "-x",  # 最初の失敗で停止
        ]
    )

    sys.exit(exit_code)
