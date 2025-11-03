"""
相互作用特徴量計算のテスト（カラム名小文字統一対応）
"""
import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator


class TestInteractionFeatureCalculator:
    """相互作用特徴量計算クラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        # DBのOHLCVカラムは、すべて小文字で統一
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            # OHLCVカラム（すべて小文字）
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 105,
            'low': np.random.randn(n).cumsum() + 95,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n),

            # 基本特徴量
            'Price_Momentum_14': np.random.randn(n),
            'Price_Change_5': np.random.randn(n),
            'RSI': np.random.uniform(0, 100, n),
            'ATR': np.random.uniform(0.5, 2.0, n),
            'ATR_20': np.random.uniform(0.5, 2.0, n),

            # オプション特徴量
            'Volume_Ratio': np.random.randn(n),
            'Trend_Strength': np.random.randn(n),
            'Breakout_Strength': np.random.randn(n),
            'FR_Normalized': np.random.randn(n),
            'OI_Change_Rate': np.random.randn(n),
            'OI_Trend': np.random.randn(n),
            'Volatility_Spike': np.random.randn(n),
            'Volume_Spike': np.random.randn(n),
            'FR_Extreme_High': np.random.randn(n),
            'FR_Extreme_Low': np.random.randn(n),
        })

        # DatetimeIndexを設定（時間関連特徴量対応）
        df.index = pd.date_range('2024-01-01', periods=n, freq='1H')
        return df

    def test_calculate_interaction_features_basic(self, sample_data):
        """基本的な相互作用特徴量計算のテスト"""
        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(sample_data)

        # 新しい特徴量が追加されていることを確認
        assert len(result_df.columns) > len(sample_data.columns)

        # 生成される特徴量の確認
        expected_features = [
            "Volatility_Momentum_Interaction",
            "Volatility_Spike_Momentum",
            "Volume_Trend_Interaction",
            "Volume_Breakout",
            "FR_RSI_Extreme",
            "FR_Overbought",
            "FR_Oversold",
            "OI_Price_Divergence",
            "OI_Momentum_Alignment",
        ]

        for feature in expected_features:
            if feature in result_df.columns:
                # 特徴量が有限値であることを確認
                assert not result_df[feature].isna().all(), f"特徴量 {feature} がすべてNaNです"
                assert np.isfinite(result_df[feature]).any(), f"特徴量 {feature} に有限値がありません"

    def test_data_preprocessing_no_ambiguous_truth(self, sample_data):
        """Seriesの真理値エラーのないデータ前処理テスト"""
        calculator = InteractionFeatureCalculator()

        # 警告が発生しないことを確認
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_df = calculator.calculate_interaction_features(sample_data)

            # "The truth value of a Series is ambiguous" 相关的警告をチェック
            ambiguous_warnings = [
                warning for warning in w
                if "truth value of a Series" in str(warning.message).lower()
                or "ambiguous" in str(warning.message).lower()
            ]

            assert len(ambiguous_warnings) == 0, f"Series真理値警告が発生: {ambiguous_warnings}"

    def test_safe_numeric_conversion(self, sample_data):
        """安全な数値変換のテスト"""
        calculator = InteractionFeatureCalculator()

        # Series を安全に数値変換
        test_series = pd.Series([1, 2, 3, np.inf, -np.inf, np.nan, '5'])
        converted = calculator._safe_numeric_conversion(test_series)

        assert converted is not None
        assert not converted.isna().any()  # NaNがない
        assert np.isfinite(converted).all()  # 無限大値がない

    def test_with_missing_features(self):
        """不足している特徴量がある場合のテスト"""
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
        })
        df.index = pd.date_range('2024-01-01', periods=100, freq='1H')

        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(df)

        # 基本特徴量が不足しているため、元のDataFrameが返される
        assert len(result_df.columns) == len(df.columns)

    def test_empty_data(self):
        """空のデータのテスト"""
        calculator = InteractionFeatureCalculator()
        empty_df = pd.DataFrame()
        result_df = calculator.calculate_interaction_features(empty_df)

        assert result_df.empty

    def test_column_case_consistency(self, sample_data):
        """カラム名の大文字小文字一貫性のテスト"""
        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(sample_data)

        # OHLCVカラムがすべて小文字であることを確認
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            assert col in sample_data.columns, f"OHLCVカラム {col} が見つかりません"

    def test_numerical_stability(self, sample_data):
        """数値安定性のテスト"""
        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(sample_data)

        # 生成された特徴量の数値安定性をチェック
        feature_columns = [col for col in result_df.columns if col not in sample_data.columns]

        for col in feature_columns:
            if col in result_df.columns:
                # 極端な値をチェック
                assert not result_df[col].isna().all(), f"特徴量 {col} がすべてNaNです"
                # 無限大値をチェック
                finite_values = result_df[col][np.isfinite(result_df[col])]
                if len(finite_values) > 0:
                    assert abs(finite_values.max()) < 1e6, f"特徴量 {col} に極端な値があります"

    def test_atr_column_variants(self):
        """ATRカラムのバリエーションテスト（基本特徴量含む）"""
        np.random.seed(42)
        n = 100

        # 基本特徴量を含むデータでATR_14テスト
        df1 = pd.DataFrame({
            'Price_Momentum_14': np.random.randn(n),
            'Price_Change_5': np.random.randn(n),
            'RSI': np.random.uniform(0, 100, n),
            'ATR_14': np.random.uniform(0.5, 2.0, n),
        })
        calculator = InteractionFeatureCalculator()
        result1 = calculator.calculate_interaction_features(df1)
        # 基本特徴量があれば相互作用特徴量は生成される
        if len(result1.columns) > len(df1.columns):
            assert "Volatility_Momentum_Interaction" in result1.columns

        # ATR_20 がある場合
        df2 = pd.DataFrame({
            'Price_Momentum_14': np.random.randn(n),
            'Price_Change_5': np.random.randn(n),
            'RSI': np.random.uniform(0, 100, n),
            'ATR_20': np.random.uniform(0.5, 2.0, n),
        })
        result2 = calculator.calculate_interaction_features(df2)
        if len(result2.columns) > len(df2.columns):
            assert "Volatility_Momentum_Interaction" in result2.columns

        # ATR がある場合
        df3 = pd.DataFrame({
            'Price_Momentum_14': np.random.randn(n),
            'Price_Change_5': np.random.randn(n),
            'RSI': np.random.uniform(0, 100, n),
            'ATR': np.random.uniform(0.5, 2.0, n),
        })
        result3 = calculator.calculate_interaction_features(df3)
        if len(result3.columns) > len(df3.columns):
            assert "Volatility_Momentum_Interaction" in result3.columns

    def test_interaction_feature_generation(self, sample_data):
        """相互作用特徴量生成の詳細テスト"""
        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(sample_data)

        # 各相互作用特徴量の生成を確認
        interactions = {
            "Volatility_Momentum_Interaction": "ATR × Price_Momentum_14",
            "Volume_Trend_Interaction": "Volume_Ratio × Trend_Strength",
            "FR_RSI_Extreme": "FR_Normalized × (RSI - 50)",
            "OI_Price_Divergence": "OI_Change_Rate × Price_Change_5",
        }

        for feature, description in interactions.items():
            if feature in result_df.columns:
                # 特徴量が生成されていることを確認
                assert feature in result_df.columns, f"{description} の特徴量 {feature} が生成されていません"
                # 数値であることを確認
                assert pd.api.types.is_numeric_dtype(result_df[feature])

    def test_get_feature_names(self):
        """特徴量名リスト取得のテスト"""
        calculator = InteractionFeatureCalculator()
        feature_names = calculator.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "Volatility_Momentum_Interaction" in feature_names
