import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.ml.feature_engineering.complexity_features import (
    ComplexityFeatureCalculator,
)


class TestComplexityFeatureCalculator:
    @pytest.fixture
    def sample_ohlcv(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.rand(100) * 1000,
            },
            index=dates,
        )
        return df

    @patch("app.services.ml.feature_engineering.complexity_features.AdvancedFeatures")
    def test_calculate_features_basic(self, mock_adv_features, sample_ohlcv):
        """標準設定での計算とカラム存在確認"""
        # モックの設定: すべての計算が同じ長さのSeriesを返すようにする
        mock_series = pd.Series(0.5, index=sample_ohlcv.index)
        mock_adv_features.hurst_exponent.return_value = mock_series
        mock_adv_features.fractal_dimension.return_value = mock_series
        mock_adv_features.sample_entropy.return_value = mock_series
        mock_adv_features.vpin_approximation.return_value = mock_series

        calc = ComplexityFeatureCalculator()
        res = calc.calculate_features(sample_ohlcv)

        # AdvancedFeaturesのメソッドが呼ばれたか確認
        assert mock_adv_features.hurst_exponent.called
        assert mock_adv_features.fractal_dimension.called
        assert mock_adv_features.sample_entropy.called
        assert mock_adv_features.vpin_approximation.called

        # 期待されるカラムが含まれているか
        expected_cols = [
            "Hurst_50",
            "Hurst_100",
            "Fractal_Dim_20",
            "Fractal_Dim_50",
            "Sample_Entropy_20",
            "VPIN_20",
            "VPIN_50",
            "Complexity_Trend_Trust",
            "Complexity_Adjusted_ER",
        ]
        for col in expected_cols:
            assert col in res.columns

        # データ長が維持されているか
        assert len(res) == len(sample_ohlcv)
        # インデックスが維持されているか
        pd.testing.assert_index_equal(res.index, sample_ohlcv.index)

    @patch("app.services.ml.feature_engineering.complexity_features.AdvancedFeatures")
    def test_calculate_features_custom_config(self, mock_adv_features, sample_ohlcv):
        """カスタム設定（期間）が反映されるか確認"""
        mock_series = pd.Series(0.5, index=sample_ohlcv.index)
        mock_adv_features.hurst_exponent.return_value = mock_series
        mock_adv_features.fractal_dimension.return_value = mock_series
        mock_adv_features.sample_entropy.return_value = mock_series
        mock_adv_features.vpin_approximation.return_value = mock_series

        config = {"lookback_periods": {"short": 10, "mid": 30, "long": 60}}

        calc = ComplexityFeatureCalculator()
        res = calc.calculate_features(sample_ohlcv, config)

        # カスタム期間のカラムが存在するか
        assert "Hurst_30" in res.columns
        assert "Hurst_60" in res.columns
        assert "Fractal_Dim_10" in res.columns

        # デフォルト期間のカラムが存在しないこと（設定が上書きされているか）
        assert "Hurst_50" not in res.columns

    def test_input_validation(self, sample_ohlcv):
        """入力データのバリデーション確認"""
        calc = ComplexityFeatureCalculator()

        # volumeカラムを削除してテスト
        df_no_vol = sample_ohlcv.drop(columns=["volume"])
        res = calc.calculate_features(df_no_vol)

        # カラムがなく、インデックスのみのDFが返されるはず
        assert res.empty or len(res.columns) == 0
        pd.testing.assert_index_equal(res.index, sample_ohlcv.index)

    def test_get_feature_names(self):
        """特徴量名リストの取得確認"""
        calc = ComplexityFeatureCalculator()
        names = calc.get_feature_names()

        assert "Hurst_50" in names
        assert "Complexity_Trend_Trust" in names
        assert isinstance(names, list)

    @patch("app.services.ml.feature_engineering.complexity_features.AdvancedFeatures")
    def test_calculation_logic_interaction(self, mock_adv_features, sample_ohlcv):
        """複合特徴量 Complexity_Trend_Trust の計算ロジック確認"""
        # Hurst = 0.8 (強いトレンド), Entropy = 0.1 (低い無秩序) -> Trust should be high (approx 8.0)
        mock_hurst = pd.Series(0.8, index=sample_ohlcv.index)
        mock_entropy = pd.Series(0.1, index=sample_ohlcv.index)
        mock_others = pd.Series(0.5, index=sample_ohlcv.index)

        mock_adv_features.hurst_exponent.return_value = mock_hurst
        mock_adv_features.sample_entropy.return_value = mock_entropy
        # 他の特徴量は計算に影響しないので適当な値
        mock_adv_features.fractal_dimension.return_value = mock_others
        mock_adv_features.vpin_approximation.return_value = mock_others

        calc = ComplexityFeatureCalculator()
        res = calc.calculate_features(sample_ohlcv)

        # Complexity_Trend_Trust = Hurst / (Entropy + 1e-9)
        expected_trust = 0.8 / (0.1 + 1e-9)

        # 最初の行の値を確認
        assert np.isclose(res["Complexity_Trend_Trust"].iloc[0], expected_trust)

    @patch("app.services.ml.feature_engineering.complexity_features.AdvancedFeatures")
    def test_calculation_logic_adjusted_er(self, mock_adv_features, sample_ohlcv):
        """複合特徴量 Complexity_Adjusted_ER の計算ロジック確認"""
        # このテストでは self.validate_input_data をパスさせるために正規のデータを使うが、
        # 計算結果はモックに依存させる部分と、内部計算（ER）の部分がある。
        # Complexity_Adjusted_ER = ER * (2.0 - Fractal_Dim)

        # Fractal Dimension を 1.5 に固定
        mock_fd = pd.Series(1.5, index=sample_ohlcv.index)

        # 他のモック
        mock_others = pd.Series(0.5, index=sample_ohlcv.index)

        mock_adv_features.fractal_dimension.return_value = mock_fd
        mock_adv_features.hurst_exponent.return_value = mock_others
        mock_adv_features.sample_entropy.return_value = mock_others
        mock_adv_features.vpin_approximation.return_value = mock_others

        # ERの計算を簡単にするため、closeを単調増加にする (diffが常に正、abs().sum() == abs(diff))
        # ER = Change / Volatility
        # 単調増加でノイズなしなら ER = 1.0 になるはずだが、ここではランダムデータなので計算させる。
        # むしろ意図的に単純なデータを与える。

        simple_df = sample_ohlcv.copy()
        # 1時間ごとに1ずつ増える -> diff=1. window=mid_p(50).
        # numerator: close.diff(50) = 50
        # denominator: close.diff(1).abs().rolling(50).sum() = 1 * 50 = 50
        # ER = 50 / 50 = 1.0
        simple_df["close"] = np.arange(len(sample_ohlcv), dtype=float)

        calc = ComplexityFeatureCalculator()
        res = calc.calculate_features(simple_df)

        # 最初の50個はNaN (rolling 50) なので、50番目以降をチェック
        # Complexity_Adjusted_ER = 1.0 * (2.0 - 1.5) = 0.5

        # fillna(0) が最後にあるため、NaNの箇所は0になっている可能性も考慮必要だが
        # ffill().fillna(0) なので、有効値があればそれが続く。
        # 今回のデータなら50行目以降は計算できるはず。

        index_to_check = 55
        expected_val = 1.0 * (2.0 - 1.5)  # 0.5

        assert np.isclose(
            res["Complexity_Adjusted_ER"].iloc[index_to_check], expected_val, atol=1e-5
        )
