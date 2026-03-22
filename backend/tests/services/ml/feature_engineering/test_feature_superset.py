"""
Feature Superset Generation Tests

create_feature_superset メソッドのユニットテスト。
複数のd値でFracDiff特徴量が正しく生成されることを確認。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)


class TestCreateFeatureSuperset:
    """create_feature_superset メソッドのテスト"""

    @pytest.fixture
    def fe_service(self):
        """FeatureEngineeringService インスタンス"""
        return FeatureEngineeringService()

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用 OHLCV データ"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")

        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        high = close + np.abs(np.random.randn(n) * 50)
        low = close - np.abs(np.random.randn(n) * 50)
        open_ = close + np.random.randn(n) * 30
        volume = np.abs(np.random.randn(n) * 1000) + 100

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_oi(self, sample_ohlcv):
        """テスト用 Open Interest データ"""
        np.random.seed(123)
        n = len(sample_ohlcv)
        return pd.DataFrame(
            {"open_interest_value": np.abs(np.random.randn(n) * 1e9) + 1e8},
            index=sample_ohlcv.index,
        )

    def test_superset_generates_multiple_d_values(self, fe_service, sample_ohlcv):
        """複数のd値でFracDiff列が生成されることを確認"""
        d_values = [0.3, 0.4, 0.5]

        result = fe_service.create_feature_superset(
            sample_ohlcv, frac_diff_d_values=d_values
        )

        # 各d値に対応するカラムが存在することを確認
        for d in d_values:
            col_name = f"FracDiff_Price_d{d}"
            assert col_name in result.columns, f"{col_name} が生成されていない"

    def test_superset_column_naming_format(self, fe_service, sample_ohlcv):
        """カラム名のフォーマットが正しいことを確認"""
        d_values = [0.3, 0.4]

        result = fe_service.create_feature_superset(
            sample_ohlcv, frac_diff_d_values=d_values
        )

        frac_cols = [c for c in result.columns if c.startswith("FracDiff_")]

        # 全てのFracDiff列が正しいフォーマットか確認
        for col in frac_cols:
            assert "_d" in col, f"{col} にd値情報がない"
            # パターン: FracDiff_<Type>_d<value>
            parts = col.split("_d")
            assert len(parts) == 2, f"{col} のフォーマットが不正"

    def test_superset_with_oi_data(self, fe_service, sample_ohlcv, sample_oi):
        """OIデータがある場合、FracDiff_OI列も生成されることを確認"""
        d_values = [0.4, 0.5]

        result = fe_service.create_feature_superset(
            sample_ohlcv,
            open_interest_data=sample_oi,
            frac_diff_d_values=d_values,
        )

        # Price と OI 両方のFracDiff列が存在することを確認
        for d in d_values:
            assert f"FracDiff_Price_d{d}" in result.columns
            assert f"FracDiff_OI_d{d}" in result.columns

    def test_superset_no_nan_in_output(self, fe_service, sample_ohlcv):
        """出力にNaN値がないことを確認（ffill + fillna(0) の確認）"""
        result = fe_service.create_feature_superset(sample_ohlcv)

        # 数値カラムにNaNがないことを確認
        num_cols = result.select_dtypes(include=[np.number]).columns
        nan_counts = result[num_cols].isna().sum().sum()

        assert nan_counts == 0, f"NaN値が {nan_counts} 個残っている"

    def test_superset_default_d_values(self, fe_service, sample_ohlcv):
        """デフォルトのd値が正しく適用されることを確認"""
        result = fe_service.create_feature_superset(sample_ohlcv)

        # デフォルト: [0.3, 0.4, 0.5, 0.6]
        expected_d_values = [0.3, 0.4, 0.5, 0.6]

        for d in expected_d_values:
            assert f"FracDiff_Price_d{d}" in result.columns

    def test_no_duplicate_columns(self, fe_service, sample_ohlcv):
        """重複カラムがないことを確認"""
        result = fe_service.create_feature_superset(sample_ohlcv)

        duplicates = result.columns[result.columns.duplicated()].tolist()
        assert len(duplicates) == 0, f"重複カラム: {duplicates}"


class TestFilterSupersetForD:
    """filter_superset_for_d ヘルパーメソッドのテスト"""

    def test_filter_selects_correct_d_columns(self):
        """指定されたd値のカラムのみ選択されることを確認"""
        # モックDataFrame
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [4, 5, 6],
                "FracDiff_Price_d0.3": [0.1, 0.2, 0.3],
                "FracDiff_Price_d0.4": [0.4, 0.5, 0.6],
                "FracDiff_OI_d0.3": [0.7, 0.8, 0.9],
                "FracDiff_OI_d0.4": [1.0, 1.1, 1.2],
            }
        )

        result = FeatureEngineeringService.filter_superset_for_d(df, d_value=0.4)

        # d=0.4 のカラムのみ残っていることを確認
        assert "FracDiff_Price_d0.4" in result.columns
        assert "FracDiff_OI_d0.4" in result.columns

        # d=0.3 のカラムは除外されていることを確認
        assert "FracDiff_Price_d0.3" not in result.columns
        assert "FracDiff_OI_d0.3" not in result.columns

        # 非FracDiff列は残っていることを確認
        assert "feature_a" in result.columns
        assert "feature_b" in result.columns

    def test_filter_preserves_all_non_fracdiff_columns(self):
        """FracDiff以外のカラムが全て保持されることを確認"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                "RSI_14": [50, 55, 60],
                "FracDiff_Price_d0.3": [0.1, 0.2, 0.3],
                "FracDiff_Price_d0.5": [0.4, 0.5, 0.6],
            }
        )

        result = FeatureEngineeringService.filter_superset_for_d(df, d_value=0.3)

        assert "close" in result.columns
        assert "volume" in result.columns
        assert "RSI_14" in result.columns
        assert len(result.columns) == 4  # 3 non-frac + 1 frac


class TestGetFracDiffColumnsForD:
    """get_frac_diff_columns_for_d ヘルパーメソッドのテスト"""

    def test_returns_correct_columns(self):
        """正しいカラム名リストが返されることを確認"""
        columns = [
            "feature_a",
            "FracDiff_Price_d0.3",
            "FracDiff_Price_d0.4",
            "FracDiff_OI_d0.3",
            "FracDiff_OI_d0.4",
        ]

        result = FeatureEngineeringService.get_frac_diff_columns_for_d(columns, 0.4)

        assert "FracDiff_Price_d0.4" in result
        assert "FracDiff_OI_d0.4" in result
        assert "FracDiff_Price_d0.3" not in result
        assert len(result) == 2

    def test_returns_empty_for_no_match(self):
        """マッチするカラムがない場合は空リストを返す"""
        columns = ["feature_a", "feature_b"]

        result = FeatureEngineeringService.get_frac_diff_columns_for_d(columns, 0.5)

        assert result == []
