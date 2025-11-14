"""
特徴量重要度分析スクリプトのテスト
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analyze_feature_importance import FeatureImportanceAnalyzer


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータを生成"""
    np.random.seed(42)
    n_samples = 200

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")
    close_prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)

    df = pd.DataFrame(
        {
            "open": close_prices + np.random.randn(n_samples) * 50,
            "high": close_prices + np.abs(np.random.randn(n_samples) * 100),
            "low": close_prices - np.abs(np.random.randn(n_samples) * 100),
            "close": close_prices,
            "volume": np.random.uniform(100, 1000, n_samples),
        },
        index=dates,
    )

    return df


@pytest.fixture
def sample_features():
    """サンプル特徴量データを生成"""
    np.random.seed(42)
    n_samples = 100

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

    df = pd.DataFrame(
        {
            "open": np.random.uniform(40000, 60000, n_samples),
            "high": np.random.uniform(40000, 60000, n_samples),
            "low": np.random.uniform(40000, 60000, n_samples),
            "close": np.random.uniform(40000, 60000, n_samples),
            "volume": np.random.uniform(100, 1000, n_samples),
            "RSI_14": np.random.uniform(30, 70, n_samples),
            "MACD": np.random.uniform(-100, 100, n_samples),
            "BB_Upper": np.random.uniform(50000, 60000, n_samples),
            "feature_high_corr_1": np.random.uniform(0, 1, n_samples),
            "feature_high_corr_2": np.random.uniform(0, 1, n_samples),
            "feature_low_importance": np.random.uniform(0, 0.01, n_samples),
        },
        index=dates,
    )

    # 高相関特徴量を作成
    df["feature_high_corr_2"] = (
        df["feature_high_corr_1"] * 0.98 + np.random.randn(n_samples) * 0.01
    )

    return df


@pytest.fixture
def sample_labels(sample_features):
    """サンプルラベルを生成"""
    np.random.seed(42)
    n_samples = len(sample_features)

    # 3クラスのラベルを生成
    labels = pd.Series(
        np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3]),
        index=sample_features.index,
    )

    return labels


class TestFeatureImportanceAnalyzer:
    """FeatureImportanceAnalyzerのテストクラス"""

    def test_init(self):
        """初期化テスト"""
        analyzer = FeatureImportanceAnalyzer(min_samples=500)
        assert analyzer.min_samples == 500
        assert analyzer.feature_service is not None
        assert analyzer.results == {}

    def test_generate_labels(self, sample_features):
        """ラベル生成テスト"""
        analyzer = FeatureImportanceAnalyzer()
        labels = analyzer.generate_labels(sample_features)

        # ラベルが生成されていることを確認
        assert len(labels) > 0
        assert labels.dtype == int

        # 3クラスに分類されていることを確認
        unique_labels = labels.unique()
        assert len(unique_labels) <= 3
        assert all(label in [0, 1, 2] for label in unique_labels)

    def test_prepare_data(self, sample_features, sample_labels):
        """データ準備テスト"""
        analyzer = FeatureImportanceAnalyzer()
        X_train, X_val, y_train, y_val = analyzer.prepare_data(
            sample_features, sample_labels
        )

        # データが分割されていることを確認
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(y_train) > 0
        assert len(y_val) > 0

        # 比率が正しいことを確認（おおよそ80:20）
        total_samples = len(X_train) + len(X_val)
        train_ratio = len(X_train) / total_samples
        assert 0.75 < train_ratio < 0.85

        # 基本カラムが除外されていることを確認
        assert "close" not in X_train.columns
        assert "volume" not in X_train.columns

    def test_analyze_lightgbm_importance(self, sample_features, sample_labels):
        """LightGBM重要度計算テスト"""
        analyzer = FeatureImportanceAnalyzer()
        X_train, X_val, y_train, y_val = analyzer.prepare_data(
            sample_features, sample_labels
        )

        importance = analyzer.analyze_lightgbm_importance(X_train, X_val, y_train)

        # 重要度が計算されていることを確認
        assert len(importance) > 0
        assert all(0 <= v <= 1 for v in importance.values())

        # 全特徴量の重要度が含まれていることを確認
        assert set(importance.keys()) == set(X_train.columns)

    def test_analyze_permutation_importance(self, sample_features, sample_labels):
        """Permutation重要度計算テスト"""
        from lightgbm import LGBMClassifier

        analyzer = FeatureImportanceAnalyzer()
        X_train, X_val, y_train, y_val = analyzer.prepare_data(
            sample_features, sample_labels
        )

        # モデルを学習
        model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        importance = analyzer.analyze_permutation_importance(model, X_val, y_val)

        # 重要度が計算されていることを確認
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

        # 全特徴量の重要度が含まれていることを確認
        assert set(importance.keys()) == set(X_val.columns)

    def test_analyze_correlation(self, sample_features):
        """相関分析テスト"""
        analyzer = FeatureImportanceAnalyzer()

        # 基本カラムを除外
        exclude_cols = ["open", "high", "low", "close", "volume"]
        X = sample_features[
            [col for col in sample_features.columns if col not in exclude_cols]
        ]

        high_corr_pairs = analyzer.analyze_correlation(X, threshold=0.95)

        # 高相関ペアが検出されることを確認
        assert isinstance(high_corr_pairs, dict)

        # 相関値が正しい範囲内であることを確認
        for feature, (corr_feature, corr_value) in high_corr_pairs.items():
            assert 0 <= corr_value <= 1
            assert feature in X.columns
            assert corr_feature in X.columns

    def test_integrate_results(self):
        """結果統合テスト"""
        analyzer = FeatureImportanceAnalyzer()

        lgbm_importance = {"feature1": 0.8, "feature2": 0.3, "feature3": 0.05}
        perm_importance = {"feature1": 0.7, "feature2": 0.4, "feature3": 0.03}
        high_corr_pairs = {"feature2": ("feature1", 0.97)}

        results_df = analyzer.integrate_results(
            lgbm_importance, perm_importance, high_corr_pairs
        )

        # DataFrameが正しく生成されていることを確認
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3

        # 必要なカラムが含まれていることを確認
        required_columns = [
            "feature_name",
            "lgbm_importance",
            "perm_importance",
            "avg_importance",
            "recommendation",
        ]
        for col in required_columns:
            assert col in results_df.columns

        # 平均重要度が正しく計算されていることを確認
        feature1_row = results_df[results_df["feature_name"] == "feature1"].iloc[0]
        expected_avg = (0.8 + 0.7) / 2
        assert abs(feature1_row["avg_importance"] - expected_avg) < 0.01

        # 削除推奨が正しく判定されていることを確認
        feature3_row = results_df[results_df["feature_name"] == "feature3"].iloc[0]
        assert feature3_row["recommendation"] == "remove"
        assert "低重要度" in feature3_row["reason"]

    def test_save_results(self, tmp_path):
        """結果保存テスト"""
        analyzer = FeatureImportanceAnalyzer()

        # テスト用の結果DataFrame
        results_df = pd.DataFrame(
            {
                "feature_name": ["feature1", "feature2"],
                "lgbm_importance": [0.8, 0.3],
                "perm_importance": [0.7, 0.4],
                "avg_importance": [0.75, 0.35],
                "recommendation": ["keep", "remove"],
            }
        )

        # 一時ファイルに保存
        output_path = tmp_path / "test_results.csv"
        analyzer.save_results(results_df, str(output_path))

        # ファイルが作成されていることを確認
        assert output_path.exists()

        # ファイルを読み込んで内容を確認
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 2
        assert list(loaded_df["feature_name"]) == ["feature1", "feature2"]

    @patch("scripts.analyze_feature_importance.SessionLocal")
    @patch.object(FeatureImportanceAnalyzer, "generate_features")
    def test_load_data_with_mock(self, mock_generate, mock_session):
        """データ読み込みテスト（モック使用）"""
        # モックデータを準備
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_repo = MagicMock()
        mock_df = pd.DataFrame(
            {
                "open": [50000],
                "high": [51000],
                "low": [49000],
                "close": [50500],
                "volume": [100],
            }
        )
        mock_repo.get_ohlcv_dataframe.return_value = mock_df

        with patch(
            "scripts.analyze_feature_importance.OHLCVRepository",
            return_value=mock_repo,
        ):
            analyzer = FeatureImportanceAnalyzer(min_samples=1)
            df = analyzer.load_data()

            # データが読み込まれていることを確認
            assert len(df) > 0
            assert "close" in df.columns

    def test_print_report(self, capsys):
        """レポート出力テスト"""
        analyzer = FeatureImportanceAnalyzer()

        results_df = pd.DataFrame(
            {
                "feature_name": ["feature1", "feature2", "feature3"],
                "lgbm_importance": [0.8, 0.3, 0.05],
                "perm_importance": [0.7, 0.4, 0.03],
                "avg_importance": [0.75, 0.35, 0.04],
                "corr_feature": ["", "", ""],
                "corr_value": [0.0, 0.0, 0.0],
                "recommendation": ["keep", "keep", "remove"],
                "reason": ["", "", "低重要度"],
            }
        )

        analyzer.print_report(results_df)

        # 出力を取得
        captured = capsys.readouterr()

        # レポートに必要な情報が含まれていることを確認
        assert "特徴量重要度分析結果" in captured.out
        assert "総特徴量数" in captured.out
        assert "上位20特徴量" in captured.out
        assert "削除推奨特徴量" in captured.out

    def test_integration(self, sample_features, sample_labels):
        """統合テスト（主要メソッドの連携）"""
        analyzer = FeatureImportanceAnalyzer()

        # データ準備
        X_train, X_val, y_train, y_val = analyzer.prepare_data(
            sample_features, sample_labels
        )

        # LightGBM重要度
        lgbm_importance = analyzer.analyze_lightgbm_importance(X_train, X_val, y_train)

        # モデル学習
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # Permutation重要度
        perm_importance = analyzer.analyze_permutation_importance(model, X_val, y_val)

        # 相関分析
        X_all = pd.concat([X_train, X_val])
        high_corr_pairs = analyzer.analyze_correlation(X_all)

        # 結果統合
        results_df = analyzer.integrate_results(
            lgbm_importance, perm_importance, high_corr_pairs
        )

        # 結果が正しく生成されていることを確認
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert "recommendation" in results_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
