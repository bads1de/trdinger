"""
Pipeline Optimization Tests

optimize_full_pipeline メソッドのユニットテスト。
CASH（Combined Algorithm Selection and Hyperparameter optimization）の動作確認。
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app.services.ml.optimization.optimization_service import OptimizationService


class TestOptimizeFullPipeline:
    """optimize_full_pipeline メソッドのテスト"""

    @pytest.fixture
    def opt_service(self):
        """OptimizationService インスタンス"""
        return OptimizationService()

    @pytest.fixture
    def sample_superset(self):
        """テスト用スーパーセットDataFrame（簡易版）"""
        np.random.seed(42)
        n = 200  # 高速化のため削減 (500 -> 200)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")

        data = {
            "open": 50000 + np.cumsum(np.random.randn(n) * 50),
            "high": 50100 + np.cumsum(np.random.randn(n) * 50),
            "low": 49900 + np.cumsum(np.random.randn(n) * 50),
            "close": 50000 + np.cumsum(np.random.randn(n) * 50),
            "volume": np.abs(np.random.randn(n) * 1000) + 100,
            "RSI_14": 50 + np.random.randn(n) * 10,
            "SMA_20": 50000 + np.cumsum(np.random.randn(n) * 30),
            "MACD": np.random.randn(n) * 100,
            # FracDiff スーパーセット（複数d値）
            "FracDiff_Price_d0.3": np.random.randn(n) * 0.1,
            "FracDiff_Price_d0.4": np.random.randn(n) * 0.1,
            "FracDiff_Price_d0.5": np.random.randn(n) * 0.1,
            "FracDiff_Price_d0.6": np.random.randn(n) * 0.1,
        }

        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_labels(self, sample_superset):
        """テスト用ラベル"""
        np.random.seed(123)
        n = len(sample_superset)
        # バイナリラベル（0 or 1）
        return pd.Series(
            np.random.randint(0, 2, n),
            index=sample_superset.index,
        )

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """重い計算を行うコンポーネントをモック化"""
        with (
            patch(
                "app.services.ml.feature_selection.feature_selector.FeatureSelector"
            ) as mock_selector_cls,
            patch("lightgbm.LGBMClassifier") as mock_lgbm_cls,
            patch(
                "app.services.ml.label_generation.presets.triple_barrier_method_preset"
            ) as mock_tbm,
        ):

            # FeatureSelector モック
            mock_selector = MagicMock()
            # 最初の5カラムを返すようにモック
            mock_selector.fit_transform.side_effect = lambda X, y: X.iloc[:, :5]
            mock_selector.transform.side_effect = lambda X: X.iloc[:, :5]
            mock_selector_cls.return_value = mock_selector

            # LGBMClassifier モック
            mock_lgbm = MagicMock()
            mock_lgbm.fit.return_value = None
            # 予測結果はランダムな0/1
            mock_lgbm.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
            mock_lgbm_cls.return_value = mock_lgbm

            # Triple Barrier Method モック (結果が返るようにする)
            def side_effect_tbm(df, **kwargs):
                return pd.Series(np.random.randint(0, 2, len(df)), index=df.index)

            mock_tbm.side_effect = side_effect_tbm

            yield

    def test_optimize_returns_required_keys(
        self, opt_service, sample_superset, sample_labels
    ):
        """最適化結果に必要なキーが含まれることを確認"""
        # 少ない試行回数で高速テスト
        result = opt_service.optimize_full_pipeline(
            ohlcv_data=sample_superset,
            feature_superset=sample_superset,
            labels=sample_labels,
            n_trials=2,  # 最小限の試行
            test_ratio=0.2,
        )

        # 必須キーの確認
        assert "best_params" in result
        assert "best_score" in result
        assert "test_score" in result
        assert "baseline_score" in result
        assert "improvement" in result
        assert "total_evaluations" in result
        assert "optimization_time" in result
        assert "n_selected_features" in result

    def test_best_params_in_search_space(
        self, opt_service, sample_superset, sample_labels
    ):
        """ベストパラメータが探索空間内にあることを確認"""
        result = opt_service.optimize_full_pipeline(
            ohlcv_data=sample_superset,
            feature_superset=sample_superset,
            labels=sample_labels,
            n_trials=2,
            frac_diff_d_values=[0.3, 0.5],
        )

        best_params = result["best_params"]

        # frac_diff_d は指定した値のいずれか
        assert best_params["frac_diff_d"] in [0.3, 0.5]

        # selection_method は有効な値
        assert best_params["selection_method"] in ["staged", "rfecv", "mutual_info"]

        # correlation_threshold は範囲内
        assert 0.85 <= best_params["correlation_threshold"] <= 0.99

        # min_features は範囲内
        assert 5 <= best_params["min_features"] <= 30

        # learning_rate は範囲内
        assert 0.005 <= best_params["learning_rate"] <= 0.1

        # num_leaves は範囲内
        assert 16 <= best_params["num_leaves"] <= 128

    def test_scores_are_valid(self, opt_service, sample_superset, sample_labels):
        """スコアが有効な範囲（0-1）であることを確認"""
        result = opt_service.optimize_full_pipeline(
            ohlcv_data=sample_superset,
            feature_superset=sample_superset,
            labels=sample_labels,
            n_trials=2,
        )

        # F1スコアは0から1の範囲
        assert 0.0 <= result["best_score"] <= 1.0
        assert 0.0 <= result["test_score"] <= 1.0
        assert 0.0 <= result["baseline_score"] <= 1.0


class TestGetPipelineParameterSpace:
    """_get_pipeline_parameter_space メソッドのテスト"""

    def test_returns_all_parameters(self):
        """全パラメータが定義されていることを確認"""
        service = OptimizationService()
        d_values = [0.3, 0.4, 0.5]

        space = service._get_pipeline_parameter_space(d_values)

        # 必須パラメータの存在確認
        assert "frac_diff_d" in space
        assert "selection_method" in space
        assert "correlation_threshold" in space
        assert "min_features" in space
        assert "learning_rate" in space
        assert "num_leaves" in space

    def test_frac_diff_d_uses_provided_values(self):
        """frac_diff_d が指定されたd値を使用することを確認"""
        service = OptimizationService()
        d_values = [0.25, 0.45, 0.65]

        space = service._get_pipeline_parameter_space(d_values)

        assert space["frac_diff_d"].type == "categorical"
        assert space["frac_diff_d"].categories == d_values

    def test_parameter_types_are_correct(self):
        """各パラメータのタイプが正しいことを確認"""
        service = OptimizationService()
        space = service._get_pipeline_parameter_space([0.4])

        assert space["frac_diff_d"].type == "categorical"
        assert space["selection_method"].type == "categorical"
        assert space["correlation_threshold"].type == "real"
        assert space["min_features"].type == "integer"
        assert space["learning_rate"].type == "real"
        assert space["num_leaves"].type == "integer"


class TestEvaluateBaseline:
    """_evaluate_baseline メソッドのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n_train, n_test = 100, 30

        def create_df(n, seed_offset=0):
            np.random.seed(42 + seed_offset)
            return pd.DataFrame(
                {
                    "close": 50000 + np.cumsum(np.random.randn(n) * 50),
                    "RSI_14": 50 + np.random.randn(n) * 10,
                    "FracDiff_Price_d0.4": np.random.randn(n) * 0.1,
                }
            )

        X_train = create_df(n_train, 0)
        X_test = create_df(n_test, 100)
        y_train = pd.Series(np.random.randint(0, 2, n_train))
        y_test = pd.Series(np.random.randint(0, 2, n_test))

        return X_train, y_train, X_test, y_test

    @pytest.fixture(autouse=True)
    def mock_dependencies_baseline(self):
        """_evaluate_baseline 用のモック"""
        with (
            patch(
                "app.services.ml.feature_selection.feature_selector.FeatureSelector"
            ) as mock_selector_cls,
            patch("lightgbm.LGBMClassifier") as mock_lgbm_cls,
        ):

            # FeatureSelector モック
            mock_selector = MagicMock()
            mock_selector.fit_transform.side_effect = lambda X, y: X.iloc[
                :, : min(5, X.shape[1])
            ]
            mock_selector.transform.side_effect = lambda X: X.iloc[
                :, : min(5, X.shape[1])
            ]
            mock_selector_cls.return_value = mock_selector

            # LGBMClassifier モック
            mock_lgbm = MagicMock()
            mock_lgbm.fit.return_value = None
            mock_lgbm.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
            mock_lgbm_cls.return_value = mock_lgbm

            yield

    def test_baseline_returns_valid_score(self, sample_data):
        """ベースライン評価が有効なスコアを返すことを確認"""
        X_train, y_train, X_test, y_test = sample_data
        service = OptimizationService()

        score = service._evaluate_baseline(X_train, y_train, X_test, y_test)

        # F1スコア（精度など）は0から1の範囲
        assert 0.0 <= score <= 1.0
