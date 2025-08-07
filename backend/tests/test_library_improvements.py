"""
ライブラリ置き換え修正のテスト

3.1, 3.3, 3.9, 3.10の問題修正が正しく動作することを確認するテストファイル
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

# テスト対象のモジュールをインポート
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
from utils.label_generation import LabelGenerator, ThresholdMethod
from services.ml.feature_engineering.optimized_crypto_features import (
    OptimizedCryptoFeatures,
)
from utils.data_validation import DataValidator
from services.ml.models.knn_wrapper import KNNModel
from services.optimization.ensemble_parameter_space import EnsembleParameterSpace


class TestAdvancedFeatureEngineer:
    """AdvancedFeatureEngineerの修正テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.engineer = AdvancedFeatureEngineer()

        # テスト用のOHLCVデータを作成
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        np.random.seed(42)

        self.test_data = pd.DataFrame(
            {
                "Open": 50000 + np.random.randn(100) * 1000,
                "High": 50000 + np.random.randn(100) * 1000 + 500,
                "Low": 50000 + np.random.randn(100) * 1000 - 500,
                "Close": 50000 + np.random.randn(100) * 1000,
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        # Closeが正の値になるように調整
        self.test_data["Close"] = np.abs(self.test_data["Close"])
        self.test_data["High"] = np.maximum(
            self.test_data["High"], self.test_data["Close"]
        )
        self.test_data["Low"] = np.minimum(
            self.test_data["Low"], self.test_data["Close"]
        )

    def test_trend_strength_calculation(self):
        """トレンド強度計算の修正テスト"""
        # 時系列特徴量を追加（トレンド強度を含む）
        result = self.engineer._add_time_series_features(self.test_data.copy())

        # トレンド強度の列が存在することを確認
        trend_columns = [col for col in result.columns if "Trend_strength" in col]
        assert len(trend_columns) == 3  # window=[10, 20, 50]

        # 各トレンド強度列をチェック
        for col in trend_columns:
            # NaNでない値が存在することを確認
            non_nan_values = result[col].dropna()
            assert len(non_nan_values) > 0, f"{col}にNaNでない値が存在しません"

            # 値が数値であることを確認
            assert all(
                isinstance(val, (int, float)) for val in non_nan_values
            ), f"{col}に数値でない値が含まれています"

    def test_no_scipy_stats_import(self):
        """scipy.statsがインポートされていないことを確認"""
        import services.ml.feature_engineering.advanced_features as module

        # モジュールのソースコードを確認
        import inspect

        source = inspect.getsource(module)

        # scipy.statsのインポートがないことを確認
        assert (
            "from scipy import stats" not in source
        ), "scipy.statsのインポートが残っています"
        assert (
            "import scipy.stats" not in source
        ), "scipy.statsのインポートが残っています"

    def test_feature_engineering_performance(self):
        """特徴量エンジニアリングのパフォーマンステスト"""
        import time

        start_time = time.time()
        result = self.engineer.create_advanced_features(self.test_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # 実行時間が合理的であることを確認（10秒以内）
        assert execution_time < 10, f"実行時間が長すぎます: {execution_time:.2f}秒"

        # 結果が適切な形状であることを確認
        assert isinstance(result, pd.DataFrame), "結果がDataFrameではありません"
        assert len(result) == len(self.test_data), "行数が一致しません"
        assert len(result.columns) > len(
            self.test_data.columns
        ), "特徴量が追加されていません"


class TestLabelGenerator:
    """LabelGeneratorの修正テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.generator = LabelGenerator()

        # テスト用の価格データを作成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=1000, freq="H")

        # トレンドのある価格データを生成
        trend = np.linspace(50000, 55000, 1000)
        noise = np.random.randn(1000) * 500
        self.price_data = pd.Series(trend + noise, index=dates, name="Close")

    def test_kbins_discretizer_method(self):
        """KBinsDiscretizerメソッドのテスト"""
        # KBinsDiscretizerを使ったラベル生成
        labels, info = self.generator.generate_labels(
            self.price_data,
            method=ThresholdMethod.KBINS_DISCRETIZER,
            strategy="quantile",
        )

        # 基本的な検証
        assert isinstance(labels, pd.Series), "ラベルがSeriesではありません"
        assert isinstance(info, dict), "情報が辞書ではありません"

        # ラベルが0, 1, 2の値を持つことを確認
        unique_labels = set(labels.unique())
        expected_labels = {0, 1, 2}
        assert (
            unique_labels == expected_labels
        ), f"期待されるラベル{expected_labels}と異なります: {unique_labels}"

        # 情報辞書の内容を確認
        assert info["method"] == "kbins_discretizer", "メソッド名が正しくありません"
        assert "threshold_up" in info, "threshold_upが含まれていません"
        assert "threshold_down" in info, "threshold_downが含まれていません"
        assert "bin_edges" in info, "bin_edgesが含まれていません"
        assert "actual_distribution" in info, "actual_distributionが含まれていません"

    def test_kbins_discretizer_strategies(self):
        """異なる戦略でのKBinsDiscretizerテスト"""
        strategies = ["uniform", "quantile", "kmeans"]

        for strategy in strategies:
            labels, info = self.generator.generate_labels(
                self.price_data,
                method=ThresholdMethod.KBINS_DISCRETIZER,
                strategy=strategy,
            )

            # 各戦略で適切にラベルが生成されることを確認
            assert len(labels) > 0, f"{strategy}戦略でラベルが生成されませんでした"
            assert (
                info["strategy"] == strategy
            ), f"戦略が正しく設定されていません: {info['strategy']}"

            # 分布が合理的であることを確認（各クラスに最低5%のデータ）
            distribution = info["actual_distribution"]
            for class_name, ratio in distribution.items():
                assert (
                    ratio >= 0.05
                ), f"{strategy}戦略の{class_name}クラスの比率が低すぎます: {ratio}"

    def test_convenience_method(self):
        """便利メソッドのテスト"""
        labels, info = self.generator.generate_labels_with_kbins_discretizer(
            self.price_data, strategy="quantile"
        )

        # 基本的な動作確認
        assert isinstance(labels, pd.Series), "ラベルがSeriesではありません"
        assert info["method"] == "kbins_discretizer", "メソッドが正しくありません"
        assert info["strategy"] == "quantile", "戦略が正しくありません"

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 空のデータでテスト
        empty_data = pd.Series([], dtype=float)

        # エラーが適切に処理されることを確認
        try:
            labels, info = self.generator.generate_labels(
                empty_data, method=ThresholdMethod.KBINS_DISCRETIZER
            )
            # フォールバックが動作することを確認
            assert (
                info["method"] != "kbins_discretizer"
            ), "フォールバックが動作していません"
        except Exception as e:
            # 適切なエラーメッセージが含まれることを確認
            assert "有効な価格変化率データがありません" in str(
                e
            ) or "価格変化率" in str(e)

    def test_comparison_with_existing_methods(self):
        """既存メソッドとの比較テスト"""
        # 複数の方法でラベルを生成
        methods_to_test = [
            (ThresholdMethod.QUANTILE, {}),
            (ThresholdMethod.STD_DEVIATION, {"std_multiplier": 0.5}),
            (ThresholdMethod.KBINS_DISCRETIZER, {"strategy": "quantile"}),
        ]

        results = {}
        for method, params in methods_to_test:
            labels, info = self.generator.generate_labels(
                self.price_data, method=method, **params
            )
            results[method.value] = {
                "labels": labels,
                "info": info,
                "distribution": info.get("actual_distribution", {}),
            }

        # すべての方法で適切にラベルが生成されることを確認
        for method_name, result in results.items():
            assert (
                len(result["labels"]) > 0
            ), f"{method_name}でラベルが生成されませんでした"

            # 分布の合理性を確認
            if "distribution" in result and result["distribution"]:
                total_ratio = sum(result["distribution"].values())
                assert (
                    abs(total_ratio - 1.0) < 0.01
                ), f"{method_name}の分布の合計が1に近くありません: {total_ratio}"


class TestOptimizedCryptoFeatures:
    """OptimizedCryptoFeaturesの修正内容をテスト（3.9対応）"""

    def setup_method(self):
        """テストデータの準備"""
        self.feature_engine = OptimizedCryptoFeatures()

        # テスト用のOHLCVデータを作成
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        np.random.seed(42)

        self.test_data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(100) * 5,
                "High": 105 + np.random.randn(100) * 5,
                "Low": 95 + np.random.randn(100) * 5,
                "Close": 100 + np.random.randn(100) * 5,
                "Volume": 1000 + np.random.randn(100) * 100,
                "open_interest": 5000 + np.random.randn(100) * 500,
                "funding_rate": np.random.randn(100) * 0.001,
                "fear_greed_value": 50 + np.random.randn(100) * 20,
            },
            index=dates,
        )

    def test_create_optimized_features(self):
        """最適化された特徴量作成のテスト"""
        result = self.feature_engine.create_optimized_features(self.test_data)

        # 結果が正常に返されることを確認
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.test_data)

        # 新しい特徴量が追加されていることを確認
        assert len(result.columns) > len(self.test_data.columns)

        # 無限値やNaN値が適切に処理されていることを確認
        assert not result.isin([np.inf, -np.inf]).any().any()

    def test_robust_return_calculation(self):
        """ロバストな変動率計算のテスト（pandasの組み込み関数使用）"""
        result = self.feature_engine.create_optimized_features(self.test_data)

        # ロバストリターン特徴量が作成されていることを確認
        robust_return_cols = [col for col in result.columns if "robust_return" in col]
        assert len(robust_return_cols) > 0

        # 値が有限であることを確認
        for col in robust_return_cols:
            assert result[col].isfinite().all()


class TestDataValidator:
    """DataValidatorの修正内容をテスト（3.9対応）"""

    def test_safe_normalize_improved(self):
        """改善されたsafe_normalize関数のテスト"""
        # テストデータ作成
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 正規化実行
        normalized = DataValidator.safe_normalize(data, window=5)

        # 結果がSeriesであることを確認
        assert isinstance(normalized, pd.Series)

        # 無限値やNaN値が適切に処理されていることを確認
        assert normalized.isfinite().all()

        # 正規化が適切に行われていることを確認（最後の5つの値の平均が0に近い）
        last_5_mean = normalized.tail(5).mean()
        assert abs(last_5_mean) < 0.5  # 許容誤差内

    def test_safe_normalize_with_constant_values(self):
        """定数値での正規化テスト"""
        # 定数値のテストデータ
        data = pd.Series([5, 5, 5, 5, 5])

        # 正規化実行
        normalized = DataValidator.safe_normalize(data, window=3)

        # 結果が有限値であることを確認
        assert normalized.isfinite().all()

        # 定数値の場合、正規化結果は0に近いはず
        assert abs(normalized.mean()) < 0.1


class TestKNNModel:
    """KNNModelの修正内容をテスト（3.10対応）"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        self.X_test = pd.DataFrame(
            np.random.randn(20, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        self.y_test = pd.Series(np.random.randint(0, 2, 20))

    def test_knn_model_initialization(self):
        """KNNモデルの初期化テスト"""
        model = KNNModel(n_neighbors=5, metric="euclidean")

        # デフォルトパラメータにleaf_sizeが含まれていることを確認
        assert "leaf_size" in model.default_params
        assert model.default_params["leaf_size"] == 30

    def test_knn_model_training(self):
        """KNNモデルの学習テスト"""
        model = KNNModel(n_neighbors=5)

        # 学習実行
        model.fit(self.X_train, self.y_train)

        # 学習が完了していることを確認
        assert model.is_trained
        assert model.model is not None

        # 予測実行
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)


class TestEnsembleParameterSpace:
    """EnsembleParameterSpaceの修正内容をテスト（3.10対応）"""

    def test_knn_parameter_space_enhanced(self):
        """拡張されたKNNパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_knn_parameter_space()

        # 新しいmetricパラメータが追加されていることを確認
        assert "knn_metric" in param_space

        # metricの選択肢が適切に設定されていることを確認
        metric_categories = param_space["knn_metric"].categories
        expected_metrics = ["minkowski", "euclidean", "manhattan", "chebyshev"]
        assert all(metric in metric_categories for metric in expected_metrics)


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v"])
