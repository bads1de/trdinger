"""
リファクタリング後のユーティリティのテスト

index_alignment.pyとlabel_generation.pyのリファクタリング後の機能をテストします。
新しい関数ベースのアプローチとscikit-learnの標準機能を活用した実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# リファクタリング後のモジュールをインポート
from app.utils.index_alignment import (
    align_data,
    validate_alignment,
    preserve_index_during_processing,
    reindex_with_intersection,
    get_index_statistics,
)

from app.utils.label_generation import (
    PriceChangeTransformer,
    SimpleLabelGenerator,
    create_label_pipeline,
    generate_labels_with_pipeline,
    optimize_label_generation_with_gridsearch,
    LabelGenerator,  # 後方互換性テスト用
)

logger = logging.getLogger(__name__)


class TestIndexAlignmentRefactored:
    """リファクタリング後のindex_alignment.pyのテスト"""

    @pytest.fixture
    def sample_features(self):
        """サンプル特徴量データ"""
        index = pd.date_range("2023-01-01", periods=100, freq="H")
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            },
            index=index,
        )

    @pytest.fixture
    def sample_labels(self):
        """サンプルラベルデータ"""
        # 特徴量より少し短いインデックス
        index = pd.date_range("2023-01-01 02:00", periods=95, freq="H")
        return pd.Series(np.random.randint(0, 3, 95), index=index, name="labels")

    def test_align_data_intersection(self, sample_features, sample_labels):
        """intersection方法でのデータ整合テスト"""
        aligned_features, aligned_labels = align_data(
            sample_features, sample_labels, method="intersection"
        )

        # インデックスが一致することを確認
        assert aligned_features.index.equals(aligned_labels.index)

        # 共通インデックスのサイズが正しいことを確認
        expected_size = len(sample_features.index.intersection(sample_labels.index))
        assert len(aligned_features) == expected_size
        assert len(aligned_labels) == expected_size

    def test_align_data_features_priority(self, sample_features, sample_labels):
        """features_priority方法でのデータ整合テスト"""
        aligned_features, aligned_labels = align_data(
            sample_features, sample_labels, method="features_priority"
        )

        # インデックスが一致することを確認
        assert aligned_features.index.equals(aligned_labels.index)

        # NaNが除去されていることを確認
        assert not aligned_features.isna().any().any()
        assert not aligned_labels.isna().any()

    def test_align_data_labels_priority(self, sample_features, sample_labels):
        """labels_priority方法でのデータ整合テスト"""
        aligned_features, aligned_labels = align_data(
            sample_features, sample_labels, method="labels_priority"
        )

        # インデックスが一致することを確認
        assert aligned_features.index.equals(aligned_labels.index)

        # NaNが除去されていることを確認
        assert not aligned_features.isna().any().any()
        assert not aligned_labels.isna().any()

    def test_align_data_outer(self, sample_features, sample_labels):
        """outer方法でのデータ整合テスト"""
        aligned_features, aligned_labels = align_data(
            sample_features, sample_labels, method="outer"
        )

        # インデックスが一致することを確認
        assert aligned_features.index.equals(aligned_labels.index)

        # NaNが除去されていることを確認
        assert not aligned_features.isna().any().any()
        assert not aligned_labels.isna().any()

    def test_align_data_invalid_method(self, sample_features, sample_labels):
        """無効な方法でのエラーテスト"""
        with pytest.raises(ValueError, match="不正な整合方法"):
            align_data(sample_features, sample_labels, method="invalid")

    def test_validate_alignment(self, sample_features, sample_labels):
        """インデックス整合性検証テスト"""
        result = validate_alignment(sample_features, sample_labels)

        # 必要なキーが含まれていることを確認
        required_keys = [
            "is_valid",
            "alignment_ratio",
            "common_rows",
            "features_rows",
            "labels_rows",
            "missing_in_features",
            "missing_in_labels",
            "issues",
        ]
        for key in required_keys:
            assert key in result

        # 数値の妥当性を確認
        assert 0 <= result["alignment_ratio"] <= 1
        assert result["common_rows"] >= 0
        assert result["features_rows"] == len(sample_features)
        assert result["labels_rows"] == len(sample_labels)

    def test_preserve_index_during_processing(self, sample_features):
        """処理中のインデックス保持テスト"""

        def dummy_processing(data):
            # データを変更するが行数は保持
            return data * 2

        result = preserve_index_during_processing(sample_features, dummy_processing)

        # インデックスが保持されていることを確認
        assert result.index.equals(sample_features.index)

        # データが変更されていることを確認
        assert not result.equals(sample_features)

    def test_reindex_with_intersection(self, sample_features, sample_labels):
        """共通インデックスでの再インデックステスト"""
        result = reindex_with_intersection(sample_features, sample_labels.index)

        # 共通インデックスのみが残っていることを確認
        expected_index = sample_features.index.intersection(sample_labels.index)
        assert result.index.equals(expected_index)

    def test_get_index_statistics(self, sample_features, sample_labels):
        """インデックス統計情報取得テスト"""
        stats = get_index_statistics(sample_features, sample_labels)

        # 必要なキーが含まれていることを確認
        required_keys = [
            "features_count",
            "labels_count",
            "common_count",
            "features_only_count",
            "labels_only_count",
            "alignment_ratio",
            "coverage_in_features",
            "coverage_in_labels",
        ]
        for key in required_keys:
            assert key in stats

        # 数値の妥当性を確認
        assert stats["features_count"] == len(sample_features)
        assert stats["labels_count"] == len(sample_labels)
        assert 0 <= stats["alignment_ratio"] <= 1
        assert 0 <= stats["coverage_in_features"] <= 1
        assert 0 <= stats["coverage_in_labels"] <= 1

    def test_backward_compatibility_mlworkflow_manager(self, sample_features):
        """後方互換性テスト - 関数ベースの互換性確認"""
        # MLWorkflowIndexManager は削除されたため、関数ベース実装で同等の動作を確認
        aligned_features, aligned_labels = align_data(
            sample_features, pd.Series([0]*len(sample_features), index=sample_features.index)
        )
        # align_data が正常に動作することを確認
        assert aligned_features.index.equals(aligned_labels.index)


class TestLabelGenerationRefactored:
    """リファクタリング後のlabel_generation.pyのテスト"""

    @pytest.fixture
    def sample_price_data(self):
        """サンプル価格データ"""
        index = pd.date_range("2023-01-01", periods=100, freq="H")
        # トレンドのある価格データを生成
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        prices = trend + noise
        return pd.Series(prices, index=index, name="close")

    def test_price_change_transformer(self, sample_price_data):
        """PriceChangeTransformerのテスト"""
        transformer = PriceChangeTransformer(periods=1)

        # フィット（何もしない）
        transformer.fit(sample_price_data)

        # 変換
        result = transformer.transform(sample_price_data)

        # 2次元配列として返されることを確認
        assert result.ndim == 2
        assert result.shape[1] == 1

        # 価格変化率が正しく計算されていることを確認
        expected_length = len(sample_price_data.pct_change().dropna())
        assert len(result) == expected_length

    def test_simple_label_generator_basic(self, sample_price_data):
        """SimpleLabelGeneratorの基本テスト"""
        generator = SimpleLabelGenerator(strategy="quantile")

        # フィット
        generator.fit(sample_price_data)
        assert generator._is_fitted

        # 変換
        labels = generator.transform(sample_price_data)

        # ラベルが正しく生成されていることを確認
        assert isinstance(labels, pd.Series)
        assert labels.dtype == int
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_simple_label_generator_fit_transform(self, sample_price_data):
        """SimpleLabelGeneratorのfit_transformテスト"""
        generator = SimpleLabelGenerator(strategy="uniform")

        labels, info = generator.fit_transform(sample_price_data)

        # ラベルが正しく生成されていることを確認
        assert isinstance(labels, pd.Series)
        assert labels.dtype == int

        # 情報辞書が正しく生成されていることを確認
        required_keys = [
            "method",
            "strategy",
            "n_bins",
            "down_count",
            "range_count",
            "up_count",
            "down_ratio",
            "range_ratio",
            "up_ratio",
            "total_count",
        ]
        for key in required_keys:
            assert key in info

        # 比率の合計が1になることを確認
        total_ratio = info["down_ratio"] + info["range_ratio"] + info["up_ratio"]
        assert abs(total_ratio - 1.0) < 1e-10

    def test_create_label_pipeline(self):
        """create_label_pipelineのテスト"""
        pipeline = create_label_pipeline(n_bins=3, strategy="quantile")

        # Pipelineが正しく作成されていることを確認
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "price_change"
        assert pipeline.steps[1][0] == "discretizer"

        # KBinsDiscretizerのパラメータが正しく設定されていることを確認
        discretizer = pipeline.steps[1][1]
        assert discretizer.n_bins == 3
        assert discretizer.strategy == "quantile"

    def test_generate_labels_with_pipeline(self, sample_price_data):
        """generate_labels_with_pipelineのテスト"""
        labels, info = generate_labels_with_pipeline(
            sample_price_data, strategy="quantile", n_bins=3
        )

        # ラベルが正しく生成されていることを確認
        assert isinstance(labels, pd.Series)
        assert labels.dtype == int
        assert set(labels.unique()).issubset({0, 1, 2})

        # 情報辞書が正しく生成されていることを確認
        assert info["method"] == "pipeline_kbins_discretizer"
        assert info["strategy"] == "quantile"
        assert info["n_bins"] == 3

        # 閾値情報が含まれていることを確認
        assert "threshold_down" in info
        assert "threshold_up" in info
        assert "bin_edges" in info

    @pytest.mark.slow
    def test_optimize_label_generation_with_gridsearch(self, sample_price_data):
        """optimize_label_generation_with_gridsearchのテスト（時間がかかるためスロー）"""
        # 小さなパラメータグリッドでテスト
        param_grid = {
            "discretizer__strategy": ["quantile", "uniform"],
            "discretizer__n_bins": [3, 4],
        }

        best_pipeline, info = optimize_label_generation_with_gridsearch(
            sample_price_data, param_grid=param_grid, cv=2
        )

        # 最適化結果が正しく返されていることを確認
        assert "best_params" in info
        assert "best_score" in info
        assert "labels" in info

        # 最適なパイプラインが返されていることを確認
        assert hasattr(best_pipeline, "transform")

        # ラベルが正しく生成されていることを確認
        labels = info["labels"]
        assert isinstance(labels, pd.Series)
        assert labels.dtype == int

    def test_backward_compatibility_label_generator(self, sample_price_data):
        """後方互換性テスト - LabelGenerator"""
        from app.utils.label_generation import ThresholdMethod

        generator = LabelGenerator()

        # 既存のメソッドが動作することを確認
        labels, threshold_info = generator.generate_labels(
            sample_price_data, method=ThresholdMethod.KBINS_DISCRETIZER
        )

        # ラベルが正しく生成されていることを確認
        assert isinstance(labels, pd.Series)
        assert isinstance(threshold_info, dict)


class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def sample_ml_data(self):
        """MLワークフロー用のサンプルデータ"""
        index = pd.date_range("2023-01-01", periods=100, freq="H")

        # 価格データ
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        prices = pd.Series(trend + noise, index=index, name="close")

        # 特徴量データ
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            },
            index=index,
        )

        return prices, features

    def test_complete_ml_workflow(self, sample_ml_data):
        """完全なMLワークフローの統合テスト"""
        prices, features = sample_ml_data

        # 1. ラベル生成
        labels, label_info = generate_labels_with_pipeline(prices, strategy="quantile")

        # 2. インデックス整合
        aligned_features, aligned_labels = align_data(
            features, labels, method="intersection"
        )

        # 3. 整合性検証
        validation_result = validate_alignment(aligned_features, aligned_labels)

        # 4. 統計情報取得
        stats = get_index_statistics(aligned_features, aligned_labels)

        # 結果の検証
        assert len(aligned_features) == len(aligned_labels)
        assert aligned_features.index.equals(aligned_labels.index)
        assert validation_result["is_valid"]
        assert stats["alignment_ratio"] == 1.0  # 完全に整合されている

        logger.info(
            f"統合テスト完了: {len(aligned_features)}行のデータが正常に処理されました"
        )

    def test_pipeline_with_different_strategies(self, sample_ml_data):
        """異なる戦略でのPipelineテスト"""
        prices, _ = sample_ml_data

        strategies = ["uniform", "quantile", "kmeans"]
        results = {}

        for strategy in strategies:
            labels, info = generate_labels_with_pipeline(
                prices, strategy=strategy, n_bins=3
            )
            results[strategy] = {
                "labels": labels,
                "info": info,
                "distribution": [
                    info["down_ratio"],
                    info["range_ratio"],
                    info["up_ratio"],
                ],
            }

        # 各戦略で異なる結果が得られることを確認
        for strategy in strategies:
            assert len(results[strategy]["labels"]) > 0
            assert results[strategy]["info"]["strategy"] == strategy

            # 分布の合計が1になることを確認
            total_ratio = sum(results[strategy]["distribution"])
            assert abs(total_ratio - 1.0) < 1e-10

        logger.info("異なる戦略でのテストが完了しました")


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
