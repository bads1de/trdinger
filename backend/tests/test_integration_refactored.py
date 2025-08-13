"""
リファクタリング後の統合テスト

既存のMLワークフローとリファクタリング後のコードが正常に統合されることを確認します。
"""

import pytest
import pandas as pd
import numpy as np
import logging

# 新しい関数ベースのアプローチ
from app.utils.index_alignment import (
    align_data,
    validate_alignment,
    get_index_statistics,
)

from app.utils.label_generation import (
    generate_labels_with_pipeline,
    SimpleLabelGenerator,
    create_label_pipeline,
)

# 既存のクラス（後方互換性確認）
# MLWorkflowIndexManager は削除されたため、テスト内で関数ベース実装を利用します
from app.utils.label_generation import LabelGenerator, ThresholdMethod

logger = logging.getLogger(__name__)


class TestBackwardCompatibility:
    """後方互換性テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        index = pd.date_range("2023-01-01", periods=100, freq="H")

        # 価格データ
        prices = pd.Series(
            np.cumsum(np.random.randn(100) * 0.01) + 100, index=index, name="close"
        )

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

    def test_existing_label_generator_still_works(self, sample_data):
        """既存のLabelGeneratorが正常に動作することを確認"""
        prices, _ = sample_data

        generator = LabelGenerator()

        # 既存のメソッドが動作することを確認
        labels, threshold_info = generator.generate_labels(
            prices, method=ThresholdMethod.STD_DEVIATION
        )

        assert isinstance(labels, pd.Series)
        assert isinstance(threshold_info, dict)
        assert len(labels) > 0
        assert "threshold_up" in threshold_info
        assert "threshold_down" in threshold_info

        logger.info("✓ 既存のLabelGeneratorが正常に動作")

    def test_existing_mlworkflow_manager_still_works(self, sample_data):
        """既存のMLWorkflowIndexManagerが正常に動作することを確認"""
        prices, features = sample_data

        # 非推奨警告が出るが動作することを確認
        manager = MLWorkflowIndexManager()
        manager.initialize_workflow(features)

        # 基本機能が動作することを確認
        assert manager.workflow_state["original_data"] is not None
        assert len(manager.workflow_state["original_data"]) == len(features)

        logger.info("✓ 既存のMLWorkflowIndexManagerが正常に動作")

    def test_new_vs_old_label_generation_consistency(self, sample_data):
        """新旧のラベル生成方法の一貫性を確認"""
        prices, _ = sample_data

        # 既存の方法
        old_generator = LabelGenerator()
        old_labels, old_info = old_generator.generate_labels(
            prices, method=ThresholdMethod.KBINS_DISCRETIZER, strategy="quantile"
        )

        # 新しい方法
        new_labels, new_info = generate_labels_with_pipeline(
            prices, strategy="quantile"
        )

        # 基本的な特性が一致することを確認
        assert len(old_labels) == len(new_labels)
        assert set(old_labels.unique()) == set(new_labels.unique())

        # 分布が類似していることを確認（完全一致は期待しない）
        old_dist = old_labels.value_counts(normalize=True).sort_index()
        new_dist = new_labels.value_counts(normalize=True).sort_index()

        # 各クラスの比率の差が20%以内であることを確認
        for label in [0, 1, 2]:
            old_ratio = old_dist.get(label, 0)
            new_ratio = new_dist.get(label, 0)
            diff = abs(old_ratio - new_ratio)
            assert diff < 0.2, f"ラベル{label}の分布差が大きすぎます: {diff:.3f}"

        logger.info("✓ 新旧のラベル生成方法の一貫性を確認")


class TestNewFeaturesIntegration:
    """新機能の統合テスト"""

    @pytest.fixture
    def ml_workflow_data(self):
        """MLワークフロー用のテストデータ"""
        index = pd.date_range("2023-01-01", periods=200, freq="H")

        # より現実的な価格データ（トレンドとノイズ）
        trend = np.linspace(100, 110, 200)
        noise = np.random.randn(200) * 2
        prices = pd.Series(trend + noise, index=index, name="close")

        # 特徴量データ（一部欠損あり）
        features = pd.DataFrame(
            {
                "sma_20": np.random.randn(200),
                "rsi": np.random.uniform(0, 100, 200),
                "volume": np.random.exponential(1000, 200),
                "volatility": np.random.exponential(0.02, 200),
            },
            index=index,
        )

        # 一部のデータを欠損させる
        features.iloc[50:55] = np.nan
        features.iloc[150:152] = np.nan

        return prices, features

    def test_complete_new_workflow(self, ml_workflow_data):
        """新しいワークフローの完全テスト"""
        prices, features = ml_workflow_data

        # 1. 新しい方法でラベル生成
        labels, label_info = generate_labels_with_pipeline(
            prices, strategy="quantile", n_bins=3
        )

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
        assert stats["alignment_ratio"] == 1.0

        # NaNが除去されていることを確認
        assert not aligned_features.isna().any().any()
        assert not aligned_labels.isna().any()

        # ラベルの分布が適切であることを確認
        label_dist = aligned_labels.value_counts(normalize=True)
        assert len(label_dist) == 3  # 3クラス
        assert all(ratio > 0.1 for ratio in label_dist.values)  # 各クラスが10%以上

        logger.info(
            f"✓ 新しいワークフロー完了: "
            f"{len(aligned_features)}行のデータが処理されました "
            f"(整合率: {stats['alignment_ratio']*100:.1f}%)"
        )

    def test_pipeline_flexibility(self, ml_workflow_data):
        """Pipelineの柔軟性テスト"""
        prices, _ = ml_workflow_data

        strategies = ["uniform", "quantile", "kmeans"]
        n_bins_options = [3, 4, 5]

        results = {}

        for strategy in strategies:
            for n_bins in n_bins_options:
                # Pipelineを作成
                pipeline = create_label_pipeline(n_bins=n_bins, strategy=strategy)

                # ラベル生成
                labels_array = pipeline.fit_transform(prices)

                # 結果を記録
                key = f"{strategy}_{n_bins}"
                results[key] = {
                    "n_unique": len(np.unique(labels_array)),
                    "shape": labels_array.shape,
                }

                # 基本的な検証
                assert results[key]["n_unique"] <= n_bins
                assert results[key]["shape"][0] > 0

        # 異なる設定で異なる結果が得られることを確認
        unique_results = set((r["n_unique"], r["shape"]) for r in results.values())
        assert len(unique_results) > 1, "異なる設定で同じ結果が得られています"

        logger.info(f"✓ Pipeline柔軟性テスト完了: {len(results)}通りの設定をテスト")

    def test_performance_comparison(self, ml_workflow_data):
        """パフォーマンス比較テスト"""
        prices, features = ml_workflow_data

        import time

        # 新しい方法のパフォーマンス測定
        start_time = time.time()

        # ラベル生成
        labels, _ = generate_labels_with_pipeline(prices, strategy="quantile")

        # インデックス整合
        aligned_features, aligned_labels = align_data(
            features, labels, method="intersection"
        )

        new_method_time = time.time() - start_time

        # 既存の方法のパフォーマンス測定
        start_time = time.time()

        # 既存のラベル生成
        old_generator = LabelGenerator()
        old_labels, _ = old_generator.generate_labels(
            prices, method=ThresholdMethod.KBINS_DISCRETIZER, strategy="quantile"
        )

        # 既存のインデックス整合（MLWorkflowIndexManagerを使用）
        manager = MLWorkflowIndexManager()
        manager.initialize_workflow(features)
        old_aligned_features, old_aligned_labels = manager.finalize_workflow(
            features, old_labels, alignment_method="intersection"
        )

        old_method_time = time.time() - start_time

        # 結果の比較
        assert len(aligned_features) == len(old_aligned_features)
        assert len(aligned_labels) == len(old_aligned_labels)

        # パフォーマンス情報をログ出力
        logger.info(
            f"✓ パフォーマンス比較: "
            f"新方法={new_method_time:.3f}s, "
            f"旧方法={old_method_time:.3f}s, "
            f"比率={new_method_time/old_method_time:.2f}"
        )

        # 新しい方法が極端に遅くないことを確認（3倍以内）
        assert new_method_time < old_method_time * 3, "新しい方法が極端に遅いです"


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_empty_data_handling(self):
        """空データのハンドリングテスト"""
        empty_series = pd.Series([], dtype=float)
        empty_df = pd.DataFrame()

        # 空データでのエラーハンドリングを確認
        try:
            result = generate_labels_with_pipeline(empty_series)
            # 空の結果が返される場合もある
            assert len(result[0]) == 0 or len(result) == 0
        except (ValueError, IndexError):
            # エラーが発生する場合もある
            pass

        try:
            result = align_data(empty_df, empty_series)
            # 空の結果が返される場合もある
            assert len(result[0]) == 0 and len(result[1]) == 0
        except (ValueError, IndexError):
            # エラーが発生する場合もある
            pass

    def test_mismatched_index_handling(self):
        """インデックス不一致のハンドリングテスト"""
        # 完全に異なるインデックス
        features = pd.DataFrame({"f1": [1, 2, 3]}, index=[1, 2, 3])

        labels = pd.Series([0, 1, 2], index=[4, 5, 6])

        # intersection方法では空の結果になることを確認
        aligned_features, aligned_labels = align_data(
            features, labels, method="intersection"
        )

        assert len(aligned_features) == 0
        assert len(aligned_labels) == 0


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
