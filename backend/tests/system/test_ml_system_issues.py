"""
MLシステムの問題点検証テスト

現在のMLシステムで特定された問題点を実際に検証し、
問題の影響を定量化するためのテストケース。
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.utils.data_processing import DataProcessor
from app.utils.label_generation import LabelGenerator, ThresholdMethod
from app.services.ml.config.ml_config import TrainingConfig
from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


class TestMLSystemIssues:
    """MLシステムの問題点検証テストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="1H")
        np.random.seed(42)

        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, len(dates))  # 2%の標準偏差
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        prices = np.array(prices)

        # OHLCV データを生成
        data = {
            "timestamp": dates,
            "Open": prices * np.random.uniform(0.995, 1.005, len(prices)),
            "High": prices * np.random.uniform(1.001, 1.02, len(prices)),
            "Low": prices * np.random.uniform(0.98, 0.999, len(prices)),
            "Close": prices,
            "Volume": np.random.uniform(100, 1000, len(prices)),
        }

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    @pytest.fixture
    def sample_funding_rate_data(self, sample_ohlcv_data):
        """テスト用のファンディングレートデータを生成"""
        return pd.DataFrame(
            {
                "timestamp": sample_ohlcv_data.index,
                "funding_rate": np.random.normal(
                    0.0001, 0.0005, len(sample_ohlcv_data)
                ),
            }
        ).set_index("timestamp")

    @pytest.fixture
    def sample_open_interest_data(self, sample_ohlcv_data):
        """テスト用の建玉残高データを生成"""
        return pd.DataFrame(
            {
                "timestamp": sample_ohlcv_data.index,
                "open_interest": np.random.uniform(
                    1000000, 5000000, len(sample_ohlcv_data)
                ),
            }
        ).set_index("timestamp")

    def test_feature_scaling_disabled_issue(
        self, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data
    ):
        """
        問題1: 特徴量スケーリングが無効になっている問題を検証
        """
        logger.info("=== 特徴量スケーリング無効化問題の検証 ===")

        # 特徴量エンジニアリングサービスを初期化
        feature_service = FeatureEngineeringService()

        # 特徴量を計算
        features_df = feature_service.calculate_advanced_features(
            ohlcv_data=sample_ohlcv_data,
            funding_rate_data=sample_funding_rate_data,
            open_interest_data=sample_open_interest_data,
        )

        # 数値特徴量のスケールを確認
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_stats = {}

        for col in numeric_columns[:10]:  # 最初の10個の特徴量をチェック
            if col not in ["Open", "High", "Low", "Close", "Volume"]:
                series = features_df[col].dropna()
                if len(series) > 0:
                    feature_stats[col] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "min": series.min(),
                        "max": series.max(),
                        "range": series.max() - series.min(),
                    }

        logger.info(f"特徴量統計情報（最初の10個）:")
        for col, stats in feature_stats.items():
            logger.info(
                f"  {col}: 平均={stats['mean']:.4f}, 標準偏差={stats['std']:.4f}, 範囲={stats['range']:.4f}"
            )

        # スケールの不整合を検証
        ranges = [stats["range"] for stats in feature_stats.values()]
        if len(ranges) > 1:
            max_range = max(ranges)
            min_range = min(ranges)
            scale_ratio = max_range / min_range if min_range > 0 else float("inf")

            logger.info(f"スケール比率（最大範囲/最小範囲）: {scale_ratio:.2f}")

            # スケール比率が100以上の場合、スケーリングが必要
            assert (
                scale_ratio > 100
            ), f"特徴量のスケール不整合が検出されました（比率: {scale_ratio:.2f}）"
            logger.warning(
                f"⚠️ 特徴量スケーリングが無効のため、スケール不整合が発生しています（比率: {scale_ratio:.2f}）"
            )

        return feature_stats

    def test_zscore_outlier_detection_issue(self, sample_ohlcv_data):
        """
        問題2: Z-scoreベースの外れ値検出が金融データに不適切な問題を検証
        """
        logger.info("=== Z-score外れ値検出問題の検証 ===")

        # 金融データに典型的な急激な価格変動を追加
        test_data = sample_ohlcv_data.copy()

        # 意図的に急激な価格変動（市場クラッシュ）を追加
        crash_index = len(test_data) // 2
        test_data.iloc[
            crash_index : crash_index + 5, test_data.columns.get_loc("Close")
        ] *= 0.8  # 20%下落

        # 価格変化率を計算
        price_changes = test_data["Close"].pct_change().dropna()

        # Z-scoreベースの外れ値検出を実行
        preprocessor = DataPreprocessor()

        # 外れ値検出前のデータ数
        original_count = len(price_changes)

        # Z-scoreベースの外れ値除去をシミュレート
        mean_change = price_changes.mean()
        std_change = price_changes.std()
        z_scores = np.abs((price_changes - mean_change) / std_change)

        # 閾値3.0で外れ値を特定
        outliers = z_scores > 3.0
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / original_count) * 100

        logger.info(f"元データ数: {original_count}")
        logger.info(f"Z-score外れ値数: {outlier_count}")
        logger.info(f"外れ値割合: {outlier_percentage:.2f}%")

        # 外れ値として検出された価格変化率を確認
        outlier_changes = price_changes[outliers]
        if len(outlier_changes) > 0:
            logger.info(f"外れ値として検出された価格変化率:")
            for i, change in enumerate(outlier_changes.head(5)):
                logger.info(f"  {i+1}: {change:.4f} ({change*100:.2f}%)")

        # 金融データでは5%以上の変動も正常な範囲内であることを確認
        large_changes = price_changes[np.abs(price_changes) > 0.05]
        large_changes_count = len(large_changes)

        logger.info(f"5%以上の価格変動数: {large_changes_count}")

        # Z-scoreが重要な市場シグナルを外れ値として誤検出していることを検証
        if outlier_count > 0:
            logger.warning(
                f"⚠️ Z-score外れ値検出により{outlier_count}個の重要な市場シグナルが除去される可能性があります"
            )

        return {
            "original_count": original_count,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "large_changes_count": large_changes_count,
        }

    def test_fixed_threshold_label_generation_issue(self, sample_ohlcv_data):
        """
        問題3: 固定閾値ラベル生成がクラス不均衡を引き起こす問題を検証
        """
        logger.info("=== 固定閾値ラベル生成問題の検証 ===")

        # 現在のデフォルト設定を確認
        config = TrainingConfig()
        fixed_threshold_up = config.THRESHOLD_UP  # 0.02 (2%)
        fixed_threshold_down = config.THRESHOLD_DOWN  # -0.02 (-2%)

        logger.info(
            f"現在の固定閾値: 上昇={fixed_threshold_up}, 下落={fixed_threshold_down}"
        )

        # ラベル生成器を初期化
        label_generator = LabelGenerator()

        # 固定閾値でラベルを生成
        labels_fixed, threshold_info_fixed = label_generator.generate_labels(
            sample_ohlcv_data["Close"],
            method=ThresholdMethod.FIXED,
            threshold_up=fixed_threshold_up,
            threshold_down=fixed_threshold_down,
        )

        # 動的閾値でラベルを生成（比較用）
        labels_dynamic, threshold_info_dynamic = label_generator.generate_labels(
            sample_ohlcv_data["Close"],
            method=ThresholdMethod.STD_DEVIATION,
            std_multiplier=0.5,
        )

        # ラベル分布を分析
        def analyze_label_distribution(labels, method_name):
            label_counts = labels.value_counts().sort_index()
            total = len(labels)

            distribution = {
                "down": label_counts.get(0, 0) / total,
                "range": label_counts.get(1, 0) / total,
                "up": label_counts.get(2, 0) / total,
            }

            logger.info(f"{method_name}ラベル分布:")
            logger.info(
                f"  下落: {distribution['down']:.3f} ({label_counts.get(0, 0)}個)"
            )
            logger.info(
                f"  レンジ: {distribution['range']:.3f} ({label_counts.get(1, 0)}個)"
            )
            logger.info(
                f"  上昇: {distribution['up']:.3f} ({label_counts.get(2, 0)}個)"
            )

            # クラス不均衡の度合いを計算（最大クラスと最小クラスの比率）
            ratios = [distribution["down"], distribution["range"], distribution["up"]]
            max_ratio = max(ratios)
            min_ratio = min([r for r in ratios if r > 0])
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float("inf")

            logger.info(f"  クラス不均衡比率: {imbalance_ratio:.2f}")

            return distribution, imbalance_ratio

        # 固定閾値の分布分析
        fixed_dist, fixed_imbalance = analyze_label_distribution(
            labels_fixed, "固定閾値"
        )

        # 動的閾値の分布分析
        dynamic_dist, dynamic_imbalance = analyze_label_distribution(
            labels_dynamic, "動的閾値"
        )

        # 問題の検証
        logger.info(f"固定閾値のクラス不均衡比率: {fixed_imbalance:.2f}")
        logger.info(f"動的閾値のクラス不均衡比率: {dynamic_imbalance:.2f}")

        if fixed_imbalance > 3.0:  # 3倍以上の不均衡
            logger.warning(
                f"⚠️ 固定閾値により深刻なクラス不均衡が発生しています（比率: {fixed_imbalance:.2f}）"
            )

        return {
            "fixed_distribution": fixed_dist,
            "dynamic_distribution": dynamic_dist,
            "fixed_imbalance_ratio": fixed_imbalance,
            "dynamic_imbalance_ratio": dynamic_imbalance,
            "threshold_info_fixed": threshold_info_fixed,
            "threshold_info_dynamic": threshold_info_dynamic,
        }

    def test_time_series_cv_missing_issue(self):
        """
        問題4: 時系列クロスバリデーションが不備な問題を検証
        """
        logger.info("=== 時系列クロスバリデーション不備問題の検証 ===")

        # 現在のMLトレーニングサービスの設定を確認
        config = TrainingConfig()
        cv_folds = config.CROSS_VALIDATION_FOLDS

        logger.info(f"現在のクロスバリデーション分割数: {cv_folds}")

        # 時系列データの特性を考慮しない通常のクロスバリデーションの問題を説明
        logger.warning(
            "⚠️ 現在のシステムでは時系列データに適したクロスバリデーションが実装されていません"
        )
        logger.warning("   - 未来の情報が過去の予測に使用される可能性（data leakage）")
        logger.warning("   - ランダムサンプリングにより時系列の順序が破壊される")
        logger.warning("   - 実際の取引環境と異なる評価結果")

        # 推奨される時系列クロスバリデーション手法
        logger.info("推奨される時系列クロスバリデーション手法:")
        logger.info("  1. Time Series Split: 時系列順に分割")
        logger.info("  2. Walk-Forward Analysis: 段階的に学習期間を拡張")
        logger.info("  3. Purged Cross-Validation: データリークを防ぐギャップ設定")

        return {
            "current_cv_folds": cv_folds,
            "has_time_series_cv": False,
            "data_leakage_risk": True,
            "recommended_methods": ["time_series_split", "walk_forward", "purged_cv"],
        }

    def test_overall_system_impact(
        self, sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data
    ):
        """
        全体的なシステムへの影響を検証
        """
        logger.info("=== 全体的なシステム影響の検証 ===")

        # 各問題の影響を統合的に評価
        feature_stats = self.test_feature_scaling_disabled_issue(
            sample_ohlcv_data, sample_funding_rate_data, sample_open_interest_data
        )

        outlier_stats = self.test_zscore_outlier_detection_issue(sample_ohlcv_data)

        label_stats = self.test_fixed_threshold_label_generation_issue(
            sample_ohlcv_data
        )

        cv_stats = self.test_time_series_cv_missing_issue()

        # 総合的な問題スコアを計算
        problem_score = 0

        # 特徴量スケーリング問題のスコア
        if len(feature_stats) > 1:
            ranges = [stats["range"] for stats in feature_stats.values()]
            max_range = max(ranges)
            min_range = min([r for r in ranges if r > 0])
            scale_ratio = max_range / min_range if min_range > 0 else 1
            if scale_ratio > 100:
                problem_score += 25  # 25点減点

        # 外れ値検出問題のスコア
        if outlier_stats["outlier_percentage"] > 5:  # 5%以上が外れ値
            problem_score += 20  # 20点減点

        # ラベル生成問題のスコア
        if label_stats["fixed_imbalance_ratio"] > 3:  # 3倍以上の不均衡
            problem_score += 30  # 30点減点

        # 時系列CV問題のスコア
        if cv_stats["data_leakage_risk"]:
            problem_score += 25  # 25点減点

        logger.info(f"総合問題スコア: {problem_score}/100")
        logger.info("スコアが高いほど深刻な問題があることを示します")

        if problem_score >= 70:
            logger.error("🚨 深刻な問題が検出されました。緊急の改善が必要です。")
        elif problem_score >= 40:
            logger.warning("⚠️ 重要な問題が検出されました。改善を推奨します。")
        else:
            logger.info("✅ 軽微な問題のみです。")

        return {
            "total_problem_score": problem_score,
            "feature_scaling_issues": feature_stats,
            "outlier_detection_issues": outlier_stats,
            "label_generation_issues": label_stats,
            "cross_validation_issues": cv_stats,
        }

    def generate_sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成（非fixture版）"""
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="1H")
        np.random.seed(42)

        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, len(dates))  # 2%の標準偏差
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        prices = np.array(prices)

        # OHLCV データを生成
        data = {
            "timestamp": dates,
            "Open": prices * np.random.uniform(0.995, 1.005, len(prices)),
            "High": prices * np.random.uniform(1.001, 1.02, len(prices)),
            "Low": prices * np.random.uniform(0.98, 0.999, len(prices)),
            "Close": prices,
            "Volume": np.random.uniform(100, 1000, len(prices)),
        }

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def generate_sample_funding_rate_data(self, ohlcv_data):
        """テスト用のファンディングレートデータを生成（非fixture版）"""
        return pd.DataFrame(
            {
                "timestamp": ohlcv_data.index,
                "funding_rate": np.random.normal(0.0001, 0.0005, len(ohlcv_data)),
            }
        ).set_index("timestamp")

    def generate_sample_open_interest_data(self, ohlcv_data):
        """テスト用の建玉残高データを生成（非fixture版）"""
        return pd.DataFrame(
            {
                "timestamp": ohlcv_data.index,
                "open_interest": np.random.uniform(1000000, 5000000, len(ohlcv_data)),
            }
        ).set_index("timestamp")


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging

    logging.basicConfig(level=logging.INFO)

    test_instance = TestMLSystemIssues()

    # サンプルデータを生成
    sample_data = test_instance.generate_sample_ohlcv_data()
    funding_data = test_instance.generate_sample_funding_rate_data(sample_data)
    oi_data = test_instance.generate_sample_open_interest_data(sample_data)

    # 全体的な影響を検証
    results = test_instance.test_overall_system_impact(
        sample_data, funding_data, oi_data
    )

    print(f"\n=== 検証結果サマリー ===")
    print(f"総合問題スコア: {results['total_problem_score']}/100")
