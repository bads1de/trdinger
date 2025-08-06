"""
改善された特徴量スケーリングのテスト

ロバストスケーリングの導入により
特徴量のスケール不整合問題が解決されるかを検証する。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)

logger = logging.getLogger(__name__)


class TestImprovedFeatureScaling:
    """改善された特徴量スケーリングのテストクラス"""

    def generate_sample_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
        np.random.seed(42)

        # 現実的な価格データを生成
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, len(dates))
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
        """テスト用のファンディングレートデータを生成"""
        return pd.DataFrame(
            {
                "timestamp": ohlcv_data.index,
                "funding_rate": np.random.normal(0.0001, 0.0005, len(ohlcv_data)),
            }
        ).set_index("timestamp")

    def generate_sample_open_interest_data(self, ohlcv_data):
        """テスト用の建玉残高データを生成"""
        return pd.DataFrame(
            {
                "timestamp": ohlcv_data.index,
                "open_interest": np.random.uniform(1000000, 5000000, len(ohlcv_data)),
            }
        ).set_index("timestamp")

    def test_feature_scaling_improvement(self):
        """特徴量スケーリングの改善をテスト"""
        logger.info("=== 特徴量スケーリング改善テスト ===")

        # サンプルデータを生成
        sample_data = self.generate_sample_data()
        funding_data = self.generate_sample_funding_rate_data(sample_data)
        oi_data = self.generate_sample_open_interest_data(sample_data)

        # 特徴量エンジニアリングサービスを初期化
        feature_service = FeatureEngineeringService()

        # 特徴量を計算（改善されたスケーリング付き）
        features_df = feature_service.calculate_advanced_features(
            ohlcv_data=sample_data,
            funding_rate_data=funding_data,
            open_interest_data=oi_data,
        )

        # 数値特徴量のスケールを確認
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_stats = {}

        for col in numeric_columns[:15]:  # 最初の15個の特徴量をチェック
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

        logger.info(f"特徴量統計情報（最初の15個、スケーリング後）:")
        for col, stats in feature_stats.items():
            logger.info(
                f"  {col}: 平均={stats['mean']:.4f}, 標準偏差={stats['std']:.4f}, 範囲={stats['range']:.4f}"
            )

        # スケールの整合性を検証
        ranges = [stats["range"] for stats in feature_stats.values()]
        if len(ranges) > 1:
            max_range = max(ranges)
            min_range = min([r for r in ranges if r > 0])
            scale_ratio = max_range / min_range if min_range > 0 else 1

            logger.info(f"スケール比率（最大範囲/最小範囲）: {scale_ratio:.2f}")

            # スケーリング後はスケール比率が大幅に改善されているはず
            if scale_ratio < 100:  # 改善前は205万倍だった
                logger.info(
                    "✅ 特徴量スケーリングにより、スケール不整合が大幅に改善されました"
                )
            else:
                logger.warning(
                    f"⚠️ スケール不整合が残っています（比率: {scale_ratio:.2f}）"
                )

        return {
            "feature_stats": feature_stats,
            "scale_ratio": scale_ratio if "scale_ratio" in locals() else 1,
            "feature_count": len(feature_stats),
        }

    def test_different_scaling_methods(self):
        """異なるスケーリング方法をテスト"""
        logger.info("=== 異なるスケーリング方法のテスト ===")

        # テスト用の特徴量データを生成
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(50000, 10000, 100),  # 大きなスケール
                "feature2": np.random.normal(0.5, 0.1, 100),  # 小さなスケール
                "feature3": np.random.normal(1000, 200, 100),  # 中間スケール
            }
        )

        # 外れ値を追加
        test_data.iloc[0, 0] = 200000  # 極端な外れ値
        test_data.iloc[1, 1] = 5.0  # 外れ値

        preprocessor = DataPreprocessor()

        scaling_methods = ["standard", "robust", "minmax"]
        results = {}

        for method in scaling_methods:
            logger.info(f"--- {method}スケーリングのテスト ---")

            # スケーリングを実行
            scaled_data = preprocessor.preprocess_features(
                test_data.copy(),
                scale_features=True,
                remove_outliers=False,  # 外れ値の影響を確認するため
                scaling_method=method,
            )

            # 統計情報を計算
            stats = {}
            for col in test_data.columns:
                series = scaled_data[col]
                stats[col] = {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "range": series.max() - series.min(),
                }

            # スケール比率を計算
            ranges = [stats[col]["range"] for col in test_data.columns]
            max_range = max(ranges)
            min_range = min([r for r in ranges if r > 0])
            scale_ratio = max_range / min_range if min_range > 0 else 1

            logger.info(f"  スケール比率: {scale_ratio:.2f}")
            logger.info(
                f"  平均値範囲: {min([stats[col]['mean'] for col in test_data.columns]):.4f} - {max([stats[col]['mean'] for col in test_data.columns]):.4f}"
            )

            results[method] = {
                "stats": stats,
                "scale_ratio": scale_ratio,
                "scaled_data": scaled_data,
            }

        # ロバストスケーリングが外れ値に強いことを確認
        robust_ratio = results["robust"]["scale_ratio"]
        standard_ratio = results["standard"]["scale_ratio"]

        logger.info(f"ロバストスケーリング比率: {robust_ratio:.2f}")
        logger.info(f"標準スケーリング比率: {standard_ratio:.2f}")

        if robust_ratio <= standard_ratio:
            logger.info("✅ ロバストスケーリングが外れ値に対してより安定しています")
        else:
            logger.warning("⚠️ ロバストスケーリングの効果が期待より低いです")

        return results

    def test_scaling_with_outlier_removal(self):
        """外れ値除去とスケーリングの組み合わせをテスト"""
        logger.info("=== 外れ値除去+スケーリングのテスト ===")

        # サンプルデータを生成
        sample_data = self.generate_sample_data()
        funding_data = self.generate_sample_funding_rate_data(sample_data)
        oi_data = self.generate_sample_open_interest_data(sample_data)

        # 特徴量エンジニアリングサービスを初期化
        feature_service = FeatureEngineeringService()

        # 特徴量を計算
        features_df = feature_service.calculate_advanced_features(
            ohlcv_data=sample_data,
            funding_rate_data=funding_data,
            open_interest_data=oi_data,
        )

        # 数値特徴量を選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_subset = features_df[numeric_columns[:10]].copy()  # 最初の10個

        preprocessor = DataPreprocessor()

        # 外れ値除去なしでスケーリング
        scaled_without_outlier_removal = preprocessor.preprocess_features(
            feature_subset.copy(),
            scale_features=True,
            remove_outliers=False,
            scaling_method="robust",
        )

        # 外れ値除去ありでスケーリング
        scaled_with_outlier_removal = preprocessor.preprocess_features(
            feature_subset.copy(),
            scale_features=True,
            remove_outliers=True,
            outlier_threshold=3.0,
            scaling_method="robust",
        )

        # 結果を比較
        def calculate_stability_metrics(df):
            """データの安定性指標を計算"""
            metrics = {}
            for col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    metrics[col] = {
                        "std": series.std(),
                        "iqr": series.quantile(0.75) - series.quantile(0.25),
                        "range": series.max() - series.min(),
                    }
            return metrics

        metrics_without = calculate_stability_metrics(scaled_without_outlier_removal)
        metrics_with = calculate_stability_metrics(scaled_with_outlier_removal)

        # 安定性の改善を評価
        improvement_count = 0
        total_features = 0

        for col in metrics_without.keys():
            if col in metrics_with:
                total_features += 1
                # IQRが小さくなっていれば改善
                if metrics_with[col]["iqr"] < metrics_without[col]["iqr"]:
                    improvement_count += 1

        improvement_ratio = (
            improvement_count / total_features if total_features > 0 else 0
        )

        logger.info(
            f"外れ値除去による安定性改善: {improvement_count}/{total_features} ({improvement_ratio*100:.1f}%)"
        )

        if improvement_ratio > 0.5:
            logger.info("✅ 外れ値除去により特徴量の安定性が改善されました")
        else:
            logger.warning("⚠️ 外れ値除去の効果が限定的です")

        return {
            "improvement_ratio": improvement_ratio,
            "metrics_without": metrics_without,
            "metrics_with": metrics_with,
        }

    def test_overall_feature_engineering_improvement(self):
        """全体的な特徴量エンジニアリング改善をテスト"""
        logger.info("=== 全体的な特徴量エンジニアリング改善テスト ===")

        # 各テストを実行
        scaling_results = self.test_feature_scaling_improvement()
        method_results = self.test_different_scaling_methods()
        outlier_results = self.test_scaling_with_outlier_removal()

        # 改善スコアを計算
        improvement_score = 0

        # スケール比率改善（最大40点）
        scale_ratio = scaling_results["scale_ratio"]
        if scale_ratio < 10:
            improvement_score += 40
        elif scale_ratio < 50:
            improvement_score += 30
        elif scale_ratio < 100:
            improvement_score += 20
        elif scale_ratio < 1000:
            improvement_score += 10

        # ロバストスケーリング効果（最大25点）
        robust_ratio = method_results["robust"]["scale_ratio"]
        standard_ratio = method_results["standard"]["scale_ratio"]
        if robust_ratio <= standard_ratio:
            improvement_score += 25
        elif robust_ratio <= standard_ratio * 1.2:
            improvement_score += 15

        # 外れ値除去効果（最大20点）
        outlier_improvement = outlier_results["improvement_ratio"]
        if outlier_improvement > 0.7:
            improvement_score += 20
        elif outlier_improvement > 0.5:
            improvement_score += 15
        elif outlier_improvement > 0.3:
            improvement_score += 10

        # 特徴量数（最大15点）
        feature_count = scaling_results["feature_count"]
        if feature_count >= 10:
            improvement_score += 15
        elif feature_count >= 5:
            improvement_score += 10

        logger.info(f"特徴量エンジニアリング改善スコア: {improvement_score}/100")

        if improvement_score >= 80:
            logger.info("🎉 優秀な改善効果が確認されました")
        elif improvement_score >= 60:
            logger.info("✅ 良好な改善効果が確認されました")
        elif improvement_score >= 40:
            logger.info("⚠️ 部分的な改善効果が確認されました")
        else:
            logger.warning("❌ 改善効果が不十分です")

        return {
            "improvement_score": improvement_score,
            "scaling_results": scaling_results,
            "method_results": method_results,
            "outlier_results": outlier_results,
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging

    logging.basicConfig(level=logging.INFO)

    test_instance = TestImprovedFeatureScaling()

    # 全体的な改善効果を検証
    results = test_instance.test_overall_feature_engineering_improvement()

    print(f"\n=== 特徴量エンジニアリング改善結果サマリー ===")
    print(f"改善スコア: {results['improvement_score']}/100")
    print(f"スケール比率: {results['scaling_results']['scale_ratio']:.2f}")
    print(
        f"外れ値除去改善率: {results['outlier_results']['improvement_ratio']*100:.1f}%"
    )
