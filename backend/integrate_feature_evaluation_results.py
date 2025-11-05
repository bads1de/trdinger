"""
3モデル（LightGBM、XGBoost、TabNet）の特徴量評価結果を統合分析

各モデルの評価結果から、共通して低寄与度と判定された特徴量を特定し、
安全に削除できる特徴量のリストを作成します。

実行方法:
    cd backend
    python integrate_feature_evaluation_results.py
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureEvaluationIntegrator:
    """3モデルの特徴量評価結果を統合分析するクラス"""

    def __init__(self):
        """初期化"""
        self.lightgbm_data = None
        self.xgboost_data = None
        self.tabnet_data = None
        self.integration_results = {}

    def load_evaluation_results(
        self,
        lightgbm_path: str = "lightgbm_feature_performance_evaluation.json",
        xgboost_path: str = "xgboost_feature_performance_evaluation.json",
        tabnet_path: str = "tabnet_feature_performance_evaluation.json",
    ) -> bool:
        """
        各モデルの評価結果を読み込み

        Args:
            lightgbm_path: LightGBM評価結果のパス
            xgboost_path: XGBoost評価結果のパス
            tabnet_path: TabNet評価結果のパス

        Returns:
            全ファイルの読み込み成功可否
        """
        all_loaded = True

        # LightGBM結果読み込み
        try:
            with open(lightgbm_path, "r", encoding="utf-8") as f:
                self.lightgbm_data = json.load(f)
            logger.info(f"LightGBM評価結果を読み込みました: {lightgbm_path}")
        except Exception as e:
            logger.warning(f"LightGBM評価結果の読み込みに失敗: {e}")
            all_loaded = False

        # XGBoost結果読み込み
        try:
            with open(xgboost_path, "r", encoding="utf-8") as f:
                self.xgboost_data = json.load(f)
            logger.info(f"XGBoost評価結果を読み込みました: {xgboost_path}")
        except Exception as e:
            logger.warning(f"XGBoost評価結果の読み込みに失敗: {e}")
            all_loaded = False

        # TabNet結果読み込み
        try:
            with open(tabnet_path, "r", encoding="utf-8") as f:
                self.tabnet_data = json.load(f)
            logger.info(f"TabNet評価結果を読み込みました: {tabnet_path}")
        except Exception as e:
            logger.warning(f"TabNet評価結果の読み込みに失敗: {e}")
            all_loaded = False

        return all_loaded

    def extract_recommended_removals(
        self, data: Dict, model_name: str
    ) -> Tuple[Set[str], float]:
        """
        各モデルの推奨削除特徴量を抽出

        Args:
            data: モデルの評価結果データ
            model_name: モデル名

        Returns:
            (推奨削除特徴量のセット, 性能変化率)
        """
        if not data or "recommendation" not in data:
            logger.warning(f"{model_name}: 推奨事項が見つかりません")
            return set(), 0.0

        recommendation = data["recommendation"]
        removed_features = recommendation.get("recommended_features_to_remove", [])
        change_pct = recommendation.get("performance_change_pct", 0.0)

        logger.info(
            f"{model_name}: {len(removed_features)}個の特徴量削除を推奨 "
            f"(性能変化: {change_pct:+.2f}%)"
        )

        return set(removed_features), change_pct

    def get_common_removals(
        self,
        min_models: int = 2,
        max_performance_change: float = 1.0,
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        複数モデルで共通して削除推奨された特徴量を取得

        Args:
            min_models: 最小モデル数（この数以上のモデルで推奨された特徴量）
            max_performance_change: 許容する最大性能変化率(%)

        Returns:
            (共通削除推奨特徴量リスト, 各特徴量の推奨モデル数)
        """
        # 各モデルの推奨削除特徴量を取得
        model_removals = {}
        model_changes = {}

        if self.lightgbm_data:
            features, change = self.extract_recommended_removals(
                self.lightgbm_data, "LightGBM"
            )
            if abs(change) <= max_performance_change:
                model_removals["LightGBM"] = features
                model_changes["LightGBM"] = change

        if self.xgboost_data:
            features, change = self.extract_recommended_removals(
                self.xgboost_data, "XGBoost"
            )
            if abs(change) <= max_performance_change:
                model_removals["XGBoost"] = features
                model_changes["XGBoost"] = change

        if self.tabnet_data:
            features, change = self.extract_recommended_removals(
                self.tabnet_data, "TabNet"
            )
            if abs(change) <= max_performance_change:
                model_removals["TabNet"] = features
                model_changes["TabNet"] = change

        if not model_removals:
            logger.warning("削除推奨特徴量が見つかりませんでした")
            return [], {}

        # 全特徴量の出現回数をカウント
        all_features = []
        for features in model_removals.values():
            all_features.extend(features)

        feature_counts = Counter(all_features)

        # min_models以上のモデルで推奨された特徴量を抽出
        common_features = [
            feat for feat, count in feature_counts.items() if count >= min_models
        ]

        logger.info(
            f"\n{len(model_removals)}モデル中{min_models}モデル以上で推奨された特徴量: "
            f"{len(common_features)}個"
        )

        return common_features, dict(feature_counts)

    def analyze_performance_impact(self) -> Dict[str, Dict[str, float]]:
        """
        各モデルでの性能影響を分析

        Returns:
            モデル別の性能指標辞書
        """
        performance_data = {}

        for model_name, data in [
            ("LightGBM", self.lightgbm_data),
            ("XGBoost", self.xgboost_data),
            ("TabNet", self.tabnet_data),
        ]:
            if not data or "scenarios" not in data:
                continue

            scenarios = data["scenarios"]
            baseline = scenarios.get("baseline", {})

            if not baseline:
                continue

            performance_data[model_name] = {
                "baseline_rmse": baseline.get("cv_rmse", 0.0),
                "baseline_mae": baseline.get("cv_mae", 0.0),
                "baseline_r2": baseline.get("cv_r2", 0.0),
            }

            # 推奨シナリオの性能変化
            recommendation = data.get("recommendation", {})
            if "performance_change_pct" in recommendation:
                performance_data[model_name]["recommended_change_pct"] = recommendation[
                    "performance_change_pct"
                ]

        return performance_data

    def create_integration_report(
        self,
        common_features: List[str],
        feature_counts: Dict[str, int],
        min_models: int = 2,
    ) -> Dict:
        """
        統合分析レポートを作成

        Args:
            common_features: 共通削除推奨特徴量
            feature_counts: 特徴量ごとの推奨モデル数
            min_models: 使用した最小モデル数閾値

        Returns:
            統合レポート辞書
        """
        # 性能影響分析
        performance_impact = self.analyze_performance_impact()

        # モデル別の推奨削除数
        model_recommendations = {}
        for model_name, data in [
            ("LightGBM", self.lightgbm_data),
            ("XGBoost", self.xgboost_data),
            ("TabNet", self.tabnet_data),
        ]:
            if data and "recommendation" in data:
                rec = data["recommendation"]
                model_recommendations[model_name] = {
                    "features_removed_count": rec.get("features_removed_count", 0),
                    "performance_change_pct": rec.get("performance_change_pct", 0.0),
                    "features": rec.get("recommended_features_to_remove", []),
                }

        # 推奨モデル数別に特徴量を分類
        features_by_agreement = {
            "all_models": [],  # 3モデルすべて
            "two_models": [],  # 2モデル
        }

        for feat in common_features:
            count = feature_counts.get(feat, 0)
            if count == 3:
                features_by_agreement["all_models"].append(feat)
            elif count == 2:
                features_by_agreement["two_models"].append(feat)

        # 統合レポート作成
        report = {
            "integration_date": datetime.now().isoformat(),
            "analysis_criteria": {
                "min_models_agreement": min_models,
                "max_performance_change_pct": 1.0,
            },
            "models_evaluated": list(model_recommendations.keys()),
            "model_recommendations": model_recommendations,
            "performance_impact": performance_impact,
            "common_features": {
                "total_count": len(common_features),
                "all_models_agree": len(features_by_agreement["all_models"]),
                "two_models_agree": len(features_by_agreement["two_models"]),
                "features_by_agreement": features_by_agreement,
                "feature_recommendation_counts": feature_counts,
            },
            "final_recommendation": {
                "safe_to_remove": features_by_agreement["all_models"],
                "cautious_removal": features_by_agreement["two_models"],
                "total_removable": len(common_features),
            },
        }

        return report

    def save_integration_results(
        self,
        report: Dict,
        json_path: str = "integrated_feature_evaluation.json",
        csv_path: str = "integrated_feature_recommendations.csv",
    ):
        """
        統合分析結果を保存

        Args:
            report: 統合レポート
            json_path: JSON保存パス
            csv_path: CSV保存パス
        """
        try:
            # JSON保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"統合結果JSON保存完了: {json_path}")

            # CSV保存（特徴量リスト）
            feature_data = []
            common = report["common_features"]

            # 全モデル一致
            for feat in common["features_by_agreement"]["all_models"]:
                feature_data.append(
                    {
                        "feature": feat,
                        "agreement_level": "all_models",
                        "model_count": 3,
                        "recommendation": "safe_to_remove",
                    }
                )

            # 2モデル一致
            for feat in common["features_by_agreement"]["two_models"]:
                feature_data.append(
                    {
                        "feature": feat,
                        "agreement_level": "two_models",
                        "model_count": 2,
                        "recommendation": "cautious_removal",
                    }
                )

            if feature_data:
                df = pd.DataFrame(feature_data)
                df = df.sort_values(["model_count", "feature"], ascending=[False, True])
                df.to_csv(csv_path, index=False)
                logger.info(f"統合結果CSV保存完了: {csv_path}")

        except Exception as e:
            logger.error(f"統合結果保存エラー: {e}")

    def print_integration_summary(self, report: Dict):
        """
        統合分析結果のサマリーを出力

        Args:
            report: 統合レポート
        """
        print("\n" + "=" * 80)
        print("特徴量評価統合分析結果")
        print("=" * 80)

        print(f"\n分析日時: {report['integration_date']}")
        print(f"評価モデル数: {len(report['models_evaluated'])}")
        print(f"評価モデル: {', '.join(report['models_evaluated'])}")

        # 各モデルの推奨
        print("\n" + "-" * 80)
        print("【モデル別推奨事項】")
        print("-" * 80)
        for model, rec in report["model_recommendations"].items():
            print(
                f"{model:12}: {rec['features_removed_count']:2}個削除推奨 "
                f"(性能変化: {rec['performance_change_pct']:+.2f}%)"
            )

        # 共通特徴量
        common = report["common_features"]
        print("\n" + "-" * 80)
        print("【統合分析結果】")
        print("-" * 80)
        print(f"共通削除推奨特徴量: {common['total_count']}個")
        print(f"  - 全モデル一致: {common['all_models_agree']}個")
        print(f"  - 2モデル一致: {common['two_models_agree']}個")

        # 最終推奨
        final = report["final_recommendation"]
        print("\n" + "-" * 80)
        print("【最終推奨】")
        print("-" * 80)
        print(f"\n✅ 安全に削除可能（全モデル一致）: {len(final['safe_to_remove'])}個")
        if final["safe_to_remove"]:
            for i, feat in enumerate(final["safe_to_remove"], 1):
                print(f"  {i:2}. {feat}")

        print(f"\n⚠️ 慎重に削除（2モデル一致）: {len(final['cautious_removal'])}個")
        if final["cautious_removal"]:
            for i, feat in enumerate(final["cautious_removal"], 1):
                print(f"  {i:2}. {feat}")

        # 性能影響
        if report["performance_impact"]:
            print("\n" + "-" * 80)
            print("【性能への影響】")
            print("-" * 80)
            for model, metrics in report["performance_impact"].items():
                print(f"\n{model}:")
                print(
                    f"  ベースライン RMSE: {metrics.get('baseline_rmse', 0.0):.6f}"
                )
                if "recommended_change_pct" in metrics:
                    print(
                        f"  推奨削除後の変化: {metrics['recommended_change_pct']:+.2f}%"
                    )

        print("\n" + "=" * 80 + "\n")

    def run_integration_analysis(self, min_models: int = 2):
        """
        統合分析を実行

        Args:
            min_models: 最小モデル数閾値
        """
        logger.info("=" * 80)
        logger.info("特徴量評価統合分析開始")
        logger.info("=" * 80)

        # 評価結果読み込み
        all_loaded = self.load_evaluation_results()

        if not all_loaded:
            logger.warning(
                "一部のモデル評価結果が読み込めませんでした。利用可能なデータで分析を続けます。"
            )

        # 少なくとも1つのモデルデータがあることを確認
        available_models = sum(
            [
                self.lightgbm_data is not None,
                self.xgboost_data is not None,
                self.tabnet_data is not None,
            ]
        )

        if available_models == 0:
            logger.error("評価結果が1つも読み込めませんでした。分析を中止します。")
            return

        logger.info(f"利用可能なモデル: {available_models}個")

        # 共通削除推奨特徴量を抽出
        common_features, feature_counts = self.get_common_removals(
            min_models=min(min_models, available_models)
        )

        if not common_features:
            logger.warning("共通削除推奨特徴量が見つかりませんでした")
            return

        # 統合レポート作成
        report = self.create_integration_report(
            common_features, feature_counts, min_models=min_models
        )

        # 結果保存
        self.save_integration_results(report)

        # サマリー出力
        self.print_integration_summary(report)

        logger.info("統合分析完了")


def main():
    """メイン実行関数"""
    try:
        integrator = FeatureEvaluationIntegrator()
        integrator.run_integration_analysis(min_models=2)

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())