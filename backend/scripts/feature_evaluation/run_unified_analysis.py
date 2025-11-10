"""
統合特徴量分析スクリプト

3つの分析を順次実行し、結果を統合:
1. detect_low_importance_features: 低重要度特徴の検出
2. analyze_feature_importance: 特徴量重要度の詳細分析
3. evaluate_feature_performance: 特徴量パフォーマンスの評価

結果をproduction allowlist更新の推奨として出力

実行方法:
    cd backend
    python -m scripts.feature_evaluation.run_unified_analysis \
        --symbol BTC/USDT:USDT \
        --timeframe 1h \
        --limit 2000 \
        --preset 4h_4bars
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.feature_evaluation.analyze_feature_importance import (
    FeatureImportanceAnalyzer,
)
from scripts.feature_evaluation.common_feature_evaluator import CommonFeatureEvaluator
from scripts.feature_evaluation.detect_low_importance_features import (
    LowImportanceFeatureDetector,
)
from scripts.feature_evaluation.evaluate_feature_performance import (
    MultiModelFeatureEvaluator,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnifiedFeatureAnalyzer:
    """統合特徴量分析クラス

    3つの分析スクリプトを統合して実行し、結果を統合します。
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 2000,
        preset_name: Optional[str] = None,
        output_dir: str = "backend/results/feature_analysis",
    ):
        """初期化

        Args:
            symbol: 取引ペア
            timeframe: 時間足
            limit: データ取得件数
            preset_name: ラベル生成プリセット名（Noneの場合は設定から読み込み）
            output_dir: 出力ディレクトリ
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.preset_name = preset_name
        self.output_dir = Path(output_dir)

        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 結果格納用
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "preset_name": preset_name,
            "label_config": {},
            "low_importance_analysis": {},
            "importance_analysis": {},
            "performance_analysis": {},
            "recommended_production_allowlist": [],
            "analysis_summary": {},
        }

    def run_analysis(self) -> Dict[str, Any]:
        """統合分析を実行

        Returns:
            Dict: 分析結果
        """
        logger.info("=" * 80)
        logger.info("統合特徴量分析開始")
        logger.info("=" * 80)

        try:
            # ラベル生成設定を取得
            self._get_label_config()

            # 分析1: 低重要度特徴の検出
            logger.info("\n" + "=" * 80)
            logger.info("分析1: 低重要度特徴の検出")
            logger.info("=" * 80)
            self._run_low_importance_detection()

            # 分析2: 特徴量重要度の詳細分析
            logger.info("\n" + "=" * 80)
            logger.info("分析2: 特徴量重要度の詳細分析")
            logger.info("=" * 80)
            self._run_importance_analysis()

            # 分析3: 特徴量パフォーマンスの評価
            logger.info("\n" + "=" * 80)
            logger.info("分析3: 特徴量パフォーマンスの評価")
            logger.info("=" * 80)
            self._run_performance_evaluation()

            # 結果を統合
            logger.info("\n" + "=" * 80)
            logger.info("結果の統合")
            logger.info("=" * 80)
            self._integrate_results()

            # 結果を保存
            self._save_results()

            # サマリーを出力
            self._print_summary()

            logger.info("\n" + "=" * 80)
            logger.info("統合特徴量分析完了")
            logger.info("=" * 80)

            return self.results

        except Exception as e:
            logger.error(f"統合分析エラー: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _get_label_config(self) -> None:
        """ラベル生成設定を取得"""
        logger.info("ラベル生成設定を取得")

        try:
            evaluator = CommonFeatureEvaluator()
            label_config = evaluator.get_label_config_info()

            # プリセット名が指定されている場合は上書き
            if self.preset_name:
                label_config["preset_name"] = self.preset_name
                label_config["use_preset"] = True

            self.results["label_config"] = label_config

            logger.info(f"ラベル生成設定: {json.dumps(label_config, indent=2)}")
            evaluator.close()

        except Exception as e:
            logger.error(f"ラベル生成設定取得エラー: {e}")
            raise

    def _run_low_importance_detection(self) -> None:
        """低重要度特徴の検出を実行"""
        try:
            with LowImportanceFeatureDetector(
                symbol=self.symbol,
                timeframe=self.timeframe,
                lookback_days=90,  # デフォルト値
                threshold=0.2,  # 下位20%
                output_dir=str(self.output_dir / "low_importance"),
            ) as detector:
                detector.run_analysis()

                # 結果を取得
                self.results["low_importance_analysis"] = {
                    "total_features": len(detector.xgb_importance),
                    "low_importance_count": len(detector.low_importance_features),
                    "low_importance_features": [
                        f["feature"] for f in detector.low_importance_features
                    ],
                }

                logger.info(
                    f"低重要度特徴検出完了: "
                    f"{len(detector.low_importance_features)}個検出"
                )

        except Exception as e:
            logger.error(f"低重要度特徴検出エラー: {e}")
            self.results["low_importance_analysis"]["error"] = str(e)

    def _run_importance_analysis(self) -> None:
        """特徴量重要度の詳細分析を実行"""
        try:
            with FeatureImportanceAnalyzer() as analyzer:
                analyzer.run_analysis(symbol=self.symbol, limit=self.limit)

                # 結果ファイルから読み込み
                analysis_file = Path("feature_importance_analysis.json")
                if analysis_file.exists():
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        importance_data = json.load(f)

                    self.results["importance_analysis"] = {
                        "total_features": importance_data.get("total_features", 0),
                        "low_importance_count": importance_data.get(
                            "low_importance_features_count", 0
                        ),
                        "low_importance_features": importance_data.get(
                            "low_importance_features", []
                        ),
                    }

                    low_count = self.results["importance_analysis"][
                        "low_importance_count"
                    ]
                    logger.info(
                        f"特徴量重要度分析完了: {low_count}個の低重要度特徴を検出"
                    )
                else:
                    logger.warning("特徴量重要度分析結果ファイルが見つかりません")
                    self.results["importance_analysis"][
                        "error"
                    ] = "結果ファイルが見つかりません"

        except Exception as e:
            logger.error(f"特徴量重要度分析エラー: {e}")
            self.results["importance_analysis"]["error"] = str(e)

    def _run_performance_evaluation(self) -> None:
        """特徴量パフォーマンスの評価を実行"""
        try:
            # LightGBMとXGBoostで評価
            evaluator = MultiModelFeatureEvaluator(["lightgbm", "xgboost"])
            results = evaluator.run_evaluation(symbol=self.symbol, limit=self.limit)

            # 結果を集約
            models_used = list(results.keys())
            recommendations = {}

            for model_name, model_result in results.items():
                recommendation = model_result.get("recommendation", {})
                if "recommended_features_to_remove" in recommendation:
                    recommendations[model_name] = recommendation.get(
                        "recommended_features_to_remove", []
                    )

            self.results["performance_analysis"] = {
                "models_used": models_used,
                "recommendations": recommendations,
            }

            logger.info(f"特徴量パフォーマンス評価完了: {len(models_used)}モデルで評価")

        except Exception as e:
            logger.error(f"特徴量パフォーマンス評価エラー: {e}")
            self.results["performance_analysis"]["error"] = str(e)

    def _integrate_results(self) -> None:
        """結果を統合してproduction allowlistの推奨を生成"""
        logger.info("結果を統合中...")

        try:
            # 各分析から削除推奨特徴を収集
            features_to_remove_sets = []

            # 分析1: 低重要度特徴
            if "low_importance_features" in self.results["low_importance_analysis"]:
                features_to_remove_sets.append(
                    set(
                        self.results["low_importance_analysis"][
                            "low_importance_features"
                        ]
                    )
                )

            # 分析2: 特徴量重要度
            if "low_importance_features" in self.results["importance_analysis"]:
                features_to_remove_sets.append(
                    set(self.results["importance_analysis"]["low_importance_features"])
                )

            # 分析3: パフォーマンス評価
            if "recommendations" in self.results["performance_analysis"]:
                for model_recommendations in self.results["performance_analysis"][
                    "recommendations"
                ].values():
                    if model_recommendations:
                        features_to_remove_sets.append(set(model_recommendations))

            # 共通して削除推奨される特徴（2つ以上の分析で推奨）
            if len(features_to_remove_sets) >= 2:
                # 少なくとも2つの分析で推奨される特徴
                common_features = set()
                for i, set1 in enumerate(features_to_remove_sets):
                    for set2 in features_to_remove_sets[i + 1 :]:
                        common_features.update(set1.intersection(set2))

                features_to_remove = sorted(list(common_features))
            elif len(features_to_remove_sets) == 1:
                # 1つしか分析結果がない場合はそれを使用
                features_to_remove = sorted(list(features_to_remove_sets[0]))
            else:
                features_to_remove = []

            # 全特徴量数を取得（いずれかの分析から）
            total_features = (
                self.results["low_importance_analysis"].get("total_features", 0)
                or self.results["importance_analysis"].get("total_features", 0)
                or 0
            )

            # 推奨production allowlistを生成（削除しない特徴）
            # 注: 実際の全特徴量リストが必要ですが、ここでは削除推奨のみを記録
            self.results["recommended_production_allowlist"] = {
                "features_to_remove": features_to_remove,
                "features_to_remove_count": len(features_to_remove),
                "note": "production allowlistは全特徴から削除推奨特徴を除外したもの",
            }

            # サマリーを作成
            self.results["analysis_summary"] = {
                "total_features": total_features,
                "analyses_completed": len(
                    [
                        k
                        for k, v in {
                            "low_importance": self.results["low_importance_analysis"],
                            "importance": self.results["importance_analysis"],
                            "performance": self.results["performance_analysis"],
                        }.items()
                        if "error" not in v
                    ]
                ),
                "recommended_removal_count": len(features_to_remove),
                "models_used": self.results["performance_analysis"].get(
                    "models_used", []
                ),
            }

            logger.info(f"統合完了: {len(features_to_remove)}個の特徴削除を推奨")

        except Exception as e:
            logger.error(f"結果統合エラー: {e}")
            raise

    def _save_results(self) -> None:
        """結果を保存"""
        logger.info("結果を保存中...")

        try:
            # タイムスタンプ付きファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON保存
            json_path = self.output_dir / f"feature_analysis_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON保存完了: {json_path}")

            # CSV保存（削除推奨特徴リスト）
            if self.results["recommended_production_allowlist"]["features_to_remove"]:
                csv_path = self.output_dir / f"features_to_remove_{timestamp}.csv"
                df = pd.DataFrame(
                    {
                        "feature_name": self.results[
                            "recommended_production_allowlist"
                        ]["features_to_remove"]
                    }
                )
                df.to_csv(csv_path, index=False)
                logger.info(f"CSV保存完了: {csv_path}")

            # 最新の結果を latest.json としても保存
            latest_path = self.output_dir / "feature_analysis_latest.json"
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"最新結果保存: {latest_path}")

        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            raise

    def _print_summary(self) -> None:
        """結果サマリーをコンソール出力"""
        print("\n" + "=" * 80)
        print("統合特徴量分析結果サマリー")
        print("=" * 80)

        summary = self.results["analysis_summary"]
        label_config = self.results["label_config"]

        print(f"\n分析日時: {self.results['timestamp']}")
        print(f"シンボル: {self.symbol}")
        print(f"時間足: {self.timeframe}")
        print(f"データ件数: {self.limit}")

        print("\nラベル生成設定:")
        if label_config.get("use_preset"):
            print(f"  プリセット: {label_config.get('preset_name', 'N/A')}")
        print(f"  時間足: {label_config.get('timeframe', 'N/A')}")
        print(f"  Horizon: {label_config.get('horizon_n', 'N/A')}本先")
        print(f"  閾値: {label_config.get('threshold', 'N/A')}")
        print(f"  閾値方法: {label_config.get('threshold_method', 'N/A')}")

        print(f"\n総特徴量数: {summary['total_features']}")
        print(f"完了した分析: {summary['analyses_completed']}/3")
        print(f"使用モデル: {', '.join(summary['models_used'])}")

        print(f"\n削除推奨特徴量数: {summary['recommended_removal_count']}")

        # 削除推奨特徴を表示（最大20個）
        features_to_remove = self.results["recommended_production_allowlist"][
            "features_to_remove"
        ]
        if features_to_remove:
            print("\n削除推奨特徴量（最大20個表示）:")
            for i, feature in enumerate(features_to_remove[:20], 1):
                print(f"  {i:2}. {feature}")

            if len(features_to_remove) > 20:
                print(f"  ... 他{len(features_to_remove) - 20}個")
        else:
            print("\n削除推奨特徴量はありません")

        print("\n" + "=" * 80)
        print(f"詳細な結果は {self.output_dir} に保存されました")
        print("=" * 80 + "\n")


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="統合特徴量分析スクリプト")

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="分析対象シンボル (デフォルト: BTC/USDT:USDT)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="時間足 (デフォルト: 1h)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="データ取得件数 (デフォルト: 2000)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="ラベル生成プリセット名 (例: 4h_4bars, 1h_4bars_dynamic)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="backend/results/feature_analysis",
        help="出力ディレクトリ (デフォルト: backend/results/feature_analysis)",
    )

    return parser.parse_args()


def main() -> None:
    """メイン実行関数"""
    try:
        # 引数パース
        args = parse_arguments()

        # 統合分析実行
        analyzer = UnifiedFeatureAnalyzer(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            preset_name=args.preset,
            output_dir=args.output_dir,
        )

        analyzer.run_analysis()

        # 成功
        sys.exit(0)

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
