"""
統合特徴量評価スクリプト

重複していた5つのスクリプトの機能を統合し、包括的な特徴量分析を提供します。
CommonFeatureEvaluatorクラスを活用し、モジュラー設計で各分析を独立して実行可能です。

実行方法:
    cd backend

    # 全分析実行
    python -m scripts.feature_evaluation.feature_evaluator

    # 特定の分析のみ実行
    python -m scripts.feature_evaluation.feature_evaluator --mode info
    python -m scripts.feature_evaluation.feature_evaluator --mode importance
    python -m scripts.feature_evaluation.feature_evaluator --mode detection
    python -m scripts.feature_evaluation.feature_evaluator --mode reduction

    # モデル指定
    python -m scripts.feature_evaluation.feature_evaluator --model lightgbm
    python -m scripts.feature_evaluation.feature_evaluator --model xgboost
    python -m scripts.feature_evaluation.feature_evaluator --model random_forest

    # Optuna最適化を有効化
    python -m scripts.feature_evaluation.feature_evaluator --optimize --n-trials 100

    # プリセット指定
    python -m scripts.feature_evaluation.feature_evaluator --preset 4h_4bars
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.unified_config import unified_config
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
)
from app.services.ml.optimization.optuna_optimizer import (
    OptunaOptimizer,
    ParameterSpace,
)
from app.services.ml.optimization.ensemble_parameter_space import EnsembleParameterSpace

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureEvaluationConfig:
    """特徴量評価の設定"""

    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    preset: Optional[str] = "4h_4bars"
    model: str = "all"
    optimize: bool = False
    n_trials: int = 50
    output_dir: str = "backend/results/feature_analysis"
    mode: str = "all"
    use_pipeline_method: bool = True  # パイプライン準拠の評価を行うか


class FeatureEvaluator:
    """統合特徴量評価クラス

    CommonFeatureEvaluatorを活用し、以下の分析を提供:
    1. 特徴量基本情報
    2. 特徴量重要度分析
    3. 低重要度特徴量検出
    4. 特徴量削減の性能評価
    """

    def __init__(
        self,
        evaluator: CommonFeatureEvaluator,
        config: FeatureEvaluationConfig,
    ):
        """初期化

        Args:
            evaluator: 共通評価ユーティリティ
            config: 評価設定
        """
        self.evaluator = evaluator
        self.config = config

        # タイムスタンプ付きサブディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 絶対パスに変換（相対パスの場合）
        base_output_dir = Path(config.output_dir)
        if not base_output_dir.is_absolute():
            # backend/results/... の場合、backendディレクトリからの相対パス
            base_output_dir = Path(
                __file__
            ).parent.parent.parent / config.output_dir.replace("backend/", "")
        self.output_dir = base_output_dir / f"feature_eval_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"結果保存先: {self.output_dir}")

        # 結果格納用
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "preset": config.preset,
                "model": config.model,
                "optimize": config.optimize,
            },
            "feature_info": {},
            "importance_scores": {},
            "low_importance_features": [],
            "reduction_comparison": {},
        }

    def analyze_info(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, Any]:
        """特徴量基本情報を分析

        Args:
            features_df: 特徴量DataFrame
            labels: ラベルSeries

        Returns:
            基本情報の辞書
        """
        logger.info("=" * 80)
        logger.info("特徴量基本情報分析")
        logger.info("=" * 80)

        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X = features_df[feature_cols]

        info = {
            "total_features": len(feature_cols),
            "total_samples": len(X),
            "label_distribution": dict(labels.value_counts().to_dict()),
            "missing_values": int(X.isnull().sum().sum()),
            "numeric_features": int(X.select_dtypes(include=[np.number]).shape[1]),
            "feature_names": feature_cols,
            "statistics": {
                "mean": X.mean().to_dict(),
                "std": X.std().to_dict(),
                "min": X.min().to_dict(),
                "max": X.max().to_dict(),
            },
        }

        logger.info(f"総特徴量数: {info['total_features']}")
        logger.info(f"総サンプル数: {info['total_samples']}")
        logger.info(f"ラベル分布: {info['label_distribution']}")
        logger.info(f"欠損値: {info['missing_values']}")

        return info

    def analyze_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "all",
    ) -> Dict[str, Any]:
        """特徴量重要度を分析

        Args:
            X: 特徴量
            y: ターゲット
            model_type: モデルタイプ

        Returns:
            重要度スコアの辞書
        """
        logger.info("=" * 80)
        logger.info("特徴量重要度分析")
        logger.info("=" * 80)

        importance_results = {}

        # Pipeline Mode (Binary)
        if self.config.use_pipeline_method:
            logger.info("パイプライン準拠（Binary/Balanced）で重要度を計算します")

            # LightGBM
            if model_type in ["lightgbm", "all"]:
                logger.info("LightGBM重要度を計算中...")
                best_params = self._optimize_hyperparameters(
                    X, y, "lightgbm", binary=True
                )
                lgb_importance = self._calculate_lightgbm_importance(
                    X, y, binary=True, params=best_params
                )
                importance_results["lightgbm"] = lgb_importance

            # XGBoost
            if model_type in ["xgboost", "all"]:
                logger.info("XGBoost重要度を計算中...")
                best_params = self._optimize_hyperparameters(
                    X, y, "xgboost", binary=True
                )
                xgb_importance = self._calculate_xgboost_importance(
                    X, y, binary=True, params=best_params
                )
                importance_results["xgboost"] = xgb_importance

            # RandomForest (Meta Labelingで使用されるため残す)
            if model_type in ["random_forest", "all"]:
                logger.info("RandomForest重要度を計算中...")
                rf_importance = self._calculate_random_forest_importance(
                    X, y
                )  # RFは多クラス/2値兼用
                importance_results["random_forest"] = rf_importance

        else:
            # Legacy Mode (Multiclass)
            # LightGBM
            if model_type in ["lightgbm", "all"]:
                logger.info("LightGBM重要度を計算中...")
                best_params = self._optimize_hyperparameters(
                    X, y, "lightgbm", binary=False
                )
                lgb_importance = self._calculate_lightgbm_importance(
                    X, y, binary=False, params=best_params
                )
                importance_results["lightgbm"] = lgb_importance

            # XGBoost
            if model_type in ["xgboost", "all"]:
                logger.info("XGBoost重要度を計算中...")
                best_params = self._optimize_hyperparameters(
                    X, y, "xgboost", binary=False
                )
                xgb_importance = self._calculate_xgboost_importance(
                    X, y, binary=False, params=best_params
                )
                importance_results["xgboost"] = xgb_importance

            # RandomForest
            if model_type in ["random_forest", "all"]:
                logger.info("RandomForest重要度を計算中...")
                rf_importance = self._calculate_random_forest_importance(X, y)
                importance_results["random_forest"] = rf_importance

            # Permutation Importance
            if model_type == "all":
                logger.info("Permutation Importance を計算中...")
                perm_importance = self._calculate_permutation_importance(X, y)
                importance_results["permutation"] = perm_importance

        # 統合スコア
        combined_scores = self._combine_importance_scores(importance_results)
        importance_results["combined"] = combined_scores

        logger.info(f"重要度分析完了: {len(importance_results)}種類")

        return importance_results

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        binary: bool = False,
    ) -> Dict[str, Any]:
        """ハイパーパラメータを最適化

        Args:
            X: 特徴量
            y: ターゲット
            model_type: モデルタイプ ('lightgbm' or 'xgboost')
            binary: 2値分類かどうか

        Returns:
            最適化されたパラメータ
        """
        if not self.config.optimize:
            return {}

        logger.info(f"{model_type}のハイパーパラメータ最適化を開始...")

        optimizer = OptunaOptimizer()

        # パラメータ空間の取得
        if model_type == "lightgbm":
            space = EnsembleParameterSpace.get_lightgbm_parameter_space()
        elif model_type == "xgboost":
            space = EnsembleParameterSpace.get_xgboost_parameter_space()
        else:
            return {}

        # 目的関数
        def objective(params: Dict[str, Any]) -> float:
            try:
                # パラメータの変換（プレフィックス除去など必要なら）
                clean_params = {}
                prefix = "lgb_" if model_type == "lightgbm" else "xgb_"
                for k, v in params.items():
                    if k.startswith(prefix):
                        clean_params[k.replace(prefix, "")] = v
                    else:
                        clean_params[k] = v

                # CV評価
                results = self._evaluate_with_cv(
                    X, y, list(X.columns), model_type, params=clean_params
                )

                # スコア（F1スコアを最大化）
                return results.get("f1_score", 0.0)
            except Exception as e:
                logger.warning(f"最適化試行エラー: {e}")
                return 0.0

        # 最適化実行
        result = optimizer.optimize(
            objective_function=objective,
            parameter_space=space,
            n_calls=self.config.n_trials,
        )

        # 最適パラメータの整形
        best_params = {}
        prefix = "lgb_" if model_type == "lightgbm" else "xgb_"
        for k, v in result.best_params.items():
            if k.startswith(prefix):
                best_params[k.replace(prefix, "")] = v
            else:
                best_params[k] = v

        logger.info(f"最適パラメータ: {best_params}")
        return best_params

    def detect_low_importance(
        self,
        importance_scores: Dict[str, Any],
        threshold: float = 0.2,
    ) -> List[str]:
        """低重要度特徴量を検出

        Args:
            importance_scores: 重要度スコア
            threshold: 閾値（下位X%）

        Returns:
            低重要度特徴量のリスト
        """
        logger.info("=" * 80)
        logger.info("低重要度特徴量検出")
        logger.info("=" * 80)

        combined = importance_scores.get("combined", {})
        if not combined:
            logger.warning("統合スコアが見つかりません")
            return []

        # スコアでソート
        sorted_features = sorted(
            combined.items(),
            key=lambda x: x[1],
        )

        # 下位threshold%を計算
        n_remove = max(1, int(len(sorted_features) * threshold))
        low_importance = [feat for feat, _ in sorted_features[:n_remove]]

        logger.info(
            f"低重要度特徴量: {len(low_importance)}個（下位{threshold * 100:.0f}%）"
        )

        return low_importance

    def evaluate_reduction(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features_to_remove: List[str],
        model_type: str = "lightgbm",
    ) -> Dict[str, Any]:
        """特徴量削減の性能を評価

        Args:
            X: 全特徴量
            y: ターゲット
            features_to_remove: 削除する特徴量
            model_type: モデルタイプ

        Returns:
            削減前後の比較結果
        """
        logger.info("=" * 80)
        logger.info("特徴量削減性能評価")
        logger.info("=" * 80)

        # ベースライン（全特徴量）
        logger.info("ベースライン評価（全特徴量）...")
        if self.config.use_pipeline_method:
            baseline_results = self._evaluate_with_purged_cv(X, y, list(X.columns))
        else:
            baseline_results = self._evaluate_with_cv(X, y, list(X.columns), model_type)

        # 削減後
        kept_features = [col for col in X.columns if col not in features_to_remove]
        logger.info(f"削減後評価（{len(kept_features)}特徴量）...")
        if self.config.use_pipeline_method:
            reduced_results = self._evaluate_with_purged_cv(X, y, kept_features)
        else:
            reduced_results = self._evaluate_with_cv(X, y, kept_features, model_type)

        # 比較
        comparison = {
            "baseline": {
                "n_features": len(X.columns),
                **baseline_results,
            },
            "reduced": {
                "n_features": len(kept_features),
                "removed_features": features_to_remove,
                **reduced_results,
            },
            "changes": self._calculate_changes(
                baseline_results,
                reduced_results,
            ),
        }

        logger.info(f"削減: {len(features_to_remove)}個の特徴量を削除")
        logger.info(f"性能変化: {comparison['changes']}")

        return comparison

    def generate_report(self) -> None:
        """マークダウンレポートを生成"""
        logger.info("=" * 80)
        logger.info("レポート生成")
        logger.info("=" * 80)

        report_path = self.output_dir / "summary_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 特徴量評価レポート\n\n")
            f.write(f"**生成日時**: {self.results['timestamp']}\n\n")

            # 設定
            f.write("## 設定\n\n")
            config = self.results["config"]
            for key, value in config.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")

            # 基本情報
            if self.results.get("feature_info"):
                info = self.results["feature_info"]
                f.write("## 特徴量基本情報\n\n")
                f.write(f"- **総特徴量数**: {info.get('total_features', 'N/A')}\n")
                f.write(f"- **総サンプル数**: {info.get('total_samples', 'N/A')}\n")
                f.write(f"- **ラベル分布**: {info.get('label_distribution', {})}\n")
                f.write(f"- **欠損値**: {info.get('missing_values', 'N/A')}\n\n")

            # 重要度分析
            if self.results.get("importance_scores"):
                f.write("## 特徴量重要度分析\n\n")
                f.write("複数のモデルで特徴量重要度を計算しました。\n\n")

            # 低重要度特徴量
            if self.results.get("low_importance_features"):
                features = self.results["low_importance_features"]
                f.write("## 低重要度特徴量\n\n")
                f.write(f"検出された低重要度特徴量: {len(features)}個\n\n")
                for i, feat in enumerate(features[:20], 1):
                    f.write(f"{i}. {feat}\n")
                if len(features) > 20:
                    f.write(f"\n... 他{len(features) - 20}個\n")
                f.write("\n")

            # 削減評価
            if self.results.get("reduction_comparison"):
                comp = self.results["reduction_comparison"]
                f.write("## 特徴量削減評価\n\n")
                if "baseline" in comp and "reduced" in comp:
                    baseline = comp["baseline"]
                    reduced = comp["reduced"]
                    f.write(
                        f"- **削減前特徴量数**: {baseline.get('n_features', 'N/A')}\n"
                    )
                    f.write(
                        f"- **削減後特徴量数**: {reduced.get('n_features', 'N/A')}\n"
                    )
                    f.write(
                        f"- **削除数**: {len(reduced.get('removed_features', []))}\n\n"
                    )

                    if "changes" in comp:
                        changes = comp["changes"]
                        f.write("### 性能変化\n\n")
                        for metric, change in changes.items():
                            f.write(f"- **{metric}**: {change:+.4f}\n")

        logger.info(f"レポート保存: {report_path}")

    def save_results(self) -> None:
        """結果を保存"""
        logger.info("=" * 80)
        logger.info("結果保存")
        logger.info("=" * 80)

        # JSON保存（全結果）
        json_path = self.output_dir / "feature_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON保存: {json_path}")

        # CSV保存（重要度スコア）
        if self.results.get("importance_scores"):
            self._save_importance_csv()

        # CSV保存（削減比較）
        if self.results.get("reduction_comparison"):
            self._save_reduction_csv()

        # 可視化
        self._create_visualizations()

    def _calculate_lightgbm_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        binary: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """LightGBM重要度を計算"""
        try:
            if binary:
                # パイプライン準拠: LGBMClassifier, class_weight='balanced'
                model_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "class_weight": "balanced",
                    "n_estimators": 100,
                    "random_state": unified_config.ml.training.random_state,
                    "verbosity": -1,
                }
                if params:
                    model_params.update(params)

                model = lgb.LGBMClassifier(**model_params)
                model.fit(X, y)
                importance = model.feature_importances_
                # gainではなくsplit(default)だが、sklearn APIではfeature_importances_はsplit/gainの指定がinitによる
                # デフォルトはsplit。gainにするには importance_type='gain'
                # ここでは一貫性のため gain を取得したいが、LGBMClassifierのfeature_importances_は
                # importance_typeパラメータに依存する
                model.set_params(importance_type="gain")
                # 再学習が必要な場合があるが、LGBMは学習後にbooster_から取得可能
                importance = model.booster_.feature_importance(importance_type="gain")
            else:
                # レガシー: Multiclass
                train_data = lgb.Dataset(X, label=y)
                lgb_params = {
                    "objective": "multiclass",
                    "num_class": 3,
                    "metric": "multi_logloss",
                    "verbose": -1,
                    "random_state": unified_config.ml.training.random_state,
                }
                if params:
                    lgb_params.update(params)
                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=100,
                )
                importance = model.feature_importance(importance_type="gain")

            if importance.sum() > 0:
                importance = importance / importance.sum()
            return dict(zip(X.columns, importance))
        except Exception as e:
            logger.error(f"LightGBM重要度計算エラー: {e}")
            return {}

    def _calculate_xgboost_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        binary: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """XGBoost重要度を計算"""
        try:
            if binary:
                xgb_params = {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "verbosity": 0,
                    "random_state": unified_config.ml.training.random_state,
                }
                if params:
                    xgb_params.update(params)
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train(
                    xgb_params,
                    dtrain,
                    num_boost_round=100,
                )
                importance_dict = model.get_score(importance_type="gain")

            result = {col: 0.0 for col in X.columns}
            result.update(importance_dict)
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result
        except Exception as e:
            logger.error(f"XGBoost重要度計算エラー: {e}")
            return {}

    def _calculate_random_forest_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """RandomForest重要度を計算"""
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=unified_config.ml.training.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            importance = rf.feature_importances_
            if importance.sum() > 0:
                importance = importance / importance.sum()
            return dict(zip(X.columns, importance))
        except Exception as e:
            logger.error(f"RandomForest重要度計算エラー: {e}")
            return {}

    def _calculate_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Permutation Importanceを計算"""
        try:
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=unified_config.ml.training.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            perm = permutation_importance(
                rf,
                X,
                y,
                n_repeats=5,
                random_state=unified_config.ml.training.random_state,
                n_jobs=-1,
            )
            importance = perm.importances_mean
            if importance.sum() > 0:
                importance = importance / importance.sum()
            return dict(zip(X.columns, importance))
        except Exception as e:
            logger.error(f"Permutation Importance計算エラー: {e}")
            return {}

    def _combine_importance_scores(
        self,
        importance_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """重要度スコアを統合"""
        if not importance_results:
            return {}

        all_features = set()
        for scores in importance_results.values():
            all_features.update(scores.keys())

        combined = {}
        for feature in all_features:
            scores = [
                importance_results[model].get(feature, 0.0)
                for model in importance_results
            ]
            combined[feature] = float(np.mean(scores))

        return combined

    def _evaluate_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        model_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """TimeSeriesSplitでCV評価"""
        X_selected = X[features]

        # NaN除去
        valid_idx = ~(X_selected.isna().any(axis=1) | y.isna())
        X_clean = X_selected[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) < 100:
            logger.warning("サンプル数不足")
            return {}

        n_splits = unified_config.ml.training.cv_folds
        tscv = TimeSeriesSplit(n_splits=n_splits)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        train_times = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean), 1):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            start_time = time.time()

            # モデル学習
            if model_type == "lightgbm":
                model = self._train_lightgbm(X_train, y_train, params)
                y_pred_proba = model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            elif model_type == "xgboost":
                model = self._train_xgboost(X_train, y_train, params)
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = model.predict(dtest)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:  # random_forest
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=unified_config.ml.training.random_state,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            train_time = time.time() - start_time
            train_times.append(train_time)

            # 評価
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            recall_scores.append(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            f1_scores.append(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )

        return {
            "accuracy": float(np.mean(accuracy_scores)),
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
            "f1_score": float(np.mean(f1_scores)),
            "train_time": float(np.mean(train_times)),
        }

    def _evaluate_with_purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
    ) -> Dict[str, float]:
        """PurgedKFoldでCV評価（パイプライン準拠）"""
        X_selected = X[features]

        # t1を取得（CommonFeatureEvaluator経由）
        # 注意: XのインデックスはDatetimeIndexであることを前提
        t1 = self.evaluator.get_t1_for_purged_kfold(
            X_selected.index, horizon_n=4
        )  # デフォルト4

        results = self.evaluator.purged_kfold_cv(X_selected, y, t1, n_splits=5)

        # キー名を合わせる
        return {
            "accuracy": 0.0,  # PurgedKFoldでは計算していないので0
            "precision": results["cv_precision"],
            "recall": results["cv_recall"],
            "f1_score": results["cv_f1"],
            "pipeline_score": results["cv_pipeline_score"],
            "train_time": results["train_time_sec"],
        }

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """LightGBMモデルを学習"""
        train_data = lgb.Dataset(X_train, label=y_train)
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbose": -1,
            "random_state": unified_config.ml.training.random_state,
        }
        if params:
            lgb_params.update(params)
        return lgb.train(lgb_params, train_data, num_boost_round=100)

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ):
        """XGBoostモデルを学習"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "random_state": unified_config.ml.training.random_state,
        }
        if params:
            xgb_params.update(params)
        return xgb.train(xgb_params, dtrain, num_boost_round=100)

    def _calculate_changes(
        self,
        baseline: Dict[str, float],
        reduced: Dict[str, float],
    ) -> Dict[str, float]:
        """性能変化を計算"""
        changes = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "pipeline_score"]:
            if metric in baseline and metric in reduced:
                changes[metric] = reduced[metric] - baseline[metric]
        return changes

    def _save_importance_csv(self) -> None:
        """重要度スコアをCSVに保存"""
        importance = self.results["importance_scores"]
        if "combined" in importance:
            df = pd.DataFrame(
                [
                    {"feature": feat, "importance": score}
                    for feat, score in importance["combined"].items()
                ]
            )
            df = df.sort_values("importance", ascending=False)
            csv_path = self.output_dir / "importance_scores.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"重要度CSV保存: {csv_path}")

            # 低重要度特徴量のCSVも保存
            if self.results.get("low_importance_features"):
                low_imp_df = pd.DataFrame(
                    {"feature": self.results["low_importance_features"]}
                )
                low_imp_csv = self.output_dir / "low_importance_features.csv"
                low_imp_df.to_csv(low_imp_csv, index=False, encoding="utf-8")
                logger.info(f"低重要度特徴量CSV保存: {low_imp_csv}")

    def _save_reduction_csv(self) -> None:
        """削減比較結果をCSVに保存"""
        comp = self.results["reduction_comparison"]
        if "baseline" in comp and "reduced" in comp:
            data = []
            for scenario, results in [
                ("baseline", comp["baseline"]),
                ("reduced", comp["reduced"]),
            ]:
                row = {"scenario": scenario}
                row.update(results)
                data.append(row)
            df = pd.DataFrame(data)
            csv_path = self.output_dir / "reduction_comparison.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"削減比較CSV保存: {csv_path}")

    def _create_visualizations(self) -> None:
        """可視化を作成"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 重要度プロット
        if self.results.get("importance_scores", {}).get("combined"):
            self._plot_importance(viz_dir)

        logger.info(f"可視化保存: {viz_dir}")

    def _plot_importance(self, viz_dir: Path) -> None:
        """重要度プロット"""
        try:
            combined = self.results["importance_scores"]["combined"]
            df = (
                pd.DataFrame(
                    [
                        {"feature": feat, "importance": score}
                        for feat, score in combined.items()
                    ]
                )
                .sort_values("importance", ascending=False)
                .head(20)
            )

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(df)), df["importance"].values)
            plt.yticks(range(len(df)), df["feature"].values)
            plt.xlabel("Importance Score")
            plt.title("Top 20 Feature Importance")
            plt.tight_layout()
            plt.savefig(viz_dir / "importance_plot.png")
            plt.close()
        except Exception as e:
            logger.warning(f"重要度プロット作成エラー: {e}")


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="統合特徴量評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "info", "importance", "detection", "reduction"],
        default="all",
        help="実行モード（デフォルト: all）",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペア（デフォルト: BTC/USDT:USDT）",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="時間軸（デフォルト: 1h）",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="4h_4bars",
        help="ラベル生成プリセット（デフォルト: 4h_4bars）",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["lightgbm", "xgboost", "random_forest", "all"],
        default="all",
        help="使用モデル（デフォルト: all）",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optuna最適化を有効化",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna試行回数（デフォルト: 50）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="backend/results/feature_analysis",
        help="結果出力ディレクトリ（デフォルト: backend/results/feature_analysis）",
    )

    return parser.parse_args()


def main() -> None:
    """メイン実行関数"""
    start_time = time.time()

    try:
        # 引数パース
        args = parse_arguments()

        # 設定作成
        config = FeatureEvaluationConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            preset=args.preset,
            model=args.model,
            optimize=args.optimize,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
            mode=args.mode,
        )

        logger.info("=" * 80)
        logger.info("統合特徴量評価開始")
        logger.info("=" * 80)
        logger.info(f"モード: {config.mode}")
        logger.info(f"シンボル: {config.symbol}")
        logger.info(f"時間軸: {config.timeframe}")
        logger.info(f"プリセット: {config.preset}")
        logger.info(f"モデル: {config.model}")
        logger.info(f"Optuna: {config.optimize}")

        # 共通評価ユーティリティ初期化
        common_evaluator = CommonFeatureEvaluator()

        try:
            # データ取得
            logger.info("\nデータ取得中...")

            # 最新のデータを使用するため、日付範囲を指定
            # OIデータは2020-08以降にしか存在しないため、最近のデータを取得する
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 直近90日分のデータ

            data = common_evaluator.fetch_data(
                symbol=config.symbol,
                timeframe=config.timeframe,
                limit=200000,  # 大きい値を指定して最新データを確実に取得
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            if data.ohlcv.empty:
                logger.error("データが取得できませんでした")
                sys.exit(1)

            # 特徴量生成
            logger.info("特徴量生成中...")
            features_df = common_evaluator.build_basic_features(
                data=data,
                skip_crypto_and_advanced=False,
            )

            # ラベル生成
            logger.info("ラベル生成中...")

            if config.use_pipeline_method:
                # パイプライン準拠のラベル生成（LabelCache使用）
                # horizon_n=4, pt=1.0, sl=1.0 はパイプラインの典型的な設定
                labels = common_evaluator.generate_pipeline_compatible_labels(
                    ohlcv_df=data.ohlcv,
                    horizon_n=4,
                    pt_factor=1.0,
                    sl_factor=1.0,
                    use_atr=True,
                    binary_label=True,  # パイプラインは2値分類
                )
                # 2値分類なのでマッピング不要（すでに0/1）
                labels_numeric = labels
                logger.info("パイプライン準拠のラベル生成完了 (Binary)")
            else:
                labels = common_evaluator.create_labels_from_config(
                    ohlcv_df=data.ohlcv,
                    preset_name=config.preset,
                )
                # ラベルを数値に変換（"UP" -> 2, "RANGE" -> 1, "DOWN" -> 0）
                label_mapping = {"DOWN": 0, "RANGE": 1, "UP": 2}
                labels_numeric = labels.map(label_mapping)
                logger.info(f"ラベルを数値に変換: {label_mapping}")

            # 評価器初期化
            evaluator = FeatureEvaluator(common_evaluator, config)

            # 特徴量準備（不要なカラムを除外）
            exclude_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "returns",
                "funding_timestamp",  # 中間計算用カラムを除外
                "timestamp",  # タイムスタンプも除外
            ]
            feature_cols = [
                col for col in features_df.columns if col not in exclude_cols
            ]
            X = features_df[feature_cols]

            # NaN除去
            combined_df = pd.concat(
                [X, labels_numeric.rename("label")], axis=1
            ).dropna()
            X_clean = combined_df[feature_cols]
            y_clean = combined_df["label"]

            logger.info(f"\n有効サンプル数: {len(X_clean)}")
            logger.info(f"特徴量数: {len(feature_cols)}")

            # 分析実行
            if config.mode in ["all", "info"]:
                info = evaluator.analyze_info(features_df, labels)
                evaluator.results["feature_info"] = info

            if config.mode in ["all", "importance"]:
                importance = evaluator.analyze_importance(
                    X_clean, y_clean, config.model
                )
                evaluator.results["importance_scores"] = importance

            if config.mode in ["all", "detection"]:
                if not evaluator.results.get("importance_scores"):
                    importance = evaluator.analyze_importance(
                        X_clean, y_clean, config.model
                    )
                    evaluator.results["importance_scores"] = importance

                low_importance = evaluator.detect_low_importance(
                    evaluator.results["importance_scores"],
                    threshold=0.2,
                )
                evaluator.results["low_importance_features"] = low_importance

            if config.mode in ["all", "reduction"]:
                if not evaluator.results.get("low_importance_features"):
                    if not evaluator.results.get("importance_scores"):
                        importance = evaluator.analyze_importance(
                            X_clean, y_clean, config.model
                        )
                        evaluator.results["importance_scores"] = importance

                    low_importance = evaluator.detect_low_importance(
                        evaluator.results["importance_scores"],
                        threshold=0.2,
                    )
                    evaluator.results["low_importance_features"] = low_importance

                reduction = evaluator.evaluate_reduction(
                    X_clean,
                    y_clean,
                    evaluator.results["low_importance_features"],
                    model_type="lightgbm" if config.model == "all" else config.model,
                )
                evaluator.results["reduction_comparison"] = reduction

            # 結果保存
            evaluator.generate_report()
            evaluator.save_results()

        finally:
            common_evaluator.close()

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"統合特徴量評価完了（処理時間: {elapsed:.2f}秒）")
        logger.info(f"結果: {config.output_dir}")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


