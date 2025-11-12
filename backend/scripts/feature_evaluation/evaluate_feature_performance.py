"""
全モデル（LightGBM、XGBoost）での特徴量性能検証統合スクリプト

既存の93特徴量を3つのモデルで評価し、
削減可能な特徴量を特定します。

TimeSeriesSplitを使用した時系列クロスバリデーションにより、
時系列データの特性を考慮した評価を実施します。

実行方法:
    cd backend
    python -m scripts.feature_evaluation.evaluate_feature_performance
    python -m scripts.feature_evaluation.evaluate_feature_performance --models lightgbm
    python -m scripts.feature_evaluation.evaluate_feature_performance \
        --models lightgbm xgboost
    python -m scripts.feature_evaluation.evaluate_feature_performance --models all

設定:
    - TimeSeriesSplit分割数: ml_config.training.cv_folds (デフォルト: 5)
    - ターゲット変数: forward return (1時間先の収益率)

注意:
    このスクリプトは個別に実行可能ですが、統合分析のため
    run_unified_analysis.py の使用を推奨します。
    統合スクリプトは一貫したラベル生成設定を使用し、
    3つの分析を統合して実行します。
"""

import argparse
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.unified_config import unified_config
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
    EvaluationData,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BaseFeatureEvaluator(ABC):  # TODO: 後続でCommonFeatureEvaluatorに完全統合予定
    """特徴量評価の基底クラス"""

    def __init__(self, model_name: str):
        """
        初期化

        Args:
            model_name: モデル名
        """
        self.model_name = model_name
        self.common = CommonFeatureEvaluator()
        self.results = {}

    def __enter__(self):
        """コンテキストマネージャー: 入場"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 退場"""
        self.common.close()

    def fetch_data(
        self, symbol: str = "BTC/USDT:USDT", limit: int = 2000
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        DBからデータを取得

        Args:
            symbol: 取引ペア
            limit: 取得件数

        Returns:
            (OHLCV, FR, OI)のタプル
        """
        data = self.common.fetch_data(symbol=symbol, timeframe="1h", limit=limit)
        return data.ohlcv, data.fr, data.oi

    def calculate_features(
        self,
        ohlcv_df: pd.DataFrame,
        fr_df: Optional[pd.DataFrame],
        oi_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        特徴量計算

        Args:
            ohlcv_df: OHLCVデータ
            fr_df: ファンディングレートデータ
            oi_df: オープンインタレストデータ

        Returns:
            特徴量DataFrame
        """
        logger.info(f"[{self.model_name}] 特徴量計算開始")

        try:
            # 暗号通貨特化特徴量とadvanced特徴量をスキップして基本特徴量のみ計算
            data = EvaluationData(ohlcv=ohlcv_df, fr=fr_df, oi=oi_df)
            features_df = self.common.build_basic_features(
                data=data,
                skip_crypto_and_advanced=True,
            )
            result_df = self.common.drop_ohlcv_columns(
                features_df,
                keep_close=True,
            )
            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量")
            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def create_target(self, df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """
        ターゲット変数作成

        Args:
            df: closeカラムを含むDataFrame
            periods: 先読み期間

        Returns:
            ターゲット変数
        """
        if "close" not in df.columns:
            raise ValueError("closeカラムが見つかりません")

        # N時間先の収益率
        target = df["close"].pct_change(periods).shift(-periods)
        return target

    @abstractmethod
    def evaluate_model_cv(
        self, X: pd.DataFrame, y: pd.Series, n_splits: Optional[int] = None
    ) -> Dict[str, float]:
        """
        TimeSeriesSplitでクロスバリデーション評価

        時系列データの特性を考慮し、過去データで学習して未来データで評価します。
        分割数はml_configから読み込まれます。

        Args:
            X: 特徴量
            y: ターゲット
            n_splits: 分割数（Noneの場合はml_configから読み込み）

        Returns:
            評価指標の辞書
        """
        pass

    @abstractmethod
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            特徴量重要度の辞書
        """
        pass

    def load_unified_scores(
        self, json_path: str = "../../feature_importance_analysis.json"
    ) -> Dict:
        """
        統合スコアをJSONから読み込み

        Args:
            json_path: JSONファイルパス

        Returns:
            統合スコアデータ
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"統合スコア読み込み完了: {json_path}")
            return data.get("feature_importance", {})
        except Exception as e:
            logger.warning(f"統合スコア読み込みエラー: {e}")
            return {}

    def select_features_by_score(
        self, features: List[str], unified_scores: Dict, bottom_pct: float
    ) -> Tuple[List[str], List[str]]:
        """
        統合スコア下位N%の特徴量を選択

        Args:
            features: 全特徴量リスト
            unified_scores: 統合スコアデータ
            bottom_pct: 下位パーセンタイル (0.1 = 10%)

        Returns:
            (削除する特徴量リスト, 保持する特徴量リスト)
        """
        # スコアでソート
        scored_features = []
        for feat in features:
            if feat in unified_scores:
                score = unified_scores[feat].get("combined_score", 0.0)
                scored_features.append((feat, score))
            else:
                # スコアがない場合は保持
                scored_features.append((feat, 1.0))

        sorted_features = sorted(scored_features, key=lambda x: x[1])

        # 下位N%を計算
        n_remove = max(1, int(len(sorted_features) * bottom_pct))

        to_remove = [feat for feat, _ in sorted_features[:n_remove]]
        to_keep = [feat for feat, _ in sorted_features[n_remove:]]

        return to_remove, to_keep

    def run_scenario(
        self,
        scenario_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        features_to_use: List[str],
        removed_features: List[str] = None,
    ) -> Dict:
        """
        1つのシナリオを実行

        Args:
            scenario_name: シナリオ名
            X: 全特徴量
            y: ターゲット
            features_to_use: 使用する特徴量リスト
            removed_features: 削除した特徴量リスト

        Returns:
            シナリオ結果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[{self.model_name}] シナリオ: {scenario_name}")
        logger.info(f"{'='*80}")
        logger.info(f"使用特徴量数: {len(features_to_use)}")

        if removed_features:
            logger.info(f"削除特徴量数: {len(removed_features)}")
            features_preview = ", ".join(removed_features[:10])
            suffix = "..." if len(removed_features) > 10 else ""
            logger.info(f"削除特徴量: {features_preview}{suffix}")

        # 特徴量選択
        X_selected = X[features_to_use]

        # NaN除去
        valid_idx = ~(X_selected.isna().any(axis=1) | y.isna())
        X_clean = X_selected[valid_idx]
        y_clean = y[valid_idx]

        logger.info(f"有効サンプル数: {len(X_clean)}行")

        if len(X_clean) < 100:
            logger.warning("サンプル数不足")
            return {}

        # クロスバリデーション評価
        cv_results = self.evaluate_model_cv(X_clean, y_clean)

        if not cv_results:
            return {}

        # 特徴量重要度取得
        feature_importance = self.get_feature_importance(X_clean, y_clean)
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

        result = {
            "n_features": len(features_to_use),
            "removed_features": removed_features or [],
            **cv_results,
            "feature_importance_top10": [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in top_features
            ],
        }

        logger.info(
            f"CV RMSE: {cv_results['cv_rmse']:.6f} "
            f"(±{cv_results['cv_rmse_std']:.6f})"
        )
        logger.info(
            f"CV MAE: {cv_results['cv_mae']:.6f} " f"(±{cv_results['cv_mae_std']:.6f})"
        )
        logger.info(
            f"CV R2: {cv_results['cv_r2']:.6f} (+/-{cv_results['cv_r2_std']:.6f})"
        )
        logger.info(f"学習時間: {cv_results['train_time_sec']:.2f}秒")

        return result

    def generate_recommendation(self, results: Dict) -> Dict:
        """
        推奨事項を生成

        Args:
            results: 各シナリオの結果

        Returns:
            推奨事項辞書
        """
        if not results.get("baseline"):
            return {
                "message": "ベースライン評価が失敗したため、推奨事項を生成できません"
            }

        # 許容範囲（RMSE変化 < 1%）で最も多く削減できるシナリオを探す
        acceptable_scenarios = []

        for key, result in results.items():
            if key == "baseline" or not result:
                continue

            change_pct = result.get("performance_change_pct", 100)
            if abs(change_pct) < 1.0:  # 1%以内の変化
                acceptable_scenarios.append(
                    {
                        "scenario": key,
                        "n_features": result["n_features"],
                        "removed_count": len(result["removed_features"]),
                        "change_pct": change_pct,
                        "removed_features": result["removed_features"],
                    }
                )

        if acceptable_scenarios:
            # 削減数が最大のシナリオを選択
            best = max(acceptable_scenarios, key=lambda x: x["removed_count"])
            return {
                "recommended_scenario": best["scenario"],
                "recommended_features_to_remove": best["removed_features"],
                "features_count_after": best["n_features"],
                "features_removed_count": best["removed_count"],
                "performance_change_pct": best["change_pct"],
                "message": f"性能劣化が1%未満で{best['removed_count']}個の特徴量削減が可能です",
            }
        else:
            return {
                "recommended_scenario": "baseline",
                "message": "性能を維持しながら削減できる特徴量は見つかりませんでした",
            }


class LightGBMEvaluator(BaseFeatureEvaluator):
    """LightGBMモデルでの特徴量性能評価クラス"""

    def __init__(self):
        """初期化"""
        super().__init__("LightGBM")

        # LightGBMパラメータ
        self.model_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

    def evaluate_model_cv(
        self, X: pd.DataFrame, y: pd.Series, n_splits: Optional[int] = None
    ) -> Dict[str, float]:
        """
        TimeSeriesSplitでクロスバリデーション評価

        Args:
            X: 特徴量
            y: ターゲット
            n_splits: 分割数（Noneの場合はml_configから読み込み）

        Returns:
            評価指標の辞書
        """
        import lightgbm as lgb

        if n_splits is None:
            n_splits = unified_config.ml.training.cv_folds

        logger.info(f"TimeSeriesSplit使用: n_splits={n_splits}")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        rmse_scores = []
        mae_scores = []
        r2_scores = []
        train_times = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # 学習時間計測
                start_time = time.time()

                # LightGBMデータセット作成
                train_data = lgb.Dataset(X_train, label=y_train)

                # モデル学習
                model = lgb.train(
                    self.model_params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10),
                        lgb.log_evaluation(0),
                    ],
                )

                train_time = time.time() - start_time
                train_times.append(train_time)

                # 予測
                y_pred = model.predict(X_test)

                # 評価
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)

                logger.info(
                    f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, "
                    f"R2={r2:.6f}, Time={train_time:.2f}s"
                )

            except Exception as e:
                logger.warning(f"Fold {fold}でエラー: {e}")
                continue

        if not rmse_scores:
            return {}

        return {
            "cv_rmse": float(np.mean(rmse_scores)),
            "cv_rmse_std": float(np.std(rmse_scores)),
            "cv_mae": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
            "cv_r2": float(np.mean(r2_scores)),
            "cv_r2_std": float(np.std(r2_scores)),
            "train_time_sec": float(np.mean(train_times)),
        }

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """LightGBMの特徴量重要度を取得"""
        import lightgbm as lgb

        try:
            # データセット作成
            train_data = lgb.Dataset(X, label=y)

            # モデル学習
            model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10),
                    lgb.log_evaluation(0),
                ],
            )

            # 重要度取得（gain）
            importance = model.feature_importance(importance_type="gain")

            # 正規化
            if importance.sum() > 0:
                importance = importance / importance.sum()

            return dict(zip(X.columns, importance))

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}


class XGBoostEvaluator(BaseFeatureEvaluator):
    """XGBoostモデルでの特徴量性能評価クラス"""

    def __init__(self):
        """初期化"""
        super().__init__("XGBoost")

        # XGBoostパラメータ
        self.model_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "random_state": 42,
            "verbosity": 0,
        }

    def evaluate_model_cv(
        self, X: pd.DataFrame, y: pd.Series, n_splits: Optional[int] = None
    ) -> Dict[str, float]:
        """
        TimeSeriesSplitでクロスバリデーション評価

        Args:
            X: 特徴量
            y: ターゲット
            n_splits: 分割数（Noneの場合はml_configから読み込み）

        Returns:
            評価指標の辞書
        """
        import xgboost as xgb

        if n_splits is None:
            n_splits = unified_config.ml.training.cv_folds

        logger.info(f"TimeSeriesSplit使用: n_splits={n_splits}")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        rmse_scores = []
        mae_scores = []
        r2_scores = []
        train_times = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # 学習時間計測
                start_time = time.time()

                # XGBoostデータセット作成
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)

                # モデル学習
                model = xgb.train(
                    self.model_params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtrain, "train")],
                    early_stopping_rounds=10,
                    verbose_eval=False,
                )

                train_time = time.time() - start_time
                train_times.append(train_time)

                # 予測
                y_pred = model.predict(dtest)

                # 評価
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)

                logger.info(
                    f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, "
                    f"R2={r2:.6f}, Time={train_time:.2f}s"
                )

            except Exception as e:
                logger.warning(f"Fold {fold}でエラー: {e}")
                continue

        if not rmse_scores:
            return {}

        return {
            "cv_rmse": float(np.mean(rmse_scores)),
            "cv_rmse_std": float(np.std(rmse_scores)),
            "cv_mae": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
            "cv_r2": float(np.mean(r2_scores)),
            "cv_r2_std": float(np.std(r2_scores)),
            "train_time_sec": float(np.mean(train_times)),
        }

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """XGBoostの特徴量重要度を取得"""
        import xgboost as xgb

        try:
            # データセット作成
            dtrain = xgb.DMatrix(X, label=y)

            # モデル学習
            model = xgb.train(
                self.model_params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )

            # 重要度取得（gain）
            importance_dict = model.get_score(importance_type="gain")

            # 全特徴量に対して重要度を設定（未使用は0）
            result = {col: 0.0 for col in X.columns}
            result.update(importance_dict)

            # 正規化
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}

            return result

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}


class MultiModelFeatureEvaluator:
    """複数モデルでの特徴量評価を統合管理するクラス"""

    def __init__(self, models: List[str]):
        """
        初期化

        Args:
            models: 評価するモデルのリスト ['lightgbm', 'xgboost']
        """
        self.models = models
        self.evaluators = {}
        self.all_results = {}

        # 評価器を初期化
        if "lightgbm" in models:
            self.evaluators["lightgbm"] = LightGBMEvaluator()
        if "xgboost" in models:
            self.evaluators["xgboost"] = XGBoostEvaluator()

    def run_evaluation(self, symbol: str = "BTC/USDT:USDT", limit: int = 2000) -> Dict:
        """
        全モデルで評価を実行

        Args:
            symbol: 分析対象シンボル
            limit: データ取得件数

        Returns:
            全モデルの評価結果
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("全モデル特徴量性能評価開始")
        logger.info(f"評価モデル: {', '.join([m.upper() for m in self.models])}")
        logger.info("=" * 80)

        # 共通データを1回だけ取得
        logger.info("共通データ取得中...")
        first_evaluator = list(self.evaluators.values())[0]
        ohlcv_df, fr_df, oi_df = first_evaluator.fetch_data(symbol, limit)

        if ohlcv_df.empty:
            logger.error("データが取得できませんでした")
            return {}

        # 特徴量計算（1回のみ）
        features_df = first_evaluator.calculate_features(ohlcv_df, fr_df, oi_df)
        target = first_evaluator.create_target(features_df, periods=1)

        # closeを除外
        feature_cols = [col for col in features_df.columns if col != "close"]
        X = features_df[feature_cols]

        # NaN除去
        combined_df = pd.concat([X, target.rename("target")], axis=1).dropna()
        X = combined_df[feature_cols]
        y = combined_df["target"]

        logger.info(f"\n分析対象サンプル数: {len(X)}行")
        logger.info(f"全特徴量数: {len(X.columns)}個")

        # 統合スコア読み込み
        unified_scores = first_evaluator.load_unified_scores()

        # 各モデルで評価実行
        for model_name, evaluator in self.evaluators.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"{model_name.upper()}モデル評価開始")
            logger.info(f"{'='*80}")

            try:
                model_results = self._run_model_scenarios(
                    evaluator, X, y, unified_scores
                )
                self.all_results[model_name] = {
                    "evaluation_date": datetime.now().isoformat(),
                    "model_name": model_name,
                    "data_samples": len(X),
                    "symbol": symbol,
                    "target": "return_1h",
                    "model_params": evaluator.model_params,
                    "scenarios": model_results["scenarios"],
                    "recommendation": model_results["recommendation"],
                }

                # 個別結果を保存
                self._save_individual_results(model_name, self.all_results[model_name])

            except Exception as e:
                logger.error(f"{model_name}評価でエラー: {e}")
                import traceback

                traceback.print_exc()
                continue

        # 統合結果を保存
        self._save_integrated_results()

        # 統合サマリーを出力
        self._print_integrated_summary()

        elapsed_time = time.time() - start_time
        logger.info(f"\n全評価完了（処理時間: {elapsed_time:.2f}秒）")

        return self.all_results

    def _run_model_scenarios(
        self,
        evaluator: BaseFeatureEvaluator,
        X: pd.DataFrame,
        y: pd.Series,
        unified_scores: Dict,
    ) -> Dict:
        """
        1つのモデルで全シナリオを実行

        Args:
            evaluator: 評価器
            X: 特徴量
            y: ターゲット
            unified_scores: 統合スコア

        Returns:
            シナリオ結果
        """
        all_features = list(X.columns)
        results = {}

        # ベースライン（全特徴量）
        results["baseline"] = evaluator.run_scenario(
            "ベースライン (93特徴量すべて)", X, y, all_features
        )

        # シナリオ2: 下位10%削除
        to_remove_10, to_keep_10 = evaluator.select_features_by_score(
            all_features, unified_scores, 0.10
        )
        results["scenario_remove_10pct"] = evaluator.run_scenario(
            "シナリオ2: 統合スコア下位10%削除", X, y, to_keep_10, to_remove_10
        )

        # シナリオ3: 下位20%削除
        to_remove_20, to_keep_20 = evaluator.select_features_by_score(
            all_features, unified_scores, 0.20
        )
        results["scenario_remove_20pct"] = evaluator.run_scenario(
            "シナリオ3: 統合スコア下位20%削除", X, y, to_keep_20, to_remove_20
        )

        # シナリオ4: 下位30%削除
        to_remove_30, to_keep_30 = evaluator.select_features_by_score(
            all_features, unified_scores, 0.30
        )
        results["scenario_remove_30pct"] = evaluator.run_scenario(
            "シナリオ4: 統合スコア下位30%削除", X, y, to_keep_30, to_remove_30
        )

        # シナリオ5: モデル固有の特徴量重要度ベース
        if results["baseline"]:
            model_importance = evaluator.get_feature_importance(X, y)
            sorted_importance = sorted(model_importance.items(), key=lambda x: x[1])
            n_remove = max(1, int(len(sorted_importance) * 0.20))
            to_remove_model = [feat for feat, _ in sorted_importance[:n_remove]]
            to_keep_model = [
                feat for feat in all_features if feat not in to_remove_model
            ]

            results[f"scenario_{evaluator.model_name.lower()}_importance"] = (
                evaluator.run_scenario(
                    f"シナリオ5: {evaluator.model_name}重要度下位20%削除",
                    X,
                    y,
                    to_keep_model,
                    to_remove_model,
                )
            )

        # 性能変化を計算
        if results["baseline"]:
            baseline_rmse = results["baseline"]["cv_rmse"]
            for key in results:
                if key != "baseline" and results[key]:
                    scenario_rmse = results[key]["cv_rmse"]
                    change_pct = ((scenario_rmse - baseline_rmse) / baseline_rmse) * 100
                    results[key]["performance_change_pct"] = float(change_pct)

        # 推奨事項生成
        recommendation = evaluator.generate_recommendation(results)

        return {"scenarios": results, "recommendation": recommendation}

    def _save_individual_results(self, model_name: str, results: Dict):
        """
        個別モデルの結果を保存

        Args:
            model_name: モデル名
            results: 評価結果
        """
        try:
            # results/feature_analysisディレクトリのパス（フラット構造）
            output_dir = (
                Path(__file__).parent.parent.parent / "results" / "feature_analysis"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # JSON保存
            json_path = output_dir / f"{model_name}_feature_performance_evaluation.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[{model_name.upper()}] JSON保存完了: {json_path}")

            # CSV保存
            csv_path = output_dir / f"{model_name}_performance_comparison.csv"
            scenarios_data = []
            for key, scenario in results.get("scenarios", {}).items():
                if scenario:
                    row = {
                        "scenario": key,
                        "n_features": scenario.get("n_features"),
                        "cv_rmse": scenario.get("cv_rmse"),
                        "cv_mae": scenario.get("cv_mae"),
                        "cv_r2": scenario.get("cv_r2"),
                        "train_time_sec": scenario.get("train_time_sec"),
                        "performance_change_pct": scenario.get(
                            "performance_change_pct", 0.0
                        ),
                        "removed_count": len(scenario.get("removed_features", [])),
                    }
                    scenarios_data.append(row)

            if scenarios_data:
                df = pd.DataFrame(scenarios_data)
                df.to_csv(csv_path, index=False)
                logger.info(f"[{model_name.upper()}] CSV保存完了: {csv_path}")

        except Exception as e:
            logger.error(f"[{model_name}] 結果保存エラー: {e}")

    def _save_integrated_results(self):
        """統合結果を保存"""
        try:
            # results/feature_analysisディレクトリのパス（フラット構造）
            output_dir = (
                Path(__file__).parent.parent.parent / "results" / "feature_analysis"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # 統合JSON保存
            integrated_json = {
                "evaluation_date": datetime.now().isoformat(),
                "evaluated_models": list(self.all_results.keys()),
                "models_results": self.all_results,
            }

            json_path = output_dir / "all_models_feature_performance_evaluation.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(integrated_json, f, indent=2, ensure_ascii=False)
            logger.info(f"統合JSON保存完了: {json_path}")

            # モデル間性能比較CSV
            comparison_data = []
            for model_name, model_result in self.all_results.items():
                for scenario_key, scenario in model_result.get("scenarios", {}).items():
                    if scenario:
                        row = {
                            "model": model_name.upper(),
                            "scenario": scenario_key,
                            "n_features": scenario.get("n_features"),
                            "cv_rmse": scenario.get("cv_rmse"),
                            "cv_mae": scenario.get("cv_mae"),
                            "cv_r2": scenario.get("cv_r2"),
                            "train_time_sec": scenario.get("train_time_sec"),
                            "performance_change_pct": scenario.get(
                                "performance_change_pct", 0.0
                            ),
                            "removed_count": len(scenario.get("removed_features", [])),
                        }
                        comparison_data.append(row)

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                csv_path = output_dir / "all_models_performance_comparison.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"統合CSV保存完了: {csv_path}")

        except Exception as e:
            logger.error(f"統合結果保存エラー: {e}")

    def _print_integrated_summary(self):
        """統合結果サマリーをコンソール出力"""
        print("\n" + "=" * 80)
        print("全モデル特徴量性能評価結果")
        print("=" * 80)

        if not self.all_results:
            print("評価結果がありません")
            return

        # 各モデルのベースライン性能を比較
        print("\n" + "-" * 80)
        print("【モデル別ベースライン性能比較】")
        print("-" * 80)
        print(
            f"{'モデル':<15} {'RMSE':<12} {'MAE':<12} {'R2':<10} {'学習時間(秒)':<15}"
        )
        print("-" * 80)

        for model_name, result in self.all_results.items():
            baseline = result.get("scenarios", {}).get("baseline", {})
            if baseline:
                print(
                    f"{model_name.upper():<15} {baseline['cv_rmse']:<12.6f} "
                    f"{baseline['cv_mae']:<12.6f} {baseline['cv_r2']:<10.4f} "
                    f"{baseline['train_time_sec']:<15.2f}"
                )

        # 各モデルの推奨事項を比較
        print("\n" + "-" * 80)
        print("【モデル別推奨事項】")
        print("-" * 80)

        best_reduction = None
        best_model = None
        best_scenario = None

        for model_name, result in self.all_results.items():
            recommendation = result.get("recommendation", {})
            print(f"\n[{model_name.upper()}]")
            print(recommendation.get("message", "推奨事項なし"))

            if "recommended_features_to_remove" in recommendation:
                removed_count = recommendation.get("features_removed_count", 0)
                if best_reduction is None or removed_count > best_reduction:
                    best_reduction = removed_count
                    best_model = model_name
                    best_scenario = recommendation

        # 総合推奨
        print("\n" + "-" * 80)
        print("【総合推奨事項】")
        print("-" * 80)

        if best_model and best_scenario:
            print(
                f"最も効果的な削減: {best_model.upper()}モデルで"
                f"{best_reduction}個の特徴量削減が可能"
            )
            print(f"性能変化: {best_scenario.get('performance_change_pct', 0):.2f}%")
            print(f"削減後の特徴量数: {best_scenario.get('features_count_after')}個")

            removed_features = best_scenario.get("recommended_features_to_remove", [])
            if removed_features:
                print(f"\n削除推奨特徴量（{len(removed_features)}個）:")
                for i, feat in enumerate(removed_features, 1):
                    print(f"  {i:2}. {feat}")
        else:
            print(
                "全モデルで性能を維持しながら" "削減できる特徴量は見つかりませんでした"
            )

        print("\n" + "=" * 80 + "\n")


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="全モデルでの特徴量性能評価スクリプト")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lightgbm", "xgboost", "all"],
        default=["all"],
        help="評価するモデルを指定 (デフォルト: all)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="分析対象シンボル (デフォルト: BTC/USDT:USDT)",
    )
    parser.add_argument(
        "--limit", type=int, default=2000, help="データ取得件数 (デフォルト: 2000)"
    )

    return parser.parse_args()


def main():
    """メイン実行関数"""
    try:
        # コマンドライン引数をパース
        args = parse_arguments()

        # モデルリストを決定
        if "all" in args.models:
            models = ["lightgbm", "xgboost"]
        else:
            models = args.models

        logger.info(f"評価対象モデル: {', '.join([m.upper() for m in models])}")

        # 評価実行
        evaluator = MultiModelFeatureEvaluator(models)
        evaluator.run_evaluation(symbol=args.symbol, limit=args.limit)

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
