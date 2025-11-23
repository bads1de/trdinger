"""
ML統合パイプライン: 最適化・学習・スタッキング・評価・分析

このスクリプト一本で以下のフローを実行します:
1. Optunaによるハイパーパラメータ＆ラベル生成パラメータの最適化 (LightGBM & XGBoost & CatBoost)
2. ベストパラメータによるモデルの再学習
3. スタッキング（アンサンブル）モデルの構築
4. 閾値調整シミュレーションとモデル比較評価
5. 特徴量重要度分析

使用例:
    # 全フロー実行 (試行回数20)
    conda run -n trading python backend/scripts/ml_optimization/run_ml_pipeline.py --n-trials 20

    # 評価と分析のみ (既存の最適化結果を使用)
    conda run -n trading python backend/scripts/ml_optimization/run_ml_pipeline.py --skip-optimize
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, List

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
import catboost as cb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windowsでの文字化け対策
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    plt.rcParams['font.family'] = 'Meiryo'

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.label_cache import LabelCache
from app.services.ml.stacking_service import StackingService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MLPipeline:
    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h", n_trials: int = 20):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        
        self.db = SessionLocal()
        self.evaluator = CommonFeatureEvaluator()
        self.feature_service = FeatureEngineeringService()
        
        # 結果保存ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "ml_pipeline"
            / f"run_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ML Pipeline初期化: {symbol} {timeframe}")
        logger.info(f"結果保存先: {self.results_dir}")

    def get_latest_results_dir(self) -> Path:
        """最新のパイプライン実行結果ディレクトリを取得"""
        base_dir = Path(__file__).parent.parent.parent / "results" / "ml_pipeline"
        list_of_dirs = glob.glob(str(base_dir / "run_*"))
        
        # best_params.jsonが存在するディレクトリのみをフィルタリング
        valid_dirs = [d for d in list_of_dirs if (Path(d) / "best_params.json").exists()]
        
        if not valid_dirs:
            # 互換性のため integrated_optimization も探す
            base_dir_old = Path(__file__).parent.parent.parent / "results" / "integrated_optimization"
            list_of_dirs = glob.glob(str(base_dir_old / "run_*"))
            valid_dirs = [d for d in list_of_dirs if (Path(d) / "best_params.json").exists()]
            
            if not valid_dirs:
                raise FileNotFoundError("過去の有効な実行結果（best_params.jsonを含む）が見つかりません。")
        
        latest_dir = max(valid_dirs, key=os.path.getctime)
        return Path(latest_dir)

    def prepare_data(self, limit: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データ取得と特徴量生成"""
        logger.info("データを準備中...")
        
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=limit
        )
        
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=data.ohlcv,
            funding_rate_data=data.fr,
            open_interest_data=data.oi,
        )
        
        feature_cols = [
            col for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        # NaN補完
        X = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())
        
        return X, data.ohlcv

    def optimize(self, X: pd.DataFrame, ohlcv_df: pd.DataFrame):
        """Optunaによる最適化 (LightGBM & XGBoost & CatBoost)"""
        logger.info("=" * 60)
        logger.info("ハイパーパラメータ最適化を開始")
        logger.info("=" * 60)

        label_cache = LabelCache(ohlcv_df)

        def objective(trial):
            # モデル選択
            model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost", "catboost"])

            # ラベル生成パラメータ
            horizon_n = trial.suggest_int("horizon_n", 4, 16, step=2)
            threshold_method = trial.suggest_categorical(
                "threshold_method", ["QUANTILE", "KBINS_DISCRETIZER", "DYNAMIC_VOLATILITY"]
            )

            if threshold_method == "QUANTILE":
                threshold = trial.suggest_float("quantile_threshold", 0.25, 0.40)
            elif threshold_method == "KBINS_DISCRETIZER":
                threshold = trial.suggest_float("kbins_threshold", 0.001, 0.005)
            else:  # DYNAMIC_VOLATILITY
                threshold = trial.suggest_float("volatility_threshold", 0.5, 2.0)

            # ラベル取得
            try:
                labels = label_cache.get_labels(
                    horizon_n=horizon_n,
                    threshold_method=threshold_method,
                    threshold=threshold,
                    timeframe=self.timeframe,
                    price_column="close",
                )
            except Exception:
                raise optuna.exceptions.TrialPruned()

            # アライメント
            common_index = X.index.intersection(labels.index)
            X_aligned = X.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            valid_idx = ~labels_aligned.isna()
            X_clean = X_aligned.loc[valid_idx]
            # TREND=1, RANGE=0
            y = labels_aligned.loc[valid_idx].map({"DOWN": 1, "RANGE": 0, "UP": 1})

            if len(y) < 100:
                raise optuna.exceptions.TrialPruned()

            # 時系列分割 (Train 60%, Val 20%, Test 20%)
            n_total = len(X_clean)
            test_start = int(n_total * 0.8)
            val_start = int(n_total * 0.6)
            
            X_train = X_clean.iloc[:val_start]
            y_train = y.iloc[:val_start]
            X_val = X_clean.iloc[val_start:test_start]
            y_val = y.iloc[val_start:test_start]

            # モデル学習
            if model_type == "lightgbm":
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "verbosity": -1,
                    "random_state": 42,
                    "class_weight": "balanced",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            
            elif model_type == "xgboost":
                neg_count = len(y_train) - sum(y_train)
                pos_count = sum(y_train)
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "random_state": 42,
                    "scale_pos_weight": scale_pos_weight,
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
            else: # catboost
                neg_count = len(y_train) - sum(y_train)
                pos_count = sum(y_train)
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    "od_type": "Iter",
                    "od_wait": 50,
                    "verbose": 0,
                    "random_seed": 42,
                    "scale_pos_weight": scale_pos_weight,
                    "allow_writing_files": False,
                }
                model = cb.CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            # 評価 (Custom Score: Trend Recall重視)
            y_pred = model.predict(X_val)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            
            # 0: RANGE, 1: TREND
            trend_rec = report['1']['recall']
            trend_prec = report['1']['precision']
            range_rec = report['0']['recall']
            
            score = (trend_rec * 1.5) + trend_prec + (range_rec * 0.5)
            
            logger.info(f"Trial {trial.number} ({model_type}): Score={score:.4f}, TrendRec={trend_rec:.2f}, RangeRec={range_rec:.2f}")
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        # ベストパラメータ保存
        best_params = study.best_params
        best_params["use_class_weight"] = True # 強制フラグ
        
        results = {
            "best_value": study.best_value,
            "best_params": best_params,
            "trials": len(study.trials)
        }
        
        with open(self.results_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"最適化完了: Best Score={study.best_value:.4f}")
        return best_params

    def train_final_models(self, X: pd.DataFrame, ohlcv_df: pd.DataFrame, best_params: Dict[str, Any]):
        """ベストパラメータでLGBM, XGBoost, CatBoostの全モデルを学習"""
        logger.info("=" * 60)
        logger.info("最終モデル学習 & スタッキング準備")
        logger.info("=" * 60)

        # ラベル生成
        label_cache = LabelCache(ohlcv_df)
        threshold_key = [k for k in best_params if "threshold" in k and k != "threshold_method"][0]
        
        labels = label_cache.get_labels(
            horizon_n=best_params["horizon_n"],
            threshold_method=best_params["threshold_method"],
            threshold=best_params[threshold_key],
            timeframe=self.timeframe,
            price_column="close",
        )

        # データセット作成
        common_index = X.index.intersection(labels.index)
        X_aligned = X.loc[common_index]
        y = labels.loc[common_index].map({"DOWN": 1, "RANGE": 0, "UP": 1})
        
        valid_idx = ~y.isna()
        X_clean = X_aligned.loc[valid_idx]
        y = y.loc[valid_idx]

        # 時系列分割 (Train+Val 80%, Test 20%)
        n_total = len(X_clean)
        test_start = int(n_total * 0.8)
        
        X_full_train = X_clean.iloc[:test_start]
        y_full_train = y.iloc[:test_start]
        X_test = X_clean.iloc[test_start:]
        y_test = y.iloc[test_start:]
        
        # スタッキング学習用にTrainをさらに分割 (TrainBase 80%, MetaVal 20%)
        # これによりメタモデルは未知のデータに対する予測値で学習できる
        n_train = len(X_full_train)
        meta_val_start = int(n_train * 0.8)
        
        X_train_base = X_full_train.iloc[:meta_val_start]
        y_train_base = y_full_train.iloc[:meta_val_start]
        X_val_meta = X_full_train.iloc[meta_val_start:]
        y_val_meta = y_full_train.iloc[meta_val_start:]
        
        logger.info(f"Train Base: {len(X_train_base)}, Meta Val: {len(X_val_meta)}, Test: {len(X_test)}")

        # クラスウェイト計算
        neg = len(y_train_base) - sum(y_train_base)
        pos = sum(y_train_base)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        # --- LightGBM ---
        lgb_params = {k: v for k, v in best_params.items() if k in [
            "n_estimators", "learning_rate", "max_depth", "num_leaves", 
            "min_child_samples", "subsample", "colsample_bytree"
        ]}
        lgb_params.update({
            "objective": "binary", "metric": "binary_logloss", 
            "verbosity": -1, "random_state": 42, "class_weight": "balanced"
        })
        
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_train_base, y_train_base)
        joblib.dump(model_lgb, self.results_dir / "model_lgb.joblib")

        # --- XGBoost ---
        xgb_params = {k: v for k, v in best_params.items() if k in [
            "n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"
        ]}
        if "min_child_weight" not in xgb_params and "min_child_samples" in best_params:
             xgb_params["min_child_weight"] = int(best_params["min_child_samples"] / 10)

        xgb_params.update({
            "objective": "binary:logistic", "eval_metric": "logloss",
            "verbosity": 0, "random_state": 42, "scale_pos_weight": scale_pos_weight
        })
        
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_train_base, y_train_base)
        joblib.dump(model_xgb, self.results_dir / "model_xgb.joblib")
        
        # --- CatBoost ---
        cat_params = {k: v for k, v in best_params.items() if k in [
            "iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature"
        ]}
        if not cat_params:
             cat_params = {
                 "iterations": best_params.get("n_estimators", 100),
                 "learning_rate": best_params.get("learning_rate", 0.05),
                 "depth": min(best_params.get("max_depth", 6), 10),
             }

        cat_params.update({
            "od_type": "Iter", "od_wait": 50, "verbose": 0, "random_seed": 42, 
            "scale_pos_weight": scale_pos_weight, "allow_writing_files": False
        })
        
        model_cat = cb.CatBoostClassifier(**cat_params)
        model_cat.fit(X_train_base, y_train_base, eval_set=[(X_val_meta, y_val_meta)], early_stopping_rounds=50, verbose=False)
        joblib.dump(model_cat, self.results_dir / "model_cat.joblib")

        # --- Stacking Meta Model (Ridge NNLS) ---
        logger.info("メタモデル (Ridge NNLS) を学習中...")
        
        prob_lgb_meta = model_lgb.predict_proba(X_val_meta)[:, 1]
        prob_xgb_meta = model_xgb.predict_proba(X_val_meta)[:, 1]
        prob_cat_meta = model_cat.predict_proba(X_val_meta)[:, 1]
        
        X_meta_train = pd.DataFrame({
            'LightGBM': prob_lgb_meta,
            'XGBoost': prob_xgb_meta,
            'CatBoost': prob_cat_meta
        })
        
        stacking_service = StackingService()
        stacking_service.train(X_meta_train, y_val_meta.values)
        
        weights = stacking_service.get_weights()
        logger.info(f"学習された重み: {weights}")
        joblib.dump(stacking_service, self.results_dir / "stacking_service.joblib")

        return model_lgb, model_xgb, model_cat, stacking_service, X_test, y_test

    def evaluate_and_stacking(self, model_lgb, model_xgb, model_cat, stacking_service, X_test, y_test):
        """閾値シミュレーションとスタッキング評価"""
        logger.info("=" * 60)
        logger.info("モデル評価 & スタッキング")
        logger.info("=" * 60)

        # 予測確率
        prob_lgb = model_lgb.predict_proba(X_test)[:, 1]
        prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
        prob_cat = model_cat.predict_proba(X_test)[:, 1]
        
        # スタッキング (Ridge NNLS)
        X_meta_test = pd.DataFrame({
            'LightGBM': prob_lgb,
            'XGBoost': prob_xgb,
            'CatBoost': prob_cat
        })
        prob_stack = stacking_service.predict(X_meta_test)
        
        # Soft Voting
        prob_avg = (prob_lgb + prob_xgb + prob_cat) / 3

        models = {
            "LightGBM": prob_lgb,
            "XGBoost": prob_xgb,
            "CatBoost": prob_cat,
            "Stacking (Avg)": prob_avg,
            "Stacking (Ridge)": prob_stack
        }

        best_result = None
        best_score = -1

        for name, probs in models.items():
            logger.info(f"\n--- {name} ---")
            logger.info(f"{ 'Threshold':<10} | {'RangeRec':<10} | {'TrendPrec':<10} | {'TrendRec':<10} | {'Acc':<10}")
            logger.info("-" * 60)

            for th in np.arange(0.5, 0.96, 0.05):
                y_pred = (probs >= th).astype(int)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                range_rec = report['0']['recall']
                trend_prec = report['1']['precision']
                trend_rec = report['1']['recall']
                acc = report['accuracy']
                
                # 評価スコア (Range回避率 > 60% を満たしつつ、Trend確度を最大化)
                score = trend_prec if range_rec > 0.6 else 0
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "model": name, "threshold": th, 
                        "range_rec": range_rec, "trend_prec": trend_prec, "trend_rec": trend_rec
                    }

                logger.info(f"{th:.2f}{' (Def)' if th==0.5 else '      ':<6} | {range_rec:.4f}     | {trend_prec:.4f}     | {trend_rec:.4f}     | {acc:.4f}")

        logger.info("\n" + "=" * 60)
        if best_result:
            logger.info(f"🏆 ベスト設定: {best_result['model']} @ Threshold {best_result['threshold']:.2f}")
            logger.info(f"   Range回避率: {best_result['range_rec']:.1%} (目標: 60%超)")
            logger.info(f"   Trend確度:   {best_result['trend_prec']:.1%} (ここを最大化)")
            logger.info(f"   Trend検出率: {best_result['trend_rec']:.1%}")
        else:
            logger.warning("目標基準（Range回避 > 60%）を満たす設定が見つかりませんでした。")

    def analyze_feature_importance(self, model_lgb, X_test):
        """LightGBMの特徴量重要度分析"""
        logger.info("\n" + "=" * 60)
        logger.info("特徴量重要度分析 (LightGBM)")
        logger.info("=" * 60)
        
        importances = model_lgb.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(feature_imp.head(20).to_string(index=False))
        
        # 保存
        feature_imp.to_csv(self.results_dir / "feature_importance.csv", index=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_imp.head(30))
        plt.title("Feature Importance (LightGBM)")
        plt.tight_layout()
        plt.savefig(self.results_dir / "feature_importance.png")
        logger.info(f"分析結果を保存: {self.results_dir}")

    def run(self, skip_optimize: bool = False):
        # 1. データ準備
        X, ohlcv = self.prepare_data()

        # 2. 最適化 or ロード
        if not skip_optimize:
            best_params = self.optimize(X, ohlcv)
        else:
            latest_dir = self.get_latest_results_dir()
            logger.info(f"既存の結果をロード: {latest_dir}")
            with open(latest_dir / "best_params.json", "r", encoding="utf-8") as f:
                best_params = json.load(f)[ "best_params"]

        # 3. 最終モデル学習 (LGBM & XGB & Stacking)
        model_lgb, model_xgb, model_cat, stacking_service, X_test, y_test = self.train_final_models(X, ohlcv, best_params)

        # 4. スタッキング評価
        self.evaluate_and_stacking(model_lgb, model_xgb, model_cat, stacking_service, X_test, y_test)

        # 5. 特徴量分析
        self.analyze_feature_importance(model_lgb, X_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--skip-optimize", action="store_true", help="最適化をスキップして学習・評価のみ実行")
    args = parser.parse_args()

    pipeline = MLPipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_trials=args.n_trials
    )
    pipeline.run(skip_optimize=args.skip_optimize)


if __name__ == "__main__":
    main()