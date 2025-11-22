"""
Phase 2: 2クラス分類（UP/DOWN）による高精度モデル構築

Phase 1.5の成果（統合最適化）を継承しつつ、RANGEクラスを除外して
純粋な方向予測（UP vs DOWN）に特化することで、精度70%超を目指します。

主な変更点:
1. ラベル生成後にRANGEクラスを除外（または統合）
2. 2クラス分類（binary）への変更
3. 評価指標の変更（Accuracy, Precision, Recall, F1, ROC-AUC）
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (  # type: ignore  # noqa: E501
    FeatureEngineeringService,
)
from app.services.ml.label_cache import LabelCache  # type: ignore
from database.connection import SessionLocal  # type: ignore
from database.repositories.ohlcv_repository import OHLCVRepository  # type: ignore
from scripts.feature_evaluation.common_feature_evaluator import (  # type: ignore
    CommonFeatureEvaluator,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase2_optimization.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class Phase2Optimizer:
    """2クラス分類特化型最適化"""

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
        mode: str = "exclude_range",  # exclude_range or trend_vs_range
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.mode = mode

        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.feature_service = FeatureEngineeringService()
        self.evaluator = CommonFeatureEvaluator()

        # 結果保存ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "phase2_2class"
            / f"run_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 2最適化初期化: {symbol} {timeframe}")
        logger.info(f"モード: {mode}")
        logger.info(f"結果保存先: {self.results_dir}")

    def prepare_data(self):
        """データ準備"""
        logger.info("データ準備開始")

        # OHLCVデータ取得
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=10000
        )

        if data.ohlcv.empty:
            raise ValueError("OHLCVデータが空です")

        logger.info(f"OHLCVデータ取得: {len(data.ohlcv)}行")

        # 特徴量生成
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=data.ohlcv,
            funding_rate_data=data.fr,
            open_interest_data=data.oi,
        )

        logger.info(f"特徴量生成完了: {len(features_df.columns)}個")

        # OHLCV列を除外
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        X = features_df[feature_cols].copy()

        # NaN補完
        if X.isna().any().any():
            X = X.fillna(X.median())

        logger.info(f"特徴量準備完了: {X.shape}")

        return X, data.ohlcv

    def run_optimization(self):
        """最適化実行"""
        start_time = time.time()

        try:
            # データ準備
            X, ohlcv_df = self.prepare_data()
            self.X = X
            self.ohlcv_df = ohlcv_df

            # LabelCacheを初期化
            label_cache = LabelCache(ohlcv_df)
            self.label_cache = label_cache

            def objective(trial):
                # ========== モデルタイプの選択 ==========
                model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost"])

                # ========== ラベル生成パラメータ ==========
                # Phase 1.5の知見を活かして探索範囲を最適化
                horizon_n = trial.suggest_int("horizon_n", 4, 16, step=2)
                threshold_method = trial.suggest_categorical(
                    "threshold_method",
                    ["DYNAMIC_VOLATILITY", "QUANTILE"],  # KBINSは除外（性能低かったため）
                )

                if threshold_method == "QUANTILE":
                    threshold = trial.suggest_float("quantile_threshold", 0.25, 0.40)
                else:  # DYNAMIC_VOLATILITY
                    threshold = trial.suggest_float("volatility_threshold", 0.5, 2.0)

                # ========== ラベル取得 ==========
                try:
                    labels = label_cache.get_labels(
                        horizon_n=horizon_n,
                        threshold_method=threshold_method,
                        threshold=threshold,
                        timeframe=self.timeframe,
                        price_column="close",
                    )
                except Exception as e:
                    logger.warning(f"ラベル生成エラー: {e}")
                    raise optuna.exceptions.TrialPruned()

                # データのアライメント
                common_index = X.index.intersection(labels.index)
                X_aligned = X.loc[common_index]
                labels_aligned = labels.loc[common_index]

                # NaN除去
                valid_idx = ~labels_aligned.isna()
                X_clean = X_aligned.loc[valid_idx]
                labels_clean = labels_aligned.loc[valid_idx]

                # ========== 2クラス変換 ==========
                if self.mode == "exclude_range":
                    # RANGEを除外してUP(1) vs DOWN(0)
                    mask = labels_clean != "RANGE"
                    X_final = X_clean.loc[mask]
                    y_final = labels_clean.loc[mask].map({"UP": 1, "DOWN": 0})
                else:
                    # TREND(1) vs RANGE(0)
                    X_final = X_clean
                    y_final = labels_clean.map(
                        {"UP": 1, "DOWN": 1, "RANGE": 0}
                    )
                
                # データ数チェック
                if len(y_final) < 100:
                    logger.warning(f"データ数不足: {len(y_final)}行")
                    raise optuna.exceptions.TrialPruned()

                # データ分割 (Train:Val:Test = 60:20:20)
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
                )

                # ========== クラス不均衡対策 ==========
                use_class_weight = trial.suggest_categorical("use_class_weight", [True, False])
                use_smote = trial.suggest_categorical("use_smote", [True, False])

                if use_smote:
                    try:
                        from imblearn.over_sampling import SMOTE
                        smote = SMOTE(random_state=42)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                    except Exception:
                        pass

                # ========== モデル学習 ==========
                try:
                    if model_type == "lightgbm":
                        params = {
                            "objective": "binary",
                            "metric": "binary_logloss",
                            "verbosity": -1,
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                            "max_depth": trial.suggest_int("max_depth", 3, 15),
                            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                            "random_state": 42,
                        }
                        if use_class_weight:
                            params["class_weight"] = "balanced"

                        model = lgb.LGBMClassifier(**params)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)]
                        )
                    else:  # XGBoost
                        params = {
                            "objective": "binary:logistic",
                            "eval_metric": "logloss",
                            "verbosity": 0,
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                            "max_depth": trial.suggest_int("max_depth", 3, 15),
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                            "gamma": trial.suggest_float("gamma", 0, 5),
                            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                            "random_state": 42,
                        }
                        if use_class_weight:
                            # 簡易的な計算: 負例数/正例数
                            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
                            params["scale_pos_weight"] = scale_pos_weight

                        model = xgb.XGBClassifier(**params)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )

                except Exception as e:
                    logger.warning(f"学習エラー: {e}")
                    raise optuna.exceptions.TrialPruned()

                # ========== 評価 ==========
                y_val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred)  # binaryなのでaverage指定なし

                # Test精度（過学習チェック）
                y_test_pred = model.predict(X_test)
                test_f1 = f1_score(y_test, y_test_pred)
                
                logger.info(f"Trial {trial.number} ({model_type}): Val_F1={val_f1:.4f}, Test_F1={test_f1:.4f}")
                
                return val_f1

            # 最適化実行
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
            study = optuna.create_study(direction="maximize", pruner=pruner)
            
            logger.info("=" * 80)
            logger.info(f"Phase 2 最適化開始 ({self.mode})")
            logger.info("=" * 80)
            
            study.optimize(objective, n_trials=self.n_trials)
            
            # 結果保存
            self._save_results(study, label_cache.get_stats(), time.time() - start_time)

        except Exception as e:
            logger.error(f"エラー発生: {e}", exc_info=True)
            raise
        finally:
            self.db.close()
            self.evaluator.close()

    def _save_results(self, study, cache_stats, elapsed_time):
        """結果保存"""
        best_params = study.best_trial.params
        final_metrics = self._evaluate_best_model(best_params)
        
        results = {
            "best_value": study.best_value,
            "best_params": best_params,
            "final_metrics": final_metrics,
            "elapsed_time": elapsed_time
        }
        
        # JSON保存
        with open(self.results_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # レポート保存
        with open(self.results_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(f"# Phase 2 最適化レポート ({self.mode})\n\n")
            f.write(f"**生成日時**: {datetime.now()}\n\n")
            
            f.write("## 最終評価結果 (Test)\n\n")
            f.write(f"- **Accuracy**: {final_metrics['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {final_metrics['precision']:.4f}\n")
            f.write(f"- **Recall**: {final_metrics['recall']:.4f}\n")
            f.write(f"- **F1 Score**: {final_metrics['f1']:.4f}\n")
            f.write(f"- **ROC-AUC**: {final_metrics['roc_auc']:.4f}\n\n")
            
            f.write("## 最適パラメータ\n\n")
            f.write("```json\n")
            f.write(json.dumps(best_params, indent=2, ensure_ascii=False))
            f.write("\n```\n")

        logger.info(f"レポート保存完了: {self.results_dir}")

    def _evaluate_best_model(self, best_params):
        """ベストモデルの最終評価"""
        # LabelCache再初期化
        label_cache = LabelCache(self.ohlcv_df)
        
        # ラベル取得
        threshold_key = [k for k in best_params if "threshold" in k and k != "threshold_method"][0]
        labels = label_cache.get_labels(
            horizon_n=best_params["horizon_n"],
            threshold_method=best_params["threshold_method"],
            threshold=best_params[threshold_key],
            timeframe=self.timeframe,
            price_column="close"
        )
        
        # データ準備
        common_index = self.X.index.intersection(labels.index)
        X_aligned = self.X.loc[common_index]
        labels_aligned = labels.loc[common_index]
        
        valid_idx = ~labels_aligned.isna()
        X_clean = X_aligned.loc[valid_idx]
        labels_clean = labels_aligned.loc[valid_idx]
        
        if self.mode == "exclude_range":
            mask = labels_clean != "RANGE"
            X_final = X_clean.loc[mask]
            y_final = labels_clean.loc[mask].map({"UP": 1, "DOWN": 0})
        else:
            X_final = X_clean
            y_final = labels_clean.map({"UP": 1, "DOWN": 1, "RANGE": 0})
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
        
        # SMOTE
        if best_params.get("use_smote", False):
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception:
                pass
                
        # モデル学習
        model_type = best_params.get("model_type", "lightgbm")
        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "num_leaves": best_params["num_leaves"],
                "min_child_samples": best_params["min_child_samples"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "random_state": 42,
            }
            if best_params["use_class_weight"]:
                params["class_weight"] = "balanced"
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
        else:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "min_child_weight": best_params["min_child_weight"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "gamma": best_params["gamma"],
                "reg_alpha": best_params["reg_alpha"],
                "reg_lambda": best_params["reg_lambda"],
                "random_state": 42,
            }
            if best_params["use_class_weight"]:
                scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
                params["scale_pos_weight"] = scale_pos_weight
                
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
        # 評価
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--mode", default="exclude_range", choices=["exclude_range", "trend_vs_range"])
    args = parser.parse_args()
    
    optimizer = Phase2Optimizer(n_trials=args.n_trials, mode=args.mode)
    optimizer.run_optimization()

if __name__ == "__main__":
    main()
