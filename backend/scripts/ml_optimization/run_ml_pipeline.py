"""
ML統合パイプライン: 最適化・学習・スタッキング・評価・分析

このスクリプト一本で以下のフローを実行します:
1. Optunaによるハイパーパラメータ＆ラベル生成パラメータの最適化 (LightGBM & XGBoost & CatBoost & GRU & LSTM)
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
from sklearn.preprocessing import StandardScaler

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
from app.services.ml.models.gru_model import GRUModel
from app.services.ml.models.lstm_model import LSTMModel
from app.utils.purged_cv import PurgedKFold # Import PurgedKFold
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
        # NaN補完 (feature_engineering_serviceで処理済みだが念のため)
        X = features_df[feature_cols].copy().fillna(0)
        
        return X, data.ohlcv

    def optimize(self, X: pd.DataFrame, ohlcv_df: pd.DataFrame):
        """Optunaによる最適化 (Purged K-Fold CVを適用)"""
        logger.info("=" * 60)
        logger.info("ハイパーパラメータ最適化を開始 (Purged K-Fold CV)")
        logger.info("=" * 60)

        label_cache = LabelCache(ohlcv_df)

        def objective(trial):
            model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost", "catboost", "gru", "lstm"])
            
            # ラベル生成パラメータ (Optunaで探索)
            horizon_n = trial.suggest_int("horizon_n", 4, 16, step=2)
            threshold_method = trial.suggest_categorical(
                "threshold_method", ["QUANTILE", "KBINS_DISCRETIZER", "DYNAMIC_VOLATILITY"]
            )
            if threshold_method == "QUANTILE":
                threshold = trial.suggest_float("quantile_threshold", 0.25, 0.40)
            elif threshold_method == "KBINS_DISCRETIZER":
                threshold = trial.suggest_float("kbins_threshold", 0.001, 0.005)
            else: # DYNAMIC_VOLATILITY
                threshold = trial.suggest_float("volatility_threshold", 0.5, 2.0)

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

            common_index = X.index.intersection(labels.index)
            X_aligned = X.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            valid_idx = ~labels_aligned.isna()
            X_clean = X_aligned.loc[valid_idx]
            y = labels_aligned.loc[valid_idx].map({"DOWN": 1, "RANGE": 0, "UP": 1})

            if len(y) < 100:
                raise optuna.exceptions.TrialPruned()

            # ここからPurged K-Fold CV
            n_splits = 3 # Optunaの探索なので少なめに
            embargo_pct = 0.01 # テスト期間の1%をエンバーゴ
            
            # X.indexは観測開始時刻、t1はラベル終了時刻
            # LabelCacheからt1（ラベル終了時刻）を取得
            t1_labels = label_cache.get_t1(X_clean.index, horizon_n) # horizon_nを使って適切なt1を取得
            
            cv = PurgedKFold(n_splits=n_splits, t1=t1_labels, embargo_pct=embargo_pct)
            
            oof_preds = np.zeros(len(X_clean))
            # oof_y = np.zeros(len(X_clean)) # これだと0で埋められてしまうので、実際にはy_val_foldを使う
            fold_scores = []
            
            # スケーラーは各フォールドで再初期化
            scaler = StandardScaler()

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_clean, y)):
                if len(train_idx) == 0 or len(val_idx) == 0:
                    logger.warning(f"Fold {fold}: Skipping due to empty train or validation set after purging.")
                    continue

                X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = None # 各フォールドでモデルを再初期化

                if model_type == "lightgbm":
                    params = {
                        "objective": "binary", "metric": "binary_logloss", "verbosity": -1, "random_state": 42, "class_weight": "balanced",
                        "n_estimators": trial.suggest_int("n_estimators_lgb", 50, 500),
                        "learning_rate": trial.suggest_float("learning_rate_lgb", 0.001, 0.5, log=True),
                        "max_depth": trial.suggest_int("max_depth_lgb", 3, 15),
                        "num_leaves": trial.suggest_int("num_leaves_lgb", 20, 200),
                        "min_child_samples": trial.suggest_int("min_child_samples_lgb", 10, 100),
                        "subsample": trial.suggest_float("subsample_lgb", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree_lgb", 0.5, 1.0),
                    }
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(50, verbose=False)])
                
                elif model_type == "xgboost":
                    neg_count = len(y_train_fold) - sum(y_train_fold)
                    pos_count = sum(y_train_fold)
                    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                    params = {
                        "objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0, "random_state": 42, "scale_pos_weight": scale_pos_weight, "missing": np.inf,
                        "n_estimators": trial.suggest_int("n_estimators_xgb", 50, 500),
                        "learning_rate": trial.suggest_float("learning_rate_xgb", 0.001, 0.5, log=True),
                        "max_depth": trial.suggest_int("max_depth_xgb", 3, 15),
                        "min_child_weight": trial.suggest_int("min_child_weight_xgb", 1, 10),
                        "subsample": trial.suggest_float("subsample_xgb", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree_xgb", 0.5, 1.0),
                        "gamma": trial.suggest_float("gamma_xgb", 0, 5),
                        "reg_alpha": trial.suggest_float("reg_alpha_xgb", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda_xgb", 1e-8, 10.0, log=True),
                    }
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                    
                elif model_type == "catboost":
                    neg_count = len(y_train_fold) - sum(y_train_fold)
                    pos_count = sum(y_train_fold)
                    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                    params = {
                        "iterations": trial.suggest_int("iterations_cat", 50, 500),
                        "learning_rate": trial.suggest_float("learning_rate_cat", 0.001, 0.5, log=True),
                        "depth": trial.suggest_int("depth_cat", 3, 10),
                        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg_cat", 1e-8, 10.0, log=True),
                        "random_strength": trial.suggest_float("random_strength_cat", 1e-8, 10.0, log=True),
                        "bagging_temperature": trial.suggest_float("bagging_temperature_cat", 0.0, 1.0),
                        "od_type": "Iter", "od_wait": 50, "verbose": 0, "random_seed": 42,
                        "scale_pos_weight": scale_pos_weight, "allow_writing_files": False,
                    }
                    model = cb.CatBoostClassifier(**params)
                    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=False)
                    
                else: # gru or lstm
                    X_train_scaled = scaler.fit_transform(X_train_fold)
                    X_val_scaled = scaler.transform(X_val_fold)
                    
                    params = {
                        "input_dim": X_clean.shape[1], # X_cleanのfull features
                        "hidden_dim": trial.suggest_categorical("hidden_dim_dl", [32, 64, 128]),
                        "num_layers": trial.suggest_int("num_layers_dl", 1, 3),
                        "seq_len": trial.suggest_categorical("seq_len_dl", [12, 24, 48]),
                        "batch_size": trial.suggest_categorical("batch_size_dl", [32, 64]),
                        "epochs": 5, # Optuna探索なのでepoch少なめ
                        "learning_rate": trial.suggest_float("learning_rate_dl", 0.0001, 0.01, log=True),
                        "dropout": trial.suggest_float("dropout_dl", 0.0, 0.5),
                    }
                    
                    if model_type == "gru":
                        model = GRUModel(**params)
                    else: # lstm
                        model = LSTMModel(**params)
                        
                    model.fit(X_train_scaled, y_train_fold)
                    y_pred_fold = model.predict_proba(X_val_scaled)
                
                # 予測 (GRU/LSTM以外)
                if model_type not in ["gru", "lstm"]:
                    y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
                
                oof_preds[val_idx] = y_pred_fold # OOF予測値を記録
                
                # スコア計算
                y_pred_bin_fold = (y_pred_fold >= 0.5).astype(int)
                fold_report = classification_report(y_val_fold, y_pred_bin_fold, output_dict=True, zero_division=0)
                fold_scores.append((fold_report['1']['recall'] * 1.5) + fold_report['1']['precision'] + (fold_report['0']['recall'] * 0.5))
            
            # OOF全体の評価
            # GRU/LSTMはパディングで0.5が入るので、0.5は評価対象外とする (oof_yはX_cleanのyなので問題なし)
            y_pred_bin = (oof_preds >= 0.5).astype(int)
            oof_report = classification_report(y, y_pred_bin, output_dict=True, zero_division=0)
            
            avg_trend_rec = oof_report['1']['recall']
            avg_trend_prec = oof_report['1']['precision']
            avg_range_rec = oof_report['0']['recall']
            
            final_score = (avg_trend_rec * 1.5) + avg_trend_prec + (avg_range_rec * 0.5)
            
            logger.info(f"Trial {trial.number} ({model_type}): Final OOF Score={final_score:.4f}, TrendRec={avg_trend_rec:.2f}, RangeRec={avg_range_rec:.2f}")
            
            return final_score

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
            
        logger.info(f"最適化完了: Best OOF Score={study.best_value:.4f}")
        return best_params

    def train_final_models(self, X: pd.DataFrame, ohlcv_df: pd.DataFrame, best_params: Dict[str, Any]):
        """ベストパラメータで全モデルを学習し、OOF予測を生成"""
        logger.info("=" * 60)
        logger.info("最終モデル学習 & スタッキング準備 (Purged K-Fold OOF)")
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

        # アライメント
        common_index = X.index.intersection(labels.index)
        X_aligned = X.loc[common_index]
        labels_aligned = labels.loc[common_index]
        
        valid_idx = ~labels_aligned.isna()
        X_clean = X_aligned.loc[valid_idx]
        y = labels_aligned.loc[valid_idx].map({"DOWN": 1, "RANGE": 0, "UP": 1})

        if len(y) < 100:
            raise ValueError("データが少なすぎます。")

        # Purged K-Fold CV for OOF prediction
        n_splits_oof = 5 # OOF生成は5分割
        embargo_pct_oof = 0.01
        t1_labels = label_cache.get_t1(X_clean.index, best_params["horizon_n"])
        cv_oof = PurgedKFold(n_splits=n_splits_oof, t1=t1_labels, embargo_pct=embargo_pct_oof)

        # Initialize OOF prediction arrays
        oof_lgb_preds = np.zeros(len(X_clean))
        oof_xgb_preds = np.zeros(len(X_clean))
        oof_cat_preds = np.zeros(len(X_clean))
        oof_gru_preds = np.zeros(len(X_clean))
        oof_lstm_preds = np.zeros(len(X_clean))
        
        # OOFループで学習したモデルはリストに保存し、後で平均するなどしてX_test予測に使う
        # または、OOFループとは別に全データで学習したモデルを作成する。今回は後者で。

        # スケーラーはOOFごとにfit_transformする
        scaler = StandardScaler()
        
        for fold, (train_idx, val_idx) in enumerate(cv_oof.split(X_clean, y)):
            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.warning(f"OOF Fold {fold}: Skipping due to empty train or validation set after purging.")
                continue

            X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # クラスウェイト計算 (各フォールドで)
            neg = len(y_train_fold) - sum(y_train_fold)
            pos = sum(y_train_fold)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            # --- LightGBM ---
            lgb_params = {k: best_params[k] for k in best_params if k.endswith("_lgb")}
            lgb_params_final = {k.replace("_lgb", ""): v for k, v in lgb_params.items()} # Optunaの接尾辞を削除
            lgb_params_final.update({
                "objective": "binary", "metric": "binary_logloss", "verbosity": -1, "random_state": 42, "class_weight": "balanced"
            })
            model_lgb = lgb.LGBMClassifier(**lgb_params_final)
            model_lgb.fit(X_train_fold, y_train_fold)
            oof_lgb_preds[val_idx] = model_lgb.predict_proba(X_val_fold)[:, 1]

            # --- XGBoost ---
            xgb_params = {k: best_params[k] for k in best_params if k.endswith("_xgb")}
            xgb_params_final = {k.replace("_xgb", ""): v for k, v in xgb_params.items()} # Optunaの接尾辞を削除
            xgb_params_final.update({
                "objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0, "random_state": 42, "scale_pos_weight": scale_pos_weight, "missing": np.inf
            })
            model_xgb = xgb.XGBClassifier(**xgb_params_final)
            model_xgb.fit(X_train_fold, y_train_fold)
            oof_xgb_preds[val_idx] = model_xgb.predict_proba(X_val_fold)[:, 1]
            
            # --- CatBoost ---
            cat_params = {k: best_params[k] for k in best_params if k.endswith("_cat")}
            cat_params_final = {k.replace("_cat", ""): v for k, v in cat_params.items()} # Optunaの接尾辞を削除
            cat_params_final.update({
                "od_type": "Iter", "od_wait": 50, "verbose": 0, "random_seed": 42, "scale_pos_weight": scale_pos_weight, "allow_writing_files": False,
            })
            model_cat = cb.CatBoostClassifier(**cat_params_final)
            model_cat.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=False)
            oof_cat_preds[val_idx] = model_cat.predict_proba(X_val_fold)[:, 1]

            # --- GRU & LSTM ---
            dl_params = {k: best_params[k] for k in best_params if k.endswith("_dl")}
            dl_params_final = {k.replace("_dl", ""): v for k, v in dl_params.items()}
            dl_params_final["input_dim"] = X_clean.shape[1]
            dl_params_final["epochs"] = 10 # 最終学習はOptunaよりepoch増やす
            
            # 各フォールドでスケーリング
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)

            model_gru = GRUModel(**dl_params_final)
            model_gru.fit(X_train_scaled, y_train_fold)
            oof_gru_preds[val_idx] = model_gru.predict_proba(X_val_scaled)

            model_lstm = LSTMModel(**dl_params_final)
            model_lstm.fit(X_train_scaled, y_train_fold)
            oof_lstm_preds[val_idx] = model_lstm.predict_proba(X_val_scaled)
            
        # Build final base models (trained on all X_clean for consistent prediction on X_test) 
        
        # --- LightGBM Final ---
        lgb_params = {k: best_params[k] for k in best_params if k.endswith("_lgb")}
        lgb_params_final = {k.replace("_lgb", ""): v for k, v in lgb_params.items()} # Optunaの接尾辞を削除
        lgb_params_final.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "random_state": 42, "class_weight": "balanced"})
        model_lgb_final = lgb.LGBMClassifier(**lgb_params_final)
        model_lgb_final.fit(X_clean, y)
        joblib.dump(model_lgb_final, self.results_dir / "model_lgb.joblib")

        # --- XGBoost Final ---
        xgb_params = {k: best_params[k] for k in best_params if k.endswith("_xgb")}
        xgb_params_final = {k.replace("_xgb", ""): v for k, v in xgb_params.items()} # Optunaの接尾辞を削除
        xgb_params_final.update({"objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0, "random_state": 42, "scale_pos_weight": scale_pos_weight, "missing": np.inf})
        model_xgb_final = xgb.XGBClassifier(**xgb_params_final)
        model_xgb_final.fit(X_clean, y)
        joblib.dump(model_xgb_final, self.results_dir / "model_xgb.joblib")

        # --- CatBoost Final ---
        cat_params = {k: best_params[k] for k in best_params if k.endswith("_cat")}
        cat_params_final = {k.replace("_cat", ""): v for k, v in cat_params.items()} # Optunaの接尾辞を削除
        cat_params_final.update({"od_type": "Iter", "od_wait": 50, "verbose": 0, "random_seed": 42, "scale_pos_weight": scale_pos_weight, "allow_writing_files": False})
        model_cat_final = cb.CatBoostClassifier(**cat_params_final)
        model_cat_final.fit(X_clean, y, eval_set=[(X_clean, y)], early_stopping_rounds=50, verbose=False) # X_test for final model validation
        joblib.dump(model_cat_final, self.results_dir / "model_cat.joblib")

        # --- GRU & LSTM Final ---
        dl_params = {k: best_params[k] for k in best_params if k.endswith("_dl")}
        dl_params_final = {k.replace("_dl", ""): v for k, v in dl_params.items()}
        dl_params_final["input_dim"] = X_clean.shape[1]
        dl_params_final["epochs"] = 20
        
        scaler_final = StandardScaler()
        X_clean_scaled = scaler_final.fit_transform(X_clean)
        joblib.dump(scaler_final, self.results_dir / "scaler.joblib")

        model_gru_final = GRUModel(**dl_params_final)
        model_gru_final.fit(X_clean_scaled, y)
        
        model_lstm_final = LSTMModel(**dl_params_final)
        model_lstm_final.fit(X_clean_scaled, y)


        # --- Stacking Meta Model (Ridge NNLS) ---
        logger.info("メタモデル (Ridge NNLS) を学習中 (OOF予測を使用)...")
        
        # Ensure OOF predictions are aligned with X_clean indices
        oof_preds_df = pd.DataFrame({
            'LightGBM': oof_lgb_preds,
            'XGBoost': oof_xgb_preds,
            'CatBoost': oof_cat_preds,
            'GRU': oof_gru_preds,
            'LSTM': oof_lstm_preds
        }, index=X_clean.index)
        
        stacking_service = StackingService()
        stacking_service.train(oof_preds_df, y.values) # Use OOF preds and full y for meta-learner
        
        weights = stacking_service.get_weights()
        logger.info(f"学習された重み: {weights}")
        joblib.dump(stacking_service, self.results_dir / "stacking_service.joblib")

        # ここではevaluate用のtest setがない（全てTrainで使った）ため、最終評価は割愛するか、
        # ອອກຈາກ OOFスコア自体を最終評価とする。
        # もし「未知のデータ」に対する評価が必要なら、最初にHold-out setを作る必要がある。
        # 今回のPurgedKFold OOFは「全データに対する予測精度」を近似できるため、OOFの結果を返すのが自然。
        
        # 変更点: X_test は存在しない（全てX_cleanとして使用）
        # そのため、evaluate_and_stacking に渡すデータは oof_preds_df と y になる
        
        return model_lgb_final, model_xgb_final, model_cat_final, model_gru_final, model_lstm_final, stacking_service, oof_preds_df, y, None

    def evaluate_and_stacking(self, model_lgb, model_xgb, model_cat, model_gru, model_lstm, stacking_service, oof_preds_df, y, X_test_scaled):
        """閾値シミュレーションとスタッキング評価 (OOF予測に基づく)"""
        logger.info("=" * 60)
        logger.info("モデル評価 & スタッキング (OOF Performance)")
        logger.info("=" * 60)

        # 各モデルのOOF予測値
        prob_lgb = oof_preds_df['LightGBM']
        prob_xgb = oof_preds_df['XGBoost']
        prob_cat = oof_preds_df['CatBoost']
        prob_gru = oof_preds_df['GRU']
        prob_lstm = oof_preds_df['LSTM']
        
        # スタッキング (Ridge NNLS) - OOF予測に対してpredictする必要はない（学習に使ったので）
        # しかし、StackingServiceのpredictは学習済み係数を掛けるだけなので、OOF予測値を入力すればOOFに対するスタッキング予測になる
        prob_stack = stacking_service.predict(oof_preds_df)
        
        # Soft Voting
        prob_avg = (prob_lgb + prob_xgb + prob_cat + prob_gru + prob_lstm) / 5

        models = {
            "LightGBM (OOF)": prob_lgb,
            "XGBoost (OOF)": prob_xgb,
            "CatBoost (OOF)": prob_cat,
            "GRU (OOF)": prob_gru,
            "LSTM (OOF)": prob_lstm,
            "Stacking (Avg OOF)": prob_avg,
            "Stacking (Ridge OOF)": prob_stack
        }

        best_result = None
        best_score = -1

        for name, probs in models.items():
            logger.info(f"\n--- {name} ---")
            logger.info(f"{ 'Threshold':<10} | {'RangeRec':<10} | {'TrendPrec':<10} | {'TrendRec':<10} | {'Acc':<10}")
            logger.info("-" * 60)

            for th in np.arange(0.5, 0.96, 0.05):
                y_pred = (probs >= th).astype(int)
                # OOFなので0.5 (padding) を除外して評価
                # ここでは簡易的に全データで評価（パディングは極少数または0.5未満として扱われる）
                
                report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                
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
            logger.warn("目標基準（Range回避 > 60%）を満たす設定が見つかりませんでした。")

    def analyze_feature_importance(self, model_lgb, X_test):
        """LightGBMの特徴量重要度分析"""
        logger.info("\n" + "=" * 60)
        logger.info("特徴量重要度分析 (LightGBM)")
        logger.info("=" * 60)
        
        # X_testは存在しないため、学習に使ったX_clean (model_lgb.feature_name_などから取得) を想定
        # model_lgbは全データで学習されているので、feature_importances_は有効
        
        importances = model_lgb.feature_importances_
        feature_names = model_lgb.feature_name_
        
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
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

        # 3. 最終モデル学習 (Purged K-Fold OOF)
        # 戻り値: 最終モデル群, メタモデル, OOF予測値DF, 正解ラベル, スケール済みX(None)
        model_lgb, model_xgb, model_cat, model_gru, model_lstm, stacking_service, oof_preds_df, y, _ = self.train_final_models(X, ohlcv, best_params)

        # 4. スタッキング評価 (OOF)
        self.evaluate_and_stacking(model_lgb, model_xgb, model_cat, model_gru, model_lstm, stacking_service, oof_preds_df, y, None)

        # 5. 特徴量分析
        # X自体を渡せばfeature_namesは取れるが、メソッド内でfeature_name_属性を使うように変更したので引数はダミーでよい
        self.analyze_feature_importance(model_lgb, X)


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
