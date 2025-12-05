import sys
import os
import glob
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import optuna
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    plt.rcParams["font.family"] = "Meiryo"

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
    FAKEOUT_DETECTION_ALLOWLIST,
)
from app.services.ml.label_cache import LabelCache
from app.services.ml.label_generation import LabelGenerationService
from app.services.ml.ensemble.meta_labeling import MetaLabelingService
from app.services.ml.models.gru_model import GRUModel
from app.services.ml.models.lstm_model import LSTMModel
from app.services.ml.cross_validation.purged_kfold import PurgedKFold
from app.config.unified_config import unified_config
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


class SimpleStackingService:
    """
    Level-2 メタ学習器 (簡易版)
    Ridge回帰 (NNLS: Non-negative Least Squares) を使用して
    各ベースモデルの予測値を最適に統合する。
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Ridge回帰の正則化パラメータ
        """
        from sklearn.linear_model import Ridge

        self.model = Ridge(
            alpha=alpha, positive=True, fit_intercept=True, random_state=42
        )
        self.weights: Optional[Dict[str, float]] = None
        self.feature_names: list = []

    def train(self, X_meta: pd.DataFrame, y_true: np.ndarray) -> None:
        """
        メタモデルを学習する
        """
        self.feature_names = X_meta.columns.tolist()

        # 学習
        self.model.fit(X_meta, y_true)

        # 重みの保存
        self.weights = dict(zip(self.feature_names, self.model.coef_))

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        予測を行う
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X_meta)

        # 確率なので [0, 1] にクリップ
        return np.clip(y_pred, 0.0, 1.0)

    def get_weights(self) -> Dict[str, float]:
        """学習された重みを取得"""
        if self.weights is None:
            return {}
        return self.weights


class MLPipeline:
    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        n_trials: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pipeline_type: str = "train_and_evaluate",
        enable_meta_labeling: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.start_date = start_date
        self.end_date = end_date
        self.pipeline_type = pipeline_type
        self.enable_meta_labeling = enable_meta_labeling

        self.db = SessionLocal()
        self.evaluator = CommonFeatureEvaluator()
        self.feature_service = FeatureEngineeringService()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "ml_pipeline"
            / f"run_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ML Pipeline Init: {symbol} {timeframe}")
        logger.info(f"Period: {start_date} - {end_date}")
        logger.info(f"Results Dir: {self.results_dir}")
        logger.info(f"Meta-Labeling Enabled: {self.enable_meta_labeling}")

    def get_latest_results_dir(self) -> Path:
        base_dir = Path(__file__).parent.parent.parent / "results" / "ml_pipeline"
        list_of_dirs = glob.glob(str(base_dir / "run_*"))

        valid_dirs = [
            d for d in list_of_dirs if (Path(d) / "best_params.json").exists()
        ]

        if not valid_dirs:
            base_dir_old = (
                Path(__file__).parent.parent.parent
                / "results"
                / "integrated_optimization"
            )
            list_of_dirs = glob.glob(str(base_dir_old / "run_*"))
            valid_dirs = [
                d for d in list_of_dirs if (Path(d) / "best_params.json").exists()
            ]

            if not valid_dirs:
                raise FileNotFoundError("No valid past results found.")

        latest_dir = max(valid_dirs, key=os.path.getctime)
        return Path(latest_dir)

    def prepare_data(self, limit: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Preparing data...")

        # start_date/end_dateが指定されている場合はそれを使用し、そうでない場合はlimitを使用
        if self.start_date and self.end_date:
            data = self.evaluator.fetch_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit,  # fetch_dataの互換性のため渡すが、start/endが優先される
                start_date=self.start_date,
                end_date=self.end_date,
            )
        else:
            data = self.evaluator.fetch_data(
                symbol=self.symbol, timeframe=self.timeframe, limit=limit
            )

        # メタラベリング有効時は特徴量リストを上書きしてダマシ検知用特徴量を使用
        original_allowlist = unified_config.ml.feature_engineering.feature_allowlist
        if self.enable_meta_labeling:
            unified_config.ml.feature_engineering.feature_allowlist = (
                FAKEOUT_DETECTION_ALLOWLIST
            )
            logger.info("🚀 Applying FAKEOUT_DETECTION_ALLOWLIST for Meta-Labeling")

        try:
            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=data.ohlcv,
                funding_rate_data=data.fr,
                open_interest_data=data.oi,
            )
        finally:
            # 設定を元に戻す
            if self.enable_meta_labeling:
                unified_config.ml.feature_engineering.feature_allowlist = (
                    original_allowlist
                )

        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        X = features_df[feature_cols].copy().fillna(0)

        return X, data.ohlcv

    def optimize(
        self, X: pd.DataFrame, ohlcv_df: pd.DataFrame, selected_models: List[str]
    ):
        logger.info("=" * 60)
        logger.info("Starting Hyperparameter Optimization (Purged K-Fold CV) [TBM]")
        logger.info("=" * 60)

        # LabelGenerationServiceを使用
        label_service = LabelGenerationService()

        def objective(trial):
            model_type = trial.suggest_categorical("model_type", selected_models)

            # Trend Scanning Parameters
            horizon_n = trial.suggest_int(
                "horizon_n", 20, 120, step=10
            )  # look_forward_window
            min_window = trial.suggest_int(
                "min_window", 10, 60, step=5
            )  # min_sample_length
            window_step = trial.suggest_int("window_step", 1, 5, step=1)  # step
            min_t_value = trial.suggest_float(
                "min_t_value", 1.5, 5.0, step=0.5
            )  # t-value threshold

            # Fixed Parameters
            use_atr = False
            threshold_method = "TREND_SCANNING"

            try:
                # LabelGenerationService.prepare_labels を使用してラベル生成とフィルタリングを一括で行う
                # Scientific Meta-Labeling: CUSUM Filter
                features_clean, labels_clean = label_service.prepare_labels(
                    features_df=X,
                    ohlcv_df=ohlcv_df,
                    use_cusum=self.enable_meta_labeling,  # CUSUMフィルターを使用
                    cusum_threshold=None,  # 動的ボラティリティ
                    cusum_vol_multiplier=2.5,  # 質と量のバランス（学術特徴量追加後）
                    horizon_n=horizon_n,
                    min_window=min_window,
                    window_step=window_step,
                    threshold=min_t_value,  # Trend Scanningの閾値
                    threshold_method=threshold_method,
                )
            except Exception as e:
                logger.error(f"Label generation failed: {e}", exc_info=True)
                raise optuna.exceptions.TrialPruned()

            X_clean = features_clean
            y = labels_clean

            # データ数チェック (緩和済み)
            if len(y) < 50:
                logger.warning(f"Not enough labels generated: {len(y)}")
                raise optuna.exceptions.TrialPruned()

            n_splits = 3
            embargo_pct = 0.01

            # LabelCacheは内部で再生成されるが、t1取得のために一時的に作成
            temp_cache = LabelCache(ohlcv_df)
            t1_labels = temp_cache.get_t1(X_clean.index, horizon_n)

            cv = PurgedKFold(n_splits=n_splits, t1=t1_labels, pct_embargo=embargo_pct)

            oof_preds = np.zeros(len(X_clean))

            scaler = StandardScaler()

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_clean, y)):
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                X_train_fold, X_val_fold = (
                    X_clean.iloc[train_idx],
                    X_clean.iloc[val_idx],
                )
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # ラベルが1種類しかない場合はスキップ (CatBoost等のエラー回避)
                if len(y_train_fold.unique()) < 2:
                    logger.warning(
                        f"Fold {fold}: Only one class in training data. Skipping."
                    )
                    continue

                model = None

                if model_type == "lightgbm":
                    params = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "verbosity": -1,
                        "random_state": 42,
                        "class_weight": "balanced",
                        "n_estimators": trial.suggest_int("n_estimators_lgb", 50, 500),
                        "learning_rate": trial.suggest_float(
                            "learning_rate_lgb", 0.001, 0.5, log=True
                        ),
                        "max_depth": trial.suggest_int("max_depth_lgb", 3, 15),
                        "num_leaves": trial.suggest_int("num_leaves_lgb", 20, 200),
                        "min_child_samples": trial.suggest_int(
                            "min_child_samples_lgb", 10, 100
                        ),
                        "subsample": trial.suggest_float("subsample_lgb", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree_lgb", 0.5, 1.0
                        ),
                    }
                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )

                elif model_type == "xgboost":
                    neg_count = len(y_train_fold) - sum(y_train_fold)
                    pos_count = sum(y_train_fold)
                    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                    params = {
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "verbosity": 0,
                        "random_state": 42,
                        "scale_pos_weight": scale_pos_weight,
                        "missing": np.inf,
                        "n_estimators": trial.suggest_int("n_estimators_xgb", 50, 500),
                        "learning_rate": trial.suggest_float(
                            "learning_rate_xgb", 0.001, 0.5, log=True
                        ),
                        "max_depth": trial.suggest_int("max_depth_xgb", 3, 15),
                        "min_child_weight": trial.suggest_int(
                            "min_child_weight_xgb", 1, 10
                        ),
                        "subsample": trial.suggest_float("subsample_xgb", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree_xgb", 0.5, 1.0
                        ),
                        "gamma": trial.suggest_float("gamma_xgb", 0, 5),
                        "reg_alpha": trial.suggest_float(
                            "reg_alpha_xgb", 1e-8, 10.0, log=True
                        ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda_xgb", 1e-8, 10.0, log=True
                        ),
                    }
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False,
                    )

                elif model_type == "catboost":
                    neg_count = len(y_train_fold) - sum(y_train_fold)
                    pos_count = sum(y_train_fold)
                    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                    params = {
                        "iterations": trial.suggest_int("iterations_cat", 50, 500),
                        "learning_rate": trial.suggest_float(
                            "learning_rate_cat", 0.001, 0.5, log=True
                        ),
                        "depth": trial.suggest_int("depth_cat", 3, 10),
                        "l2_leaf_reg": trial.suggest_float(
                            "l2_leaf_reg_cat", 1e-8, 10.0, log=True
                        ),
                        "random_strength": trial.suggest_float(
                            "random_strength_cat", 1e-8, 10.0, log=True
                        ),
                        "bagging_temperature": trial.suggest_float(
                            "bagging_temperature_cat", 0.0, 1.0
                        ),
                        "od_type": "Iter",
                        "od_wait": 50,
                        "verbose": 0,
                        "random_seed": 42,
                        "scale_pos_weight": scale_pos_weight,
                        "allow_writing_files": False,
                    }
                    model = cb.CatBoostClassifier(**params)
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=50,
                        verbose=False,
                    )

                else:  # gru or lstm
                    X_train_scaled = scaler.fit_transform(X_train_fold)
                    X_val_scaled = scaler.transform(X_val_fold)

                    params = {
                        "input_dim": X_clean.shape[1],  # X_cleanのfull features
                        "hidden_dim": trial.suggest_categorical(
                            "hidden_dim_dl", [32, 64, 128]
                        ),
                        "num_layers": trial.suggest_int("num_layers_dl", 1, 3),
                        "seq_len": trial.suggest_categorical(
                            "seq_len_dl", [12, 24, 48]
                        ),
                        "batch_size": trial.suggest_categorical(
                            "batch_size_dl", [32, 64]
                        ),
                        "epochs": 5,
                        "learning_rate": trial.suggest_float(
                            "learning_rate_dl", 0.0001, 0.01, log=True
                        ),
                        "dropout": trial.suggest_float("dropout_dl", 0.0, 0.5),
                    }

                    if model_type == "gru":
                        model = GRUModel(**params)
                    else:
                        model = LSTMModel(**params)

                    model.fit(X_train_scaled, y_train_fold)
                    y_pred_fold = model.predict_proba(X_val_scaled)

                if model_type not in ["gru", "lstm"]:
                    y_pred_fold = model.predict_proba(X_val_fold)[:, 1]

                oof_preds[val_idx] = y_pred_fold

            non_padded_idx = oof_preds != 0
            if len(oof_preds[non_padded_idx]) == 0:
                return 0.0

            y_pred_bin = (oof_preds >= 0.5).astype(int)
            oof_report = classification_report(
                y, y_pred_bin, output_dict=True, zero_division=0
            )

            trend_metrics = oof_report["1"]
            precision = trend_metrics["precision"]
            f1 = trend_metrics["f1-score"]

            # トレード回数（ポジティブ判定数）
            n_trades = y_pred_bin.sum()

            # 評価スコアの計算
            if self.enable_meta_labeling:
                # メタラベリング時は Precision（勝率）を最重要視
                # Precision * log(1 + trades)
                # ただし最低限のトレード数は必要
                if n_trades < 5:
                    final_score = 0.0
                else:
                    final_score = precision * np.log1p(n_trades)
            else:
                # 従来: F1スコア × log(トレード回数)
                # 最低限の精度(0.55)と回数(10)がないとスコア0
                if precision < 0.55 or n_trades < 10:
                    final_score = 0.0
                else:
                    final_score = f1 * np.log1p(n_trades)

            logger.info(
                f"Trial {trial.number} ({model_type}): Final Score={final_score:.4f}, F1={f1:.4f}, Trades={n_trades}, Prec={precision:.4f}"
            )

            return final_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_params["use_class_weight"] = (
            True  # Optuna trial is not supposed to optimize this parameter.
        )
        best_params["threshold_method"] = "TREND_SCANNING"  # 明示的に保存

        results = {
            "best_value": study.best_value,
            "best_params": best_params,
            "trials": len(study.trials),
        }

        with open(self.results_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Optimization Done: Best Score={study.best_value:.4f}")
        return best_params

    def train_final_models(
        self,
        X: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        best_params: Dict[str, Any],
        selected_models: List[str],
    ):
        logger.info("=" * 60)
        logger.info(
            "Training Final Models & Stacking (Purged K-Fold OOF) [Trend Scanning]"
        )
        logger.info("=" * 60)

        # 最適化されたモデルのみを使用する
        best_model_type = best_params["model_type"]
        logger.info(f"Using Best Model Only: {best_model_type}")
        selected_models = [best_model_type]

        label_service = LabelGenerationService()

        # Trend Scanning parameters
        horizon_n = best_params["horizon_n"]
        min_window = best_params.get("min_window", 10)
        window_step = best_params.get("window_step", 1)
        min_t_value = best_params.get("min_t_value", 2.0)
        threshold_method = "TREND_SCANNING"

        # Fixed parameters
        use_atr = False

        try:
            # Scientific Meta-Labeling: CUSUM Filter
            features_clean, labels_clean = label_service.prepare_labels(
                features_df=X,
                ohlcv_df=ohlcv_df,
                use_cusum=self.enable_meta_labeling,
                cusum_threshold=None,  # 動的ボラティリティ
                cusum_vol_multiplier=2.5,  # 質と量のバランス（学術特徴量追加後）
                horizon_n=horizon_n,
                min_window=min_window,
                window_step=window_step,
                threshold=min_t_value,
                threshold_method=threshold_method,
            )
        except Exception as e:
            logger.error(f"Final model label generation failed: {e}")
            raise

        X_clean = features_clean
        y = labels_clean

        if len(y) < 50:
            raise ValueError("Not enough data for final training.")

        # Save feature names for later analysis
        with open(self.results_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(X_clean.columns.tolist(), f, indent=2, ensure_ascii=False)
        logger.info(f"Feature names saved to {self.results_dir / 'feature_names.json'}")

        n_splits_oof = 5
        embargo_pct_oof = 0.01

        # t1取得用の一時キャッシュ
        temp_cache = LabelCache(ohlcv_df)
        t1_labels = temp_cache.get_t1(X_clean.index, horizon_n)

        cv_oof = PurgedKFold(
            n_splits=n_splits_oof, t1=t1_labels, pct_embargo=embargo_pct_oof
        )

        oof_preds = {model: np.zeros(len(X_clean)) for model in selected_models}
        models_result = {}

        scaler = StandardScaler()

        for fold, (train_idx, val_idx) in enumerate(cv_oof.split(X_clean, y)):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            neg = len(y_train_fold) - sum(y_train_fold)
            pos = sum(y_train_fold)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            # --- LightGBM ---
            if "lightgbm" in selected_models:
                lgb_params = {
                    k: best_params[k] for k in best_params if k.endswith("_lgb")
                }
                lgb_params_final = {
                    k.replace("_lgb", ""): v for k, v in lgb_params.items()
                }
                lgb_params_final.update(
                    {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "verbosity": -1,
                        "random_state": 42,
                        "class_weight": "balanced",
                    }
                )
                model_lgb = lgb.LGBMClassifier(**lgb_params_final)
                model_lgb.fit(X_train_fold, y_train_fold)
                oof_preds["lightgbm"][val_idx] = model_lgb.predict_proba(X_val_fold)[
                    :, 1
                ]

            # --- XGBoost ---
            if "xgboost" in selected_models:
                xgb_params = {
                    k: best_params[k] for k in best_params if k.endswith("_xgb")
                }
                xgb_params_final = {
                    k.replace("_xgb", ""): v for k, v in xgb_params.items()
                }
                xgb_params_final.update(
                    {
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "verbosity": 0,
                        "random_state": 42,
                        "scale_pos_weight": scale_pos_weight,
                        "missing": np.inf,
                    }
                )
                model_xgb = xgb.XGBClassifier(**xgb_params_final)
                model_xgb.fit(X_train_fold, y_train_fold)
                oof_preds["xgboost"][val_idx] = model_xgb.predict_proba(X_val_fold)[
                    :, 1
                ]

            # --- CatBoost ---
            if "catboost" in selected_models:
                cat_params = {
                    k: best_params[k] for k in best_params if k.endswith("_cat")
                }
                cat_params_final = {
                    k.replace("_cat", ""): v for k, v in cat_params.items()
                }
                cat_params_final.update(
                    {
                        "od_type": "Iter",
                        "od_wait": 50,
                        "verbose": 0,
                        "random_seed": 42,
                        "scale_pos_weight": scale_pos_weight,
                        "allow_writing_files": False,
                    }
                )
                model_cat = cb.CatBoostClassifier(**cat_params_final)
                model_cat.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                oof_preds["catboost"][val_idx] = model_cat.predict_proba(X_val_fold)[
                    :, 1
                ]

            # --- GRU & LSTM ---
            if "gru" in selected_models or "lstm" in selected_models:
                dl_params = {
                    k: best_params[k] for k in best_params if k.endswith("_dl")
                }
                dl_params_final = {
                    k.replace("_dl", ""): v for k, v in dl_params.items()
                }
                dl_params_final["input_dim"] = X_clean.shape[1]
                dl_params_final["epochs"] = 10

                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)

                if "gru" in selected_models:
                    model_gru = GRUModel(**dl_params_final)
                    model_gru.fit(X_train_scaled, y_train_fold)
                    oof_preds["gru"][val_idx] = model_gru.predict_proba(X_val_scaled)

                if "lstm" in selected_models:
                    model_lstm = LSTMModel(**dl_params_final)
                    model_lstm.fit(X_train_scaled, y_train_fold)
                    oof_preds["lstm"][val_idx] = model_lstm.predict_proba(X_val_scaled)

        # Final Models
        if "lightgbm" in selected_models:
            lgb_params = {k: best_params[k] for k in best_params if k.endswith("_lgb")}
            lgb_params_final = {k.replace("_lgb", ""): v for k, v in lgb_params.items()}
            lgb_params_final.update(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "verbosity": -1,
                    "random_state": 42,
                    "class_weight": "balanced",
                }
            )
            model_lgb_final = lgb.LGBMClassifier(**lgb_params_final)
            model_lgb_final.fit(X_clean, y)
            joblib.dump(model_lgb_final, self.results_dir / "model_lgb.joblib")
            models_result["lightgbm"] = model_lgb_final

        if "xgboost" in selected_models:
            xgb_params = {k: best_params[k] for k in best_params if k.endswith("_xgb")}
            xgb_params_final = {k.replace("_xgb", ""): v for k, v in xgb_params.items()}
            xgb_params_final.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "random_state": 42,
                    "scale_pos_weight": scale_pos_weight,
                    "missing": np.inf,
                }
            )
            model_xgb_final = xgb.XGBClassifier(**xgb_params_final)
            model_xgb_final.fit(X_clean, y)
            joblib.dump(model_xgb_final, self.results_dir / "model_xgb.joblib")
            models_result["xgboost"] = model_xgb_final

        if "catboost" in selected_models:
            cat_params = {k: best_params[k] for k in best_params if k.endswith("_cat")}
            cat_params_final = {k.replace("_cat", ""): v for k, v in cat_params.items()}
            cat_params_final.update(
                {
                    "od_type": "Iter",
                    "od_wait": 50,
                    "verbose": 0,
                    "random_seed": 42,
                    "scale_pos_weight": scale_pos_weight,
                    "allow_writing_files": False,
                }
            )
            model_cat_final = cb.CatBoostClassifier(**cat_params_final)
            model_cat_final.fit(
                X_clean,
                y,
                eval_set=[(X_clean, y)],
                early_stopping_rounds=50,
                verbose=False,
            )
            joblib.dump(model_cat_final, self.results_dir / "model_cat.joblib")
            models_result["catboost"] = model_cat_final

        if "gru" in selected_models or "lstm" in selected_models:
            dl_params = {k: best_params[k] for k in best_params if k.endswith("_dl")}
            dl_params_final = {k.replace("_dl", ""): v for k, v in dl_params.items()}
            dl_params_final["input_dim"] = X_clean.shape[1]
            dl_params_final["epochs"] = 20

            scaler_final = StandardScaler()
            X_clean_scaled = scaler_final.fit_transform(X_clean)
            joblib.dump(scaler_final, self.results_dir / "scaler.joblib")

            if "gru" in selected_models:
                model_gru_final = GRUModel(**dl_params_final)
                model_gru_final.fit(X_clean_scaled, y)
                joblib.dump(model_gru_final, self.results_dir / "model_gru.joblib")
                models_result["gru"] = model_gru_final

            if "lstm" in selected_models:
                model_lstm_final = LSTMModel(**dl_params_final)
                model_lstm_final.fit(X_clean_scaled, y)
                joblib.dump(model_lstm_final, self.results_dir / "model_lstm.joblib")
                models_result["lstm"] = model_lstm_final

        # --- Stacking (Level 2) ---
        oof_preds_df = pd.DataFrame(oof_preds, index=X_clean.index)

        stacking_service = SimpleStackingService()
        stacking_service.train(oof_preds_df, y.values)

        weights = stacking_service.get_weights()
        logger.info(f"Learned Weights: {weights}")
        joblib.dump(stacking_service, self.results_dir / "stacking_service.joblib")

        # --- Meta Labeling ---
        if self.enable_meta_labeling:
            logger.info("Training Meta-Labeling Model...")

            # スタッキングモデルのOOF予測値を取得 (これが一次モデルの確信度となる)
            # Ridgeスタッキングは不安定なため、単純平均を使用する
            primary_oof_proba = oof_preds_df.mean(axis=1).values
            primary_oof_series = pd.Series(primary_oof_proba, index=X_clean.index)

            # メタラベリング専用の特徴量リストを取得
            from app.services.ml.feature_engineering.feature_engineering_service import (
                FAKEOUT_DETECTION_ALLOWLIST,
            )

            # 存在する特徴量のみをフィルタリング
            available_features = [
                f for f in FAKEOUT_DETECTION_ALLOWLIST if f in X_clean.columns
            ]
            if len(available_features) < len(FAKEOUT_DETECTION_ALLOWLIST):
                missing = set(FAKEOUT_DETECTION_ALLOWLIST) - set(available_features)
                logger.warning(
                    f"Some fakeout detection features are missing: {missing}"
                )

            X_meta_input = X_clean[available_features].copy()
            logger.info(f"Using {len(available_features)} features for Meta-Labeling.")

            meta_service = MetaLabelingService()

            # メタモデルのOOF予測を生成 (評価用)
            logger.info("Generating Meta-Model OOF Predictions...")
            meta_oof_preds = meta_service.cross_validate(
                X=X_meta_input,
                y=y,
                primary_proba=primary_oof_series,
                base_model_probs_df=oof_preds_df,
                threshold=0.5,
                n_splits=5,
                t1=t1_labels,  # PurgedKFold用
                pct_embargo=0.01,
            )
            models_result["meta_oof_preds"] = meta_oof_preds

            # メタモデルの学習 (最終モデル用)
            # X_meta_inputを使用
            meta_result = meta_service.train(
                X_train=X_meta_input,
                y_train=y,
                primary_proba_train=primary_oof_series,
                base_model_probs_df=oof_preds_df,  # 個別ベースモデルのOOF予測確率DataFrameを渡す
                threshold=0.5,
            )

            if meta_result["status"] == "success":
                joblib.dump(
                    meta_service, self.results_dir / "meta_labeling_service.joblib"
                )
                logger.info("Meta-Labeling Model trained and saved.")
                models_result["meta_service"] = meta_service
            else:
                logger.warning(
                    f"Meta-Labeling training skipped: {meta_result.get('reason')}"
                )
                logger.warning(f"Meta-Labeling result details: {meta_result}")
                models_result["meta_service"] = None
        else:
            logger.info("Meta-Labeling disabled.")
            models_result["meta_service"] = None

        models_result["stacking_service"] = stacking_service
        models_result["oof_preds_df"] = oof_preds_df
        models_result["y"] = y
        models_result["X_clean"] = X_clean
        models_result["X_meta_input"] = (
            X_meta_input if self.enable_meta_labeling else None
        )

        return models_result

    def evaluate_and_stacking(
        self,
        model_lgb,
        model_xgb,
        model_cat,
        model_gru,
        model_lstm,
        stacking_service,
        oof_preds_df,
        y,
        meta_service,
        X_clean,
        meta_oof_preds=None,
        X_meta_input=None,
    ):
        logger.info("=" * 60)
        logger.info("Model Evaluation & Stacking (OOF Performance)")
        logger.info("=" * 60)

        models = {}
        prob_sum = np.zeros(len(y))
        model_count = 0

        if "lightgbm" in oof_preds_df.columns:
            prob_lgb = oof_preds_df["lightgbm"]
            models["LightGBM (OOF)"] = prob_lgb
            prob_sum += prob_lgb
            model_count += 1

        if "xgboost" in oof_preds_df.columns:
            prob_xgb = oof_preds_df["xgboost"]
            models["XGBoost (OOF)"] = prob_xgb
            prob_sum += prob_xgb
            model_count += 1

        if "catboost" in oof_preds_df.columns:
            prob_cat = oof_preds_df["catboost"]
            models["CatBoost (OOF)"] = prob_cat
            prob_sum += prob_cat
            model_count += 1

        if "gru" in oof_preds_df.columns:
            prob_gru = oof_preds_df["gru"]
            models["GRU (OOF)"] = prob_gru
            prob_sum += prob_gru
            model_count += 1

        if "lstm" in oof_preds_df.columns:
            prob_lstm = oof_preds_df["lstm"]
            models["LSTM (OOF)"] = prob_lstm
            prob_sum += prob_lstm
            model_count += 1

        prob_stack = stacking_service.predict(oof_preds_df)
        models["Stacking (Ridge OOF)"] = prob_stack

        if model_count > 0:
            prob_avg = prob_sum / model_count
            models["Stacking (Avg OOF)"] = prob_avg

        best_result = None
        best_score = -1

        for name, probs in models.items():
            logger.info(f"\n--- {name} ---")
            logger.info(
                f"{ 'Threshold':<10} | {'RangeRec':<10} | {'TrendPrec':<10} | {'TrendRec':<10} | {'Acc':<10} | {'Trades':<5} | {'Score':<10}"
            )
            logger.info("-" * 80)

            for th in np.arange(0.5, 0.96, 0.05):
                y_pred = (probs >= th).astype(int)
                report = classification_report(
                    y, y_pred, output_dict=True, zero_division=0
                )

                range_rec = report["0"]["recall"]
                trend_prec = report["1"]["precision"]
                trend_rec = report["1"]["recall"]
                trend_f1 = report["1"]["f1-score"]
                acc = report["accuracy"]

                n_trades = y_pred.sum()

                # optimizeと同じスコア基準: 最低限の精度(0.55)と回数(10)が必要
                if trend_prec < 0.55 or n_trades < 10:
                    score = 0.0
                else:
                    # F1スコア × log(トレード回数)
                    score = trend_f1 * np.log1p(n_trades)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "model": name,
                        "threshold": th,
                        "range_rec": range_rec,
                        "trend_prec": trend_prec,
                        "trend_rec": trend_rec,
                        "trend_f1": trend_f1,
                        "n_trades": n_trades,
                        "score": score,
                    }

                logger.info(
                    f"{th:.2f}{' (Def)' if th==0.5 else '      ':<6} | {range_rec:.4f}     | {trend_prec:.4f}     | {trend_rec:.4f}     | {acc:.4f}     | {n_trades:<5} | {score:.4f}"
                )

        logger.info("\n" + "=" * 60)
        if best_result:
            logger.info(
                f"🏆 Best Config: {best_result['model']} @ Threshold {best_result['threshold']:.2f}"
            )
            logger.info(f"   Score: {best_result['score']:.4f} (F1 * log(Trades))")
            logger.info(f"   Trades: {best_result['n_trades']}")
            logger.info(f"   TrendPrec: {best_result['trend_prec']:.1%}")
            logger.info(f"   TrendRec:  {best_result['trend_rec']:.1%}")
            logger.info(f"   TrendF1:   {best_result['trend_f1']:.1%}")
        else:
            logger.warning("No configuration met the criteria.")

        # --- Meta Labeling Evaluation ---
        if meta_service and meta_service.is_trained:
            logger.info("\n" + "=" * 60)
            logger.info("Meta-Labeling Performance (In-Sample Check)")
            logger.info("=" * 60)

            # スタッキングのOOF予測値を使用
            # Ridgeスタッキングは不安定なため、単純平均を使用する
            primary_oof_proba = oof_preds_df.mean(axis=1).values
            primary_oof_series = pd.Series(primary_oof_proba, index=oof_preds_df.index)

            threshold = 0.5
            trend_mask = primary_oof_series >= threshold
            n_primary_trends = trend_mask.sum()

            if n_primary_trends > 0:
                logger.info(
                    f"Primary Model Trend Predictions: {n_primary_trends} / {len(y)} ({n_primary_trends/len(y):.1%})"
                )

                actual_hits = y[trend_mask].sum()
                primary_precision = actual_hits / n_primary_trends
                logger.info(f"Primary Model Precision: {primary_precision:.1%}")

            # メタモデルの評価 (OOF予測を使用)
            # meta_oof_preds is passed as argument

            if meta_oof_preds is not None:
                # classification_report is already imported at module level

                # メタモデルのOOF予測と正解ラベルで評価
                # メタモデルは、一次モデルがTrendと予測した箇所について、
                # それが本当にTrend(1)かRange(0)かを予測している。
                # したがって、一次モデルがTrendと予測した箇所のみで評価する。

                trend_mask = primary_oof_series >= threshold
                if trend_mask.sum() > 0:
                    y_true_meta = y[trend_mask]
                    y_pred_meta = meta_oof_preds[trend_mask]

                    report = classification_report(
                        y_true_meta, y_pred_meta, output_dict=True, zero_division=0
                    )

                    meta_precision = report["1"]["precision"]
                    meta_recall = report["1"]["recall"]
                    meta_f1 = report["1"]["f1-score"]

                    logger.info(
                        f"Meta-Model Precision (OOF): {meta_precision:.1%} (vs Primary {primary_precision:.1%})"
                    )
                    logger.info(f"Meta-Model Recall (OOF):    {meta_recall:.1%}")
                    logger.info(f"Meta-Model F1 (OOF):        {meta_f1:.1%}")
                else:
                    logger.warning(
                        "Primary model made no trend predictions. Cannot evaluate meta-model."
                    )
            else:
                # 従来のIn-Sample評価 (フォールバック)
                # X_meta_inputがあればそれを使用、なければX_clean
                X_eval = X_meta_input if X_meta_input is not None else X_clean

                meta_eval_results = meta_service.evaluate(
                    X_test=X_eval,
                    y_test=y,
                    primary_proba_test=primary_oof_series,
                    base_model_probs_df=oof_preds_df,
                )

                logger.info(
                    f"Meta-Model Precision (In-Sample): {meta_eval_results['meta_precision']:.1%} (vs Primary {primary_precision:.1%})"
                )
                logger.info(
                    f"Meta-Model Recall (In-Sample):    {meta_eval_results['meta_recall']:.1%}"
                )
                logger.info(
                    f"Meta-Model F1 (In-Sample):        {meta_eval_results['meta_f1']:.1%}"
                )

    def run(self):
        # 1. データ準備
        X, ohlcv_df = self.prepare_data(limit=50000)

        selected_models = ["lightgbm", "xgboost", "catboost"]

        if self.pipeline_type == "train_and_evaluate":
            # 2. ハイパーパラメータ最適化
            best_params = self.optimize(X, ohlcv_df, selected_models)

            # 3. 最終モデル学習
            models_result = self.train_final_models(
                X, ohlcv_df, best_params, selected_models
            )

            # 4. 評価
            self.evaluate_and_stacking(
                models_result.get("lightgbm"),
                models_result.get("xgboost"),
                models_result.get("catboost"),
                models_result.get("gru"),
                models_result.get("lstm"),
                models_result["stacking_service"],
                models_result["oof_preds_df"],
                models_result["y"],
                models_result["meta_service"],
                models_result["X_clean"],
                models_result.get("meta_oof_preds"),
                models_result.get("X_meta_input"),
            )

        elif self.pipeline_type == "evaluate_only":
            logger.info("Evaluation only mode not fully implemented yet.")
            pass


if __name__ == "__main__":
    # パイプライン実行
    pipeline = MLPipeline(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        n_trials=5,  # 試行回数
        start_date="2023-12-01",  # 直近1年に限定（市場環境の一貫性）
        pipeline_type="train_and_evaluate",
        enable_meta_labeling=True,  # Meta-Labeling有効化
    )
    pipeline.run()
