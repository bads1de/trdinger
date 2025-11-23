"""
閾値調整によるモデル性能評価スクリプト

最新の統合最適化結果（best_params.json）を読み込み、
トレンド判定の閾値（Probability Threshold）を調整した場合の
クラス別精度（Precision/Recall）の変化をシミュレーションします。

目的:
- レンジ相場の検出率（Recall）を向上させる最適な閾値を見つける
- トレンド判定の確実性（Precision）を高める
"""

import json
import logging
import sys
import time
from pathlib import Path
import glob
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windowsでの文字化け対策
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.label_cache import LabelCache
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


class ThresholdEvaluator:
    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.db = SessionLocal()
        self.evaluator = CommonFeatureEvaluator()
        self.feature_service = FeatureEngineeringService()

    def get_latest_results_dir(self) -> Path:
        """最新の最適化結果ディレクトリを取得"""
        base_dir = Path(__file__).parent.parent.parent / "results" / "integrated_optimization"
        list_of_dirs = glob.glob(str(base_dir / "run_*"))
        if not list_of_dirs:
            raise FileNotFoundError("最適化結果が見つかりません。先に最適化を実行してください。")
        latest_dir = max(list_of_dirs, key=os.path.getctime)
        return Path(latest_dir)

    def load_best_params(self, results_dir: Path) -> dict:
        """パラメータ読み込み"""
        json_path = results_dir / "best_params.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["best_params"]

    def prepare_data(self, best_params: dict):
        """データ準備（最適化時と同じロジック）"""
        logger.info("データを準備中...")
        
        # データ取得
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=10000
        )
        
        # 特徴量生成
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=data.ohlcv,
            funding_rate_data=data.fr,
            open_interest_data=data.oi,
        )
        
        feature_cols = [
            col for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        X = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())

        # ラベル生成
        label_cache = LabelCache(data.ohlcv)
        
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
        labels_clean = labels_aligned.loc[valid_idx]

        # 2値化 (TREND=1, RANGE=0)
        y = labels_clean.map({"DOWN": 1, "RANGE": 0, "UP": 1})

        return X_clean, y

    def train_model(self, X, y, best_params):
        """モデル学習"""
        logger.info(f"モデル学習開始: {best_params.get('model_type', 'lightgbm')}")
        
        # データ分割 (時系列順: Train+Val 80%, Test 20%)
        # 閾値評価用なので、Train+Valで学習し、Testで評価する
        n_total = len(X)
        test_start_idx = int(n_total * 0.8)
        
        X_train = X.iloc[:test_start_idx]
        y_train = y.iloc[:test_start_idx]
        
        X_test = X.iloc[test_start_idx:]
        y_test = y.iloc[test_start_idx:]
        
        # Validation用に内部でさらに分割 (LightGBM/XGBoostのearly_stopping用)
        val_start_idx = int(len(X_train) * 0.8)
        X_tr = X_train.iloc[:val_start_idx]
        y_tr = y_train.iloc[:val_start_idx]
        X_val = X_train.iloc[val_start_idx:]
        y_val = y_train.iloc[val_start_idx:]

        model_type = best_params.get("model_type", "lightgbm")
        
        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "random_state": 42,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "num_leaves": best_params["num_leaves"],
                "min_child_samples": best_params["min_child_samples"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
            }
            if best_params.get("use_class_weight", True):
                params["class_weight"] = "balanced"
                
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
        else: # xgboost
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "min_child_weight": best_params["min_child_weight"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "gamma": best_params["gamma"],
                "reg_alpha": best_params["reg_alpha"],
                "reg_lambda": best_params["reg_lambda"],
            }
            if best_params.get("use_class_weight", True):
                neg = len(y_tr) - sum(y_tr)
                pos = sum(y_tr)
                params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
                
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        return model, X_test, y_test

    def run_simulation(self):
        try:
            results_dir = self.get_latest_results_dir()
            logger.info(f"最新の結果ディレクトリ: {results_dir}")
            
            best_params = self.load_best_params(results_dir)
            
            X, y = self.prepare_data(best_params)
            model, X_test, y_test = self.train_model(X, y, best_params)
            
            # 予測確率 (TREND=1 の確率)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
            else:
                y_probs = model.predict(X_test)

            logger.info("\n" + "="*60)
            logger.info("閾値調整シミュレーション (Trend vs Range)")
            logger.info("="*60)
            logger.info(f"{ 'Threshold':<10} | {'RANGE Recall':<12} | {'RANGE Prec':<12} | {'TREND Prec':<12} | {'TREND Recall':<12} | {'Accuracy':<10}")
            logger.info("-" * 80)

            results = []
            
            # 閾値を0.5から0.95まで変化
            for threshold in np.arange(0.5, 0.96, 0.05):
                # 確率 >= threshold なら TREND(1), そうでなければ RANGE(0)
                y_pred = (y_probs >= threshold).astype(int)
                
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # クラス0 = RANGE, クラス1 = TREND
                range_recall = report['0']['recall']
                range_prec = report['0']['precision']
                trend_prec = report['1']['precision']
                trend_recall = report['1']['recall']
                acc = report['accuracy']
                
                logger.info(f"{threshold:.2f}{' (Default)' if threshold==0.5 else '         ':<10} | {range_recall:.4f}       | {range_prec:.4f}       | {trend_prec:.4f}       | {trend_recall:.4f}       | {acc:.4f}")
                
                results.append({
                    "threshold": threshold,
                    "range_recall": range_recall,
                    "range_precision": range_prec,
                    "trend_precision": trend_prec,
                    "trend_recall": trend_recall,
                    "accuracy": acc
                })

            # 最適な閾値の提案 (バランス重視: RANGE Recall > 0.6 かつ TREND Precision > 0.7)
            best_balanced = [r for r in results if r['range_recall'] > 0.6 and r['trend_precision'] > 0.75]
            
            logger.info("="*60)
            if best_balanced:
                rec = max(best_balanced, key=lambda x: x['accuracy'])
                logger.info(f"推奨閾値: {rec['threshold']:.2f}")
                logger.info(f"  - レンジ検出率: {rec['range_recall']:.1%}")
                logger.info(f"  - トレンド確度: {rec['trend_precision']:.1%}")
                logger.info("  (解説: この設定にすると、レンジの6割以上を見抜きつつ、")
                logger.info("   トレンドと判定した時の信頼度は75%以上を維持できます)")
            else:
                logger.info("バランスの良い閾値が見つかりませんでした。特徴量の追加検討を推奨します。")

        finally:
            self.db.close()
            self.evaluator.close()

if __name__ == "__main__":
    evaluator = ThresholdEvaluator()
    evaluator.run_simulation()