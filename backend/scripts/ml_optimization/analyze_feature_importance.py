"""
特徴量重要度分析スクリプト

最新の最適化済みモデルをロードし、特徴量重要度（Gain/Split）を算出・可視化します。
不要な特徴量を特定し、削減の提案を行います。
"""

import json
import logging
import sys
from pathlib import Path
import glob
import os

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windowsでの文字化け対策
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    # 日本語フォント設定 (Seaborn/Matplotlib用)
    plt.rcParams['font.family'] = 'Meiryo'

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.label_cache import LabelCache
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"):
        self.symbol = symbol
        self.timeframe = timeframe
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
        """データ準備"""
        logger.info("データを準備中...")
        
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=10000
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
        X = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())

        label_cache = LabelCache(data.ohlcv)
        threshold_key = [k for k in best_params if "threshold" in k and k != "threshold_method"][0]
        
        labels = label_cache.get_labels(
            horizon_n=best_params["horizon_n"],
            threshold_method=best_params["threshold_method"],
            threshold=best_params[threshold_key],
            timeframe=self.timeframe,
            price_column="close",
        )

        common_index = X.index.intersection(labels.index)
        X_aligned = X.loc[common_index]
        labels_aligned = labels.loc[common_index]

        valid_idx = ~labels_aligned.isna()
        X_clean = X_aligned.loc[valid_idx]
        labels_clean = labels_aligned.loc[valid_idx]

        y = labels_clean.map({"DOWN": 1, "RANGE": 0, "UP": 1})

        return X_clean, y

    def train_and_analyze(self):
        results_dir = self.get_latest_results_dir()
        logger.info(f"最新の結果ディレクトリ: {results_dir}")
        
        best_params = self.load_best_params(results_dir)
        X, y = self.prepare_data(best_params)
        
        # 時系列分割
        n_total = len(X)
        train_size = int(n_total * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]

        logger.info("モデル学習中...")
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
                "class_weight": "balanced"
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            importances = model.feature_importances_
            
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
            neg = len(y_train) - sum(y_train)
            pos = sum(y_train)
            params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            importances = model.feature_importances_

        # 重要度DataFrame作成
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # 結果出力
        logger.info("\n" + "="*60)
        logger.info("特徴量重要度ランキング (Top 20)")
        logger.info("="*60)
        print(feature_imp.head(20).to_string(index=False))
        
        logger.info("\n" + "="*60)
        logger.info("不要な特徴量 (Importance = 0)")
        logger.info("="*60)
        zero_imp = feature_imp[feature_imp['Importance'] == 0]
        if not zero_imp.empty:
            print(zero_imp['Feature'].tolist())
            logger.info(f"合計: {len(zero_imp)}個")
        else:
            logger.info("なし")

        # CSV保存
        csv_path = results_dir / "feature_importance.csv"
        feature_imp.to_csv(csv_path, index=False)
        logger.info(f"\n詳細結果を保存しました: {csv_path}")

        # 可視化 (Top 30)
        plt.figure(figsize=(12, 10))
        sns.barplot(x="Importance", y="Feature", data=feature_imp.head(30))
        plt.title(f"Feature Importance ({model_type.upper()})")
        plt.tight_layout()
        plt.savefig(results_dir / "feature_importance.png")
        logger.info(f"グラフを保存しました: {results_dir / 'feature_importance.png'}")

if __name__ == "__main__":
    analyzer = FeatureImportanceAnalyzer()
    analyzer.train_and_analyze()
