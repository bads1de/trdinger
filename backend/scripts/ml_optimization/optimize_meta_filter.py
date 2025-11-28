import sys
import os
import logging
from pathlib import Path
import json
import pandas as pd
import pandas_ta as ta
import optuna
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import joblib
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
backend_dir = project_root / "backend"
scripts_dir = backend_dir / "scripts" / "ml_optimization"
sys.path.append(str(backend_dir))
sys.path.append(str(scripts_dir))

from run_ml_pipeline import MLPipeline
from app.services.ml.ensemble.meta_labeling import MetaLabelingService
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from scripts.feature_evaluation.common_feature_evaluator import CommonFeatureEvaluator

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 戦略クラス定義 (backtest_meta_strategy.py から移植) ---
class MetaEmaCrossStrategy(Strategy):
    """EMAクロス + MLフィルタリング戦略"""

    n1 = 10
    n2 = 20

    # 外部から注入されるパラメータ
    meta_model = None  # クラス変数としてモデルを保持
    features = None  # 特徴量データフレーム
    threshold = 0.5  # メタモデルの判定閾値
    scaling_factor = 5.0  # 追加: EMA乖離のスケーリング係数

    def init(self):
        self.ema1 = self.I(pd.Series(self.data.Close).ewm(span=self.n1).mean)
        self.ema2 = self.I(pd.Series(self.data.Close).ewm(span=self.n2).mean)

        # ATR計算 (バックテストデータ内で計算)
        # self.data.ATR が存在しない可能性があるため計算しておく
        self.atr = self.I(
            ta.atr,
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            14,
        )

    def next(self):
        # 現在の時刻
        current_time = self.data.index[-1]

        # シグナル発生チェック
        signal_type = None

        # ゴールデンクロス (Buy)
        if crossover(self.ema1, self.ema2):
            signal_type = "buy"
        # デッドクロス (Sell)
        elif crossover(self.ema2, self.ema1):
            signal_type = "sell"

        if signal_type:
            # EMAの乖離絶対値をATRで正規化し、Primary Modelの予測確率として使用
            ema_diff = abs(self.ema1[-1] - self.ema2[-1])

            # ATR取得
            atr_value = self.atr[-1]
            if np.isnan(atr_value) or atr_value <= 0:
                atr_value = self.data.Close[-1] * 0.01  # fallback

            normalized_ema_diff = ema_diff / atr_value

            # primary_proba_value は 0-1 の範囲にスケーリング
            # 乖離が大きいほど「トレンドが強い」とみなし、モデルへの入力値を高くする
            primary_proba_value = min(1.0, normalized_ema_diff * self.scaling_factor)

            # MLフィルタリング (ボラティリティがあるかどうかを判定)
            if self.check_meta_filter(current_time, primary_proba_value):
                if signal_type == "buy":
                    if self.position.is_short:
                        self.position.close()
                    self.buy()
                elif signal_type == "sell":
                    if self.position.is_long:
                        self.position.close()
                    self.sell()

    def check_meta_filter(self, current_time, primary_proba_value):
        if self.meta_model is None or self.features is None:
            return False

        try:
            # 現在時刻の特徴量を取得
            if current_time not in self.features.index:
                return False

            feature_row_raw = self.features.loc[[current_time]]

            # MetaLabelingServiceが内部で追加する特徴量名
            meta_service_added_suffixes = [
                "_proba",
                "_prob_mean",
                "_prob_std",
                "_prob_min",
                "_prob_max",
            ]
            base_model_names_from_oof = self.meta_model.base_model_names

            # meta_model.model (RandomForestClassifier) の feature_names_in_ を取得
            rf_feature_names = list(self.meta_model.model.feature_names_in_)

            pure_X_feature_names = []
            for rf_feat in rf_feature_names:
                is_added = False
                if any(
                    rf_feat.endswith(suffix) for suffix in meta_service_added_suffixes
                ):
                    is_added = True
                if rf_feat in base_model_names_from_oof:
                    is_added = True

                if not is_added:
                    pure_X_feature_names.append(rf_feat)

            feature_row_for_X = feature_row_raw[pure_X_feature_names]

            # Primary Proba
            primary_proba = pd.Series(
                [primary_proba_value], index=feature_row_for_X.index
            )

            # Dummy Base Model Probs
            dummy_base_model_probs_data = {
                col: [1.0] for col in base_model_names_from_oof
            }
            base_model_probs_df = pd.DataFrame(
                dummy_base_model_probs_data, index=feature_row_for_X.index
            )

            # 予測
            prob = self.meta_model.predict(
                X=feature_row_for_X,
                primary_proba=primary_proba,
                base_model_probs_df=base_model_probs_df,
                threshold=self.threshold,  # ここでクラス変数のthresholdを使用
            )

            # predict は 0 or 1 を返すが、predict_proba 相当のロジックは内部で threshold を使って判定済み
            # しかし、Optunaで最適化したいのは「threshold」そのものである。
            # MetaLabelingService.predict は内部で (proba >= threshold).astype(int) している。
            # なので、返り値が 1 なら合格。

            return prob.iloc[0] == 1

        except Exception as e:
            # logger.error(f"Error in meta filter: {e}") # 高速化のためコメントアウト
            return False


# --- データ準備 ---
def prepare_data():
    symbol = "BTC/USDT:USDT"
    timeframe = "1h"

    # 1. モデル探索
    results_dir = project_root / "backend" / "results" / "ml_pipeline"
    logger.info(f"Searching for results in: {results_dir}")

    valid_runs = []
    for d in results_dir.iterdir():
        if d.is_dir():
            model_path = d / "meta_labeling_service.joblib"  # 正しいファイル名
            if model_path.exists():
                valid_runs.append(d)

    if not valid_runs:
        raise FileNotFoundError(
            f"No valid run directory with meta_labeling_service.joblib found in {results_dir}"
        )

    latest_run_dir = sorted(valid_runs, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"Using latest valid run directory: {latest_run_dir}")

    # モデルロード
    meta_model = joblib.load(latest_run_dir / "meta_labeling_service.joblib")

    # 特徴量リストロード
    with open(latest_run_dir / "best_params.json", "r", encoding="utf-8") as f:
        best_params_data = json.load(f)
    meta_feature_names = best_params_data.get("feature_names", [])

    # 2. データ準備
    logger.info("Preparing backtest data...")
    evaluator = CommonFeatureEvaluator()
    feature_service = FeatureEngineeringService()

    data = evaluator.fetch_data(symbol=symbol, timeframe=timeframe, limit=10000)
    ohlcv = data.ohlcv.copy()

    features = feature_service.calculate_advanced_features(
        ohlcv_data=ohlcv, funding_rate_data=data.fr, open_interest_data=data.oi
    )

    # 特徴量のフィルタリング
    features = features[meta_feature_names].copy()
    features = features.fillna(0)

    # Backtest用データ
    bt_data = ohlcv.copy()
    bt_data.columns = [col.capitalize() for col in bt_data.columns]
    bt_data.index.name = "timestamp"

    return bt_data, features, meta_model


# --- グローバル変数 ---
GLOBAL_DATA = None
GLOBAL_FEATURES = None
GLOBAL_MODEL = None


def objective(trial):
    global GLOBAL_DATA, GLOBAL_FEATURES, GLOBAL_MODEL

    # パラメータ探索範囲
    # threshold: 0.05 ~ 0.60 (高すぎるとトレードしなくなるため)
    threshold = trial.suggest_float("threshold", 0.05, 0.60, step=0.05)
    # scaling_factor: 1.0 ~ 10.0
    scaling_factor = trial.suggest_float("scaling_factor", 1.0, 10.0, step=0.5)

    # クラス変数設定
    MetaEmaCrossStrategy.threshold = threshold
    MetaEmaCrossStrategy.scaling_factor = scaling_factor
    MetaEmaCrossStrategy.meta_model = GLOBAL_MODEL
    MetaEmaCrossStrategy.features = GLOBAL_FEATURES

    # バックテスト実行
    bt = Backtest(GLOBAL_DATA, MetaEmaCrossStrategy, cash=10000, commission=0.001)
    stats = bt.run()

    trades = stats["# Trades"]
    ret = stats["Return [%]"]
    sqn = stats["SQN"]
    win_rate = stats["Win Rate [%]"]

    # 制約条件
    if trades < 10:
        return -1000.0  # トレード回数が少なすぎる場合は除外

    # 目的関数: SQN を最大化
    return sqn


if __name__ == "__main__":
    try:
        # データ準備
        GLOBAL_DATA, GLOBAL_FEATURES, GLOBAL_MODEL = prepare_data()

        logger.info("Starting optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)  # 回数を30回に設定 (時間がかかるため)

        logger.info("Optimization finished!")
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best value (SQN): {study.best_value}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
