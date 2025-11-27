import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
import glob
import os
import pandas_ta as ta

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.services.ml.label_generation.presets import forward_classification_preset
from database.connection import SessionLocal
from scripts.feature_evaluation.common_feature_evaluator import CommonFeatureEvaluator
from scripts.ml_optimization.run_ml_pipeline import MLPipeline

# バックテストライブラリ (backtesting.pyを使用と仮定)
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmaCrossStrategy(Strategy):
    """単純なEMAクロス戦略"""
    n1 = 10
    n2 = 20
    
    def init(self):
        self.ema1 = self.I(pd.Series(self.data.Close).ewm(span=self.n1).mean)
        self.ema2 = self.I(pd.Series(self.data.Close).ewm(span=self.n2).mean)

    def next(self):
        if crossover(self.ema1, self.ema2):
            self.buy()
        elif crossover(self.ema2, self.ema1):
            self.position.close()

class MetaEmaCrossStrategy(Strategy):
    """EMAクロス + MLフィルタリング戦略 (Long/Short)"""
    n1 = 10
    n2 = 20
    
    # 外部から注入されるパラメータ
    meta_model = None 
    features = None   
    threshold = 0.2   # 最適化結果
    scaling_factor = 3.0 # 最適化結果

    def init(self):
        self.ema1 = self.I(pd.Series(self.data.Close).ewm(span=self.n1).mean)
        self.ema2 = self.I(pd.Series(self.data.Close).ewm(span=self.n2).mean)
        # ATR計算
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), 14)

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
                atr_value = self.data.Close[-1] * 0.01 # fallback
                
            normalized_ema_diff = ema_diff / atr_value
            
            # primary_proba_value は 0-1 の範囲にスケーリング
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
            meta_service_added_suffixes = ["_proba", "_prob_mean", "_prob_std", "_prob_min", "_prob_max"]
            base_model_names_from_oof = self.meta_model.base_model_names

            # meta_model.model (RandomForestClassifier) の feature_names_in_ を取得
            rf_feature_names = list(self.meta_model.model.feature_names_in_)

            pure_X_feature_names = []
            for rf_feat in rf_feature_names:
                is_added = False
                if any(rf_feat.endswith(suffix) for suffix in meta_service_added_suffixes):
                    is_added = True
                if rf_feat in base_model_names_from_oof:
                    is_added = True
                
                if not is_added:
                    pure_X_feature_names.append(rf_feat)
            
            # 現在のfeature_rowから、pure_X_feature_names に対応するカラムを抽出
            feature_row_for_X = feature_row_raw[pure_X_feature_names]
            
            # Primary Proba
            primary_proba = pd.Series([primary_proba_value], index=feature_row_for_X.index)
            
            # Dummy Base Model Probs
            dummy_base_model_probs_data = {col: [1.0] for col in base_model_names_from_oof}
            base_model_probs_df = pd.DataFrame(dummy_base_model_probs_data, index=feature_row_for_X.index)
            
            # 予測
            prob = self.meta_model.predict(
                X=feature_row_for_X, 
                primary_proba=primary_proba, 
                base_model_probs_df=base_model_probs_df,
                threshold=self.threshold
            )
            
            return prob.iloc[0] == 1
            
        except Exception as e:
            logger.error(f"Error in meta filter: {e}")
            return False

def run_backtest():
    symbol = "BTC/USDT:USDT"
    timeframe = "1h"
    
    # データ準備
    logger.info("Preparing data for backtest...")
    
    # MLPipelineインスタンスを作成 (パス解決用)
    pipeline_instance = MLPipeline(enable_meta_labeling=False) 
    
    evaluator = CommonFeatureEvaluator()
    feature_service = FeatureEngineeringService()
    
    # データ取得
    data = evaluator.fetch_data(symbol=symbol, timeframe=timeframe, limit=10000)
    ohlcv = data.ohlcv.copy()
    
    # 特徴量生成
    features = feature_service.calculate_advanced_features(
        ohlcv_data=ohlcv,
        funding_rate_data=data.fr,
        open_interest_data=data.oi
    )
    
    feature_cols_to_exclude = ["open", "high", "low", "volume"]
    features = features.drop(columns=feature_cols_to_exclude, errors='ignore')
    features = features.fillna(0)
    
    bt_data = ohlcv.copy()
    bt_data.columns = [col.capitalize() for col in bt_data.columns]
    bt_data.index.name = "timestamp"

    # モデルロード
    try:
        # 最新の実行ディレクトリを探す（meta_labeling_service.joblibがあるもの）
        results_dir = Path("backend/results/ml_pipeline")
        valid_runs = []
        for d in results_dir.iterdir():
            if d.is_dir() and (d / "meta_labeling_service.joblib").exists():
                valid_runs.append(d)
        
        if not valid_runs:
            logger.error("No trained model found.")
            return

        latest_dir = sorted(valid_runs, key=lambda x: x.stat().st_mtime)[-1]
        logger.info(f"Loading meta model from: {latest_dir}")
        
        meta_model = joblib.load(latest_dir / "meta_labeling_service.joblib")
        
        with open(latest_dir / "best_params.json", "r", encoding="utf-8") as f:
            best_params_data = json.load(f)
        meta_feature_names = best_params_data["feature_names"]
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 1. ベース戦略のバックテスト
    logger.info("\n=== Running Base Strategy (EMA Cross) ===")
    bt_base = Backtest(bt_data, EmaCrossStrategy, cash=10000, commission=0.001)
    stats_base = bt_base.run()
    print(stats_base)
    
    # 2. メタ戦略のバックテスト
    logger.info("\n=== Running Meta Strategy (EMA Cross + ML Filter [Long/Short]) ===")
    
    MetaEmaCrossStrategy.features = features[meta_feature_names].copy()
    MetaEmaCrossStrategy.meta_model = meta_model
    MetaEmaCrossStrategy.threshold = 0.2 # 最適化値
    MetaEmaCrossStrategy.scaling_factor = 3.0 # 最適化値
    
    bt_meta = Backtest(bt_data, MetaEmaCrossStrategy, cash=10000, commission=0.001)
    stats_meta = bt_meta.run()
    print(stats_meta)
    
    # 比較
    logger.info("\n=== Comparison ===")
    logger.info(f"Return: {stats_base['Return [%]']:.2f}% -> {stats_meta['Return [%]']:.2f}%")
    logger.info(f"Sharpe Ratio: {stats_base['Sharpe Ratio']:.4f} -> {stats_meta['Sharpe Ratio']:.4f}")
    logger.info(f"Win Rate: {stats_base['Win Rate [%]']:.2f}% -> {stats_meta['Win Rate [%]']:.2f}%")
    logger.info(f"Trades: {stats_base['# Trades']} -> {stats_meta['# Trades']}")
    logger.info(f"Max Drawdown: {stats_base['Max. Drawdown [%]']:.2f}% -> {stats_meta['Max. Drawdown [%]']:.2f}%")

if __name__ == "__main__":
    run_backtest()