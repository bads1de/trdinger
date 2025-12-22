import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトルートをパスに追加
backend_path = str(Path(__file__).resolve().parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from database.connection import SessionLocal
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.long_short_ratio_repository import LongShortRatioRepository
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.label_generation.presets import triple_barrier_method_preset

# ロガー設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_index(df):
    if df is None or df.empty:
        return df
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df[~df.index.duplicated(keep="first")].sort_index()


def fetch_all_data(symbol, timeframe, limit):
    """全ての必要なデータを取得し、インデックスを正規化して返す"""
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        fr_repo = FundingRateRepository(db)
        ls_repo = LongShortRatioRepository(db)
        oi_repo = OpenInterestRepository(db)

        # 助走期間(1週間分)を含めて多めにロード
        warmup = 200
        total_limit = limit + warmup

        ohlcv_df = normalize_index(
            ohlcv_repo.get_ohlcv_dataframe(symbol, timeframe, limit=total_limit)
        )
        if ohlcv_df.empty:
            return None, None, None, None, None

        start_time = ohlcv_df.index.min()

        # FR
        fr_data = fr_repo.get_funding_rate_data(symbol, start_time=start_time)
        fr_df = normalize_index(fr_repo.to_dataframe(fr_data, index_column="timestamp"))

        # LS
        ls_symbol = symbol.split(":")[0]
        ls_data = ls_repo.get_long_short_ratio_data(
            ls_symbol, period="1h", start_time=start_time
        )
        ls_df = normalize_index(ls_repo.to_dataframe(ls_data, index_column="timestamp"))

        # OI
        oi_data = oi_repo.get_open_interest_data(symbol, start_time=start_time)
        oi_df = normalize_index(oi_repo.to_dataframe(oi_data, index_column="timestamp"))

        # 1m OHLCV (Intraday用)
        ohlcv_1m = normalize_index(
            ohlcv_repo.get_ohlcv_dataframe(symbol, "1m", start_time=start_time)
        )

        return ohlcv_df, fr_df, ls_df, oi_df, ohlcv_1m
    finally:
        db.close()


def run_verify_mode(symbol, timeframe, limit):
    print(f"\n{'='*60}\n VERIFY MODE: Full Microstructure Power\n{'='*60}")

    ohlcv_df, fr_df, ls_df, oi_df, ohlcv_1m = fetch_all_data(symbol, timeframe, limit)
    if ohlcv_df is None:
        print("Error: No data found.")
        return

    fe_service = FeatureEngineeringService()

    # 1. スーパーセット生成 (魔改造LS含む)
    print("[*] Generating feature superset (including Advanced Sentiment)...")
    X_raw = fe_service.create_feature_superset(ohlcv_df, fr_df, oi_df, ls_df)

    # 2. 1分足統計 (Intraday)
    print("[*] Processing 1m intraday statistics...")
    agg_1m = fe_service.aggregate_intraday_features(ohlcv_1m)

    # 3. 統合
    X_full = X_raw.join(agg_1m, how="left").ffill().fillna(0)

    # 4. ラベルとトリガー (20h Breakout)
    y_raw = triple_barrier_method_preset(
        ohlcv_df, timeframe="1h", horizon_n=8, pt=1.5, sl=1.0
    )
    rolling_high = ohlcv_df["high"].rolling(window=20).max().shift(1)
    trigger_mask = ohlcv_df["close"] > rolling_high

    # 助走期間を避けて有効なインデックスを抽出
    warmup_period = 200
    valid_idx = X_full.index.intersection(y_raw.index).intersection(
        trigger_mask[trigger_mask].index
    )
    valid_idx = [i for i in valid_idx if i >= X_full.index[warmup_period]]

    X_model_all = X_full.loc[valid_idx]
    y_model_all = y_raw.loc[valid_idx]

    # 【Elite Set】不整合・歪み指標のみを厳選
    elite_keywords = [
        "LS_Price_Incongruence",
        "LS_Sentiment_Elasticity",
        "LS_FR_Stress_Index",
        "Intraday_Volatility_Zscore",
        "Intraday_Absorption",
        "Intraday_Volume_Buy_Ratio",
        "FR_Extremity_Zscore",
        "Returns_Kurtosis_50",
        "FracDiff_Price",
        "Intraday_Volume_Concentration",
    ]
    elite_cols = [c for c in X_model_all.columns if any(k in c for k in elite_keywords)]
    X_elite = X_model_all[elite_cols]

    print(f"[*] Total valid signals: {len(X_model_all)}")
    print(f"[*] Using {len(elite_cols)} elite features for final verification.")

    # 評価 (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=5)
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )

    scores = []
    for train_idx, test_idx in tscv.split(X_elite):
        model.fit(X_elite.iloc[train_idx], y_model_all.iloc[train_idx])
        y_pred = model.predict(X_elite.iloc[test_idx])
        scores.append(balanced_accuracy_score(y_model_all.iloc[test_idx], y_pred))

    final_acc = np.mean(scores)
    print(
        f"\n--- Final Results ---\n  Balanced Accuracy (5-fold CV): {final_acc:.4f}\n"
    )

    # JSON出力
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "total_valid_signals": len(X_model_all),
        "original_features_count": len(X_model_all.columns),
        "original_features": X_model_all.columns.tolist(),
        "elite_features_count": len(elite_cols),
        "elite_features": elite_cols,
        "cv_scores": scores,
        "final_balanced_accuracy": float(final_acc),
    }

    output_path = Path(__file__).parent / "verify_feature_reduction_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"[*] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "optimize"], default="verify")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    if args.mode == "verify":
        run_verify_mode(args.symbol, args.timeframe, args.limit)
