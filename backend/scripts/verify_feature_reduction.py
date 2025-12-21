"""
Feature Reduction & Accuracy Impact Verification Script (Real DB Data Version)

実際のDBデータを使用した特徴量削減の動作確認と精度への影響を検証するスクリプト
- DBからOHLCV/FR/OIデータを取得
- FeatureEngineeringServiceによる特徴量生成
- 特徴量削減の各ステージの動作確認
- 削減前後の精度比較
"""

import logging
import sys
import os
import pandas as pd
import numpy as np
import time
from typing import Optional, Tuple
from datetime import datetime

# プロジェクトルートをパスに追加
from pathlib import Path

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.services.ml.feature_selection.feature_selector import FeatureSelector
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.label_generation.presets import apply_preset_by_name
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# ロガー設定
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def fetch_db_data(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    limit: int = 10000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """DBからOHLCV/FR/OIデータを取得"""
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        fr_repo = FundingRateRepository(db)
        oi_repo = OpenInterestRepository(db)

        start_time = datetime.fromisoformat(start_date) if start_date else None
        end_time = datetime.fromisoformat(end_date) if end_date else None

        ohlcv_df = ohlcv_repo.get_ohlcv_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )

        if ohlcv_df.empty:
            return pd.DataFrame(), None, None

        actual_start, actual_end = ohlcv_df.index.min(), ohlcv_df.index.max()

        fr_df = None
        try:
            fr_records = fr_repo.get_funding_rate_data(
                symbol=symbol, start_time=actual_start, end_time=actual_end
            )
            if fr_records:
                fr_df = fr_repo.to_dataframe(
                    fr_records,
                    {"funding_timestamp": "timestamp", "funding_rate": "funding_rate"},
                    "timestamp",
                )
                if "funding_timestamp" in fr_df.columns:
                    fr_df = fr_df.drop(columns=["funding_timestamp"])
        except Exception:
            pass

        oi_df = None
        try:
            oi_records = oi_repo.get_open_interest_data(
                symbol=symbol, start_time=actual_start, end_time=actual_end
            )
            if oi_records:
                oi_df = pd.DataFrame(
                    [
                        {
                            "timestamp": r.data_timestamp,
                            "open_interest_value": r.open_interest_value,
                        }
                        for r in oi_records
                    ]
                )
                oi_df.set_index("timestamp", inplace=True)
        except Exception:
            pass

        return ohlcv_df, fr_df, oi_df
    finally:
        db.close()


def run_verification(symbol="BTC/USDT:USDT", timeframe="1h", limit=10000):
    print(f"\n--- Feature Reduction Test with REAL Data ({symbol}, {timeframe}) ---")

    ohlcv_df, fr_df, oi_df = fetch_db_data(symbol, timeframe, limit)
    if ohlcv_df.empty:
        print("Error: No data found.")
        return

    # 特徴量生成
    fe_service = FeatureEngineeringService()
    features_df = fe_service.calculate_advanced_features(ohlcv_df, fr_df, oi_df)

    # ラベル生成 (Trend Scanning 1h)
    labels, _ = apply_preset_by_name(ohlcv_df, "trend_scanning_1h")

    # データ整列
    common_idx = features_df.index.intersection(labels.index)
    X = features_df.loc[common_idx].copy()
    y = labels.loc[common_idx]

    # クリーニング
    exclude_cols = ["open", "high", "low", "close", "volume"]
    X = X[[c for c in X.columns if c not in exclude_cols]]
    if y.dtype == object:
        y = (y == "UP").astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    n_samples, n_features = X.shape
    print(f"Dataset prepared: {n_samples} samples, {n_features} features")

    # 特徴量選択（デモ：全データでの挙動確認）
    # 注意: ここではデータ全体の傾向を見るために全データでfitしていますが、
    # 実際の精度検証にはこれ（X_selected_demo）を使ってはいけません（リークするため）。
    print("\n--- Feature Selection Demo (Using All Data for Visualization) ---")
    selector_demo = FeatureSelector(
        method="staged",
        correlation_threshold=0.85,
        min_features=10,
        cv_folds=3,
        n_jobs=-1,
    )

    start_time = time.time()
    X_selected_demo = selector_demo.fit_transform(X, y)
    details = getattr(selector_demo, "selection_details_", {})
    selection_time = time.time() - start_time

    print(
        f"Final feature count: {X_selected_demo.shape[1]} (Selection time: {selection_time:.2f}s)"
    )
    print(f"Reduction rate: {(1 - X_selected_demo.shape[1]/n_features)*100:.1f}%")

    if "stages" in details:
        for i, stage in enumerate(details["stages"]):
            print(
                f"Stage {i+1} [{stage['method']}]: {stage['selected_count']} features remaining"
            )

    # 精度比較（厳密な検証）
    # Pipelineを使用することで、CVの各フォールド内で特徴量選択を行い、リークを防ぎます。
    from sklearn.pipeline import Pipeline

    tscv = TimeSeriesSplit(n_splits=3)
    clf = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)

    print("\n--- Accuracy Check (Balanced Accuracy, No Leakage) ---")

    # 1. 全特徴量（ベースライン）
    print("Evaluating Baseline (All Features)...")
    score_all = cross_val_score(
        clf, X, y, cv=tscv, scoring="balanced_accuracy", n_jobs=-1
    ).mean()
    print(f" All Features Score: {score_all:.4f}")

    # 2. 特徴量選択あり（正しい検証）
    print("Evaluating Feature Selection (Inside CV Pipeline)...")
    pipe = Pipeline(
        [
            (
                "selector",
                FeatureSelector(
                    method="staged",
                    correlation_threshold=0.85,
                    min_features=10,
                    cv_folds=3,
                    n_jobs=-1,
                ),
            ),
            ("clf", clf),
        ]
    )

    # パイプライン全体をCVにかけることで、trainデータのみでselect -> testデータでevaluateが行われる
    score_sel = cross_val_score(
        pipe, X, y, cv=tscv, scoring="balanced_accuracy", n_jobs=-1
    ).mean()

    print(f" Selected Features Score: {score_sel:.4f}")
    print(f" Difference: {score_sel - score_all:+.4f}")


if __name__ == "__main__":
    run_verification()
