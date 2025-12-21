"""
Feature Reduction & Optimization Tool (Real DB Data Version)

実際のDBデータを使用した特徴量削減の検証（verify）と、
Optuna による完全パイプライン最適化（optimize）を実行する統合ツール。

Usage:
    # 検証モード（既存機能）
    python verify_feature_reduction.py --mode verify

    # 最適化モード（CASH: Combined Algorithm Selection and Hyperparameter）
    python verify_feature_reduction.py --mode optimize --n_trials 50

    # データ制限付き
    python verify_feature_reduction.py --mode optimize --n_trials 10 --data_limit 500
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# プロジェクトルートをパスに追加
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.services.ml.feature_engineering.feature_engineering_service import (  # noqa: E402
    FeatureEngineeringService,
)
from app.services.ml.feature_selection.feature_selector import (  # noqa: E402
    FeatureSelector,
)
from app.services.ml.label_generation.presets import apply_preset_by_name  # noqa: E402
from app.services.ml.optimization.optimization_service import (  # noqa: E402
    OptimizationService,
)
from database.connection import SessionLocal  # noqa: E402
from database.repositories.funding_rate_repository import (  # noqa: E402
    FundingRateRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository  # noqa: E402
from database.repositories.open_interest_repository import (  # noqa: E402
    OpenInterestRepository,
)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


def run_verify_mode(symbol: str, timeframe: str, limit: int):
    """検証モード: 既存の特徴量削減検証を実行"""
    print(f"\n{'='*60}")
    print(f" VERIFY MODE: Feature Reduction Test ({symbol}, {timeframe})")
    print(f"{ '='*60}\n")

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

    # 特徴量選択デモ
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
    selection_time = time.time() - start_time

    details = getattr(selector_demo, "selection_details_", {})

    print(
        f"Final feature count: {X_selected_demo.shape[1]} "
        f"(Selection time: {selection_time:.2f}s)"
    )
    print(f"Reduction rate: {(1 - X_selected_demo.shape[1]/n_features)*100:.1f}%")

    if "stages" in details:
        print("\n--- Selection Stages ---")
        for i, stage in enumerate(details["stages"]):
            print(
                f"Stage {i+1} [{stage['method']}]: "
                f"{stage['selected_count']} features remaining"
            )

    print("\n--- Selected Features List ---")
    selected_cols = X_selected_demo.columns.tolist()
    for i, col in enumerate(selected_cols):
        print(f"{i+1:2d}. {col}")

    # 今回追加した特徴量が含まれているかチェック
    new_features = [
        "Void_Oscillator",
        "Crypto_Leverage_Index",
        "FracDiff_Price",
        "FracDiff_OI",
        "OI_Price_Regime",
        "OI_Price_Confirmation",
        "OI_Volume_Ratio",
        "Hurst_Exponent_100",
        "Triplet_Imbalance",
        "Fakeout_Volume_Divergence",
    ]
    found_new = [f for f in new_features if f in selected_cols]
    print(f"\n[Check] New features survived: {len(found_new)} / {len(new_features)}")
    for f in found_new:
        print(f"  - {f}")

    # 精度比較
    tscv = TimeSeriesSplit(n_splits=3)
    clf = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)

    print("\n--- Accuracy Check (Balanced Accuracy, No Leakage) ---")

    print("Evaluating Baseline (All Features)...")
    scores_all_acc = cross_val_score(
        clf, X, y, cv=tscv, scoring="balanced_accuracy", n_jobs=-1
    )
    scores_all_f1 = cross_val_score(clf, X, y, cv=tscv, scoring="f1_macro", n_jobs=-1)
    print(
        f" All Features - Balanced Acc: {scores_all_acc.mean():.4f}, F1 Macro: {scores_all_f1.mean():.4f}"
    )

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

    scores_sel_acc = cross_val_score(
        pipe, X, y, cv=tscv, scoring="balanced_accuracy", n_jobs=-1
    )
    scores_sel_f1 = cross_val_score(pipe, X, y, cv=tscv, scoring="f1_macro", n_jobs=-1)

    print(
        f" Selected Features - Balanced Acc: {scores_sel_acc.mean():.4f}, F1 Macro: {scores_sel_f1.mean():.4f}"
    )
    print(f" Difference (Acc): {scores_sel_acc.mean() - scores_all_acc.mean():+.4f}")
    print(f" Difference (F1):  {scores_sel_f1.mean() - scores_all_f1.mean():+.4f}")

    print("\n--- Hold-out Test (Last 20%) ---")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ベースライン（全特徴量）
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import balanced_accuracy_score

    score_holdout = balanced_accuracy_score(y_test, y_pred)
    print(f"Baseline Score on Test Data: {score_holdout:.4f}")

    # 特徴量選択後
    pipe.fit(X_train, y_train)
    y_pred_sel = pipe.predict(X_test)
    score_holdout_sel = balanced_accuracy_score(y_test, y_pred_sel)
    print(f"Selected Features Score on Test Data: {score_holdout_sel:.4f}")


def run_optimize_mode(
    symbol: str,
    timeframe: str,
    limit: int,
    n_trials: int,
    output_path: Optional[str] = None,
):
    """最適化モード: CASH（Combined Algorithm Selection and Hyperparameter）を実行"""
    print(f"\n{'='*60}")
    print(" OPTIMIZE MODE: Full Pipeline Optimization (CASH)")
    print(f" Symbol: {symbol}, Timeframe: {timeframe}")
    print(f" Trials: {n_trials}, Data Limit: {limit}")
    print(f"{ '='*60}\n")

    # データ取得
    print("[*] Loading data from database...")
    ohlcv_df, fr_df, oi_df = fetch_db_data(symbol, timeframe, limit)
    if ohlcv_df.empty:
        print("Error: No data found.")
        return

    print(f"    Loaded {len(ohlcv_df)} OHLCV records")

    # スーパーセット生成
    print("\n[*] Generating feature superset...")
    fe_service = FeatureEngineeringService()
    superset_df = fe_service.create_feature_superset(ohlcv_df, fr_df, oi_df)
    print(f"    Generated {len(superset_df.columns)} columns (superset)")

    # ラベル生成
    print("\n[*] Generating labels...")
    labels, _ = apply_preset_by_name(ohlcv_df, "trend_scanning_1h")

    # データ整列
    common_idx = superset_df.index.intersection(labels.index)
    X = superset_df.loc[common_idx].copy()
    y = labels.loc[common_idx]

    if y.dtype == object:
        y = (y == "UP").astype(int)

    print(f"    Aligned data: {len(X)} samples")

    # 最適化実行
    print(f"\n[*] Starting optimization with {n_trials} trials...")
    print("    (This may take a while...)\n")

    opt_service = OptimizationService()
    result = opt_service.optimize_full_pipeline(
        feature_superset=X,
        labels=y,
        ohlcv_data=ohlcv_df,  # Added
        n_trials=n_trials,
        test_ratio=0.2,
    )

    # 結果表示
    print("\n" + "=" * 60)
    print(" [RESULTS] OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\n--- Best Parameters ---")
    for key, value in result["best_params"].items():
        print(f"    {key}: {value}")

    print("\n--- Scores ---")
    print(f"    Best Validation Score: {result['best_score']:.4f}")
    print(f"    Test Score (Best Params): {result['test_score']:.4f}")
    print(f"    Baseline Score (Default): {result['baseline_score']:.4f}")
    print(f"    Improvement: {result['improvement']:+.4f}")

    print("\n--- Statistics ---")
    print(f"    Total Evaluations: {result['total_evaluations']}")
    print(f"    Optimization Time: {result['optimization_time']:.1f}s")
    print(f"    Selected Features: {result['n_selected_features']}")

    # JSON出力
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_limit": limit,
            "n_trials": n_trials,
        },
        "results": {
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "test_score": result["test_score"],
            "baseline_score": result["baseline_score"],
            "improvement": result["improvement"],
            "total_evaluations": result["total_evaluations"],
            "optimization_time": result["optimization_time"],
            "n_selected_features": result["n_selected_features"],
        },
    }

    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(backend_root / "scripts" / "optimization_result.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="ML Pipeline Optimization & Verification Tool"
    )

    parser.add_argument(
        "--mode",
        choices=["verify", "optimize"],
        default="verify",
        help="Execution mode: 'verify' for feature reduction test, "
        "'optimize' for full pipeline optimization (CASH)",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT:USDT",
        help="Trading symbol (default: BTC/USDT:USDT)",
    )
    parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--data_limit",
        type=int,
        default=10000,
        help="Maximum data rows to use (default: 10000)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50, optimize mode only)",
    )
    parser.add_argument("--output", help="Output JSON file path (optimize mode only)")

    args = parser.parse_args()

    if args.mode == "verify":
        run_verify_mode(args.symbol, args.timeframe, args.data_limit)
    else:
        run_optimize_mode(
            args.symbol,
            args.timeframe,
            args.data_limit,
            args.n_trials,
            args.output,
        )


if __name__ == "__main__":
    main()
