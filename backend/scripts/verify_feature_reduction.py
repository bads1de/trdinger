import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトルートをパスに追加
backend_path = str(Path(__file__).resolve().parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from database.connection import SessionLocal  # noqa: E402
from database.repositories.funding_rate_repository import (  # noqa: E402
    FundingRateRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository  # noqa: E402
from database.repositories.open_interest_repository import (  # noqa: E402
    OpenInterestRepository,
)
from database.repositories.long_short_ratio_repository import (  # noqa: E402
    LongShortRatioRepository,
)
from app.services.ml.feature_engineering.feature_engineering_service import (  # noqa: E402
    FeatureEngineeringService,
)
from app.services.ml.feature_selection.feature_selector import (  # noqa: E402
    FeatureSelector,
)
from app.services.ml.label_generation.presets import (  # noqa: E402
    triple_barrier_method_preset,
)
from app.services.ml.ensemble.meta_labeling import MetaLabelingService  # noqa: E402

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


def run_analysis_pipeline(
    symbol, timeframe, limit, labeling_method="trend_scanning", save_json=True
):
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
    
    # 3.5 特徴量拡張 (ラグ、相互作用等)
    print(f"[*] Expanding features (current: {len(X_full.columns)})...")
    X_full = fe_service.expand_features(X_full)
    print(f"[*] Expanded to {len(X_full.columns)} features.")

    y_raw = None
    w_model_all = None

    if labeling_method == "trend_scanning":
        # 4. ラベル (Trend Scanning)
        print("[*] Generating labels using Trend Scanning with t-values as weights...")
        from app.services.ml.label_generation.trend_scanning import TrendScanning

        ts_scanner = TrendScanning(
            min_window=5,
            max_window=20,
            step=1,
            min_t_value=2.0,
        )

        labels = ts_scanner.get_labels(ohlcv_df["close"], use_log_price=True)
        y_raw = labels["bin"]
        # トレンドスキャンニングのみ、t値をウェイトとして使用
        raw_weights = labels["t_value"].abs()
        
        # 検証用：バイナリ分類
        y_raw = (y_raw == 1).astype(int)

    elif labeling_method == "triple_barrier":
        # 4. ラベル (Triple Barrier)
        print("[*] Generating labels using Triple Barrier Method...")
        y_raw = triple_barrier_method_preset(
            ohlcv_df,
            timeframe=timeframe,
            horizon_n=24,
            pt=1.0,
            sl=1.0,
            min_ret=0.001,
            use_atr=True,
            atr_period=14,
        )
        y_raw = y_raw.dropna().astype(int)
        # TBMはウェイトなし (None)
        raw_weights = None

    else:
        print(f"Error: Unknown labeling method '{labeling_method}'")
        return

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
    if raw_weights is not None:
        w_model_all = raw_weights.loc[valid_idx]
    else:
        w_model_all = None

    # 自動特徴量選択 (FeatureSelector利用)
    print("[*] Running Automatic Feature Selection (Staged Strategy)...")
    from app.services.ml.feature_selection.feature_selector import FeatureSelector

    selector = FeatureSelector(
        method="staged",
        cv_folds=5,
        cv_strategy="timeseries",
        n_jobs=-1,  # 並列処理
        variance_threshold=0.0001,
        correlation_threshold=0.95,
        min_features=10,
        random_state=42,
    )

    # 学習と変換
    X_elite = selector.fit_transform(X_model_all, y_model_all)
    elite_cols = X_elite.columns.tolist()

    print(f"[*] Total valid signals: {len(X_model_all)}")
    print(
        f"[*] Selected {len(elite_cols)} features from {len(X_model_all.columns)} original features."
    )
    print(f"[*] Top 10 Features: {elite_cols[:10]}")

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

    # Meta-Labeling Service
    meta_service = MetaLabelingService(
        use_feature_selection=True,
        feature_selection_params={
            "method": "staged",
            "min_features": 10,  # 情報を保持
            "random_state": 42,
        },
    )

    primary_scores = []
    meta_scores = []

    print(f"\n{'='*20} Cross Validation Results {'='*20}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_elite)):
        # --- Training Data ---
        X_train, y_train = X_elite.iloc[train_idx], y_model_all.iloc[train_idx]
        X_test, y_test = X_elite.iloc[test_idx], y_model_all.iloc[test_idx]
        
        # ウェイトの抽出 (TSの場合はt値、TBMの場合はNone)
        w_train = w_model_all.iloc[train_idx] if w_model_all is not None else None

        # 1. Train Primary Model
        model.fit(X_train, y_train, sample_weight=w_train)

        # Predictions
        primary_proba_train = pd.Series(
            model.predict_proba(X_train)[:, 1], index=X_train.index
        )
        primary_proba_test = pd.Series(
            model.predict_proba(X_test)[:, 1], index=X_test.index
        )

        # Primary Metrics
        y_pred = (primary_proba_test >= 0.5).astype(int)

        p_acc = balanced_accuracy_score(y_test, y_pred)
        p_prec = precision_score(y_test, y_pred, zero_division=0)

        # Count primary trades
        n_primary_trades = int(y_pred.sum())
        y_test_np = y_test.values.astype(int)
        y_pred_np = (
            y_pred if isinstance(y_pred, np.ndarray) else y_pred.values.astype(int)
        )
        n_primary_hits = int((y_pred_np * y_test_np).sum())

        primary_scores.append(
            {
                "acc": p_acc,
                "prec": p_prec,
                "trades": n_primary_trades,
                "hits": n_primary_hits,
            }
        )

        # 2. Train Meta Model
        base_probs_train = pd.DataFrame(index=X_train.index)
        base_probs_test = pd.DataFrame(index=X_test.index)

        # train (Using the improved MetaLabelingService with DynamicMetaSelector)
        meta_res = meta_service.train(
            X_train, y_train, primary_proba_train, base_probs_train, threshold=0.5
        )

        if meta_res["status"] == "skipped":
            print(
                f"Fold {fold+1}: Primary Acc={p_acc:.4f} | Meta SKIPPED ({meta_res['reason']})"
            )
            # Fallback stats (same as primary)
            meta_scores.append(
                {
                    "acc": p_acc,
                    "prec": p_prec,
                    "trades": n_primary_trades,
                    "hits": n_primary_hits,
                }
            )
        else:
            meta_pred = meta_service.predict(
                X_test, primary_proba_test, base_probs_test, threshold=0.5
            )

            # Calculate Metrics
            m_acc = balanced_accuracy_score(y_test, meta_pred)
            m_prec = precision_score(y_test, meta_pred, zero_division=0)

            # Count trades
            n_meta_trades = int(meta_pred.sum())
            meta_pred_np = (
                meta_pred
                if isinstance(meta_pred, np.ndarray)
                else meta_pred.values.astype(int)
            )
            n_meta_hits = int((meta_pred_np * y_test_np).sum())

            meta_scores.append(
                {
                    "acc": m_acc,
                    "prec": m_prec,
                    "trades": n_meta_trades,
                    "hits": n_meta_hits,
                }
            )

            print(
                f"Fold {fold+1}: Primary Trades={n_primary_trades:<4} ({n_primary_hits:<4} hits) -> "
                f"Meta Trades={n_meta_trades:<4} ({n_meta_hits:<4} hits) | Meta Prec={m_prec:.4f}"
            )

    avg_p_acc = np.mean([s["acc"] for s in primary_scores])
    avg_m_acc = np.mean([s["acc"] for s in meta_scores])
    avg_p_prec = np.mean([s["prec"] for s in primary_scores])
    avg_m_prec = np.mean([s["prec"] for s in meta_scores])

    total_p_trades = sum([s["trades"] for s in primary_scores])
    total_m_trades = sum([s["trades"] for s in meta_scores])

    print(f"\n{'='*20} Final Comparison {'='*20}")
    print(
        f"Primary Model: Balanced Acc = {avg_p_acc:.4f} | Precision = {avg_p_prec:.4f} | Total Trades = {total_p_trades}"
    )
    print(
        f"Meta Model:    Balanced Acc = {avg_m_acc:.4f} | Precision = {avg_m_prec:.4f} | Total Trades = {total_m_trades}"
    )
    print(
        f"Improvement:   Balanced Acc = {avg_m_acc - avg_p_acc:+.4f} | Precision = {avg_m_prec - avg_p_prec:+.4f}"
    )

    winner = "Meta Model" if avg_m_prec > avg_p_prec else "Primary Model"
    print(f"\n WINNER: {winner} ")

    # JSON出力
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "labeling_method": labeling_method,
        "total_valid_signals": len(X_model_all),
        "elite_features_count": len(elite_cols),
        "elite_features": elite_cols,
        "primary_scores": primary_scores,
        "meta_scores": meta_scores,
        "final_primary_acc": float(avg_p_acc),
        "final_meta_acc": float(avg_m_acc),
        "winner": winner,
    }

    if save_json:
        output_path = Path(__file__).parent / f"verify_result_{labeling_method}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"[*] Results saved to {output_path}")

    return result_data


def run_verify_mode(symbol, timeframe, limit, labeling_method):
    run_analysis_pipeline(symbol, timeframe, limit, labeling_method, save_json=True)


def run_compare_mode(symbol, timeframe, limit):
    print(f"\n{'#'*80}\n COMPARISON MODE: Trend Scanning vs Triple Barrier\n{'#'*80}")

    # 1. Trend Scanning
    print("\n>>> [1/2] Running Trend Scanning Pipeline...")
    res_ts = run_analysis_pipeline(
        symbol, timeframe, limit, "trend_scanning", save_json=False
    )

    # 2. Triple Barrier
    print("\n>>> [2/2] Running Triple Barrier Pipeline...")
    res_tb = run_analysis_pipeline(
        symbol, timeframe, limit, "triple_barrier", save_json=False
    )

    print(f"\n{'='*60}")
    print(f" FINAL COMPARISON: {symbol} ({timeframe})")
    print(f"{'='*60}")

    # ヘッダー
    print(
        f"{'Metric':<20} | {'Trend Scanning':<20} | {'Triple Barrier':<20} | {'Diff (TS - TB)':<15}"
    )
    print("-" * 85)

    metrics = [
        ("Meta Model Acc", "final_meta_acc", "{:.4f}"),
        ("Primary Model Acc", "final_primary_acc", "{:.4f}"),
        ("Elite Features", "elite_features_count", "{}"),
        ("Valid Signals", "total_valid_signals", "{}"),
    ]

    for label, key, fmt in metrics:
        val_ts = res_ts[key]
        val_tb = res_tb[key]

        diff_str = "-"
        if isinstance(val_ts, (int, float)) and isinstance(val_tb, (int, float)):
            diff = val_ts - val_tb
            diff_str = f"{diff:+.4f}" if isinstance(val_ts, float) else f"{diff:+d}"

        # フォーマット
        v_ts_str = fmt.format(val_ts)
        v_tb_str = fmt.format(val_tb)

        print(f"{label:<20} | {v_ts_str:<20} | {v_tb_str:<20} | {diff_str:<15}")

    # トレード数合計の計算
    ts_trades = sum(s["trades"] for s in res_ts["meta_scores"])
    tb_trades = sum(s["trades"] for s in res_tb["meta_scores"])
    print(
        f"{'Meta Total Trades':<20} | {ts_trades:<20} | {tb_trades:<20} | {ts_trades - tb_trades:<+15}"
    )

    # 平均Precision
    ts_prec = np.mean([s["prec"] for s in res_ts["meta_scores"]])
    tb_prec = np.mean([s["prec"] for s in res_tb["meta_scores"]])
    print(
        f"{'Meta Avg Precision':<20} | {ts_prec:<20.4f} | {tb_prec:<20.4f} | {ts_prec - tb_prec:+.4f}"
    )

    print("-" * 85)
    print("Top 3 Features (Trend Scanning):", res_ts["elite_features"][:3])
    print("Top 3 Features (Triple Barrier):", res_tb["elite_features"][:3])
    print("=" * 60)

    # 比較結果保存
    cmp_result = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "trend_scanning": res_ts,
        "triple_barrier": res_tb,
    }
    output_path = Path(__file__).parent / "comparison_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cmp_result, f, indent=4, ensure_ascii=False)
    print(f"[*] Comparison results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["verify", "optimize", "compare"], default="verify"
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument(
        "--labeling-method",
        choices=["trend_scanning", "triple_barrier"],
        default="trend_scanning",
        help="Choose labeling method: trend_scanning or triple_barrier",
    )
    args = parser.parse_args()

    if args.mode == "verify":
        run_verify_mode(args.symbol, args.timeframe, args.limit, args.labeling_method)
    elif args.mode == "compare":
        run_compare_mode(args.symbol, args.timeframe, args.limit)
