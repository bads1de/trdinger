"""
データベースの全データを使用したファンディングレート特徴量評価

90日間ではなく、DBに存在する全期間のデータを使用して評価を実施します。
"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app.services.ml.feature_engineering.funding_rate_features import (
    FundingRateFeatureCalculator,
)
from database.connection import get_session
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository

warnings.filterwarnings("ignore")


def load_all_data(symbol: str = "BTC/USDT:USDT"):
    """
    データベースから全データを読み込み

    期間指定なしで全データを取得します。

    Args:
        symbol: 取引ペア

    Returns:
        (ohlcv_df, fr_df): OHLCVとファンディングレートのDataFrame
    """
    session = get_session()

    ohlcv_repo = OHLCVRepository(session)
    fr_repo = FundingRateRepository(session)

    print("データベースから全データを読み込み中...")

    # 全OHLCVデータ（期間指定なし）
    ohlcv_data = ohlcv_repo.get_all_by_symbol(symbol=symbol, timeframe="1h")

    # 全Funding Rateデータ
    fr_data = fr_repo.get_all_by_symbol(symbol=symbol)

    if not ohlcv_data:
        raise ValueError(f"データがありません: {symbol}")

    print(f"OHLCV: {len(ohlcv_data)}行を取得")
    print(f"FR: {len(fr_data)}行を取得")

    # DataFrameに変換
    ohlcv_df = pd.DataFrame(
        [
            {
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume,
            }
            for d in ohlcv_data
        ]
    )

    fr_df = (
        pd.DataFrame(
            [
                {
                    "timestamp": d.funding_timestamp,
                    "funding_rate": d.funding_rate,
                }
                for d in fr_data
            ]
        )
        if fr_data
        else pd.DataFrame()
    )

    # データ期間を表示
    if not ohlcv_df.empty:
        start_date = ohlcv_df["timestamp"].min()
        end_date = ohlcv_df["timestamp"].max()
        days = (end_date - start_date).days
        print(f"データ期間: {start_date} - {end_date} ({days}日間)")

    session.close()
    return ohlcv_df, fr_df


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ベースライン特徴量（FR特徴量なし）

    Args:
        df: 入力DataFrame

    Returns:
        特徴量を追加したDataFrame
    """
    df = df.copy()

    # 価格変化率
    for period in [1, 3, 6, 12, 24]:
        df[f"returns_{period}h"] = df["close"].pct_change(period)

    # 移動平均
    for period in [7, 14, 30, 50]:
        df[f"ma_{period}"] = df["close"].rolling(period).mean()
        df[f"ma_ratio_{period}"] = df["close"] / df[f"ma_{period}"]

    # ボラティリティ
    df["volatility_24h"] = df["close"].rolling(24).std()
    df["volatility_168h"] = df["close"].rolling(168).std()

    # 出来高
    df["volume_ma_24h"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_24h"] + 1e-10)

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    return df


def evaluate_with_cross_validation(
    df_baseline: pd.DataFrame, df_fr: pd.DataFrame, n_splits: int = 5
):
    """
    時系列交差検証で性能を評価

    Args:
        df_baseline: ベースライン特徴量のDataFrame
        df_fr: FR特徴量ありのDataFrame
        n_splits: 交差検証の分割数

    Returns:
        評価結果の辞書
    """
    # ターゲット変数
    df_baseline["target"] = df_baseline["close"].pct_change(1).shift(-1)
    df_fr["target"] = df_fr["close"].pct_change(1).shift(-1)

    # 欠損値除去
    df_baseline = df_baseline.dropna().reset_index(drop=True)
    df_fr = df_fr.dropna().reset_index(drop=True)
    
    # 両方のDataFrameのサイズを最小の方に揃える
    min_len = min(len(df_baseline), len(df_fr))
    df_baseline = df_baseline.iloc[:min_len]
    df_fr = df_fr.iloc[:min_len]

    print(f"\n有効データ: {len(df_baseline)}行")

    # 特徴量列を特定
    baseline_features = [
        col
        for col in df_baseline.columns
        if col
        not in ["timestamp", "target", "open", "high", "low", "close", "volume"]
    ]

    # FR DataFrameから既存のベースライン特徴量を除外
    fr_only_features = [
        col
        for col in df_fr.columns
        if col.startswith("fr_")
        or col.startswith("funding_")
        or col.startswith("regime_")
    ]
    
    # FR DataFrameで使用する全特徴量（ベースライン特徴量が含まれている場合は使用）
    available_baseline_in_fr = [
        col for col in baseline_features if col in df_fr.columns
    ]
    
    all_features = available_baseline_in_fr + fr_only_features

    print(f"ベースライン特徴量: {len(baseline_features)}個")
    print(f"FR特徴量のみ: {len(fr_only_features)}個")
    print(f"FR側で利用可能なベースライン特徴量: {len(available_baseline_in_fr)}個")
    print(f"合計（FR側）: {len(all_features)}個")

    # 時系列交差検証
    print(f"\n{n_splits}-fold TimeSeriesSplit評価中...")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    baseline_results = []
    fr_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_baseline), 1):
        print(f"  Fold {fold}/{n_splits}...", end=" ")

        # ベースライン評価
        X_train_base = df_baseline.iloc[train_idx][baseline_features]
        X_test_base = df_baseline.iloc[test_idx][baseline_features]
        y_train = df_baseline.iloc[train_idx]["target"]
        y_test = df_baseline.iloc[test_idx]["target"]

        model_base = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        model_base.fit(X_train_base, y_train)
        y_pred_base = model_base.predict(X_test_base)

        baseline_results.append(
            {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_base)),
                "mae": mean_absolute_error(y_test, y_pred_base),
                "r2": r2_score(y_test, y_pred_base),
            }
        )

        # FR特徴量あり評価
        X_train_fr = df_fr.iloc[train_idx][all_features]
        X_test_fr = df_fr.iloc[test_idx][all_features]
        y_train_fr = df_fr.iloc[train_idx]["target"]
        y_test_fr = df_fr.iloc[test_idx]["target"]

        model_fr = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        model_fr.fit(X_train_fr, y_train_fr)
        y_pred_fr = model_fr.predict(X_test_fr)

        fr_results.append(
            {
                "rmse": np.sqrt(mean_squared_error(y_test_fr, y_pred_fr)),
                "mae": mean_absolute_error(y_test_fr, y_pred_fr),
                "r2": r2_score(y_test_fr, y_pred_fr),
            }
        )

        print("OK")

    # 結果集計
    baseline_avg = {
        "rmse": np.mean([r["rmse"] for r in baseline_results]),
        "rmse_std": np.std([r["rmse"] for r in baseline_results]),
        "mae": np.mean([r["mae"] for r in baseline_results]),
        "r2": np.mean([r["r2"] for r in baseline_results]),
    }

    fr_avg = {
        "rmse": np.mean([r["rmse"] for r in fr_results]),
        "rmse_std": np.std([r["rmse"] for r in fr_results]),
        "mae": np.mean([r["mae"] for r in fr_results]),
        "r2": np.mean([r["r2"] for r in fr_results]),
    }

    # 改善率
    improvements = {
        "rmse": ((baseline_avg["rmse"] - fr_avg["rmse"]) / baseline_avg["rmse"])
        * 100,
        "mae": ((baseline_avg["mae"] - fr_avg["mae"]) / baseline_avg["mae"]) * 100,
        "r2": (
            ((fr_avg["r2"] - baseline_avg["r2"]) / abs(baseline_avg["r2"]) * 100)
            if baseline_avg["r2"] != 0
            else 0
        ),
    }

    return {
        "baseline": baseline_avg,
        "fr_enhanced": fr_avg,
        "improvements": improvements,
        "fold_results": {"baseline": baseline_results, "fr_enhanced": fr_results},
    }


def main():
    print("=" * 70)
    print("ファンディングレート特徴量 - 全データ評価")
    print("=" * 70)

    # 全データ読み込み
    try:
        ohlcv_df, fr_df = load_all_data()
    except Exception as e:
        print(f"\nエラー: {e}")
        print("\nデータ収集が必要な場合:")
        print("  python -m app.services.data_collection.bybit.ohlcv_service")
        print("  python -m app.services.data_collection.bybit.funding_rate_service")
        return

    # ベースライン特徴量作成
    print("\nベースライン特徴量作成中...")
    baseline_df = create_baseline_features(ohlcv_df)

    # FR特徴量作成
    print("FR特徴量計算中...")
    calculator = FundingRateFeatureCalculator()
    fr_enhanced_df = calculator.calculate_features(ohlcv_df, fr_df)

    # 評価実行
    results = evaluate_with_cross_validation(baseline_df, fr_enhanced_df, n_splits=5)

    # 結果表示
    print("\n" + "=" * 70)
    print("[RESULTS] 全データ評価結果")
    print("=" * 70)

    print("\n[Baseline - No FR Features]")
    print(
        f"  RMSE:     {results['baseline']['rmse']:.6f} (+/-{results['baseline']['rmse_std']:.6f})"
    )
    print(f"  MAE:      {results['baseline']['mae']:.6f}")
    print(f"  R2:       {results['baseline']['r2']:.4f}")

    print("\n[FR Feature Enhanced]")
    print(
        f"  RMSE:     {results['fr_enhanced']['rmse']:.6f} (+/-{results['fr_enhanced']['rmse_std']:.6f})"
    )
    print(f"  MAE:      {results['fr_enhanced']['mae']:.6f}")
    print(f"  R2:       {results['fr_enhanced']['r2']:.4f}")

    print("\n[Improvements]")
    rmse_imp = results["improvements"]["rmse"]
    mae_imp = results["improvements"]["mae"]
    r2_imp = results["improvements"]["r2"]

    print(f"  {'[+]' if rmse_imp > 0 else '[-]'} RMSE:    {rmse_imp:+.2f}%")
    print(f"  {'[+]' if mae_imp > 0 else '[-]'} MAE:     {mae_imp:+.2f}%")
    print(f"  {'[+]' if r2_imp > 0 else '[-]'} R2:      {r2_imp:+.2f}%")

    # 各foldの詳細
    print("\n[Fold Details]")
    print("Fold | Baseline RMSE | FR RMSE    | Improvement")
    print("-" * 55)
    for i, (base, fr) in enumerate(
        zip(
            results["fold_results"]["baseline"],
            results["fold_results"]["fr_enhanced"],
        ),
        1,
    ):
        imp = ((base["rmse"] - fr["rmse"]) / base["rmse"]) * 100
        print(f"{i:4d} | {base['rmse']:.6f}    | {fr['rmse']:.6f} | {imp:+.2f}%")

    print("\n" + "=" * 70)
    print("[CONCLUSION]")
    print("=" * 70)

    if rmse_imp > 0 and mae_imp > 0:
        if rmse_imp >= 5:
            print("[SUCCESS] FR特徴量により顕著な改善が確認されました (5%以上)")
        elif rmse_imp >= 2:
            print("[SUCCESS] FR特徴量により明確な改善が確認されました (2%以上)")
        else:
            print("[WARNING] FR特徴量により改善が見られますが、効果は限定的です")
    else:
        print("[FAILED] FR特徴量による明確な改善は確認できませんでした")

    print("\n[Comparison with 90-day results (2.37% improvement)]:")
    print(f"  Full data improvement: {rmse_imp:.2f}%")
    print(f"  Difference: {rmse_imp - 2.37:.2f} points")

    if abs(rmse_imp - 2.37) < 1.0:
        print("  => Consistent with 90-day results, high reliability")
    else:
        print("  => Discrepancy with 90-day results, further analysis needed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()