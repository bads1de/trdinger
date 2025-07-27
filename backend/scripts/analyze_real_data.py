#!/usr/bin/env python3
"""
実際のOHLCV、FG、OI、FRデータの分析スクリプト

データの期間不一致、欠損値、特徴量の有効性を分析します。
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import get_db
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.services.backtest.backtest_data_service import BacktestDataService


def analyze_data_coverage():
    """データのカバレッジを分析"""
    print("=== データカバレッジ分析 ===")

    # データベース接続
    db = next(get_db())

    try:
        # リポジトリ初期化
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        fg_repo = FearGreedIndexRepository(db)

        # 分析期間（過去30日）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        symbol = "BTC/USDT"

        print(
            f"分析期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"対象シンボル: {symbol}")

        # 各データソースの件数を確認
        ohlcv_data = ohlcv_repo.get_ohlcv_data(
            symbol=symbol, timeframe="1h", start_time=start_date, end_time=end_date
        )
        oi_data = oi_repo.get_open_interest_data(
            symbol=symbol, start_time=start_date, end_time=end_date
        )
        fr_data = fr_repo.get_funding_rate_data(
            symbol=symbol, start_time=start_date, end_time=end_date
        )
        fg_data = fg_repo.get_fear_greed_data(start_time=start_date, end_time=end_date)

        print(f"\n📊 データ件数:")
        print(f"  OHLCV (1h): {len(ohlcv_data)}件")
        print(f"  Open Interest: {len(oi_data)}件")
        print(f"  Funding Rate: {len(fr_data)}件")
        print(f"  Fear & Greed: {len(fg_data)}件")

        # 期間分析
        if ohlcv_data:
            ohlcv_start = min(d.timestamp for d in ohlcv_data)
            ohlcv_end = max(d.timestamp for d in ohlcv_data)
            print(f"\n📈 OHLCV期間: {ohlcv_start} - {ohlcv_end}")

        if oi_data:
            oi_start = min(d.timestamp for d in oi_data)
            oi_end = max(d.timestamp for d in oi_data)
            print(f"📊 OI期間: {oi_start} - {oi_end}")

        if fr_data:
            fr_start = min(d.timestamp for d in fr_data)
            fr_end = max(d.timestamp for d in fr_data)
            print(f"💰 FR期間: {fr_start} - {fr_end}")

        if fg_data:
            fg_start = min(d.data_timestamp for d in fg_data)
            fg_end = max(d.data_timestamp for d in fg_data)
            print(f"😨 FG期間: {fg_start} - {fg_end}")

        return {
            "ohlcv": ohlcv_data,
            "oi": oi_data,
            "fr": fr_data,
            "fg": fg_data,
        }

    finally:
        db.close()


def analyze_data_integration():
    """データ統合処理の分析"""
    print("\n=== データ統合分析 ===")

    # データベース接続
    db = next(get_db())

    try:
        # BacktestDataServiceを使用してデータ統合
        service = BacktestDataService(db)

        # 分析期間（過去7日）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        symbol = "BTC/USDT"
        timeframe = "1h"

        print(f"統合データ取得: {symbol} {timeframe}")
        print(
            f"期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
        )

        # 統合データを取得
        df = service.get_data_for_backtest(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        print(f"\n📊 統合データ形状: {df.shape}")
        print(f"📊 カラム: {list(df.columns)}")

        # 欠損値分析
        print(f"\n🔍 欠損値分析:")
        missing_analysis = df.isnull().sum()
        for col, missing_count in missing_analysis.items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count}件 ({missing_pct:.1f}%)")

        # データ型分析
        print(f"\n📋 データ型:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        # 統計情報
        print(f"\n📈 統計情報:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe()
        print(stats)

        return df

    finally:
        db.close()


def analyze_feature_effectiveness(df: pd.DataFrame):
    """特徴量の有効性を分析"""
    print("\n=== 特徴量有効性分析 ===")

    if df.empty:
        print("データが空のため、分析をスキップします")
        return

    # ターゲット変数を作成（次の時間の価格変動率）
    if "Close" in df.columns:
        df["target"] = df["Close"].pct_change().shift(-1)
    else:
        print("Closeカラムが見つかりません")
        return

    # 数値カラムのみを選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "target"]

    print(f"分析対象特徴量: {len(numeric_cols)}個")

    # ターゲットとの相関分析
    correlations = []
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # 有効なデータが10個以上
            corr = df[col].corr(df["target"])
            if not pd.isna(corr):
                correlations.append(
                    {
                        "feature": col,
                        "correlation": abs(corr),
                        "correlation_raw": corr,
                        "valid_count": df[col].notna().sum(),
                        "missing_pct": (df[col].isna().sum() / len(df)) * 100,
                    }
                )

    # 相関の高い順にソート
    correlations = sorted(correlations, key=lambda x: x["correlation"], reverse=True)

    print(f"\n🎯 ターゲットとの相関（上位20位）:")
    for i, item in enumerate(correlations[:20]):
        print(
            f"  {i+1:2d}. {item['feature']:<30} "
            f"相関: {item['correlation_raw']:+.4f} "
            f"(有効: {item['valid_count']:4d}件, "
            f"欠損: {item['missing_pct']:5.1f}%)"
        )

    # 低相関特徴量
    print(f"\n❌ 低相関特徴量（相関<0.01）:")
    low_corr = [item for item in correlations if item["correlation"] < 0.01]
    for item in low_corr[:10]:
        print(f"  - {item['feature']:<30} 相関: {item['correlation_raw']:+.4f}")

    # 欠損値の多い特徴量
    print(f"\n⚠️  欠損値の多い特徴量（>50%）:")
    high_missing = [item for item in correlations if item["missing_pct"] > 50]
    for item in high_missing:
        print(f"  - {item['feature']:<30} 欠損: {item['missing_pct']:5.1f}%")

    return correlations


def analyze_data_frequency():
    """データ頻度の分析"""
    print("\n=== データ頻度分析 ===")

    db = next(get_db())

    try:
        # 各データソースの更新頻度を分析
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        fg_repo = FearGreedIndexRepository(db)

        symbol = "BTC/USDT"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)  # 3日間で分析

        # OHLCV（1時間）
        ohlcv_data = ohlcv_repo.get_ohlcv_data(
            symbol=symbol, timeframe="1h", start_time=start_date, end_time=end_date
        )
        if ohlcv_data:
            ohlcv_times = [d.timestamp for d in ohlcv_data]
            ohlcv_intervals = [
                (ohlcv_times[i] - ohlcv_times[i - 1]).total_seconds() / 3600
                for i in range(1, len(ohlcv_times))
            ]
            print(
                f"📈 OHLCV間隔: 平均 {np.mean(ohlcv_intervals):.2f}時間, "
                f"標準偏差 {np.std(ohlcv_intervals):.2f}時間"
            )

        # Open Interest
        oi_data = oi_repo.get_open_interest_data(
            symbol=symbol, start_time=start_date, end_time=end_date
        )
        if oi_data and len(oi_data) > 1:
            oi_times = [d.timestamp for d in oi_data]
            oi_intervals = [
                (oi_times[i] - oi_times[i - 1]).total_seconds() / 3600
                for i in range(1, len(oi_times))
            ]
            print(
                f"📊 OI間隔: 平均 {np.mean(oi_intervals):.2f}時間, "
                f"標準偏差 {np.std(oi_intervals):.2f}時間"
            )

        # Funding Rate
        fr_data = fr_repo.get_funding_rate_data(
            symbol=symbol, start_time=start_date, end_time=end_date
        )
        if fr_data and len(fr_data) > 1:
            fr_times = [d.timestamp for d in fr_data]
            fr_intervals = [
                (fr_times[i] - fr_times[i - 1]).total_seconds() / 3600
                for i in range(1, len(fr_times))
            ]
            print(
                f"💰 FR間隔: 平均 {np.mean(fr_intervals):.2f}時間, "
                f"標準偏差 {np.std(fr_intervals):.2f}時間"
            )

        # Fear & Greed
        fg_data = fg_repo.get_fear_greed_data(start_time=start_date, end_time=end_date)
        if fg_data and len(fg_data) > 1:
            fg_times = [d.data_timestamp for d in fg_data]
            fg_intervals = [
                (fg_times[i] - fg_times[i - 1]).total_seconds() / 3600
                for i in range(1, len(fg_times))
            ]
            print(
                f"😨 FG間隔: 平均 {np.mean(fg_intervals):.2f}時間, "
                f"標準偏差 {np.std(fg_intervals):.2f}時間"
            )

    finally:
        db.close()


def main():
    """メイン実行関数"""
    print("実際のデータ分析開始")
    print("=" * 50)

    try:
        # データカバレッジ分析
        data_sources = analyze_data_coverage()

        # データ統合分析
        integrated_df = analyze_data_integration()

        # 特徴量有効性分析
        if integrated_df is not None and not integrated_df.empty:
            correlations = analyze_feature_effectiveness(integrated_df)

        # データ頻度分析
        analyze_data_frequency()

        print("\n" + "=" * 50)
        print("📊 分析完了")

        # 推奨事項
        print("\n🎯 推奨事項:")
        print("1. 欠損値の多い特徴量は除外または改善が必要")
        print("2. 低相関特徴量は特徴量エンジニアリングで改善")
        print("3. データ頻度の違いを考慮した補間方法の最適化")
        print("4. より効果的な特徴量の開発")

    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
