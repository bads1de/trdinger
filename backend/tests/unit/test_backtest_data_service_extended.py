#!/usr/bin/env python3
"""
拡張されたBacktestDataServiceのテスト

OI/FRデータ統合機能のテストを行います。
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.backtest_data_service import BacktestDataService
from database.models import OHLCVData, OpenInterestData, FundingRateData


class MockOHLCVRepository:
    """テスト用のモックOHLCVリポジトリ"""

    def __init__(self, sample_data):
        self.sample_data = sample_data

    def get_ohlcv_data(self, symbol, timeframe, start_time, end_time):
        return self.sample_data


class MockOpenInterestRepository:
    """テスト用のモックOpen Interestリポジトリ"""

    def __init__(self, sample_data):
        self.sample_data = sample_data

    def get_open_interest_data(self, symbol, start_time, end_time):
        return self.sample_data


class MockFundingRateRepository:
    """テスト用のモックFunding Rateリポジトリ"""

    def __init__(self, sample_data):
        self.sample_data = sample_data

    def get_funding_rate_data(self, symbol, start_time, end_time):
        return self.sample_data


def create_sample_ohlcv_data():
    """サンプルOHLCVデータを作成"""
    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(10):
        timestamp = base_date + timedelta(days=i)
        data.append(
            OHLCVData(
                symbol="BTC/USDT",
                timeframe="1d",
                timestamp=timestamp,
                open=50000 + i * 100,
                high=51000 + i * 100,
                low=49000 + i * 100,
                close=50500 + i * 100,
                volume=1000 + i * 10,
            )
        )

    return data


def create_sample_oi_data():
    """サンプルOpen Interestデータを作成"""
    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(10):
        timestamp = base_date + timedelta(days=i)
        data.append(
            OpenInterestData(
                symbol="BTC/USDT",
                data_timestamp=timestamp,
                open_interest_value=1000000 + i * 10000,
                timestamp=timestamp,
            )
        )

    return data


def create_sample_fr_data():
    """サンプルFunding Rateデータを作成"""
    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(10):
        timestamp = base_date + timedelta(days=i)
        data.append(
            FundingRateData(
                symbol="BTC/USDT",
                funding_timestamp=timestamp,
                funding_rate=0.0001 + i * 0.00001,
                mark_price=50000 + i * 100,
                index_price=50000 + i * 100,
            )
        )

    return data


def test_extended_backtest_data_service():
    """拡張されたBacktestDataServiceのテスト"""
    print("🧪 拡張BacktestDataServiceテスト開始")
    print("=" * 60)

    try:
        # 1. サンプルデータ作成
        print("1. サンプルデータ作成中...")
        ohlcv_data = create_sample_ohlcv_data()
        oi_data = create_sample_oi_data()
        fr_data = create_sample_fr_data()
        print(f"  ✅ OHLCV: {len(ohlcv_data)} 件")
        print(f"  ✅ OI: {len(oi_data)} 件")
        print(f"  ✅ FR: {len(fr_data)} 件")

        # 2. モックリポジトリ作成
        print("\n2. モックリポジトリ作成中...")
        ohlcv_repo = MockOHLCVRepository(ohlcv_data)
        oi_repo = MockOpenInterestRepository(oi_data)
        fr_repo = MockFundingRateRepository(fr_data)
        print("  ✅ モックリポジトリ作成完了")

        # 3. 拡張BacktestDataService作成
        print("\n3. 拡張BacktestDataService作成中...")
        data_service = BacktestDataService(
            ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
        )
        print("  ✅ 拡張BacktestDataService作成完了")

        # 4. 統合データ取得テスト
        print("\n4. 統合データ取得テスト中...")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        df = data_service.get_data_for_backtest(
            symbol="BTC/USDT", timeframe="1d", start_date=start_date, end_date=end_date
        )

        print(f"  ✅ データ取得成功: {len(df)} 行")
        print(f"  ✅ カラム: {df.columns.tolist()}")

        # 5. データ内容確認
        print("\n5. データ内容確認中...")
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "OpenInterest",
            "FundingRate",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"  ❌ 不足カラム: {missing_columns}")
            return False
        else:
            print("  ✅ 全必要カラムが存在")

        # 6. データ統計表示
        print("\n6. データ統計:")
        print(f"  📊 OHLCV統計:")
        print(f"    - 価格範囲: {df['Low'].min():.2f} - {df['High'].max():.2f}")
        print(f"    - 平均出来高: {df['Volume'].mean():.2f}")

        print(f"  📊 OI統計:")
        print(
            f"    - OI範囲: {df['OpenInterest'].min():.2f} - {df['OpenInterest'].max():.2f}"
        )
        print(f"    - 平均OI: {df['OpenInterest'].mean():.2f}")

        print(f"  📊 FR統計:")
        print(
            f"    - FR範囲: {df['FundingRate'].min():.6f} - {df['FundingRate'].max():.6f}"
        )
        print(f"    - 平均FR: {df['FundingRate'].mean():.6f}")

        # 7. データ概要取得テスト
        print("\n7. データ概要取得テスト中...")
        summary = data_service.get_data_summary(df)
        print(f"  ✅ 概要取得成功")
        print(f"  📋 総レコード数: {summary['total_records']}")

        if "open_interest_stats" in summary:
            print(f"  📋 OI統計含む: ✅")
        if "funding_rate_stats" in summary:
            print(f"  📋 FR統計含む: ✅")

        print("\n🎉 拡張BacktestDataServiceテスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n🔄 後方互換性テスト開始")
    print("=" * 60)

    try:
        # OHLCVのみでのテスト
        ohlcv_data = create_sample_ohlcv_data()
        ohlcv_repo = MockOHLCVRepository(ohlcv_data)

        # OI/FRリポジトリなしでサービス作成
        data_service = BacktestDataService(ohlcv_repo=ohlcv_repo)

        # 古いメソッドのテスト
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        df_old = data_service.get_ohlcv_for_backtest(
            symbol="BTC/USDT", timeframe="1d", start_date=start_date, end_date=end_date
        )

        print(f"  ✅ 古いメソッド動作: {len(df_old)} 行")
        print(f"  ✅ カラム: {df_old.columns.tolist()}")

        # 新しいメソッドのテスト（OI/FRはデフォルト値）
        df_new = data_service.get_data_for_backtest(
            symbol="BTC/USDT", timeframe="1d", start_date=start_date, end_date=end_date
        )

        print(f"  ✅ 新しいメソッド動作: {len(df_new)} 行")
        print(f"  ✅ カラム: {df_new.columns.tolist()}")

        # OI/FRがデフォルト値（0.0）で埋められているかチェック
        if "OpenInterest" in df_new.columns and "FundingRate" in df_new.columns:
            oi_unique = df_new["OpenInterest"].unique()
            fr_unique = df_new["FundingRate"].unique()

            if len(oi_unique) == 1 and oi_unique[0] == 0.0:
                print("  ✅ OIデフォルト値設定: 正常")
            else:
                print(f"  ⚠️ OIデフォルト値: {oi_unique}")

            if len(fr_unique) == 1 and fr_unique[0] == 0.0:
                print("  ✅ FRデフォルト値設定: 正常")
            else:
                print(f"  ⚠️ FRデフォルト値: {fr_unique}")

        print("\n🎉 後方互換性テスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ 後方互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_extended_backtest_data_service()
    success2 = test_backward_compatibility()

    if success1 and success2:
        print("\n🎊 全テスト成功！")
    else:
        print("\n💥 一部テスト失敗")
        sys.exit(1)
