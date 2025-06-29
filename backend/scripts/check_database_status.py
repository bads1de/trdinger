#!/usr/bin/env python3
"""
データベース状況確認スクリプト

実際のDBデータを確認して、GA戦略生成に使用可能なデータを把握します。
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_database_status():
    """データベースの状況を詳細に確認"""
    if (
        SessionLocal is None
        or OHLCVRepository is None
        or OpenInterestRepository is None
        or FundingRateRepository is None
    ):
        print("エラー: データベースコンポーネントが利用できません。")
        return

    print("🔍 データベース状況確認開始")
    print("=" * 80)

    try:
        db = SessionLocal()
        try:
            # リポジトリ初期化
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)

            # 1. OHLCVデータの確認
            print("\n📊 OHLCV データ状況:")
            print("-" * 40)

            # 利用可能なシンボルを取得
            symbols = ohlcv_repo.get_available_symbols()
            print(f"利用可能シンボル数: {len(symbols)}")
            print(f"シンボル一覧: {symbols}")

            # 各シンボルの詳細確認
            for symbol in symbols[:5]:  # 最初の5つのシンボルを詳細確認
                print(f"\n📈 {symbol} の詳細:")

                timeframes = ohlcv_repo.get_available_timeframes(symbol)
                print(f"  利用可能時間軸: {timeframes}")

                for timeframe in timeframes[:3]:  # 最初の3つの時間軸を確認
                    count = ohlcv_repo.get_data_count(symbol, timeframe)
                    if count > 0:
                        ohlcv_repo.get_date_range(symbol, timeframe)
                        latest = ohlcv_repo.get_latest_timestamp(symbol, timeframe)
                        oldest = ohlcv_repo.get_oldest_timestamp(symbol, timeframe)
                        if oldest and latest:
                            print(
                                f"    期間: {oldest.strftime('%Y-%m-%d')} ～ {latest.strftime('%Y-%m-%d')}"
                            )

                        # 最新データのサンプル表示
                        latest_data = ohlcv_repo.get_ohlcv_data(
                            symbol, timeframe, limit=1
                        )
                        if latest_data:
                            data = latest_data[0]
                            print(
                                f"      最新価格: O:{data.open:.2f} H:{data.high:.2f} L:{data.low:.2f} C:{data.close:.2f}"
                            )

            # 2. Open Interest データの確認
            print("\n\n🔢 Open Interest データ状況:")
            print("-" * 40)

            oi_symbols = []
            for symbol in symbols[:5]:
                try:
                    oi_data = oi_repo.get_open_interest_data(symbol, limit=1)
                    if oi_data:
                        oi_symbols.append(symbol)

                        # 期間確認
                        all_oi_data = oi_repo.get_open_interest_data(symbol)
                        if all_oi_data:
                            oldest_oi = min(data.data_timestamp for data in all_oi_data)
                            latest_oi = max(data.data_timestamp for data in all_oi_data)

                            print(f"📊 {symbol}:")
                            print(f"    データ件数: {len(all_oi_data):,} 件")
                            print(
                                f"    期間: {oldest_oi.strftime('%Y-%m-%d')} ～ {latest_oi.strftime('%Y-%m-%d')}"
                            )
                            print(f"    最新OI: {oi_data[0].open_interest_value:,.0f}")
                except Exception as e:
                    print(f"  {symbol}: OIデータなし ({e})")

            print(f"\nOIデータ利用可能シンボル: {oi_symbols}")

            # 3. Funding Rate データの確認
            print("\n\n💰 Funding Rate データ状況:")
            print("-" * 40)

            fr_symbols = []
            for symbol in symbols[:5]:
                try:
                    fr_data = fr_repo.get_funding_rate_data(symbol, limit=1)
                    if fr_data:
                        fr_symbols.append(symbol)

                        # 期間確認
                        all_fr_data = fr_repo.get_funding_rate_data(symbol)
                        if all_fr_data:
                            oldest_fr = min(
                                data.funding_timestamp for data in all_fr_data
                            )
                            latest_fr = max(
                                data.funding_timestamp for data in all_fr_data
                            )

                            print(f"💸 {symbol}:")
                            print(f"    データ件数: {len(all_fr_data):,} 件")
                            print(
                                f"    期間: {oldest_fr.strftime('%Y-%m-%d')} ～ {latest_fr.strftime('%Y-%m-%d')}"
                            )
                            print(
                                f"    最新FR: {fr_data[0].funding_rate:.6f} ({fr_data[0].funding_rate*100:.4f}%)"
                            )
                except Exception as e:
                    print(f"  {symbol}: FRデータなし ({e})")

            print(f"\nFRデータ利用可能シンボル: {fr_symbols}")

            # 4. GA戦略生成に最適なデータセットの推奨
            print("\n\n🎯 GA戦略生成推奨データセット:")
            print("-" * 40)

            # OHLCV、OI、FRすべてが利用可能なシンボルを特定
            complete_symbols = []
            for symbol in symbols:
                has_ohlcv = (
                    ohlcv_repo.get_data_count(symbol, "1d") > 30
                )  # 30日以上のデータ
                has_oi = symbol in oi_symbols
                has_fr = symbol in fr_symbols

                if has_ohlcv and (
                    has_oi or has_fr
                ):  # OHLCVと少なくともOIまたはFRがある
                    complete_symbols.append(
                        {
                            "symbol": symbol,
                            "has_oi": has_oi,
                            "has_fr": has_fr,
                            "ohlcv_count": ohlcv_repo.get_data_count(symbol, "1d"),
                        }
                    )

            print("完全データセット利用可能シンボル:")
            for data in complete_symbols:
                oi_status = "✅" if data["has_oi"] else "❌"
                fr_status = "✅" if data["has_fr"] else "❌"
                print(
                    f"  📊 {data['symbol']}: OHLCV({data['ohlcv_count']}件) OI({oi_status}) FR({fr_status})"
                )

            # 5. 推奨設定の提案
            if complete_symbols:
                best_symbol = max(complete_symbols, key=lambda x: x["ohlcv_count"])

                print("\n🚀 GA戦略生成推奨設定:")
                print(f"  シンボル: {best_symbol['symbol']}")
                print("  時間軸: 1d (日足)")
                print("  期間: 過去30-90日")
                print(f"  OI利用: {'可能' if best_symbol['has_oi'] else '不可'}")
                print(f"  FR利用: {'可能' if best_symbol['has_fr'] else '不可'}")

                return best_symbol
            else:
                print("⚠️ 完全なデータセットが見つかりません")
                print("サンプルデータの生成を推奨します")
                return None

        finally:
            db.close()

    except Exception as e:
        logger.error(f"データベース確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def suggest_ga_config(symbol_data):
    """GA設定の提案"""
    if not symbol_data:
        return None

    print("\n⚙️ 推奨GA設定:")
    print("-" * 40)

    config = {
        "symbol": symbol_data["symbol"],
        "timeframe": "1d",
        "population_size": 20,
        "generations": 10,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "fitness_weights": {
            "total_return": 0.35,
            "sharpe_ratio": 0.35,
            "max_drawdown": 0.25,
            "win_rate": 0.05,
        },
        "backtest_period_days": 60,
        "use_oi": symbol_data["has_oi"],
        "use_fr": symbol_data["has_fr"],
    }

    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


if __name__ == "__main__":
    print("🔍 データベース状況確認とGA設定提案")
    print("=" * 80)

    # データベース状況確認
    symbol_data = check_database_status()

    # GA設定提案
    ga_config = suggest_ga_config(symbol_data)

    print("\n" + "=" * 80)
    if symbol_data and ga_config:
        print("✅ データベース確認完了")
        print("🚀 GA戦略生成の準備が整いました")
    else:
        print("⚠️ データ不足のため、サンプルデータ生成が必要です")
        print("python scripts/create_sample_data.py を実行してください")
