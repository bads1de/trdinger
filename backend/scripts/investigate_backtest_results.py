"""
バックテスト結果調査スクリプト

SmartConditionGeneratorの実装が実際のバックテストに反映されていない問題を調査
"""

import sys
import os
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.connection import SessionLocal
from database.models import BacktestResult, GeneratedStrategy, GAExperiment
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_decoder import GeneDecoder


def check_database_structure():
    """データベース構造を確認"""
    import sqlite3

    try:
        # SQLiteデータベースに直接接続
        conn = sqlite3.connect('backend/trdinger.db')
        cursor = conn.cursor()

        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("📊 データベース内のテーブル:")
        for table in tables:
            print(f"   - {table[0]}")

        # backtest_resultsテーブルの存在確認
        if ('backtest_results',) in tables:
            print("\n✅ backtest_resultsテーブルが存在します")

            # レコード数を確認
            cursor.execute("SELECT COUNT(*) FROM backtest_results")
            count = cursor.fetchone()[0]
            print(f"   レコード数: {count}")

            if count > 0:
                # 最新のレコードを確認
                cursor.execute("""
                    SELECT strategy_name, symbol, created_at
                    FROM backtest_results
                    ORDER BY created_at DESC
                    LIMIT 3
                """)
                recent = cursor.fetchall()
                print("   最新のレコード:")
                for record in recent:
                    print(f"     - {record[0]} ({record[1]}) - {record[2]}")
        else:
            print("\n❌ backtest_resultsテーブルが存在しません")

        conn.close()

    except Exception as e:
        print(f"❌ データベース構造確認エラー: {e}")


def analyze_backtest_data_directly():
    """SQLiteから直接バックテストデータを分析"""
    import sqlite3

    try:
        conn = sqlite3.connect('backend/trdinger.db')
        cursor = conn.cursor()

        print("📊 バックテスト結果の詳細分析:")

        # 最新のバックテスト結果を取得
        cursor.execute("""
            SELECT id, strategy_name, symbol, timeframe, start_date, end_date,
                   config_json, performance_metrics, created_at
            FROM backtest_results
            ORDER BY created_at DESC
            LIMIT 5
        """)

        results = cursor.fetchall()

        for i, result in enumerate(results):
            print(f"\n--- 結果 {i+1} ---")
            print(f"ID: {result[0]}")
            print(f"戦略名: {result[1]}")
            print(f"シンボル: {result[2]}")
            print(f"時間軸: {result[3]}")
            print(f"期間: {result[4]} - {result[5]}")
            print(f"作成日時: {result[8]}")

            # config_jsonを解析
            config_json = result[6]
            if config_json:
                try:
                    config = json.loads(config_json)
                    strategy_config = config.get('strategy_config', {})
                    strategy_type = strategy_config.get('strategy_type')

                    print(f"戦略タイプ: {strategy_type}")

                    # 戦略遺伝子の詳細を確認
                    if strategy_type in ['GENERATED_AUTO', 'GENERATED_TEST']:
                        parameters = strategy_config.get('parameters', {})
                        strategy_gene_dict = parameters.get('strategy_gene', {})

                        if strategy_gene_dict:
                            print("🧬 戦略遺伝子詳細:")

                            # ロング・ショート条件の確認
                            long_conditions = strategy_gene_dict.get('long_entry_conditions', [])
                            short_conditions = strategy_gene_dict.get('short_entry_conditions', [])
                            exit_conditions = strategy_gene_dict.get('exit_conditions', [])

                            print(f"   ロング条件数: {len(long_conditions)}")
                            print(f"   ショート条件数: {len(short_conditions)}")
                            print(f"   エグジット条件数: {len(exit_conditions)}")

                            # 条件の内容を確認
                            if long_conditions:
                                print("   ロング条件例:")
                                for j, cond in enumerate(long_conditions[:2]):
                                    print(f"     {j+1}. {cond.get('left_operand')} {cond.get('operator')} {cond.get('right_operand')}")

                            if short_conditions:
                                print("   ショート条件例:")
                                for j, cond in enumerate(short_conditions[:2]):
                                    print(f"     {j+1}. {cond.get('left_operand')} {cond.get('operator')} {cond.get('right_operand')}")

                            # TP/SL設定の確認
                            tpsl_gene = strategy_gene_dict.get('tpsl_gene', {})
                            if tpsl_gene:
                                tpsl_enabled = tpsl_gene.get('enabled', False)
                                print(f"   TP/SL有効: {tpsl_enabled}")
                                if tpsl_enabled:
                                    print(f"   SL率: {tpsl_gene.get('stop_loss_pct', 'N/A')}")
                                    print(f"   TP率: {tpsl_gene.get('take_profit_pct', 'N/A')}")

                            # 指標の確認
                            indicators = strategy_gene_dict.get('indicators', [])
                            print(f"   指標数: {len(indicators)}")
                            if indicators:
                                enabled_indicators = [ind for ind in indicators if ind.get('enabled', False)]
                                print(f"   有効指標数: {len(enabled_indicators)}")
                                for ind in enabled_indicators[:3]:
                                    print(f"     - {ind.get('type')} (期間: {ind.get('parameters', {}).get('period', 'N/A')})")

                            # SmartConditionGeneratorの使用確認
                            if len(long_conditions) > 0 and len(short_conditions) > 0:
                                # フォールバック条件かどうかチェック
                                long_str = str(long_conditions)
                                short_str = str(short_conditions)

                                if "close" in long_str and "open" in long_str and "close" in short_str and "open" in short_str:
                                    if ">" in long_str and "<" in short_str:
                                        print("   ⚠️  フォールバック条件が使用されている可能性")
                                    else:
                                        print("   ✅ 多様な条件が生成されている")
                                else:
                                    print("   ✅ 指標ベースの条件が生成されている")
                            else:
                                print("   ❌ ロング・ショート条件が不完全")

                except json.JSONDecodeError as e:
                    print(f"   ❌ config_json解析エラー: {e}")

            # performance_metricsを解析
            performance_metrics = result[7]
            if performance_metrics:
                try:
                    metrics = json.loads(performance_metrics)
                    total_trades = metrics.get('total_trades', 0)
                    win_rate = metrics.get('win_rate', 0)
                    total_return = metrics.get('total_return', 0)

                    print(f"📈 パフォーマンス:")
                    print(f"   総取引数: {total_trades}")
                    print(f"   勝率: {win_rate:.2%}")
                    print(f"   総リターン: {total_return:.2%}")

                    # 問題の兆候をチェック
                    if total_trades < 10:
                        print("   ⚠️  取引数が異常に少ない")
                    if total_trades == 0:
                        print("   ❌ 取引が全く発生していない")
                    if win_rate == 0 or win_rate == 1:
                        print("   ⚠️  勝率が極端")

                except json.JSONDecodeError as e:
                    print(f"   ❌ performance_metrics解析エラー: {e}")

        conn.close()

    except Exception as e:
        print(f"❌ 直接分析エラー: {e}")
        import traceback
        traceback.print_exc()


def investigate_backtest_results():
    """バックテスト結果の詳細調査"""
    print("🔍 バックテスト結果詳細調査開始")
    print("="*60)

    # まずデータベースのテーブル構造を確認
    print("\n0. データベース構造を確認...")
    check_database_structure()

    db = SessionLocal()

    try:
        # 1. 最新のバックテスト結果を取得
        print("\n1. 最新のバックテスト結果を調査...")

        # SQLiteから直接データを取得
        analyze_backtest_data_directly()

        print(f"📊 最新のバックテスト結果 {len(recent_results)} 件を分析:")

        for i, result in enumerate(recent_results):
            print(f"\n--- 結果 {i+1} ---")
            print(f"戦略名: {result.strategy_name}")
            print(f"シンボル: {result.symbol}")
            print(f"期間: {result.start_date} - {result.end_date}")
            print(f"作成日時: {result.created_at}")

            # パフォーマンス指標を確認
            if result.performance_metrics:
                metrics = result.performance_metrics
                total_trades = metrics.get('total_trades', 0)
                win_rate = metrics.get('win_rate', 0)
                total_return = metrics.get('total_return', 0)

                print(f"総取引数: {total_trades}")
                print(f"勝率: {win_rate:.2%}")
                print(f"総リターン: {total_return:.2%}")

                # 問題の兆候をチェック
                if total_trades < 10:
                    print("⚠️  取引数が異常に少ない")
                if win_rate == 0 or win_rate == 1:
                    print("⚠️  勝率が極端")

            # 戦略設定を確認
            if result.config_json:
                config = result.config_json
                strategy_config = config.get('strategy_config', {})
                strategy_type = strategy_config.get('strategy_type')

                print(f"戦略タイプ: {strategy_type}")

                # 戦略遺伝子の詳細を確認
                if strategy_type in ['GENERATED_AUTO', 'GENERATED_TEST']:
                    parameters = strategy_config.get('parameters', {})
                    strategy_gene_dict = parameters.get('strategy_gene', {})

                    if strategy_gene_dict:
                        print("🧬 戦略遺伝子詳細:")

                        # ロング・ショート条件の確認
                        long_conditions = strategy_gene_dict.get('long_entry_conditions', [])
                        short_conditions = strategy_gene_dict.get('short_entry_conditions', [])
                        exit_conditions = strategy_gene_dict.get('exit_conditions', [])

                        print(f"   ロング条件数: {len(long_conditions)}")
                        print(f"   ショート条件数: {len(short_conditions)}")
                        print(f"   エグジット条件数: {len(exit_conditions)}")

                        # 条件の内容を確認
                        if long_conditions:
                            print("   ロング条件例:")
                            for j, cond in enumerate(long_conditions[:2]):
                                print(f"     {j+1}. {cond.get('left_operand')} {cond.get('operator')} {cond.get('right_operand')}")

                        if short_conditions:
                            print("   ショート条件例:")
                            for j, cond in enumerate(short_conditions[:2]):
                                print(f"     {j+1}. {cond.get('left_operand')} {cond.get('operator')} {cond.get('right_operand')}")

                        # TP/SL設定の確認
                        tpsl_gene = strategy_gene_dict.get('tpsl_gene', {})
                        if tpsl_gene:
                            tpsl_enabled = tpsl_gene.get('enabled', False)
                            print(f"   TP/SL有効: {tpsl_enabled}")
                            if tpsl_enabled:
                                print(f"   SL率: {tpsl_gene.get('stop_loss_pct', 'N/A')}")
                                print(f"   TP率: {tpsl_gene.get('take_profit_pct', 'N/A')}")

                        # 指標の確認
                        indicators = strategy_gene_dict.get('indicators', [])
                        print(f"   指標数: {len(indicators)}")
                        if indicators:
                            enabled_indicators = [ind for ind in indicators if ind.get('enabled', False)]
                            print(f"   有効指標数: {len(enabled_indicators)}")
                            for ind in enabled_indicators[:3]:
                                print(f"     - {ind.get('type')} (期間: {ind.get('parameters', {}).get('period', 'N/A')})")

        # 2. 戦略生成の現在の状況を確認
        print(f"\n2. 現在の戦略生成状況を確認...")
        test_smart_condition_generation()

    except Exception as e:
        print(f"❌ 調査中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


def test_smart_condition_generation():
    """SmartConditionGeneratorの現在の動作を確認"""
    print("\n🧪 SmartConditionGenerator動作確認:")

    try:
        # 1. 直接テスト
        smart_generator = SmartConditionGenerator(enable_smart_generation=True)
        legacy_generator = SmartConditionGenerator(enable_smart_generation=False)

        from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene

        test_indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        # SmartConditionGenerator
        smart_long, smart_short, smart_exit = smart_generator.generate_balanced_conditions(test_indicators)
        print(f"   Smart - ロング: {len(smart_long)}, ショート: {len(smart_short)}")

        # 従来方式
        legacy_long, legacy_short, legacy_exit = legacy_generator.generate_balanced_conditions(test_indicators)
        print(f"   Legacy - ロング: {len(legacy_long)}, ショート: {len(legacy_short)}")

        # 2. RandomGeneGeneratorでの使用確認
        ga_config = GAConfig.create_fast()

        # enable_smart_generation=Trueで生成
        smart_gene_generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)
        smart_gene = smart_gene_generator.generate_random_gene()

        print(f"   RandomGeneGenerator(Smart) - ロング: {len(smart_gene.long_entry_conditions)}, ショート: {len(smart_gene.short_entry_conditions)}")

        # enable_smart_generation=Falseで生成
        legacy_gene_generator = RandomGeneGenerator(ga_config, enable_smart_generation=False)
        legacy_gene = legacy_gene_generator.generate_random_gene()

        print(f"   RandomGeneGenerator(Legacy) - ロング: {len(legacy_gene.long_entry_conditions)}, ショート: {len(legacy_gene.short_entry_conditions)}")

    except Exception as e:
        print(f"   ❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    investigate_backtest_results()