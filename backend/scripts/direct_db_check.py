"""
データベースを直接確認するスクリプト

SQLiteデータベースを直接クエリして最新の結果を確認します。
"""

import json
from datetime import datetime
from sqlalchemy import text

from utils.db_utils import get_db_session, get_repositories


def check_database_directly():
    """データベースを直接確認"""
    print("=== データベース直接確認 ===")

    try:
        with get_db_session() as db:
            repos = get_repositories(db)

            # 1. 最新のバックテスト結果を確認
            print("\n📊 最新バックテスト結果:")
            results = repos["backtest_result"].get_all_results(limit=5)

            for i, result in enumerate(results, 1):
                print(f"\n  結果 {i}:")
                print(f"    戦略名: {result.strategy_name}")
                print(f"    シンボル: {result.symbol}")
                print(f"    期間: {result.start_date} - {result.end_date}")
                print(f"    作成日時: {result.created_at}")

                if result.performance_metrics:
                    try:
                        metrics = (
                            json.loads(result.performance_metrics)
                            if isinstance(result.performance_metrics, str)
                            else result.performance_metrics
                        )
                        total_trades = metrics.get("total_trades", 0)
                        total_return = metrics.get("total_return", 0)
                        equity_final = metrics.get("equity_final", 0)

                        print(f"    総取引数: {total_trades}")
                        print(
                            f"    総リターン: {total_return:.4f} ({total_return*100:.2f}%)"
                        )
                        print(f"    最終資産: {equity_final:,.0f}")

                        if total_trades > 0:
                            print(f"    ✅ 取引が実行されました！")
                        else:
                            print(f"    ❌ 取引回数0")

                    except Exception as e:
                        print(f"    メトリクス解析エラー: {e}")

            # 2. GA実験の状況を確認
            print(f"\n🧬 GA実験状況:")
            experiments = repos["ga_experiment"].get_recent_experiments(limit=5)

            for i, exp in enumerate(experiments, 1):
                print(f"\n  実験 {i}:")
                print(f"    名前: {exp.name}")
                print(f"    ステータス: {exp.status}")
                print(f"    進捗: {exp.progress:.2%}")
                print(f"    最高フィットネス: {exp.best_fitness}")
                print(f"    世代: {exp.current_generation}/{exp.total_generations}")
                print(f"    作成日時: {exp.created_at}")
                print(f"    完了日時: {exp.completed_at}")

            # 3. 生成戦略の確認
            print(f"\n🎯 生成戦略:")
            strategies = repos["generated_strategy"].get_best_strategies(limit=5)

            for i, strategy in enumerate(strategies, 1):
                print(f"\n  戦略 {i}:")
                print(f"    実験ID: {strategy.experiment_id}")
                print(f"    世代: {strategy.generation}")
                print(f"    フィットネス: {strategy.fitness_score}")
                print(f"    作成日時: {strategy.created_at}")

                if strategy.gene_data:
                    try:
                        gene_data = strategy.gene_data
                        indicators = gene_data.get("indicators", [])
                        entry_conditions = gene_data.get("entry_conditions", [])
                        exit_conditions = gene_data.get("exit_conditions", [])

                        print(f"    指標数: {len(indicators)}")
                        print(f"    エントリー条件数: {len(entry_conditions)}")
                        print(f"    エグジット条件数: {len(exit_conditions)}")

                        if entry_conditions:
                            print(f"    エントリー条件例: {entry_conditions[0]}")

                    except Exception as e:
                        print(f"    遺伝子データ解析エラー: {e}")

            # 4. 最新の修正版結果を特定 (カスタムクエリ)
            print(f"\n🔧 修正版結果の特定:")
            query = text(
                """
                SELECT br.strategy_name, br.performance_metrics, br.created_at,
                       ge.name as experiment_name, ge.status as experiment_status
                FROM backtest_results br
                LEFT JOIN ga_experiments ge ON br.strategy_name LIKE '%' || REPLACE(ge.name, 'FIXED_AUTO_STRATEGY_', '') || '%'
                WHERE br.strategy_name LIKE '%FIXED%' OR br.strategy_name LIKE '%FIX%'
                ORDER BY br.created_at DESC
                LIMIT 3
            """
            )
            fixed_results = db.execute(query).mappings().all()

            if fixed_results:
                print(f"  修正版結果数: {len(fixed_results)}")
                for i, result in enumerate(fixed_results, 1):
                    print(f"\n    修正版結果 {i}:")
                    print(f"      戦略名: {result['strategy_name']}")
                    print(f"      実験名: {result['experiment_name']}")
                    print(f"      実験ステータス: {result['experiment_status']}")
                    print(f"      作成日時: {result['created_at']}")

                    if result["performance_metrics"]:
                        try:
                            metrics = json.loads(result["performance_metrics"])
                            total_trades = metrics.get("total_trades", 0)
                            total_return = metrics.get("total_return", 0)

                            print(f"      総取引数: {total_trades}")
                            print(f"      総リターン: {total_return:.4f}")

                            if total_trades > 0:
                                print(f"      🎉 修正版で取引が発生しました！")
                            else:
                                print(f"      ❌ 修正版でも取引回数0")
                        except Exception as e:
                            print(f"      メトリクス解析エラー: {e}")
            else:
                print(f"  修正版結果が見つかりませんでした")

    except Exception as e:
        print(f"❌ データベース確認エラー: {e}")
        import traceback

        traceback.print_exc()


def analyze_condition_patterns():
    """条件パターンの分析"""
    print(f"\n=== 条件パターン分析 ===")

    try:
        with get_db_session() as db:
            repos = get_repositories(db)
            strategies = repos["generated_strategy"].get_all_strategies(limit=10)

            condition_patterns = {}

            for strategy in strategies:
                if strategy.gene_data:
                    try:
                        gene_data = strategy.gene_data
                        entry_conditions = gene_data.get("entry_conditions", [])

                        for condition in entry_conditions:
                            left = condition.get("left_operand", "")
                            op = condition.get("operator", "")
                            right = condition.get("right_operand", "")

                            pattern = f"{left} {op} {right}"
                            condition_patterns[pattern] = (
                                condition_patterns.get(pattern, 0) + 1
                            )

                    except Exception:
                        continue

            print(f"よく使われる条件パターン:")
            sorted_patterns = sorted(
                condition_patterns.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for pattern, count in sorted_patterns:
                print(f"  {pattern}: {count}回")

    except Exception as e:
        print(f"❌ 条件パターン分析エラー: {e}")


def main():
    """メイン実行関数"""
    print("🔍 データベース直接確認開始")
    print(f"実行時刻: {datetime.now()}")

    # 1. データベース直接確認
    check_database_directly()

    # 2. 条件パターン分析
    analyze_condition_patterns()

    print(f"\n🔍 確認完了")


if __name__ == "__main__":
    main()
