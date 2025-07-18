"""
最新のバックテスト結果を確認するスクリプト

修正版で実行された戦略の取引結果を詳しく確認します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from utils.db_utils import get_db_session, get_repositories
import json
from datetime import datetime


def check_latest_backtest_results():
    """最新のバックテスト結果を確認"""
    print("=== 最新バックテスト結果確認 ===")

    try:
        with get_db_session() as db:
            # 直接SQLクエリで最新結果を取得
            query = text(
                """
                SELECT * FROM backtest_results
                WHERE strategy_name LIKE '%FIXED_AUTO_STRATEGY%'
                OR strategy_name LIKE '%AUTO_STRATEGY%'
                ORDER BY created_at DESC
                LIMIT 5
            """
            )
            results = db.execute(query).mappings().all()

            print(f"最新のバックテスト結果数: {len(results)}")

            for i, result in enumerate(results, 1):
                print(f"\n📊 結果 {i}:")
                print(f"  戦略名: {result['strategy_name']}")
                print(f"  シンボル: {result['symbol']}")
                print(f"  時間軸: {result['timeframe']}")
                print(f"  期間: {result['start_date']} - {result['end_date']}")
                print(f"  作成日時: {result['created_at']}")

                # パフォーマンス指標を確認
                if result["performance_metrics"]:
                    metrics = (
                        json.loads(result["performance_metrics"])
                        if isinstance(result["performance_metrics"], str)
                        else result["performance_metrics"]
                    )

                    total_trades = metrics.get("total_trades", 0)
                    total_return = metrics.get("total_return", 0)
                    win_rate = metrics.get("win_rate", 0)
                    max_drawdown = metrics.get("max_drawdown", 0)
                    equity_final = metrics.get("equity_final", 0)

                    print(f"  📈 パフォーマンス:")
                    print(f"    総取引数: {total_trades}")
                    print(
                        f"    総リターン: {total_return:.4f} ({total_return*100:.2f}%)"
                    )
                    print(f"    最終資産: {equity_final:,.0f}")
                    print(
                        f"    勝率: {win_rate:.4f}"
                        if win_rate and str(win_rate) != "nan"
                        else "    勝率: N/A"
                    )
                    print(
                        f"    最大ドローダウン: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)"
                    )

                    # 取引回数の詳細分析
                    if total_trades > 0:
                        print(f"    ✅ 取引が実行されました！")

                        winning_trades = metrics.get("winning_trades", 0)
                        losing_trades = metrics.get("losing_trades", 0)
                        avg_win = metrics.get("avg_win", 0)
                        avg_loss = metrics.get("avg_loss", 0)

                        print(f"    勝ちトレード: {winning_trades}")
                        print(f"    負けトレード: {losing_trades}")
                        print(f"    平均利益: {avg_win:.4f}")
                        print(f"    平均損失: {avg_loss:.4f}")

                    else:
                        print(f"    ❌ 取引回数0")

                        # 戦略設定を確認
                        if result["config_json"]:
                            config = (
                                json.loads(result["config_json"])
                                if isinstance(result["config_json"], str)
                                else result["config_json"]
                            )

                            strategy_config = config.get("strategy_config", {})
                            parameters = strategy_config.get("parameters", {})
                            strategy_gene = parameters.get("strategy_gene", {})

                            print(f"    🔍 戦略分析:")

                            # エントリー条件の確認
                            entry_conditions = strategy_gene.get("entry_conditions", [])
                            print(f"      エントリー条件数: {len(entry_conditions)}")
                            for j, condition in enumerate(entry_conditions, 1):
                                left = condition.get("left_operand", "")
                                op = condition.get("operator", "")
                                right = condition.get("right_operand", "")
                                print(f"        {j}. {left} {op} {right}")

                            # エグジット条件の確認
                            exit_conditions = strategy_gene.get("exit_conditions", [])
                            print(f"      エグジット条件数: {len(exit_conditions)}")
                            for j, condition in enumerate(exit_conditions, 1):
                                left = condition.get("left_operand", "")
                                op = condition.get("operator", "")
                                right = condition.get("right_operand", "")
                                print(f"        {j}. {left} {op} {right}")

                            # 問題の分析
                            analyze_no_trades_issue(entry_conditions, exit_conditions)

                print("-" * 50)

    except Exception as e:
        print(f"❌ バックテスト結果取得エラー: {e}")
        import traceback

        traceback.print_exc()


def analyze_no_trades_issue(entry_conditions, exit_conditions):
    """取引回数0の問題を分析"""
    print(f"      🔍 取引回数0の原因分析:")

    issues = []

    # エントリー条件の問題をチェック
    for condition in entry_conditions:
        left = condition.get("left_operand", "")
        op = condition.get("operator", "")
        right = condition.get("right_operand", "")

        # 価格と指標の比較問題
        if (
            left == "close"
            and isinstance(right, str)
            and right not in ["open", "high", "low", "volume"]
        ):
            if right not in ["SMA", "EMA", "BB"]:  # 価格系指標以外
                issues.append(f"価格({left})と非価格系指標({right})の比較")

        # 数値文字列の問題
        if isinstance(right, str) and right.replace(".", "").replace("-", "").isdigit():
            try:
                num_value = float(right)
                if left in ["RSI", "CCI", "STOCH"] and (
                    num_value < 0 or num_value > 100
                ):
                    issues.append(f"{left}に対する範囲外の値({right})")
            except:
                pass

    # 条件の厳しさをチェック
    if len(entry_conditions) > 2:
        issues.append(f"エントリー条件が多すぎる({len(entry_conditions)}個)")

    if len(exit_conditions) > 2:
        issues.append(f"エグジット条件が多すぎる({len(exit_conditions)}個)")

    if issues:
        for issue in issues:
            print(f"        ⚠️ {issue}")
    else:
        print(f"        ✅ 明らかな問題は見つかりませんでした")


def check_recent_experiments():
    """最近の実験を確認"""
    print("\n=== 最近の実験確認 ===")

    try:
        with get_db_session() as db:
            repos = get_repositories(db)
            exp_repo = repos["ga_experiment"]
            strategy_repo = repos["generated_strategy"]

            # 最近の実験を取得
            recent_experiments = exp_repo.get_recent_experiments(limit=5)

            for exp in recent_experiments:
                print(f"\n実験: {exp.name}")
                print(f"  ID: {exp.id}")
                print(f"  ステータス: {exp.status}")
                print(f"  進捗: {exp.progress:.2%}")
                print(f"  最高フィットネス: {exp.best_fitness}")
                print(f"  作成日時: {exp.created_at}")
                print(f"  完了日時: {exp.completed_at}")

                if "FIXED" in exp.name:
                    print(f"  🎯 修正版実験を発見")

                    # この実験の戦略を確認
                    strategies = strategy_repo.get_strategies_by_experiment(exp.id)
                    print(f"  生成戦略数: {len(strategies)}")

                    if strategies:
                        best_strategy = max(
                            strategies, key=lambda s: s.fitness_score or 0
                        )
                        print(f"  最高戦略フィットネス: {best_strategy.fitness_score}")

    except Exception as e:
        print(f"❌ 最近の実験確認中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def main():
    """メイン実行関数"""
    print("🔍 最新結果確認開始")
    print(f"実行時刻: {datetime.now()}")

    # 1. 最新のバックテスト結果を確認
    check_latest_backtest_results()

    # 2. 最近の実験を確認
    check_recent_experiments()

    print(f"\n🔍 確認完了")


if __name__ == "__main__":
    main()
