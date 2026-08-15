"""
バックテスト結果の遡及リンクスクリプト

フェーズ1以前に保存された詳細バックテスト結果（backtest_results）を、
対応する生成戦略（generated_strategies）へ遡及的にリンクします。

リンク条件:
  - backtest_results.config_json.db_experiment_id == generated_strategies.experiment_id
  - まだ backtest_result_id が未設定（NULL）の戦略のみ対象
  - 実験内で最も fitness_score が高い戦略1件へリンク（最良戦略）

冪等: すでにリンク済みのBT結果はスキップされるため、何度実行しても安全です。

使用方法:
    cd backend
    uv run python -m scripts.backfill_backtest_links [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import SessionLocal  # noqa: E402
from database.models import BacktestResult, GeneratedStrategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_unlinked_strategies(db) -> list[GeneratedStrategy]:
    """リンク未設定の戦略を取得する。"""
    return (
        db.query(GeneratedStrategy)
        .filter(GeneratedStrategy.backtest_result_id.is_(None))
        .all()
    )


def link_backtest_results(db, dry_run: bool) -> int:
    """BT結果を最良戦略へリンクし、リンクした件数を返す。"""
    unlinked_strategies = find_unlinked_strategies(db)
    if not unlinked_strategies:
        logger.info("未リンクの戦略はありません")
        return 0

    # 実験ID → 最良戦略（fitness_score 最大、同値ならID最小）のマップ
    best_by_experiment: dict[int, GeneratedStrategy] = {}
    for strategy in unlinked_strategies:
        current = best_by_experiment.get(strategy.experiment_id)
        if current is None or _is_better(strategy, current):
            best_by_experiment[strategy.experiment_id] = strategy

    # BT結果を走査してマッチング
    linked_count = 0
    for bt_result in db.query(BacktestResult).all():
        config_json = bt_result.config_json or {}
        db_experiment_id = config_json.get("db_experiment_id")
        if db_experiment_id is None:
            continue

        # このBT結果がすでにどこかの戦略にリンク済みか確認
        already_linked = (
            db.query(GeneratedStrategy)
            .filter(GeneratedStrategy.backtest_result_id == bt_result.id)
            .first()
            is not None
        )
        if already_linked:
            continue

        target = best_by_experiment.get(int(db_experiment_id))
        if target is None:
            logger.warning(
                f"BT結果 ID={bt_result.id} の実験 {db_experiment_id} に"
                "未リンクの最良戦略が見つかりません（スキップ）"
            )
            continue

        logger.info(
            f"BT結果 ID={bt_result.id} (戦略名: {bt_result.strategy_name}) "
            f"→ 戦略 ID={target.id} (実験 {target.experiment_id}) へリンク"
        )
        linked_count += 1
        if not dry_run:
            target.backtest_result_id = bt_result.id

    if not dry_run and linked_count:
        db.commit()
        logger.info(f"コミット完了: {linked_count} 件をリンクしました")

    return linked_count


def _is_better(a: GeneratedStrategy, b: GeneratedStrategy) -> bool:
    """戦略aが戦略bより最良か判定する（fitness降順、同値ならID昇順）。"""
    fa = a.fitness_score or 0.0
    fb = b.fitness_score or 0.0
    if fa != fb:
        return fa > fb
    return a.id < b.id


def main() -> None:
    parser = argparse.ArgumentParser(description="BT結果の遡及リンク")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="リンクせず対象の表示のみ行う",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        count = link_backtest_results(db, dry_run=args.dry_run)
        if args.dry_run:
            logger.info(f"[dry-run] リンク対象: {count} 件")
        else:
            logger.info(f"完了: {count} 件をリンクしました")
    finally:
        db.close()


if __name__ == "__main__":
    main()
