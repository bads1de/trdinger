"""
無効な指標を含む戦略遺伝子のクリーンアップスクリプト

MININDEX、MAXINDEX、MINMAXなどの無効な指標を含む戦略遺伝子を
データベースから検出し、修正または無効化する。
"""

import logging
import sys
from typing import List, Dict, Any
from pathlib import Path

# バックエンドディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from database.connection import get_db, SessionLocal
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository
from database.models import GeneratedStrategy

# 無効な指標リスト
INVALID_INDICATORS = ["MINMAX", "MINMAXINDEX", "MAXINDEX", "MININDEX", "MAMA"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyGeneCleaner:
    """戦略遺伝子のクリーンアップクラス"""

    def __init__(self):
        self.db_session = SessionLocal()
        self.repository = GeneratedStrategyRepository(self.db_session)

    def find_strategies_with_invalid_indicators(self) -> List[GeneratedStrategy]:
        """
        無効な指標を含む戦略を検索

        Returns:
            無効な指標を含む戦略のリスト
        """
        logger.info("無効な指標を含む戦略遺伝子の検索を開始します...")

        # すべての戦略を取得 (フィルターなし)
        all_strategies = self.repository.get_filtered_data()

        invalid_strategies = []
        for strategy in all_strategies:
            if self._has_invalid_indicator(strategy):
                invalid_strategies.append(strategy)
                logger.warning(f"無効な指標を含む戦略を発見: ID={strategy.id}, "
                             f"Experiment={strategy.experiment_id}, "
                             f"Generation={strategy.generation}")

        logger.info(f"無効な戦略数: {len(invalid_strategies)}")
        return invalid_strategies

    def _has_invalid_indicator(self, strategy: GeneratedStrategy) -> bool:
        """
        戦略が無効な指標を含むかをチェック

        Args:
            strategy: 戦略インスタンス

        Returns:
            無効な指標を含む場合True
        """
        try:
            gene_data = strategy.gene_data
            if not gene_data or not isinstance(gene_data, dict):
                return False

            # indicatorsセクションから無効な指標を検索
            indicators = gene_data.get("indicators", [])
            if isinstance(indicators, dict):
                indicators = [indicators] if indicators else []
            elif not isinstance(indicators, list):
                return False

            for indicator in indicators:
                if isinstance(indicator, dict):
                    indicator_type = indicator.get("type")
                elif isinstance(indicator, str):
                    indicator_type = indicator
                else:
                    continue

                if indicator_type in INVALID_INDICATORS:
                    logger.debug(f"無効な指標 {indicator_type} が見つかりました")
                    return True

            return False

        except Exception as e:
            logger.error(f"戦略ID {strategy.id} の解析中にエラー: {e}")
            return False

    def clean_strategy_gene_data(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """
        戦略遺伝子データから無効な指標を削除

        Args:
            strategy: 戦略インスタンス

        Returns:
            クリーンアップ後の遺伝子データ
        """
        gene_data = strategy.gene_data.copy() if strategy.gene_data else {}
        original_indicators = gene_data.get("indicators", [])

        cleaned_indicators = []
        for indicator in original_indicators:
            if isinstance(indicator, dict):
                indicator_type = indicator.get("type")
            elif isinstance(indicator, str):
                indicator_type = indicator
            else:
                continue

            if indicator_type not in INVALID_INDICATORS:
                cleaned_indicators.append(indicator)
            else:
                logger.info(f"無効な指標 {indicator_type} を削除: 戦略ID={strategy.id}")

        gene_data["indicators"] = cleaned_indicators
        return gene_data

    def update_strategy_gene_data(self, strategy_id: int, new_gene_data: Dict[str, Any]) -> None:
        """
        戦略遺伝子データを更新

        Args:
            strategy_id: 戦略ID
            new_gene_data: 新しい遺伝子データ
        """
        try:
            # SQLAlchemy 2.0形式で更新
            strategy = self.db_session.get(GeneratedStrategy, strategy_id)
            if strategy:
                strategy.gene_data = new_gene_data
                self.db_session.commit()
                logger.info(f"戦略ID {strategy_id} の遺伝子データを更新しました")
            else:
                logger.warning(f"戦略ID {strategy_id} が見つかりません")

        except Exception as e:
            logger.error(f"戦略ID {strategy_id} の更新エラー: {e}")
            self.db_session.rollback()

    def mark_strategy_invalid(self, strategy: GeneratedStrategy) -> None:
        """
        戦略遺伝子を無効としてマーク

        Args:
            strategy: 戦略インスタンス
        """
        try:
            # 戦略に無効マークを追加
            gene_data = dict(strategy.gene_data) if strategy.gene_data else {}
            gene_data["__invalid__"] = True
            gene_data["__invalid_reason__"] = f"戦略に無効な指標を含みます: {INVALID_INDICATORS}"

            # データベースを更新
            self.update_strategy_gene_data(strategy.id, gene_data)
            logger.info(f"戦略ID {strategy.id} を無効としてマークしました")

        except Exception as e:
            logger.error(f"戦略ID {strategy.id} の無効化エラー: {e}")

    def delete_strategy(self, strategy: GeneratedStrategy) -> None:
        """
        戦略を削除

        Args:
            strategy: 戦略インスタンス
        """
        try:
            # SQLAlchemy 2.0形式で削除
            strategy_to_delete = self.db_session.get(GeneratedStrategy, strategy.id)
            if strategy_to_delete:
                self.db_session.delete(strategy_to_delete)
                self.db_session.commit()
                logger.info(f"戦略ID {strategy.id} を削除しました")
            else:
                logger.warning(f"戦略ID {strategy.id} が見つかりません")

        except Exception as e:
            logger.error(f"戦略ID {strategy.id} の削除エラー: {e}")
            self.db_session.rollback()

    def cleanup_invalid_strategies(self, action: str = "mark") -> None:
        """
        無効な戦略のクリーンアップを実行

        Args:
            action: 実行するアクション ("mark" または "delete")
        """
        invalid_strategies = self.find_strategies_with_invalid_indicators()

        if not invalid_strategies:
            logger.info("無効な戦略は見つかりませんでした")
            return

        for strategy in invalid_strategies:
            if action == "delete":
                self.delete_strategy(strategy)
            elif action == "clean":
                cleaned_data = self.clean_strategy_gene_data(strategy)
                self.update_strategy_gene_data(strategy.id, cleaned_data)
                logger.info(f"戦略ID {strategy.id} の遺伝子データをクリーンアップしました")
            else:  # mark
                self.mark_strategy_invalid(strategy)

    def display_invalid_strategies_summary(self) -> None:
        """無効な戦略のサマリーを表示"""
        invalid_strategies = self.find_strategies_with_invalid_indicators()

        if not invalid_strategies:
            print("SUCCESS: 無効な戦略は見つかりませんでした")
            return

        print(f"\n=== 無効な戦略サマリー ({len(invalid_strategies)} 件) ===")
        print("=" * 80)

        for strategy in invalid_strategies:
            print(f"戦略ID: {strategy.id}")
            print(f"  実験ID: {strategy.experiment_id}")
            print(f"  世代: {strategy.generation}")
            print(f"  作成日時: {strategy.created_at}")

            # indicators内容を表示
            gene_data = strategy.gene_data
            if gene_data and "indicators" in gene_data:
                indicators = gene_data["indicators"]
                invalid_indicator_types = []
                for ind in indicators:
                    ind_type = ind.get("type") if isinstance(ind, dict) else str(ind)
                    if ind_type in INVALID_INDICATORS:
                        invalid_indicator_types.append(ind_type)
                print(f"  無効な指標: {invalid_indicator_types}")
            print("-" * 40)

    def close(self):
        """リソースを解放"""
        if self.db_session:
            self.db_session.close()

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        データベースの統計情報を取得

        Returns:
            データベース統計情報
        """
        try:
            # データベース内の全戦略数を取得
            total_strategies = len(self.repository.get_filtered_data())

            # データベース接続情報
            stats = {
                "total_strategies": total_strategies,
                "database_healthy": self.db_session.connection().scalar("SELECT 1") == 1,
                "database_type": self.db_session.connection().engine.driver,
            }

            return stats

        except Exception as e:
            logger.error(f"データベース統計取得エラー: {e}")
            return {"error": str(e)}


def main():
    """メイン実行関数"""
    print("戦略遺伝子クリーンアップスクリプト")
    print("=" * 50)

    cleaner = StrategyGeneCleaner()

    try:
        # サマリーを表示
        cleaner.display_invalid_strategies_summary()

        # ユーザーアクションの選択
        if input("\nクリーンアップを実行しますか？ (y/N): ").lower().startswith('y'):
            action = input("アクションを選択 (mark/clean/delete) [mark]: ").strip().lower()
            if action not in ["mark", "clean", "delete"]:
                action = "mark"

            print(f"\n選択されたアクション: {action}")
            cleaner.cleanup_invalid_strategies(action)
            print("SUCCESS: クリーンアップが完了しました")

        else:
            print("キャンセルしました")

        # データベース統計を表示
        print("\n=== データベース統計 ===")
        stats = cleaner.get_database_statistics()
        print(f"総戦略数: {stats.get('total_strategies', '不明')}")
        print(f"DB正常: {stats.get('database_healthy', '不明')}")
        print(f"DBタイプ: {stats.get('database_type', '不明')}")

        print("\n完了しました")

    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()