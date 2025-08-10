"""
バックテスト結果リポジトリ

"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, date, time
import logging
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import desc, select, func, delete
import numpy as np
import pandas as pd

from .base_repository import BaseRepository
from database.models import BacktestResult

logger = logging.getLogger(__name__)


class BacktestResultRepository(BaseRepository):
    """バックテスト結果のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, BacktestResult)

    def _to_json_safe(self, obj: Any) -> Any:
        """JSONにシリアライズ可能な形へ再帰的に変換"""
        try:
            # Noneやプリミティブ型
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            # 日付・時刻
            if isinstance(obj, (datetime, date, time)):
                return obj.isoformat()
            # pandas Timestamp
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            # Decimal
            if isinstance(obj, Decimal):
                return float(obj)
            # numpyスカラ
            if isinstance(obj, np.generic):
                return obj.item()
            # numpy配列
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # 辞書
            if isinstance(obj, dict):
                return {k: self._to_json_safe(v) for k, v in obj.items()}
            # リスト/タプル/セット
            if isinstance(obj, (list, tuple, set)):
                return [self._to_json_safe(v) for v in obj]
            # それ以外は文字列化の最後の手段
            return obj
        except Exception:
            return str(obj)

    def _normalize_result_data(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """入力データを正規化し、BacktestResultモデルの形式に変換する"""

        # 日付の処理
        start_date = result_data.get("start_date")
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        end_date = result_data.get("end_date")
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # パフォーマンス指標の正規化（後方互換性対応）
        performance_metrics = result_data.get("performance_metrics", {})
        if not performance_metrics:
            performance_metrics = result_data.get("results_json", {}).get(
                "performance_metrics", {}
            )
        if not performance_metrics:
            performance_metrics = {
                "total_return": result_data.get("total_return", 0.0),
                "sharpe_ratio": result_data.get("sharpe_ratio", 0.0),
                "max_drawdown": result_data.get("max_drawdown", 0.0),
                "total_trades": result_data.get("total_trades", 0),
                "win_rate": result_data.get("win_rate", 0.0),
                "profit_factor": result_data.get("profit_factor", 0.0),
            }

        # その他のデータの正規化
        equity_curve = result_data.get(
            "equity_curve", result_data.get("results_json", {}).get("equity_curve", [])
        )
        trade_history = result_data.get(
            "trade_history",
            result_data.get("results_json", {}).get("trade_history", []),
        )

        # JSON列はシリアライズ可能に正規化
        config_json = self._to_json_safe(result_data.get("config_json", {}))
        performance_metrics = self._to_json_safe(performance_metrics)
        equity_curve = self._to_json_safe(equity_curve)
        trade_history = self._to_json_safe(trade_history)

        return {
            "strategy_name": result_data["strategy_name"],
            "symbol": result_data["symbol"],
            "timeframe": result_data["timeframe"],
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": result_data["initial_capital"],
            "commission_rate": result_data.get("commission_rate", 0.001),
            "config_json": config_json,
            "performance_metrics": performance_metrics,
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "execution_time": result_data.get("execution_time"),
            "status": result_data.get("status", "completed"),
            "error_message": result_data.get("error_message"),
        }

    def save_backtest_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        バックテスト結果を保存

        Args:
            result_data: バックテスト結果データ

        Returns:
            保存されたバックテスト結果（ID付き）
        """
        try:
            log_data = {
                k: v
                for k, v in result_data.items()
                if k not in ["equity_curve", "trade_history", "config_json"]
            }
            log_data["config_json"] = "..."
            logger.info(f"バックテスト結果を保存中: {log_data}")

            normalized_data = self._normalize_result_data(result_data)
            backtest_result = BacktestResult(**normalized_data)

            self.db.add(backtest_result)
            self.db.commit()
            self.db.refresh(backtest_result)

            logger.info(f"バックテスト結果を保存しました (ID: {backtest_result.id})")
            return backtest_result.to_dict()

        except Exception as e:
            self.db.rollback()
            logger.error(f"バックテスト結果の保存に失敗しました: {e}", exc_info=True)
            raise Exception(f"バックテスト結果の保存に失敗しました: {str(e)}")

    def get_backtest_results(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        バックテスト結果一覧を取得

        Args:
            limit: 取得件数
            offset: オフセット
            symbol: 取引ペアフィルター
            strategy_name: 戦略名フィルター

        Returns:
            バックテスト結果のリスト
        """
        try:
            # フィルター条件を構築
            filters = {}
            if symbol:
                filters["symbol"] = symbol
            if strategy_name:
                filters["strategy_name"] = strategy_name

            # BaseRepositoryの汎用メソッドを使用
            results = self.get_filtered_data(
                filters=filters,
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
                offset=offset,
            )

            # 辞書形式に変換
            return [result.to_dict() for result in results]

        except Exception as e:
            raise Exception(f"バックテスト結果の取得に失敗しました: {str(e)}")

    def get_backtest_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """
        ID指定でバックテスト結果を取得

        Args:
            result_id: バックテスト結果ID

        Returns:
            バックテスト結果、見つからない場合はNone
        """
        try:
            # BaseRepositoryの汎用メソッドを使用
            results = self.get_filtered_data(
                filters={"id": result_id},
                limit=1,
            )
            if results:
                # BaseRepositoryのget_filtered_dataは適切な型を返すため、
                # キャストは不要です。
                result = results[0]
                return result.to_dict()
            return None

        except Exception as e:
            raise Exception(
                f"ID指定によるバックテスト結果の取得に失敗しました: {str(e)}"
            )

    def delete_backtest_result(self, result_id: int) -> bool:
        """
        バックテスト結果を削除

        Args:
            result_id: バックテスト結果ID

        Returns:
            削除成功時True、見つからない場合False
        """
        try:
            result = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.id == result_id)
                .first()
            )

            if result:
                self.db.delete(result)
                self.db.commit()
                return True
            return False

        except Exception as e:
            self.db.rollback()
            # 削除中にエラーが発生した場合、トランザクションをロールバックしてデータベースの状態を
            # 変更前の状態に戻します。これにより、部分的な削除を防ぎ、データの一貫性を保ちます。
            raise Exception(f"バックテスト結果の削除に失敗しました: {str(e)}")

    def delete_all_backtest_results(self) -> int:
        """
        すべてのバックテスト結果を削除

        Returns:
            削除された件数
        """
        try:
            deleted_count = self.db.query(BacktestResult).delete()
            self.db.commit()
            return deleted_count

        except Exception as e:
            self.db.rollback()
            # すべての削除中にエラーが発生した場合、トランザクションをロールバックします。
            # これにより、データベースの部分的な変更を防ぎ、整合性を維持します。
            raise Exception(f"すべてのバックテスト結果の削除に失敗しました: {str(e)}")

    def count_backtest_results(
        self, symbol: Optional[str] = None, strategy_name: Optional[str] = None
    ) -> int:
        """
        バックテスト結果の総数を取得

        Args:
            symbol: 取引ペアフィルター
            strategy_name: 戦略名フィルター

        Returns:
            バックテスト結果の総数
        """
        try:
            query = self.db.query(BacktestResult)

            # フィルター適用
            # symbolまたはstrategy_nameが指定された場合、対応する条件でクエリをフィルタリングします。
            # これにより、特定の取引ペアや戦略の結果のみを検索できます。
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)

            return query.count()

        except Exception as e:
            raise Exception(f"バックテスト結果の総数取得に失敗しました: {str(e)}")

    def get_backtest_results_by_strategy(
        self, strategy_name: str
    ) -> List[Dict[str, Any]]:
        """
        戦略名でバックテスト結果を取得

        Args:
            strategy_name: 戦略名

        Returns:
            バックテスト結果のリスト
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = (
                select(BacktestResult)
                .where(BacktestResult.strategy_name == strategy_name)
                .order_by(desc(BacktestResult.created_at))
            )
            results = list(self.db.scalars(stmt).all())

            return [result.to_dict() for result in results]

        except Exception as e:
            raise Exception(
                f"戦略名によるバックテスト結果の取得に失敗しました: {str(e)}"
            )

    def get_backtest_results_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        取引ペアでバックテスト結果を取得

        Args:
            symbol: 取引ペア

        Returns:
            バックテスト結果のリスト
        """
        try:
            results = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.symbol == symbol)
                .order_by(desc(BacktestResult.created_at))
                .all()
            )

            # データベースから取得したBacktestResultオブジェクトのリストを、
            # クライアントに返しやすい辞書形式のリストに変換します。
            # 各BacktestResultインスタンスはto_dictメソッドを持っています。
            return [result.to_dict() for result in results]

        except Exception as e:
            raise Exception(
                f"取引ペアによるバックテスト結果の取得に失敗しました: {str(e)}"
            )

    def get_recent_backtest_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        最近のバックテスト結果を取得

        BaseRepositoryの汎用メソッドを利用して、最新のバックテスト結果を効率的に取得します。
        これにより、クエリロジックの重複を防ぎ、コードの再利用性と保守性を高めています。

        Args:
            limit: 取得件数

        Returns:
            バックテスト結果のリスト
        """
        try:
            # BaseRepositoryの汎用メソッドを使用
            results = self.get_latest_records(
                timestamp_column="created_at",
                limit=limit,
            )
            # 取得した結果を辞書形式のリストに変換して返します。
            return [result.to_dict() for result in results]

        except Exception as e:
            raise Exception(f"最近のバックテスト結果の取得に失敗しました: {str(e)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンスサマリーを取得

        このメソッドは、データベースに保存されているすべてのバックテスト結果の概要を計算します。
        具体的には、総結果数、戦略のユニーク数、そして最近の5件の結果を提供します。
        結果が一つもない場合は、すべてのメトリクスをゼロで返します。

        Returns:
            パフォーマンスサマリー
        """
        try:
            # 総バックテスト結果数を取得（SQLAlchemy 2.0 標準API使用）
            total_results = (
                self.db.scalar(select(func.count()).select_from(BacktestResult)) or 0
            )

            if total_results == 0:
                # 結果がない場合は、デフォルトのサマリーを返す
                return {
                    "total_results": 0,
                    "avg_return": 0,
                    "best_return": 0,
                    "worst_return": 0,
                    "strategies_count": 0,
                }

            # 戦略数を取得（SQLAlchemy 2.0 標準API使用）
            # func.distinctを使用して、重複しない戦略名の数をカウントします。
            # これにより、異なる戦略がいくつ実行されたかを把握できます。
            strategies_count = (
                self.db.scalar(
                    select(func.count(func.distinct(BacktestResult.strategy_name)))
                )
                or 0
            )

            return {
                "total_results": total_results,
                "strategies_count": strategies_count,
                "recent_results": self.get_recent_backtest_results(5),
            }

        except Exception as e:
            raise Exception(f"パフォーマンスサマリーの取得に失敗しました: {str(e)}")

    def cleanup_old_results(self, days_to_keep: int = 30) -> int:
        """
        古いバックテスト結果を削除

        このメソッドは、指定された日数よりも古いバックテスト結果をデータベースから削除します。
        データの肥大化を防ぎ、パフォーマンスを維持するために定期的なクリーンアップを目的としています。

        Args:
            days_to_keep: 保持する日数。この日数よりも古いデータが削除されます。

        Returns:
            削除された件数
        """
        try:
            logger.info(
                f"{days_to_keep}日より古いバックテスト結果のクリーンアップを開始します。"
            )

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            logger.debug(f"削除対象の基準日: {cutoff_date}")

            # SQLAlchemy 2.0の標準的なdelete文を使用
            stmt = delete(BacktestResult).where(BacktestResult.created_at < cutoff_date)
            result = self.db.execute(stmt)
            deleted_count = result.rowcount

            self.db.commit()

            if deleted_count > 0:
                logger.info(f"{deleted_count}件の古いバックテスト結果を削除しました。")
            else:
                logger.info("削除対象の古いバックテスト結果はありませんでした。")

            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(
                f"古いバックテスト結果のクリーンアップ中にエラーが発生しました: {e}",
                exc_info=True,
            )
            raise Exception(
                f"古いバックテスト結果のクリーンアップに失敗しました: {str(e)}"
            )
