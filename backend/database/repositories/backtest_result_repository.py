"""
バックテスト結果リポジトリ

BacktestResultモデルのデータアクセス機能を提供します。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from .base_repository import BaseRepository
from database.models import BacktestResult


class BacktestResultRepository(BaseRepository):
    """バックテスト結果のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, BacktestResult)

    def save_backtest_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        バックテスト結果を保存

        Args:
            result_data: バックテスト結果データ

        Returns:
            保存されたバックテスト結果（ID付き）
        """
        try:
            # 日付の処理
            start_date = result_data.get("start_date")
            end_date = result_data.get("end_date")

            # datetimeオブジェクトの場合はそのまま使用、文字列の場合は変換
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)

            # パフォーマンス指標の構築
            performance_metrics = result_data.get("results_json", {}).get(
                "performance_metrics", {}
            )
            if not performance_metrics:
                # 個別フィールドからパフォーマンス指標を構築
                performance_metrics = {
                    "total_return": result_data.get("total_return", 0.0),
                    "sharpe_ratio": result_data.get("sharpe_ratio", 0.0),
                    "max_drawdown": result_data.get("max_drawdown", 0.0),
                    "total_trades": result_data.get("total_trades", 0),
                    "win_rate": result_data.get("win_rate", 0.0),
                    "profit_factor": result_data.get("profit_factor", 0.0),
                }

            # BacktestResultインスタンスを作成
            backtest_result = BacktestResult(
                strategy_name=result_data["strategy_name"],
                symbol=result_data["symbol"],
                timeframe=result_data["timeframe"],
                start_date=start_date,
                end_date=end_date,
                initial_capital=result_data["initial_capital"],
                commission_rate=result_data.get("commission_rate", 0.001),
                config_json=result_data.get("config_json", {}),
                performance_metrics=performance_metrics,
                equity_curve=result_data.get("results_json", {}).get(
                    "equity_curve", []
                ),
                trade_history=result_data.get("results_json", {}).get(
                    "trade_history", []
                ),
                execution_time=result_data.get("execution_time"),
                status=result_data.get("status", "completed"),
                error_message=result_data.get("error_message"),
            )

            # データベースに保存
            self.db.add(backtest_result)
            self.db.commit()
            self.db.refresh(backtest_result)

            # BacktestResultオブジェクトを返す（to_dictメソッドがない場合に備えて）
            return backtest_result

        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to save backtest result: {str(e)}")

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
            query = self.db.query(BacktestResult)

            # フィルター適用
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)

            # 作成日時の降順でソート
            query = query.order_by(desc(BacktestResult.created_at))

            # ページネーション
            query = query.offset(offset).limit(limit)

            results = query.all()

            # 辞書形式に変換（to_dictメソッドがない場合に備えて）
            return [self._result_to_dict(result) for result in results]

        except Exception as e:
            raise Exception(f"Failed to get backtest results: {str(e)}")

    def get_backtest_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """
        ID指定でバックテスト結果を取得

        Args:
            result_id: バックテスト結果ID

        Returns:
            バックテスト結果、見つからない場合はNone
        """
        try:
            result = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.id == result_id)
                .first()
            )

            if result:
                return result
            return None

        except Exception as e:
            raise Exception(f"Failed to get backtest result by ID: {str(e)}")

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
            raise Exception(f"Failed to delete backtest result: {str(e)}")

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
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)

            return query.count()

        except Exception as e:
            raise Exception(f"Failed to count backtest results: {str(e)}")

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
            results = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.strategy_name == strategy_name)
                .order_by(desc(BacktestResult.created_at))
                .all()
            )

            return [self._result_to_dict(result) for result in results]

        except Exception as e:
            raise Exception(f"Failed to get backtest results by strategy: {str(e)}")

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

            return [self._result_to_dict(result) for result in results]

        except Exception as e:
            raise Exception(f"Failed to get backtest results by symbol: {str(e)}")

    def get_recent_backtest_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        最新のバックテスト結果を取得

        Args:
            limit: 取得件数

        Returns:
            最新のバックテスト結果のリスト
        """
        try:
            results = (
                self.db.query(BacktestResult)
                .order_by(desc(BacktestResult.created_at))
                .limit(limit)
                .all()
            )

            return [self._result_to_dict(result) for result in results]

        except Exception as e:
            raise Exception(f"Failed to get recent backtest results: {str(e)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンスサマリーを取得

        Returns:
            パフォーマンスサマリー
        """
        try:
            total_results = self.db.query(BacktestResult).count()

            if total_results == 0:
                return {
                    "total_results": 0,
                    "avg_return": 0,
                    "best_return": 0,
                    "worst_return": 0,
                    "strategies_count": 0,
                }

            # 基本統計を計算（JSONBフィールドから値を抽出）
            from sqlalchemy import func

            # 戦略数
            strategies_count = self.db.query(
                func.count(func.distinct(BacktestResult.strategy_name))
            ).scalar()

            return {
                "total_results": total_results,
                "strategies_count": strategies_count,
                "recent_results": self.get_recent_backtest_results(5),
            }

        except Exception as e:
            raise Exception(f"Failed to get performance summary: {str(e)}")

    def cleanup_old_results(self, days_to_keep: int = 30) -> int:
        """
        古いバックテスト結果を削除

        Args:
            days_to_keep: 保持する日数

        Returns:
            削除された件数
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            deleted_count = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.created_at < cutoff_date)
                .delete()
            )

            self.db.commit()
            return deleted_count

        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to cleanup old results: {str(e)}")

    def _result_to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """
        BacktestResultオブジェクトを辞書に変換

        Args:
            result: BacktestResultオブジェクト

        Returns:
            辞書形式のデータ
        """
        # パフォーマンス指標から個別の値を抽出
        performance_metrics = result.performance_metrics or {}

        return {
            "id": result.id,
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "start_date": result.start_date.isoformat() if result.start_date else None,
            "end_date": result.end_date.isoformat() if result.end_date else None,
            "initial_capital": result.initial_capital,
            "commission_rate": result.commission_rate,
            "config_json": result.config_json,
            "performance_metrics": performance_metrics,
            "equity_curve": result.equity_curve,
            "trade_history": result.trade_history,
            # 個別のパフォーマンス指標（後方互換性のため）
            "total_return": performance_metrics.get("total_return", 0.0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": performance_metrics.get("max_drawdown", 0.0),
            "total_trades": performance_metrics.get("total_trades", 0),
            "win_rate": performance_metrics.get("win_rate", 0.0),
            "profit_factor": performance_metrics.get("profit_factor", 0.0),
            "execution_time": result.execution_time,
            "status": result.status,
            "error_message": result.error_message,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "updated_at": result.updated_at.isoformat() if result.updated_at else None,
        }
