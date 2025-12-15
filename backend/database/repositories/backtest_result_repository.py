"""
バックテスト結果リポジトリ

"""

import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import BacktestResult

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class BacktestResultRepository(BaseRepository):
    """バックテスト結果のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, BacktestResult)

    def to_dict(self, model_instance: BacktestResult) -> dict:
        """バックテスト結果を辞書に変換

        Args:
            model_instance: 変換するモデルインスタンス

        Returns:
            変換された辞書
        """
        d = super().to_dict(model_instance)
        performance_metrics = d.get("performance_metrics") or {}
        # 保守性のため個別メトリクスをトップレベルに追加（後方互換性）
        d.setdefault("total_return", performance_metrics.get("total_return", 0.0))
        d.setdefault("sharpe_ratio", performance_metrics.get("sharpe_ratio", 0.0))
        d.setdefault("max_drawdown", performance_metrics.get("max_drawdown", 0.0))
        d.setdefault("total_trades", performance_metrics.get("total_trades", 0))
        d.setdefault("win_rate", performance_metrics.get("win_rate", 0.0))
        d.setdefault("profit_factor", performance_metrics.get("profit_factor", 0.0))
        d.setdefault("final_balance", performance_metrics.get("final_balance", 0.0))
        return d

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="バックテスト結果保存", is_api_call=False)
        def _save_result():
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
            return self.to_dict(backtest_result)

        return _save_result()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="バックテスト結果一覧取得", is_api_call=False)
        def _get_results():
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
            return [self.to_dict(result) for result in results]

        return _get_results()

    def get_backtest_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """
        ID指定でバックテスト結果を取得

        Args:
            result_id: バックテスト結果ID

        Returns:
            バックテスト結果、見つからない場合はNone
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="ID指定バックテスト結果取得", is_api_call=False)
        def _get_result_by_id():
            # BaseRepositoryの汎用メソッドを使用
            results = self.get_filtered_data(
                filters={"id": result_id},
                limit=1,
            )
            if results:
                # BaseRepositoryのget_filtered_dataは適切な型を返すため、
                # キャストは不要です。
                result = results[0]
                return self.to_dict(result)
            return None

        return _get_result_by_id()

    def delete_backtest_result(self, result_id: int) -> bool:
        """
        バックテスト結果を削除

        Args:
            result_id: バックテスト結果ID

        Returns:
            削除成功時True、見つからない場合False
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="バックテスト結果削除", is_api_call=False)
        def _delete_result():
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

        return _delete_result()

    def delete_all_backtest_results(self) -> int:
        """
        すべてのバックテスト結果を削除

        Returns:
            削除された件数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="全バックテスト結果削除", is_api_call=False)
        def _delete_all_results():
            deleted_count = self.db.query(BacktestResult).delete()
            self.db.commit()
            return deleted_count

        return _delete_all_results()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="バックテスト結果総数取得", is_api_call=False)
        def _count_results():
            query = self.db.query(BacktestResult)

            # フィルター適用
            # symbolまたはstrategy_nameが指定された場合、対応する条件でクエリをフィルタリングします。
            # これにより、特定の取引ペアや戦略の結果のみを検索できます。
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)

            return query.count()

        return _count_results()

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context="最近バックテスト結果取得", is_api_call=False)
        def _get_recent_results():
            # BaseRepositoryの汎用メソッドを使用
            results = self.get_latest_records(
                timestamp_column="created_at",
                limit=limit,
            )
            # 取得した結果を辞書形式のリストに変換して返します。
            return [self.to_dict(result) for result in results]

        return _get_recent_results()


