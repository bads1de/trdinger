"""
戦略ショーケースリポジトリ

StrategyShowcaseテーブルへのデータアクセスを提供します。
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_

# from database.models import StrategyShowcase
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class StrategyShowcaseRepository(BaseRepository):
    """戦略ショーケースのリポジトリクラス（一時的に無効化）"""

    def __init__(self, db: Session):
        # super().__init__(db, StrategyShowcase)
        self.db = db

    def get_strategies_with_filters(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        フィルター条件付きで戦略を取得

        Args:
            limit: 取得件数制限
            offset: オフセット
            filters: フィルター条件
            sort_by: ソート項目
            sort_order: ソート順序

        Returns:
            戦略データと統計情報
        """
        try:
            query = self.db.query(StrategyShowcase)

            # フィルター適用
            if filters:
                if filters.get("category"):
                    query = query.filter(
                        StrategyShowcase.category == filters["category"]
                    )

                if filters.get("risk_level"):
                    query = query.filter(
                        StrategyShowcase.risk_level == filters["risk_level"]
                    )

                if filters.get("min_return"):
                    query = query.filter(
                        StrategyShowcase.expected_return >= filters["min_return"]
                    )

                if filters.get("max_return"):
                    query = query.filter(
                        StrategyShowcase.expected_return <= filters["max_return"]
                    )

                if filters.get("min_sharpe"):
                    query = query.filter(
                        StrategyShowcase.sharpe_ratio >= filters["min_sharpe"]
                    )

                if filters.get("max_sharpe"):
                    query = query.filter(
                        StrategyShowcase.sharpe_ratio <= filters["max_sharpe"]
                    )

                if filters.get("search_query"):
                    search_term = f"%{filters['search_query']}%"
                    query = query.filter(
                        or_(
                            StrategyShowcase.name.ilike(search_term),
                            StrategyShowcase.description.ilike(search_term),
                        )
                    )

            # 総件数を取得
            total_count = query.count()

            # ソート適用
            if sort_by == "expected_return":
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(StrategyShowcase.expected_return))
                else:
                    query = query.order_by(asc(StrategyShowcase.expected_return))
            elif sort_by == "sharpe_ratio":
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(StrategyShowcase.sharpe_ratio))
                else:
                    query = query.order_by(asc(StrategyShowcase.sharpe_ratio))
            elif sort_by == "max_drawdown":
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(StrategyShowcase.max_drawdown))
                else:
                    query = query.order_by(asc(StrategyShowcase.max_drawdown))
            elif sort_by == "win_rate":
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(StrategyShowcase.win_rate))
                else:
                    query = query.order_by(asc(StrategyShowcase.win_rate))
            else:  # created_at or default
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(StrategyShowcase.created_at))
                else:
                    query = query.order_by(asc(StrategyShowcase.created_at))

            # ページネーション適用
            if offset > 0:
                query = query.offset(offset)
            if limit > 0:
                query = query.limit(limit)

            strategies = query.all()

            # 辞書形式に変換
            strategy_list = []
            for strategy in strategies:
                strategy_dict = {
                    "id": strategy.id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "category": strategy.category,
                    "indicators": strategy.indicators if strategy.indicators else [],
                    "parameters": strategy.parameters if strategy.parameters else {},
                    "expected_return": strategy.expected_return,
                    "sharpe_ratio": strategy.sharpe_ratio,
                    "max_drawdown": strategy.max_drawdown,
                    "win_rate": strategy.win_rate,
                    "risk_level": strategy.risk_level,
                    "recommended_timeframe": strategy.recommended_timeframe,
                    "is_active": strategy.is_active,
                    "created_at": (
                        strategy.created_at.isoformat() if strategy.created_at else None
                    ),
                    "updated_at": (
                        strategy.updated_at.isoformat() if strategy.updated_at else None
                    ),
                }
                strategy_list.append(strategy_dict)

            return {
                "strategies": strategy_list,
                "total_count": total_count,
                "has_more": offset + limit < total_count,
            }

        except Exception as e:
            logger.error(f"フィルター付き戦略取得エラー: {e}")
            return {"strategies": [], "total_count": 0, "has_more": False}

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """
        IDで戦略を取得

        Args:
            strategy_id: 戦略ID

        Returns:
            戦略データ
        """
        try:
            strategy = (
                self.db.query(StrategyShowcase)
                .filter(StrategyShowcase.id == strategy_id)
                .first()
            )

            if not strategy:
                return None

            return {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "category": strategy.category,
                "indicators": strategy.indicators if strategy.indicators else [],
                "parameters": strategy.parameters if strategy.parameters else {},
                "expected_return": strategy.expected_return,
                "sharpe_ratio": strategy.sharpe_ratio,
                "max_drawdown": strategy.max_drawdown,
                "win_rate": strategy.win_rate,
                "risk_level": strategy.risk_level,
                "recommended_timeframe": strategy.recommended_timeframe,
                "is_active": strategy.is_active,
                "created_at": (
                    strategy.created_at.isoformat() if strategy.created_at else None
                ),
                "updated_at": (
                    strategy.updated_at.isoformat() if strategy.updated_at else None
                ),
            }

        except Exception as e:
            logger.error(f"戦略取得エラー: {e}")
            return None

    def create_strategy(
        self, strategy_data: Dict[str, Any]
    ) -> Optional[StrategyShowcase]:
        """
        新しい戦略を作成

        Args:
            strategy_data: 戦略データ

        Returns:
            作成された戦略
        """
        try:
            strategy = StrategyShowcase(
                name=strategy_data["name"],
                description=strategy_data.get("description", ""),
                category=strategy_data.get("category", "unknown"),
                indicators=strategy_data.get("indicators", []),
                parameters=strategy_data.get("parameters", {}),
                expected_return=strategy_data.get("expected_return", 0.0),
                sharpe_ratio=strategy_data.get("sharpe_ratio", 0.0),
                max_drawdown=strategy_data.get("max_drawdown", 0.0),
                win_rate=strategy_data.get("win_rate", 0.0),
                risk_level=strategy_data.get("risk_level", "medium"),
                recommended_timeframe=strategy_data.get("recommended_timeframe", "1h"),
                is_active=strategy_data.get("is_active", True),
            )

            self.db.add(strategy)
            self.db.commit()
            self.db.refresh(strategy)

            logger.info(f"戦略を作成しました: {strategy.id} ({strategy.name})")
            return strategy

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略作成エラー: {e}")
            return None

    def update_strategy(self, strategy_id: int, update_data: Dict[str, Any]) -> bool:
        """
        戦略を更新

        Args:
            strategy_id: 戦略ID
            update_data: 更新データ

        Returns:
            更新成功フラグ
        """
        try:
            strategy = (
                self.db.query(StrategyShowcase)
                .filter(StrategyShowcase.id == strategy_id)
                .first()
            )

            if not strategy:
                return False

            # 更新可能なフィールドのみ更新
            updatable_fields = [
                "name",
                "description",
                "category",
                "indicators",
                "parameters",
                "expected_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "risk_level",
                "recommended_timeframe",
                "is_active",
            ]

            for field in updatable_fields:
                if field in update_data:
                    setattr(strategy, field, update_data[field])

            self.db.commit()
            logger.info(f"戦略を更新しました: {strategy_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略更新エラー: {e}")
            return False

    def delete_strategy(self, strategy_id: int) -> bool:
        """
        戦略を削除

        Args:
            strategy_id: 戦略ID

        Returns:
            削除成功フラグ
        """
        try:
            strategy = (
                self.db.query(StrategyShowcase)
                .filter(StrategyShowcase.id == strategy_id)
                .first()
            )

            if not strategy:
                return False

            self.db.delete(strategy)
            self.db.commit()

            logger.info(f"戦略を削除しました: {strategy_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"戦略削除エラー: {e}")
            return False

    def get_categories(self) -> List[str]:
        """
        利用可能なカテゴリ一覧を取得

        Returns:
            カテゴリリスト
        """
        try:
            categories = self.db.query(StrategyShowcase.category).distinct().all()
            return [category[0] for category in categories if category[0]]

        except Exception as e:
            logger.error(f"カテゴリ取得エラー: {e}")
            return []

    def get_risk_levels(self) -> List[str]:
        """
        利用可能なリスクレベル一覧を取得

        Returns:
            リスクレベルリスト
        """
        try:
            risk_levels = self.db.query(StrategyShowcase.risk_level).distinct().all()
            return [level[0] for level in risk_levels if level[0]]

        except Exception as e:
            logger.error(f"リスクレベル取得エラー: {e}")
            return []
