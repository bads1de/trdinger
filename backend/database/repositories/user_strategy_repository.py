"""
ユーザー戦略リポジトリ

ストラテジービルダーで作成されたユーザー定義戦略のデータアクセス層
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
import logging

from ..models import UserStrategy

logger = logging.getLogger(__name__)


class UserStrategyRepository:
    """ユーザー戦略リポジトリクラス"""

    def __init__(self, db: Session):
        """
        初期化

        Args:
            db: データベースセッション
        """
        self.db = db

    def create(self, strategy_data: Dict[str, Any]) -> UserStrategy:
        """
        新しいユーザー戦略を作成

        Args:
            strategy_data: 戦略データ
                - name: 戦略名
                - description: 戦略の説明（オプション）
                - strategy_config: 戦略設定（StrategyGene形式）
                - is_active: アクティブフラグ（オプション、デフォルト: True）

        Returns:
            作成されたUserStrategyオブジェクト

        Raises:
            Exception: データベースエラーの場合
        """
        try:
            user_strategy = UserStrategy(
                name=strategy_data["name"],
                description=strategy_data.get("description"),
                strategy_config=strategy_data["strategy_config"],
                is_active=strategy_data.get("is_active", True),
            )

            self.db.add(user_strategy)
            self.db.commit()
            self.db.refresh(user_strategy)

            logger.info(f"ユーザー戦略を作成しました: ID={user_strategy.id}, 名前={user_strategy.name}")
            return user_strategy

        except Exception as e:
            self.db.rollback()
            logger.error(f"ユーザー戦略作成エラー: {e}")
            raise

    def get_by_id(self, strategy_id: int) -> Optional[UserStrategy]:
        """
        IDでユーザー戦略を取得

        Args:
            strategy_id: 戦略ID

        Returns:
            UserStrategyオブジェクト（見つからない場合はNone）
        """
        try:
            strategy = self.db.query(UserStrategy).filter(
                UserStrategy.id == strategy_id
            ).first()

            if strategy:
                logger.debug(f"ユーザー戦略を取得しました: ID={strategy_id}")
            else:
                logger.debug(f"ユーザー戦略が見つかりません: ID={strategy_id}")

            return strategy

        except Exception as e:
            logger.error(f"ユーザー戦略取得エラー (ID={strategy_id}): {e}")
            raise

    def get_all(self, active_only: bool = True, limit: Optional[int] = None) -> List[UserStrategy]:
        """
        全てのユーザー戦略を取得

        Args:
            active_only: アクティブな戦略のみを取得するか
            limit: 取得件数制限

        Returns:
            UserStrategyオブジェクトのリスト
        """
        try:
            query = self.db.query(UserStrategy)

            if active_only:
                query = query.filter(UserStrategy.is_active == True)

            # 作成日時の降順でソート
            query = query.order_by(desc(UserStrategy.created_at))

            if limit:
                query = query.limit(limit)

            strategies = query.all()

            logger.info(f"ユーザー戦略を取得しました: {len(strategies)}件")
            return strategies

        except Exception as e:
            logger.error(f"ユーザー戦略一覧取得エラー: {e}")
            raise

    def get_by_name(self, name: str, active_only: bool = True) -> List[UserStrategy]:
        """
        名前でユーザー戦略を検索

        Args:
            name: 戦略名（部分一致）
            active_only: アクティブな戦略のみを取得するか

        Returns:
            UserStrategyオブジェクトのリスト
        """
        try:
            query = self.db.query(UserStrategy).filter(
                UserStrategy.name.ilike(f"%{name}%")
            )

            if active_only:
                query = query.filter(UserStrategy.is_active == True)

            strategies = query.order_by(desc(UserStrategy.created_at)).all()

            logger.debug(f"名前検索結果: '{name}' -> {len(strategies)}件")
            return strategies

        except Exception as e:
            logger.error(f"ユーザー戦略名前検索エラー (name={name}): {e}")
            raise

    def update(self, strategy_id: int, update_data: Dict[str, Any]) -> Optional[UserStrategy]:
        """
        ユーザー戦略を更新

        Args:
            strategy_id: 戦略ID
            update_data: 更新データ
                - name: 戦略名（オプション）
                - description: 戦略の説明（オプション）
                - strategy_config: 戦略設定（オプション）
                - is_active: アクティブフラグ（オプション）

        Returns:
            更新されたUserStrategyオブジェクト（見つからない場合はNone）

        Raises:
            Exception: データベースエラーの場合
        """
        try:
            strategy = self.get_by_id(strategy_id)
            if not strategy:
                logger.warning(f"更新対象のユーザー戦略が見つかりません: ID={strategy_id}")
                return None

            # 更新可能なフィールドのみを更新
            if "name" in update_data:
                strategy.name = update_data["name"]
            if "description" in update_data:
                strategy.description = update_data["description"]
            if "strategy_config" in update_data:
                strategy.strategy_config = update_data["strategy_config"]
            if "is_active" in update_data:
                strategy.is_active = update_data["is_active"]

            self.db.commit()
            self.db.refresh(strategy)

            logger.info(f"ユーザー戦略を更新しました: ID={strategy_id}")
            return strategy

        except Exception as e:
            self.db.rollback()
            logger.error(f"ユーザー戦略更新エラー (ID={strategy_id}): {e}")
            raise

    def delete(self, strategy_id: int) -> bool:
        """
        ユーザー戦略を削除（論理削除）

        Args:
            strategy_id: 戦略ID

        Returns:
            削除成功の場合True、見つからない場合False

        Raises:
            Exception: データベースエラーの場合
        """
        try:
            strategy = self.get_by_id(strategy_id)
            if not strategy:
                logger.warning(f"削除対象のユーザー戦略が見つかりません: ID={strategy_id}")
                return False

            # 論理削除（is_activeをFalseに設定）
            strategy.is_active = False
            self.db.commit()

            logger.info(f"ユーザー戦略を論理削除しました: ID={strategy_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"ユーザー戦略削除エラー (ID={strategy_id}): {e}")
            raise

    def hard_delete(self, strategy_id: int) -> bool:
        """
        ユーザー戦略を物理削除

        Args:
            strategy_id: 戦略ID

        Returns:
            削除成功の場合True、見つからない場合False

        Raises:
            Exception: データベースエラーの場合
        """
        try:
            strategy = self.get_by_id(strategy_id)
            if not strategy:
                logger.warning(f"削除対象のユーザー戦略が見つかりません: ID={strategy_id}")
                return False

            self.db.delete(strategy)
            self.db.commit()

            logger.info(f"ユーザー戦略を物理削除しました: ID={strategy_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"ユーザー戦略物理削除エラー (ID={strategy_id}): {e}")
            raise

    def count(self, active_only: bool = True) -> int:
        """
        ユーザー戦略の件数を取得

        Args:
            active_only: アクティブな戦略のみをカウントするか

        Returns:
            戦略の件数
        """
        try:
            query = self.db.query(UserStrategy)

            if active_only:
                query = query.filter(UserStrategy.is_active == True)

            count = query.count()

            logger.debug(f"ユーザー戦略件数: {count}件")
            return count

        except Exception as e:
            logger.error(f"ユーザー戦略件数取得エラー: {e}")
            raise
