"""
ベイジアン最適化結果リポジトリ

ベイジアン最適化結果の永続化処理を管理します。
"""

from typing import List, Optional, Dict, Any, cast
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
import logging
import numpy as np

from .base_repository import BaseRepository
from database.models import BayesianOptimizationResult
from app.core.utils.database_utils import DatabaseQueryHelper

logger = logging.getLogger(__name__)


class BayesianOptimizationRepository(BaseRepository):
    """ベイジアン最適化結果のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, BayesianOptimizationResult)

    def _convert_numpy_types(self, obj):
        """NumPy型をPythonの標準型に再帰的に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _normalize_model_type(self, model_type: str) -> str:
        """モデルタイプの大文字小文字を正規化"""
        if not model_type:
            return model_type

        # 一般的なモデルタイプの正規化
        model_type_mapping = {
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "catboost": "CatBoost",
            "svm": "SVM",
            "logisticregression": "LogisticRegression",
            "decisiontree": "DecisionTree",
            "gradientboosting": "GradientBoosting",
        }

        return model_type_mapping.get(model_type.lower(), model_type)

    def create_optimization_result(
        self,
        profile_name: str,
        optimization_type: str,
        best_params: Dict[str, Any],
        best_score: float,
        total_evaluations: int,
        optimization_time: float,
        convergence_info: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        model_type: Optional[str] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: bool = True,
        is_default: bool = False,
        target_model_type: Optional[str] = None,
    ) -> BayesianOptimizationResult:
        """
        新しいベイジアン最適化結果を作成

        Args:
            profile_name: プロファイル名
            optimization_type: 最適化タイプ
            best_params: 最適パラメータ
            best_score: 最高スコア
            total_evaluations: 総評価回数
            optimization_time: 最適化時間
            convergence_info: 収束情報
            optimization_history: 最適化履歴
            model_type: モデルタイプ（オプション）
            experiment_name: 実験名（オプション）
            description: 説明（オプション）
            is_active: アクティブフラグ
            is_default: デフォルトプロファイルかどうか
            target_model_type: 対象モデルタイプ（オプション）

        Returns:
            作成されたベイジアン最適化結果

        Raises:
            Exception: 作成に失敗した場合
        """
        try:
            # NumPy型を変換
            best_params = self._convert_numpy_types(best_params)
            best_score = self._convert_numpy_types(best_score)
            total_evaluations = self._convert_numpy_types(total_evaluations)
            optimization_time = self._convert_numpy_types(optimization_time)
            convergence_info = self._convert_numpy_types(convergence_info)
            optimization_history = self._convert_numpy_types(optimization_history)

            # モデルタイプを正規化
            model_type = (
                self._normalize_model_type(model_type) if model_type else model_type
            )
            target_model_type = (
                self._normalize_model_type(target_model_type)
                if target_model_type
                else self._normalize_model_type(model_type)
            )

            # 同じプロファイル名が存在する場合は更新
            existing_result = self.get_by_profile_name(profile_name)
            if existing_result:
                return self.update_optimization_result(
                    existing_result.id,
                    best_params=best_params,
                    best_score=best_score,
                    total_evaluations=total_evaluations,
                    optimization_time=optimization_time,
                    convergence_info=convergence_info,
                    optimization_history=optimization_history,
                    model_type=model_type,
                    experiment_name=experiment_name,
                    description=description,
                    is_active=is_active,
                    is_default=is_default,
                    target_model_type=target_model_type,
                )

            result = BayesianOptimizationResult(
                profile_name=profile_name,
                optimization_type=optimization_type,
                model_type=model_type,
                experiment_name=experiment_name,
                best_params=best_params,
                best_score=best_score,
                total_evaluations=total_evaluations,
                optimization_time=optimization_time,
                convergence_info=convergence_info,
                optimization_history=optimization_history,
                is_active=is_active,
                is_default=is_default,
                target_model_type=target_model_type,
                description=description,
            )

            self.db.add(result)
            self.db.commit()
            self.db.refresh(result)

            logger.info(f"ベイジアン最適化結果を作成: {profile_name}")
            return result

        except Exception as e:
            self.db.rollback()
            logger.error(f"ベイジアン最適化結果作成エラー: {e}")
            raise

    def update_optimization_result(
        self, result_id: int, **kwargs
    ) -> Optional[BayesianOptimizationResult]:
        """
        ベイジアン最適化結果を更新

        Args:
            result_id: 結果ID
            **kwargs: 更新するフィールド

        Returns:
            更新されたベイジアン最適化結果
        """
        try:
            result = (
                self.db.query(BayesianOptimizationResult)
                .filter(BayesianOptimizationResult.id == result_id)
                .first()
            )

            if not result:
                logger.warning(f"ベイジアン最適化結果が見つかりません: ID={result_id}")
                return None

            # 更新可能なフィールドのみ更新
            updatable_fields = [
                "best_params",
                "best_score",
                "total_evaluations",
                "optimization_time",
                "convergence_info",
                "optimization_history",
                "model_type",
                "experiment_name",
                "description",
                "is_active",
                "is_default",
                "target_model_type",
            ]

            for field, value in kwargs.items():
                if field in updatable_fields and hasattr(result, field):
                    # NumPy型を変換
                    converted_value = self._convert_numpy_types(value)
                    setattr(result, field, converted_value)

            result.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(result)

            logger.info(f"ベイジアン最適化結果を更新: ID={result_id}")
            return result

        except Exception as e:
            self.db.rollback()
            logger.error(f"ベイジアン最適化結果更新エラー: {e}")
            raise

    def get_by_profile_name(
        self, profile_name: str
    ) -> Optional[BayesianOptimizationResult]:
        """
        プロファイル名でベイジアン最適化結果を取得

        Args:
            profile_name: プロファイル名

        Returns:
            ベイジアン最適化結果（見つからない場合はNone）
        """
        try:
            return (
                self.db.query(BayesianOptimizationResult)
                .filter(BayesianOptimizationResult.profile_name == profile_name)
                .first()
            )

        except Exception as e:
            logger.error(f"プロファイル名による検索エラー: {e}")
            return None

    def get_by_id(self, result_id: int) -> Optional[BayesianOptimizationResult]:
        """
        IDでベイジアン最適化結果を取得

        Args:
            result_id: 結果ID

        Returns:
            ベイジアン最適化結果（見つからない場合はNone）
        """
        try:
            return (
                self.db.query(BayesianOptimizationResult)
                .filter(BayesianOptimizationResult.id == result_id)
                .first()
            )

        except Exception as e:
            logger.error(f"ID による検索エラー: {e}")
            return None

    def get_active_results(
        self,
        optimization_type: Optional[str] = None,
        model_type: Optional[str] = None,
        target_model_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BayesianOptimizationResult]:
        """
        アクティブなベイジアン最適化結果を取得

        Args:
            optimization_type: 最適化タイプでフィルタ（オプション）
            model_type: モデルタイプでフィルタ（オプション）
            target_model_type: 対象モデルタイプでフィルタ（オプション）
            limit: 取得件数制限（オプション）

        Returns:
            ベイジアン最適化結果のリスト
        """
        try:
            query = self.db.query(BayesianOptimizationResult).filter(
                BayesianOptimizationResult.is_active == True
            )

            if optimization_type:
                query = query.filter(
                    BayesianOptimizationResult.optimization_type == optimization_type
                )

            if model_type:
                query = query.filter(
                    BayesianOptimizationResult.model_type == model_type
                )

            if target_model_type:
                query = query.filter(
                    BayesianOptimizationResult.target_model_type == target_model_type
                )

            query = query.order_by(desc(BayesianOptimizationResult.created_at))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"アクティブ結果取得エラー: {e}")
            return []

    def get_all_results(
        self, include_inactive: bool = False, limit: Optional[int] = None
    ) -> List[BayesianOptimizationResult]:
        """
        すべてのベイジアン最適化結果を取得

        Args:
            include_inactive: 非アクティブな結果も含めるか
            limit: 取得件数制限（オプション）

        Returns:
            ベイジアン最適化結果のリスト
        """
        try:
            query = self.db.query(BayesianOptimizationResult)

            if not include_inactive:
                query = query.filter(BayesianOptimizationResult.is_active == True)

            query = query.order_by(desc(BayesianOptimizationResult.created_at))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"全結果取得エラー: {e}")
            return []

    def delete_result(self, result_id: int) -> bool:
        """
        ベイジアン最適化結果を削除

        Args:
            result_id: 結果ID

        Returns:
            削除成功の場合True
        """
        try:
            result = (
                self.db.query(BayesianOptimizationResult)
                .filter(BayesianOptimizationResult.id == result_id)
                .first()
            )

            if not result:
                logger.warning(
                    f"削除対象のベイジアン最適化結果が見つかりません: ID={result_id}"
                )
                return False

            self.db.delete(result)
            self.db.commit()

            logger.info(f"ベイジアン最適化結果を削除: ID={result_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"ベイジアン最適化結果削除エラー: {e}")
            return False

    # プロファイル関連のメソッド
    def get_default_profile(
        self, target_model_type: str
    ) -> Optional[BayesianOptimizationResult]:
        """
        指定されたモデルタイプのデフォルトプロファイルを取得

        Args:
            target_model_type: 対象モデルタイプ

        Returns:
            デフォルトプロファイル（見つからない場合はNone）
        """
        try:
            # モデルタイプを正規化
            normalized_model_type = self._normalize_model_type(target_model_type)

            return (
                self.db.query(BayesianOptimizationResult)
                .filter(
                    and_(
                        BayesianOptimizationResult.target_model_type
                        == normalized_model_type,
                        BayesianOptimizationResult.is_default == True,
                        BayesianOptimizationResult.is_active == True,
                    )
                )
                .first()
            )

        except Exception as e:
            logger.error(f"デフォルトプロファイル取得エラー: {e}")
            return None

    def set_default_profile(self, profile_id: int, target_model_type: str) -> bool:
        """
        指定されたプロファイルをデフォルトに設定

        Args:
            profile_id: プロファイルID
            target_model_type: 対象モデルタイプ

        Returns:
            設定成功の場合True
        """
        try:
            # 既存のデフォルトプロファイルを無効化
            self.db.query(BayesianOptimizationResult).filter(
                and_(
                    BayesianOptimizationResult.target_model_type == target_model_type,
                    BayesianOptimizationResult.is_default == True,
                )
            ).update({"is_default": False})

            # 新しいデフォルトプロファイルを設定
            result = (
                self.db.query(BayesianOptimizationResult)
                .filter(BayesianOptimizationResult.id == profile_id)
                .first()
            )

            if not result:
                logger.warning(f"プロファイルが見つかりません: ID={profile_id}")
                return False

            result.is_default = True
            result.target_model_type = target_model_type
            self.db.commit()

            logger.info(
                f"デフォルトプロファイルを設定: ID={profile_id}, モデルタイプ={target_model_type}"
            )
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"デフォルトプロファイル設定エラー: {e}")
            return False

    def get_profiles_by_model_type(
        self,
        target_model_type: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
    ) -> List[BayesianOptimizationResult]:
        """
        指定されたモデルタイプのプロファイルを取得

        Args:
            target_model_type: 対象モデルタイプ
            include_inactive: 非アクティブなプロファイルも含めるか
            limit: 取得件数制限（オプション）

        Returns:
            プロファイルのリスト
        """
        try:
            # モデルタイプを正規化
            normalized_model_type = self._normalize_model_type(target_model_type)

            query = self.db.query(BayesianOptimizationResult).filter(
                BayesianOptimizationResult.target_model_type == normalized_model_type
            )

            if not include_inactive:
                query = query.filter(BayesianOptimizationResult.is_active == True)

            query = query.order_by(
                desc(BayesianOptimizationResult.is_default),
                desc(BayesianOptimizationResult.created_at),
            )

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"モデルタイプ別プロファイル取得エラー: {e}")
            return []

    def deactivate_result(self, result_id: int) -> bool:
        """
        ベイジアン最適化結果を非アクティブ化

        Args:
            result_id: 結果ID

        Returns:
            非アクティブ化成功の場合True
        """
        try:
            result = self.update_optimization_result(result_id, is_active=False)
            return result is not None

        except Exception as e:
            logger.error(f"ベイジアン最適化結果非アクティブ化エラー: {e}")
            return False
