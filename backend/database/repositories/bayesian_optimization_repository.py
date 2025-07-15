"""
ベイジアン最適化結果リポジトリ

ベイジアン最適化結果の永続化処理を管理します。
"""

from typing import List, Optional, Dict, Any, Union, cast
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
import numpy as np

from .base_repository import BaseRepository
from database.models import BayesianOptimizationResult

logger = logging.getLogger(__name__)

# NumPyの型を許容するための型エイリアス
Numeric = Union[int, float, np.integer, np.floating]
Primitive = Union[str, int, float, bool, None]
# 再帰的なJSONデータ型を定義
JsonData = Union[Dict[str, "JsonData"], List["JsonData"], Primitive]


class BayesianOptimizationRepository(BaseRepository[BayesianOptimizationResult]):
    """ベイジアン最適化結果のリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, BayesianOptimizationResult)

    def _convert_numpy_types(self, obj: Any) -> JsonData:
        """NumPy型をPythonの標準型に再帰的に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            # JSONのキーは常に文字列であるべき
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _normalize_model_type(self, model_type: Optional[str]) -> Optional[str]:
        """モデルタイプの大文字小文字を正規化"""
        if not model_type:
            return model_type

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
        best_score: Numeric,
        total_evaluations: Numeric,
        optimization_time: Numeric,
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
        """
        try:
            # NumPy型をPythonの標準型に変換
            clean_best_params = self._convert_numpy_types(best_params)
            # _convert_numpy_typesは適切な型を返すので、ここでキャストする
            converted_best_score = self._convert_numpy_types(best_score)
            clean_best_score = (
                float(converted_best_score)
                if isinstance(converted_best_score, (int, float))
                else 0.0
            )

            converted_total_evaluations = self._convert_numpy_types(total_evaluations)
            clean_total_evaluations = (
                int(converted_total_evaluations)
                if isinstance(converted_total_evaluations, (int, float))
                else 0
            )

            converted_optimization_time = self._convert_numpy_types(optimization_time)
            clean_optimization_time = (
                float(converted_optimization_time)
                if isinstance(converted_optimization_time, (int, float))
                else 0.0
            )

            clean_convergence_info = self._convert_numpy_types(convergence_info)
            clean_optimization_history = self._convert_numpy_types(optimization_history)

            # モデルタイプを正規化
            normalized_model_type = self._normalize_model_type(model_type)
            normalized_target_model_type = self._normalize_model_type(
                target_model_type
            ) or self._normalize_model_type(model_type)

            # 同じプロファイル名が存在する場合は更新
            existing_result = self.get_by_profile_name(profile_name)
            if existing_result:
                update_data = {
                    "best_params": clean_best_params,
                    "best_score": clean_best_score,
                    "total_evaluations": clean_total_evaluations,
                    "optimization_time": clean_optimization_time,
                    "convergence_info": clean_convergence_info,
                    "optimization_history": clean_optimization_history,
                    "model_type": normalized_model_type,
                    "experiment_name": experiment_name,
                    "description": description,
                    "is_active": is_active,
                    "is_default": is_default,
                    "target_model_type": normalized_target_model_type,
                }
                updated_result = self.update_optimization_result(
                    cast(int, existing_result.id), **update_data
                )
                if updated_result:
                    return updated_result
                # 更新に失敗した場合は、予期せぬエラーとしてログに残し、例外を発生させる
                logger.error(f"プロファイル更新に失敗: {profile_name}")
                raise Exception(f"Failed to update profile: {profile_name}")

            new_result = BayesianOptimizationResult(
                profile_name=profile_name,
                optimization_type=optimization_type,
                model_type=normalized_model_type,
                experiment_name=experiment_name,
                best_params=clean_best_params,
                best_score=clean_best_score,
                total_evaluations=clean_total_evaluations,
                optimization_time=clean_optimization_time,
                convergence_info=clean_convergence_info,
                optimization_history=clean_optimization_history,
                is_active=is_active,
                is_default=is_default,
                target_model_type=normalized_target_model_type,
                description=description,
            )

            self.db.add(new_result)
            self.db.commit()
            self.db.refresh(new_result)

            logger.info(f"ベイジアン最適化結果を作成: {profile_name}")
            return new_result

        except Exception as e:
            self.db.rollback()
            logger.error(f"ベイジアン最適化結果作成エラー: {e}", exc_info=True)
            raise

    def update_optimization_result(
        self, result_id: int, **kwargs: Any
    ) -> Optional[BayesianOptimizationResult]:
        """
        ベイジアン最適化結果を更新
        """
        try:
            result = self.get_by_id(result_id)
            if not result:
                logger.warning(
                    f"更新対象のベイジアン最適化結果が見つかりません: ID={result_id}"
                )
                return None

            updatable_fields = {
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
            }

            for field, value in kwargs.items():
                if field in updatable_fields:
                    converted_value = self._convert_numpy_types(value)
                    setattr(result, field, converted_value)

            # updated_atはDBのonupdate機能で自動更新されるため、明示的な更新は不要
            # result.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(result)

            logger.info(f"ベイジアン最適化結果を更新: ID={result_id}")
            return result

        except Exception as e:
            self.db.rollback()
            logger.error(f"ベイジアン最適化結果更新エラー: {e}", exc_info=True)
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
                self.db.query(self.model_class)
                .filter(self.model_class.profile_name == profile_name)
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
            return self.db.query(self.model_class).filter_by(id=result_id).first()

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
            query = self.db.query(self.model_class).filter_by(is_active=True)

            if optimization_type:
                query = query.filter_by(optimization_type=optimization_type)

            if model_type:
                query = query.filter_by(model_type=model_type)

            if target_model_type:
                query = query.filter_by(target_model_type=target_model_type)

            query = query.order_by(desc(self.model_class.created_at))

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
            query = self.db.query(self.model_class)

            if not include_inactive:
                query = query.filter_by(is_active=True)

            query = query.order_by(desc(self.model_class.created_at))

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
            result = self.get_by_id(result_id)

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
                self.db.query(self.model_class)
                .filter_by(
                    target_model_type=normalized_model_type,
                    is_default=True,
                    is_active=True,
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
            normalized_model_type = self._normalize_model_type(target_model_type)
            if not normalized_model_type:
                logger.warning(f"無効なモデルタイプです: {target_model_type}")
                return False

            # 既存のデフォルトプロファイルを無効化
            (
                self.db.query(self.model_class)
                .filter_by(target_model_type=normalized_model_type, is_default=True)
                .update({"is_default": False})
            )

            # 新しいデフォルトプロファイルを設定
            update_count = (
                self.db.query(self.model_class)
                .filter_by(id=profile_id)
                .update(
                    {
                        "is_default": True,
                        "target_model_type": normalized_model_type,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
            )

            if update_count == 0:
                logger.warning(f"プロファイルが見つかりません: ID={profile_id}")
                self.db.rollback()
                return False

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

            query = self.db.query(self.model_class).filter_by(
                target_model_type=normalized_model_type
            )

            if not include_inactive:
                query = query.filter_by(is_active=True)

            query = query.order_by(
                desc(self.model_class.is_default),
                desc(self.model_class.created_at),
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
