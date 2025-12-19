"""ハイブリッドGA+ML予測器の実装。"""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from app.services.ml.common.exceptions import MLPredictionError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from app.services.ml.orchestration.ml_training_service import MLTrainingService
    from app.services.ml.model_manager import ModelManager


class HybridPredictor:
    """
    ハイブリッドGA+ML予測器

    MLTrainingServiceを使用してGA個体評価用の予測を提供します。
    単一モデルまたは複数モデルのアンサンブル予測をサポートします。
    """

    def __init__(
        self,
        trainer_type: str = "single",
        model_type: Optional[str] = None,
        model_types: Optional[List[str]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
        use_time_series_cv: bool = False,
        training_service_cls: Optional[Type["MLTrainingService"]] = None,
        model_manager_instance: Optional["ModelManager"] = None,
    ):
        """初期化"""

        self.trainer_type = trainer_type
        self.model_types = model_types
        self.single_model_config = single_model_config or {}
        self.ensemble_config = ensemble_config
        self.use_time_series_cv = use_time_series_cv

        if model_type:
            self.model_type = model_type
        elif self.single_model_config.get("model_type"):
            self.model_type = self.single_model_config["model_type"]
        else:
            self.model_type = "lightgbm"

        if trainer_type == "single":
            self.single_model_config["model_type"] = self.model_type

        self.training_service_cls = self._resolve_training_service_cls(
            training_service_cls
        )
        self.model_manager = self._resolve_model_manager(model_manager_instance)

        if model_types and len(model_types) > 1:
            self.services: List["MLTrainingService"] = []
            for mt in model_types:
                config = self.single_model_config.copy()
                config["model_type"] = mt
                service = self.training_service_cls(
                    trainer_type="single",
                    single_model_config=config,
                )
                self.services.append(service)
            self.service = self.services[0]
        else:
            service = self.training_service_cls(
                trainer_type=trainer_type,
                single_model_config=(
                    self.single_model_config if trainer_type == "single" else None
                ),
                ensemble_config=ensemble_config,
            )
            self.services = [service]
            self.service = service

    @staticmethod
    def _default_prediction() -> Dict[str, float]:
        """
        デフォルトの予測値を返す（二値分類 / ダマシ予測専用）

        Returns:
            デフォルトの予測確率辞書
        """
        # is_valid = 0.5 は「判断不能」を意味する
        return {"is_valid": 0.5}

    @staticmethod
    def get_available_models() -> List[str]:
        """
        利用可能なモデル一覧を取得

        Returns:
            モデルタイプのリスト
        """
        service_cls = HybridPredictor._resolve_training_service_cls(None)
        return service_cls.get_available_single_models()

    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """予測を実行（統合版）"""
        try:
            if features_df is None or features_df.empty:
                raise MLPredictionError("特徴量DataFrameが空です")

            # 全サービスの予測結果を収集
            preds = []
            for s in self.services:
                if self._service_is_trained(s):
                    self._run_time_series_cv(s, features_df)
                    preds.append(s.generate_signals(features_df))

            if not preds:
                logger.warning("予測可能なモデルがありません")
                return self._default_prediction()

            # 複数モデルの結果を平均化
            if len(preds) == 1:
                ml_prediction = preds[0]
            else:
                keys = preds[0].keys()
                ml_prediction = {
                    k: float(np.mean([p.get(k, 0.0) for p in preds])) for k in keys
                }

            return self._normalise_prediction(ml_prediction)

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise MLPredictionError(f"予測失敗: {e}")

    def predict_batch(
        self, features_batch: List[pd.DataFrame]
    ) -> List[Dict[str, float]]:
        """
        バッチ予測

        Args:
            features_batch: 特徴量DataFrameのリスト

        Returns:
            予測結果のリスト
        """
        results = []
        for features_df in features_batch:
            try:
                result = self.predict(features_df)
                results.append(result)
            except Exception as e:
                logger.error(f"バッチ予測エラー: {e}")
                results.append(self._default_prediction())

        return results

    def _run_time_series_cv(
        self, service: "MLTrainingService", features_df: pd.DataFrame
    ) -> None:
        """
        必要に応じて時系列クロスバリデーション（Walk-Forward予測）を実行します。

        特徴量データに基づいてモデルの学習状態を最新、または特定の時点に
        合わせるための処理を各サービスで呼び出します。

        Args:
            service: MLトレーニングサービスインスタンス
            features_df: 特徴量DataFrame
        """

        if not self.use_time_series_cv:
            return

        try:
            if hasattr(service, "run_time_series_cv"):
                service.run_time_series_cv(features_df)
            elif hasattr(service, "time_series_cross_validate"):
                service.time_series_cross_validate(features_df)
            elif hasattr(service, "trainer") and hasattr(
                service.trainer, "time_series_cross_validate"
            ):
                service.trainer.time_series_cross_validate(features_df)
        except Exception as exc:
            logger.warning(f"時系列クロスバリデーション実行エラー: {exc}")

    def load_model(self, model_path: str) -> bool:
        """
        モデルをロード

        Args:
            model_path: モデルファイルパス

        Returns:
            ロード成功フラグ
        """
        try:
            for service in self.services:
                success = service.load_model(model_path)
                if not success:
                    logger.warning(f"モデルロード失敗: {model_path}")
                    return False
            return True
        except Exception as e:
            logger.error(f"モデルロードエラー: {e}")
            return False

    def get_latest_model(self) -> Optional[str]:
        """
        最新モデルのパスを取得

        Returns:
            モデルパス（存在しない場合はNone）
        """
        try:
            # model_typeに基づいて最新モデルを取得
            model_path = self.model_manager.get_latest_model()
            return model_path
        except Exception as e:
            logger.error(f"最新モデル取得エラー: {e}")
            return None

    def is_trained(self) -> bool:
        """
        モデルが学習済みかチェック

        Returns:
            学習済みフラグ
        """
        for service in self.services:
            if not self._service_is_trained(service):
                return False
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報辞書
        """
        if len(self.services) == 1:
            return self.services[0].get_training_status()
        else:
            # 複数モデルの場合は各モデルの情報をリストで返す
            return {
                "trainer_type": "multi_model",
                "model_count": len(self.services),
                "models": [s.get_training_status() for s in self.services],
            }

    @staticmethod
    def _resolve_training_service_cls(
        override: Optional[Type["MLTrainingService"]],
    ) -> Type["MLTrainingService"]:
        if override is not None:
            return override

        module = importlib.import_module("app.services.ml.orchestration.ml_training_service")
        return getattr(module, "MLTrainingService")

    @staticmethod
    def _resolve_model_manager(override: Optional["ModelManager"]) -> "ModelManager":
        if override is not None:
            return override

        module = importlib.import_module("app.services.ml.models.model_manager")
        manager_cls = getattr(module, "ModelManager")
        return manager_cls()

    @staticmethod
    def _service_is_trained(service: "MLTrainingService") -> bool:
        """
        サービス配下のトレーナーが学習済み（モデル保持状態）か判定します。

        Args:
            service: MLトレーニングサービス

        Returns:
            学習済みであればTrue
        """
        trainer = getattr(service, "trainer", None)
        if trainer is None:
            return True

        is_trained_attr = getattr(trainer, "is_trained", None)
        if is_trained_attr is False:
            return False

        model_attr = getattr(trainer, "model", object())
        if model_attr is None and is_trained_attr is not None:
            return False

        return True

    @staticmethod
    def _normalise_prediction(prediction: Dict[str, float]) -> Dict[str, float]:
        """
        予測結果を正規化（二値分類 / ダマシ予測専用）

        Args:
            prediction: 生の予測結果

        Returns:
            正規化された予測結果
        """
        # is_validが含まれている場合
        if "is_valid" in prediction:
            is_valid = float(prediction.get("is_valid", 0.5))
            # 0-1の範囲に制限
            is_valid = max(0.0, min(1.0, is_valid))
            if not np.isfinite(is_valid):
                is_valid = 0.5
            return {"is_valid": is_valid}

        # 未知のフォーマットの場合はデフォルトを返す
        logger.warning(f"未知の予測フォーマット: {prediction.keys()}")
        return {"is_valid": 0.5}
