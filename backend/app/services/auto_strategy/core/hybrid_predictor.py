"""Hybrid GA+ML predictor implementation."""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from app.services.ml.exceptions import MLPredictionError
from .drl_policy_adapter import DRLPolicyAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from app.services.ml.ml_training_service import MLTrainingService
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
        automl_config: Optional[Dict[str, Any]] = None,
        use_time_series_cv: bool = False,
        training_service_cls: Optional[Type["MLTrainingService"]] = None,
        model_manager_instance: Optional["ModelManager"] = None,
        drl_policy_adapter: Optional[DRLPolicyAdapter] = None,
    ):
        """初期化"""

        self.trainer_type = trainer_type
        self.model_types = model_types
        self.single_model_config = single_model_config or {}
        self.ensemble_config = ensemble_config
        self.automl_config = automl_config
        self.use_time_series_cv = use_time_series_cv
        self.drl_config = (automl_config or {}).get("drl") if automl_config else None
        self._drl_enabled = bool(
            self.drl_config and self.drl_config.get("enabled", False)
        )
        self._drl_weight = (
            float(self.drl_config.get("policy_weight", 0.5)) if self.drl_config else 0.0
        )
        if not 0.0 <= self._drl_weight <= 1.0:
            logger.warning("DRLポリシーの重みが範囲外のため0.5に調整しました")
            self._drl_weight = 0.5

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
                    automl_config=automl_config,
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
                automl_config=automl_config,
            )
            self.services = [service]
            self.service = service

        self.drl_policy_adapter = self._resolve_drl_adapter(drl_policy_adapter)

    @staticmethod
    def _default_prediction() -> Dict[str, float]:
        return {"up": 0.33, "down": 0.33, "range": 0.34}

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
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率辞書 {"up": float, "down": float, "range": float}

        Raises:
            MLPredictionError: 予測に失敗した場合
        """
        try:
            # 入力検証
            if features_df is None or features_df.empty:
                raise MLPredictionError("特徴量DataFrameが空です")

            # 複数モデルの場合は平均化
            if len(self.services) > 1:
                predictions = []
                for service in self.services:
                    try:
                        self._run_time_series_cv(service, features_df)
                        pred = service.generate_signals(features_df)
                        predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"モデル予測エラー: {e}")
                        continue

                if not predictions:
                    logger.warning("全てのモデルで予測失敗、デフォルト値を返します")
                    ml_prediction = self._default_prediction()
                else:
                    ml_prediction = {
                        "up": float(np.mean([p.get("up", 0.0) for p in predictions])),
                        "down": float(
                            np.mean([p.get("down", 0.0) for p in predictions])
                        ),
                        "range": float(
                            np.mean([p.get("range", 0.0) for p in predictions])
                        ),
                    }
            else:
                # 単一モデルの場合
                service = self.services[0]

                # モデルが学習されているかチェック
                if not self._service_is_trained(service):
                    logger.warning("モデル未学習、デフォルト値を返します")
                    ml_prediction = self._default_prediction()
                else:
                    # 予測実行
                    self._run_time_series_cv(service, features_df)
                    ml_prediction = service.generate_signals(features_df)

            blended = self._blend_with_drl(ml_prediction, features_df)
            return blended

        except MLPredictionError:
            raise
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise MLPredictionError(f"予測に失敗しました: {e}")

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
        """必要に応じて時系列クロスバリデーションを実行"""

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

        module = importlib.import_module("app.services.ml.ml_training_service")
        return getattr(module, "MLTrainingService")

    @staticmethod
    def _resolve_model_manager(override: Optional["ModelManager"]) -> "ModelManager":
        if override is not None:
            return override

        module = importlib.import_module("app.services.ml.model_manager")
        manager_cls = getattr(module, "ModelManager")
        return manager_cls()

    @staticmethod
    def _service_is_trained(service: "MLTrainingService") -> bool:
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

    def _resolve_drl_adapter(
        self, override: Optional[DRLPolicyAdapter]
    ) -> Optional[DRLPolicyAdapter]:
        if override is not None:
            return override

        if not self._drl_enabled:
            return None

        try:
            policy_type = (
                self.drl_config.get("policy_type", "ppo") if self.drl_config else "ppo"
            )
            return DRLPolicyAdapter(
                policy_type=policy_type, policy_config=self.drl_config
            )
        except Exception as exc:
            logger.warning("DRLPolicyAdapterの初期化に失敗しました: %s", exc)
            return None

    def _blend_with_drl(
        self, ml_prediction: Dict[str, float], features_df: pd.DataFrame
    ) -> Dict[str, float]:
        if not self._drl_enabled or self.drl_policy_adapter is None:
            return self._normalise_prediction(ml_prediction)

        try:
            drl_prediction = self.drl_policy_adapter.predict_signals(features_df)
        except Exception as exc:
            logger.warning(
                "DRLポリシー予測が失敗したためML予測のみを使用します: %s", exc
            )
            return self._normalise_prediction(ml_prediction)

        weight = self._drl_weight
        weight = min(max(weight, 0.0), 1.0)
        ml_weight = 1.0 - weight

        blended = {
            key: weight * float(drl_prediction.get(key, 0.0))
            + ml_weight * float(ml_prediction.get(key, 0.0))
            for key in {"up", "down", "range"}
        }

        return self._normalise_prediction(blended)

    @staticmethod
    def _normalise_prediction(prediction: Dict[str, float]) -> Dict[str, float]:
        up = float(prediction.get("up", 0.0))
        down = float(prediction.get("down", 0.0))
        range_score = float(prediction.get("range", 0.0))
        total = up + down + range_score
        if total <= 0 or not np.isfinite(total):
            return {"up": 1 / 3, "down": 1 / 3, "range": 1 / 3}
        return {
            "up": up / total,
            "down": down / total,
            "range": range_score / total,
        }
