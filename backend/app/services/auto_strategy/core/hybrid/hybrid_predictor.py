"""ハイブリッドGA+ML予測器の実装。"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from app.services.ml.common.exceptions import MLPredictionError
from app.services.ml.common.utils import prepare_data_for_prediction
from app.services.ml.models.model_manager import ModelManager
from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingService,
)

logger = logging.getLogger(__name__)


class RuntimeModelPredictorAdapter:
    """
    保存済みモデルアセットを runtime predictor 契約へ変換する薄いアダプタ。

    `model_manager.load_model()` の戻り値辞書、または直接モデルオブジェクトを受け取り、
    `predict(DataFrame) -> {"forecast_log_rv", "forecast_vol", "gate_open"}` /
    `is_trained() -> bool` を提供します。
    """

    def __init__(self, model_data: Any):
        self._loaded_from_model_artifacts = isinstance(model_data, dict)
        if self._loaded_from_model_artifacts:
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            self.feature_columns = model_data.get("feature_columns")
            self.metadata = model_data.get("metadata", {})
        else:
            self.model = model_data
            self.scaler = None
            self.feature_columns = None
            self.metadata = {}

    def is_trained(self) -> bool:
        """モデルが推論可能状態かを判定する。"""
        if self.model is None:
            return False
        if self._loaded_from_model_artifacts:
            if self.metadata.get("task_type") != "volatility_regression":
                return False
            if self.metadata.get("target_kind") != "log_realized_vol":
                return False

        is_trained_attr = getattr(self.model, "is_trained", None)
        if isinstance(is_trained_attr, bool):
            return is_trained_attr
        if callable(is_trained_attr):
            try:
                result = is_trained_attr()
                if isinstance(result, bool):
                    return result
            except Exception:
                logger.debug("runtime predictor の is_trained 呼び出しに失敗しました")

        is_fitted_attr = getattr(self.model, "is_fitted", None)
        if isinstance(is_fitted_attr, bool):
            return is_fitted_attr

        return True

    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """保存済みモデルを使ってボラティリティ予測を返す。"""
        try:
            if features_df is None or features_df.empty or not self.is_trained():
                return HybridPredictor._default_prediction()

            prepared_features = self._prepare_features(features_df)
            raw_prediction = self._run_model(prepared_features)
            normalised = self._normalise_runtime_prediction(raw_prediction)
            gate_cutoff_log_rv = float(self.metadata.get("gate_cutoff_log_rv", 0.0))
            normalised["gate_open"] = bool(
                normalised.get("forecast_log_rv", 0.0) >= gate_cutoff_log_rv
            )
            return normalised
        except Exception as exc:
            logger.warning(f"runtime predictor 予測エラー: {exc}")
            return HybridPredictor._default_prediction()

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """保存時の feature_columns / scaler を使って推論入力を揃える。"""
        if self.feature_columns:
            return prepare_data_for_prediction(
                features_df,
                expected_columns=list(self.feature_columns),
                scaler=self.scaler,
            )

        prepared = features_df.copy()
        return prepared.ffill().fillna(0.0)

    def _run_model(self, features_df: pd.DataFrame) -> Any:
        """モデル固有の予測関数を呼び出す。"""
        predict_volatility = getattr(self.model, "predict_volatility", None)
        if callable(predict_volatility):
            return predict_volatility(features_df)

        if self.metadata.get("task_type") == "volatility_regression":
            predict = getattr(self.model, "predict", None)
            if callable(predict):
                return predict(features_df)

        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            return predict_proba(features_df)

        predict = getattr(self.model, "predict", None)
        if callable(predict):
            return predict(features_df)

        raise MLPredictionError(
            "runtime predictor が predict/predict_proba を持っていません"
        )

    @staticmethod
    def _normalise_runtime_prediction(raw_prediction: Any) -> Dict[str, float]:
        """モデル出力を volatility gate 形式に正規化する。"""
        if isinstance(raw_prediction, dict):
            return HybridPredictor._normalise_prediction(raw_prediction)

        predictions = np.asarray(raw_prediction)
        if predictions.ndim == 0:
            latest_pred = predictions.item()
        elif predictions.ndim == 1:
            latest_pred = predictions[-1]
        else:
            latest_pred = predictions[-1]

        latest_array = np.asarray(latest_pred)
        if latest_array.ndim == 0:
            forecast_log_rv = float(latest_array)
        else:
            forecast_log_rv = float(latest_array.reshape(-1)[-1])

        return HybridPredictor._normalise_prediction(
            {"forecast_log_rv": forecast_log_rv}
        )


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
        デフォルトの予測値を返す（volatility gate 専用）

        Returns:
            デフォルトの予測確率辞書
        """
        return {
            "forecast_log_rv": 0.0,
            "forecast_vol": 1.0,
            "gate_open": True,
        }

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
                    generate_forecast = getattr(s, "generate_forecast", None)
                    if callable(generate_forecast):
                        preds.append(generate_forecast(features_df))
                    else:
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
                service.run_time_series_cv(features_df)  # type: ignore[reportAttributeAccessIssue]
            elif hasattr(service, "time_series_cross_validate"):
                service.time_series_cross_validate(features_df)  # type: ignore[reportAttributeAccessIssue]
            elif hasattr(service, "trainer") and hasattr(
                service.trainer, "time_series_cross_validate"
            ):
                service.trainer.time_series_cross_validate(features_df)  # type: ignore[reportAttributeAccessIssue]
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

    def get_latest_model(
        self, model_name_pattern: Optional[str] = None
    ) -> Optional[str]:
        """
        最新モデルのパスを取得

        Returns:
            モデルパス（存在しない場合はNone）
        """
        try:
            pattern = model_name_pattern or self.model_type or "*"
            model_path = self.model_manager.get_latest_model(
                pattern,
                metadata_filters={
                    "task_type": "volatility_regression",
                    "target_kind": "log_realized_vol",
                },
            )
            return model_path
        except Exception as e:
            logger.error(f"最新モデル取得エラー: {e}")
            return None

    def load_latest_models(self) -> bool:
        """
        利用可能な最新モデルを各サービスへロードする。

        Returns:
            少なくとも1つのモデルをロードできた場合はTrue
        """
        loaded_any = False

        if self.model_types and len(self.model_types) > 1:
            for model_type, service in zip(self.model_types, self.services):
                latest_model = self.get_latest_model(model_type)
                if latest_model is None:
                    logger.info(
                        f"最新モデルが見つからないためスキップします: {model_type}"
                    )
                    continue
                if service.load_model(latest_model):
                    loaded_any = True
                else:
                    logger.warning(f"最新モデルのロードに失敗しました: {latest_model}")
            return loaded_any

        latest_model = self.get_latest_model(self.model_type)
        if latest_model is None:
            logger.info(f"最新モデルが見つかりませんでした: {self.model_type}")
            return False

        return self.load_model(latest_model)

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

    async def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報辞書
        """
        if len(self.services) == 1:
            return await self.services[0].get_training_status()
        else:
            # 複数モデルの場合は各モデルの情報をリストで返す
            models = []
            for s in self.services:
                models.append(await s.get_training_status())
            return {
                "trainer_type": "multi_model",
                "model_count": len(self.services),
                "models": models,
            }

    @staticmethod
    def _resolve_training_service_cls(
        override: Optional[Type["MLTrainingService"]],
    ) -> Type["MLTrainingService"]:
        if override is not None:
            return override

        module = importlib.import_module(
            "app.services.ml.orchestration.ml_training_orchestration_service"
        )
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
        予測結果を正規化（volatility gate 専用）

        Args:
            prediction: 生の予測結果

        Returns:
            正規化された予測結果
        """
        if "forecast_log_rv" in prediction:
            forecast_log_rv = float(prediction.get("forecast_log_rv", 0.0))
            if not np.isfinite(forecast_log_rv):
                forecast_log_rv = 0.0
            forecast_vol = prediction.get("forecast_vol")
            if forecast_vol is None:
                forecast_vol = float(np.exp(forecast_log_rv))
            forecast_vol = float(forecast_vol)
            if not np.isfinite(forecast_vol) or forecast_vol < 0.0:
                forecast_vol = float(np.exp(forecast_log_rv))
            gate_cutoff_log_rv = float(prediction.get("gate_cutoff_log_rv", 0.0))
            raw_gate_open = prediction.get("gate_open")
            if isinstance(raw_gate_open, bool):
                gate_open = raw_gate_open
            elif raw_gate_open is None:
                gate_open = forecast_log_rv >= gate_cutoff_log_rv
            else:
                try:
                    gate_open = float(raw_gate_open) >= 0.5
                except (TypeError, ValueError):
                    gate_open = forecast_log_rv >= gate_cutoff_log_rv
            return {
                "forecast_log_rv": forecast_log_rv,
                "forecast_vol": forecast_vol,
                "gate_open": gate_open,
            }

        if "is_valid" in prediction:
            is_valid = float(prediction.get("is_valid", 0.5))
            return {
                "forecast_log_rv": 0.0,
                "forecast_vol": 1.0,
                "gate_open": bool(is_valid >= 0.5),
            }

        # 未知のフォーマットの場合はデフォルトを返す
        logger.warning(f"未知の予測フォーマット: {prediction.keys()}")
        return HybridPredictor._default_prediction()
