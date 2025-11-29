import logging
from typing import Any, Callable, Dict, Optional, List
import pandas as pd

from ...utils.error_handler import safe_operation
from .optuna_optimizer import OptunaOptimizer, ParameterSpace
from ..ml.ensemble.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class OptimizationSettings:
    """最適化設定クラス"""

    def __init__(
        self,
        enabled: bool = False,
        n_calls: int = 50,
        parameter_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enabled = enabled
        self.n_calls = n_calls
        self.parameter_space = parameter_space or {}


class OptimizationService:
    """
    最適化サービス（統一トレーナー対応）

    MLモデルのハイパーパラメータ最適化を管理します。
    SingleModelTrainerは廃止し、全てEnsembleTrainerで統一。
    """

    def __init__(self):
        self.optimizer = OptunaOptimizer()

    @safe_operation(context="パラメータ最適化", is_api_call=False)
    def optimize_parameters(
        self,
        trainer: Any,
        training_data: pd.DataFrame,
        optimization_settings: OptimizationSettings,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        パラメータ最適化を実行
        """
        logger.info("🚀 最適化プロセスを開始")

        # パラメータ空間を準備
        parameter_space = self._prepare_parameter_space(trainer, optimization_settings)

        # 目的関数を作成
        objective_function = self._create_objective_function(
            trainer=trainer,
            training_data=training_data,
            optimization_settings=optimization_settings,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            **training_params,
        )

        # 最適化を実行
        result = self.optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=optimization_settings.n_calls,
        )

        return {
            "method": "optuna",
            "best_params": result.best_params,
            "best_score": result.best_score,
            "total_evaluations": result.total_evaluations,
            "optimization_time": result.optimization_time,
        }

    def _prepare_parameter_space(
        self, trainer: Any, optimization_settings: OptimizationSettings
    ) -> Dict[str, ParameterSpace]:
        """パラメータ空間を準備"""
        if not optimization_settings.parameter_space:
            # EnsembleTrainerの場合（単一モデルも含む）
            if hasattr(trainer, "ensemble_config"):
                ensemble_method = trainer.ensemble_config.get("method", "stacking")
                enabled_models = trainer.ensemble_config.get(
                    "models", ["lightgbm", "xgboost"]
                )
                return self.optimizer.get_ensemble_parameter_space(
                    ensemble_method, enabled_models
                )
            else:
                # フォールバック: デフォルトのLightGBMパラメータ空間
                return self.optimizer.get_default_parameter_space()
        else:
            return self._convert_parameter_space_config(
                optimization_settings.parameter_space
            )

    def _convert_parameter_space_config(
        self, parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """設定辞書をParameterSpaceオブジェクトに変換"""
        parameter_space = {}
        for param_name, param_config in parameter_space_config.items():
            param_type = param_config["type"]
            low = param_config.get("low")
            high = param_config.get("high")

            if param_type == "integer" and low is not None and high is not None:
                low = int(low)
                high = int(high)

            parameter_space[param_name] = ParameterSpace(
                type=param_type,
                low=low,
                high=high,
                categories=param_config.get("categories"),
            )
        return parameter_space

    def _create_objective_function(
        self,
        trainer: Any,
        training_data: pd.DataFrame,
        optimization_settings: OptimizationSettings,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        **base_training_params,
    ) -> Callable[[Dict[str, Any]], float]:
        """目的関数を作成"""
        evaluation_count = 0

        def objective_function(params: Dict[str, Any]) -> float:
            nonlocal evaluation_count
            evaluation_count += 1

            try:
                logger.info(
                    f"🔍 試行 {evaluation_count}/{optimization_settings.n_calls}: {params}"
                )

                # パラメータのマージ
                training_params = {**base_training_params, **params}

                # 一時的なトレーナーを作成
                temp_trainer = self._create_temp_trainer(trainer, params)

                # 学習実行（保存なし）
                result = temp_trainer.train_model(
                    training_data=training_data,
                    funding_rate_data=funding_rate_data,
                    open_interest_data=open_interest_data,
                    save_model=False,
                    model_name=None,
                    **training_params,
                )

                # スコア取得
                f1_score = result.get("f1_score", 0.0)
                if "classification_report" in result:
                    f1_score = (
                        result["classification_report"]
                        .get("macro avg", {})
                        .get("f1-score", f1_score)
                    )

                return f1_score

            except Exception as e:
                logger.warning(f"目的関数評価エラー: {e}")
                return 0.0

        return objective_function

    def _create_temp_trainer(
        self, original_trainer: Any, params: Dict[str, Any]
    ) -> Any:
        """一時的なトレーナーを作成（全てEnsembleTrainerで統一）"""
        # オリジナルのトレーナーがEnsembleTrainerであることを前提
        if hasattr(original_trainer, "ensemble_config"):
            temp_config = original_trainer.ensemble_config.copy()

            # 最適化用にCV foldsを減らす（速度向上）
            if "stacking_params" in temp_config:
                stacking_params = temp_config["stacking_params"].copy()
                stacking_params["cv_folds"] = 3
                temp_config["stacking_params"] = stacking_params

            return EnsembleTrainer(ensemble_config=temp_config)
        else:
            # フォールバック: デフォルト設定でEnsembleTrainer作成
            return EnsembleTrainer(ensemble_config={"method": "stacking"})

    def cleanup(self):
        """リソースクリーンアップ"""
        self.optimizer.cleanup()
