"""
Optunaベースの最適化エンジン

既存の複雑な最適化システムを置き換える、シンプルで効率的な実装。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import optuna

from app.utils.error_handler import safe_operation

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """最適化結果"""

    best_params: Dict[str, Any]
    best_score: float
    total_evaluations: int
    optimization_time: float
    study: optuna.Study


@dataclass
class ParameterSpace:
    """パラメータ空間の定義"""

    type: str  # "real", "integer", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[list] = None


class OptunaOptimizer:
    """
    Optunaベースの最適化エンジン

    既存の複雑なシステムを置き換える、シンプルで効率的な実装。
    """

    def __init__(self):
        """初期化"""
        self.study: Optional[optuna.Study] = None

    @safe_operation(context="Optuna最適化", is_api_call=False)
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
    ) -> OptimizationResult:
        """
        Optunaを使用した最適化を実行

        Args:
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 最適化試行回数

        Returns:
            最適化結果
        """
        logger.info(f"🚀 Optuna最適化を開始: 試行回数={n_calls}")
        start_time = datetime.now()

        # Optunaスタディを作成
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # 目的関数をOptunaに適応
        def optuna_objective(trial: optuna.Trial) -> float:
            params = self._suggest_parameters(trial, parameter_space)
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.warning(f"目的関数評価中にエラー: {e}")
                raise optuna.TrialPruned()

        # 最適化実行
        self.study.optimize(optuna_objective, n_trials=n_calls)

        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()

        # 結果を作成
        if self.study is None or len(self.study.trials) == 0:
            raise RuntimeError("最適化が実行されていないか、試行がありません")

        best_trial = self.study.best_trial
        if best_trial is None:
            raise RuntimeError("最適な試行が見つかりません")

        best_score = best_trial.value if best_trial.value is not None else 0.0
        if best_trial.value is None:
            logger.warning("ベスト試行の値がNoneです。デフォルト値0.0を使用します")

        result = OptimizationResult(
            best_params=best_trial.params,
            best_score=best_score,
            total_evaluations=len(self.study.trials),
            optimization_time=optimization_time,
            study=self.study,
        )

        logger.info(
            f"✅ Optuna最適化完了: ベストスコア={result.best_score:.4f}, 時間={optimization_time:.2f}秒"
        )
        return result

    @safe_operation(context="Optunaリソースクリーンアップ", is_api_call=False)
    def cleanup(self):
        """
        Optunaリソースのクリーンアップ
        メモリーリーク防止のため、最適化完了後に呼び出す
        """
        if self.study is not None:
            # Studyの内部データをクリア
            if hasattr(self.study, "trials"):
                self.study.trials.clear()

            # Studyオブジェクト自体をクリア
            self.study = None

            # 強制ガベージコレクション
            import gc

            gc.collect()

    def __del__(self):
        """デストラクタでクリーンアップを確実に実行"""
        try:
            self.cleanup()
        except Exception:
            pass  # デストラクタでは例外を発生させない

    def _suggest_parameters(
        self, trial: optuna.Trial, parameter_space: Dict[str, ParameterSpace]
    ) -> Dict[str, Any]:
        """パラメータをサジェスト"""
        params = {}

        for param_name, param_config in parameter_space.items():
            if param_config.type == "real":
                assert param_config.low is not None and param_config.high is not None
                params[param_name] = trial.suggest_float(
                    param_name, param_config.low, param_config.high
                )
            elif param_config.type == "integer":
                assert param_config.low is not None and param_config.high is not None
                params[param_name] = trial.suggest_int(
                    param_name, int(param_config.low), int(param_config.high)
                )
            elif param_config.type == "categorical":
                assert param_config.categories is not None
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config.categories
                )

        return params

    @staticmethod
    def get_default_parameter_space() -> Dict[str, ParameterSpace]:
        """LightGBMのデフォルトパラメータ空間（後方互換性のため）"""
        return {
            "num_leaves": ParameterSpace(type="integer", low=10, high=100),
            "learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "bagging_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "min_data_in_leaf": ParameterSpace(type="integer", low=5, high=50),
            "max_depth": ParameterSpace(type="integer", low=3, high=15),
        }

    @staticmethod
    @safe_operation(context="アンサンブルパラメータ空間取得", is_api_call=False)
    def get_ensemble_parameter_space(
        ensemble_method: str, enabled_models: list
    ) -> Dict[str, ParameterSpace]:
        """
        アンサンブル学習用のパラメータ空間を取得

        Args:
            ensemble_method: アンサンブル手法 ("stacking")
            enabled_models: 有効なベースモデルのリスト

        Returns:
            アンサンブル用パラメータ空間
        """
        from .ensemble_parameter_space import EnsembleParameterSpace

        return EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method, enabled_models
        )
