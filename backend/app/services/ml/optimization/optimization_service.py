"""
ML モデル最適化サービスモジュール

このモジュールは、機械学習モデル（特にアンサンブルモデル）のハイパーパラメータ最適化を実行するための
高レベルサービスを提供します。Optuna をバックエンドとして使用し、目的関数の定義、パラメータ空間の探索、
学習プロセスの実行を管理します。

主なクラス:
    - OptimizationSettings: 最適化の実行設定（試行回数、パラメータ空間など）を定義するデータクラス。
    - OptimizationService: 最適化プロセス全体を統括するサービスクラス。トレーナーと連携して最適なパラメータを探索します。
"""

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.utils.error_handler import safe_operation

from .optuna_optimizer import OptunaOptimizer, ParameterSpace

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
    ML モデルのハイパーパラメータ最適化を統括するサービス

    `OptunaOptimizer` をバックエンドとして使用し、アンサンブルモデルを構成する
    各ベースモデルやメタモデルの最適なパラメータセットを自動探索します。
    目的関数（Objective Function）の生成、探索空間の設定、
    CV（交差検証）回数の調整等を行い、指定された試行回数内で
    モデル性能（主にマクロ F1 スコア）を最大化します。
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
        アンサンブルモデルのハイパーパラメータ最適化を実行

        Optuna を用いて、指定された試行回数（n_calls）の中で
        最も高い F1 スコア（マクロ平均）を出すパラメータの組み合わせを探索します。
        内部で一時的なトレーナーを生成し、評価を行います。

        Args:
            trainer: ベースとなるトレーナーインスタンス
            training_data: 学習用データ
            optimization_settings: 最適化の設定（有効化、試行回数、探索空間）
            funding_rate_data: オプションの FR データ
            open_interest_data: オプションの OI データ
            model_name: 保存時のモデル名（最適化中は保存されません）
            **training_params: 追加の学習パラメータ

        Returns:
            最適パラメータ、ベストスコア、評価時間等を含む結果辞書
        """
        try:
            logger.info("🚀 最適化プロセスを開始")

            # パラメータ空間を準備
            parameter_space = self._prepare_parameter_space(
                trainer, optimization_settings
            )

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

        finally:
            # 確実にリソースを解放
            self.optimizer.cleanup()

    def _prepare_parameter_space(
        self, trainer: Any, optimization_settings: OptimizationSettings
    ) -> Dict[str, ParameterSpace]:
        """
        探索対象となるパラメータ空間を定義

        設定で探索空間が明示されている場合はそれを使用し、
        そうでない場合はトレーナーのアンサンブル設定に基づいたデフォルト空間を生成します。

        Args:
            trainer: 対象のトレーナー
            optimization_settings: 最適化設定

        Returns:
            パラメータ名をキー、探索範囲（ParameterSpace）を値とする辞書
        """
        if optimization_settings.parameter_space:
            return self._convert_parameter_space_config(
                optimization_settings.parameter_space
            )

        # EnsembleTrainerの場合（単一モデルも含む）
        if hasattr(trainer, "ensemble_config"):
            c = trainer.ensemble_config
            return self.optimizer.get_ensemble_parameter_space(
                c.get("method", "stacking"), c.get("models", ["lightgbm", "xgboost"])
            )

        return self.optimizer.get_default_parameter_space()

    def _convert_parameter_space_config(
        self, parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """設定辞書をParameterSpaceオブジェクトに変換"""
        return {
            name: ParameterSpace(
                type=cfg["type"],
                low=int(cfg["low"]) if cfg["type"] == "integer" else cfg.get("low"),
                high=int(cfg["high"]) if cfg["type"] == "integer" else cfg.get("high"),
                categories=cfg.get("categories"),
            )
            for name, cfg in parameter_space_config.items()
        }

    def _create_objective_function(
        self,
        trainer: Any,
        training_data: pd.DataFrame,
        optimization_settings: OptimizationSettings,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        **base_training_params,
    ) -> Callable[[Dict[str, Any]], float]:
        """
        Optuna に渡す目的関数（Objective Function）を作成

        各試行で渡されるパラメータを受け取り、モデル学習と評価を行い、
        最大化すべきスコア（F1 スコア）を返します。

        Args:
            trainer: テンプレートとなるトレーナー
            training_data: 学習データ
            optimization_settings: 最適化設定
            **base_training_params: 固定の学習パラメータ

        Returns:
            パラメータ辞書を受け取りスコア（float）を返す関数
        """
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
