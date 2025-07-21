"""
ML学習サービス

MLモデルの学習機能を専門的に扱うサービス。
BaseMLTrainerを使用してコードの重複を解消し、責任を明確化します。
ハイパーパラメータ最適化機能も提供します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Callable


from .config import ml_config
from ...utils.unified_error_handler import safe_ml_operation
from .lightgbm_trainer import LightGBMTrainer
from .model_manager import model_manager
from ..optimization.optimizer_factory import OptimizerFactory
from ..optimization.base_optimizer import ParameterSpace

logger = logging.getLogger(__name__)


class OptimizationSettings:
    """最適化設定クラス"""

    def __init__(
        self,
        enabled: bool = False,
        method: str = "bayesian",
        n_calls: int = 50,
        parameter_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enabled = enabled
        self.method = method
        self.n_calls = n_calls
        self.parameter_space = parameter_space or {}


class MLTrainingService:
    """
    ML学習サービス

    BaseMLTrainerを使用してMLモデルの学習、評価、保存を専門的に行うサービス。
    コードの重複を解消し、保守性を向上させます。
    """

    def __init__(self, trainer_type: str = "lightgbm"):
        """
        初期化

        Args:
            trainer_type: 使用するトレーナーのタイプ（現在は"lightgbm"のみサポート）
        """
        self.config = ml_config

        # トレーナーを選択
        if trainer_type.lower() == "lightgbm":
            self.trainer = LightGBMTrainer()
        else:
            raise ValueError(f"サポートされていないトレーナータイプ: {trainer_type}")

        self.trainer_type = trainer_type

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        optimization_settings: Optional[OptimizationSettings] = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習（最適化機能付き）

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            model_name: モデル名（オプション）
            optimization_settings: 最適化設定（オプション）
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書

        Raises:
            MLDataError: データが無効な場合
            MLModelError: 学習に失敗した場合
        """
        # 最適化が有効な場合は最適化ワークフローを実行
        if optimization_settings and optimization_settings.enabled:
            return self._train_with_optimization(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                optimization_settings=optimization_settings,
                **training_params,
            )
        else:
            # 通常のトレーニング
            return self.trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                **training_params,
            )

    def evaluate_model(
        self,
        test_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        学習済みモデルを評価

        Args:
            test_data: テスト用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            評価結果の辞書
        """
        # BaseMLTrainerに委譲
        return self.trainer.evaluate_model(
            test_data=test_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
        )

    def get_training_status(self) -> Dict[str, Any]:
        """
        学習状態を取得

        Returns:
            学習状態の辞書
        """
        if hasattr(self.trainer, "get_model_info"):
            model_info = self.trainer.get_model_info()
            model_info["trainer_type"] = self.trainer_type
            return model_info
        else:
            return {
                "is_trained": self.trainer.is_trained,
                "feature_columns": self.trainer.feature_columns,
                "feature_count": (
                    len(self.trainer.feature_columns)
                    if self.trainer.feature_columns
                    else 0
                ),
                "model_type": (
                    type(self.trainer.model).__name__ if self.trainer.model else None
                ),
                "trainer_type": self.trainer_type,
            }

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        predictions = self.trainer.predict(features_df)
        return {
            "predictions": predictions,
            "model_type": self.trainer_type,
            "feature_count": (
                len(self.trainer.feature_columns) if self.trainer.feature_columns else 0
            ),
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if hasattr(self.trainer, "get_feature_importance"):
            return self.trainer.get_feature_importance()
        else:
            return {}

    @safe_ml_operation(
        default_return=False, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        return self.trainer.load_model(model_path)

    def get_latest_model_path(self) -> Optional[str]:
        """最新のモデルパスを取得"""
        return model_manager.get_latest_model("*")

    def list_available_models(self) -> list:
        """利用可能なモデルの一覧を取得"""
        return model_manager.list_models("*")

    def _train_with_optimization(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        optimization_settings: OptimizationSettings = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        最適化を使用してMLモデルを学習

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            model_name: モデル名（オプション）
            optimization_settings: 最適化設定
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書（最適化情報を含む）
        """
        try:
            logger.info("🚀 ハイパーパラメータ最適化を開始")
            logger.info(f"🔧 使用する最適化手法: {optimization_settings.method}")
            logger.info(f"🎯 目標試行回数: {optimization_settings.n_calls}")

            # オプティマイザーを作成
            optimizer = OptimizerFactory.create_optimizer(optimization_settings.method)
            logger.info(f"✅ {optimizer.__class__.__name__} を作成しました")

            # パラメータ空間を準備
            parameter_space = self._prepare_parameter_space(
                optimization_settings.parameter_space
            )
            logger.info(
                f"📊 パラメータ空間を準備: {len(parameter_space)}個のパラメータ"
            )

            # 目的関数を作成
            objective_function = self._create_objective_function(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                optimization_settings=optimization_settings,
                **training_params,
            )
            logger.info("🎯 目的関数を作成しました")

            # 最適化を実行
            logger.info("🔄 最適化処理を開始...")
            optimization_result = optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=optimization_settings.n_calls,
            )

            logger.info("🎉 最適化が完了しました！")
            logger.info(f"🏆 ベストスコア: {optimization_result.best_score:.4f}")
            logger.info(f"⚙️  最適パラメータ: {optimization_result.best_params}")
            logger.info(f"📈 総評価回数: {optimization_result.total_evaluations}")
            logger.info(f"⏱️  最適化時間: {optimization_result.optimization_time:.2f}秒")

            # 最適化されたパラメータで最終モデルを学習
            final_training_params = {
                **training_params,
                **optimization_result.best_params,
            }
            final_result = self.trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                **final_training_params,
            )

            # 最適化情報を結果に追加
            final_result["optimization_result"] = {
                "method": optimization_settings.method,
                "best_params": optimization_result.best_params,
                "best_score": optimization_result.best_score,
                "total_evaluations": optimization_result.total_evaluations,
                "optimization_time": optimization_result.optimization_time,
                "convergence_info": optimization_result.convergence_info,
            }

            return final_result

        except Exception as e:
            logger.error(f"最適化学習中にエラーが発生しました: {e}")
            raise

    def _prepare_parameter_space(
        self, parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """
        パラメータ空間設定をParameterSpaceオブジェクトに変換

        Args:
            parameter_space_config: パラメータ空間設定

        Returns:
            ParameterSpaceオブジェクトの辞書
        """
        parameter_space = {}

        for param_name, param_config in parameter_space_config.items():
            param_type = param_config["type"]
            low = param_config.get("low")
            high = param_config.get("high")

            # integer型の場合は、lowとhighを整数に変換
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
        training_data: pd.DataFrame,
        optimization_settings: "OptimizationSettings",
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        **base_training_params,
    ) -> Callable[[Dict[str, Any]], float]:
        """
        最適化のための目的関数を作成

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            **base_training_params: ベースとなる学習パラメータ

        Returns:
            目的関数（パラメータを受け取りスコアを返す関数）
        """

        # 試行回数カウンター
        evaluation_count = 0

        def objective_function(params: Dict[str, Any]) -> float:
            """
            目的関数：与えられたハイパーパラメータでミニトレーニングを実行し、評価スコアを返す

            Args:
                params: 最適化対象のハイパーパラメータ

            Returns:
                評価スコア（F1スコア）
            """
            nonlocal evaluation_count
            evaluation_count += 1

            try:
                logger.info(
                    f"🔍 最適化試行 {evaluation_count}/{optimization_settings.n_calls}"
                )
                logger.info(f"📋 試行パラメータ: {params}")

                # ベースパラメータと最適化パラメータをマージ
                training_params = {**base_training_params, **params}

                # 一時的なトレーナーを作成（元のトレーナーに影響しないように）
                temp_trainer = LightGBMTrainer()

                # ミニトレーニングを実行（保存はしない）
                result = temp_trainer.train_model(
                    training_data=training_data,
                    funding_rate_data=funding_rate_data,
                    open_interest_data=open_interest_data,
                    save_model=False,  # 最適化中は保存しない
                    model_name=None,
                    **training_params,
                )

                # F1スコアを評価指標として使用
                f1_score = result.get("f1_score", 0.0)

                # マクロ平均F1スコアがある場合はそれを優先
                if "classification_report" in result:
                    macro_f1 = (
                        result["classification_report"]
                        .get("macro avg", {})
                        .get("f1-score", f1_score)
                    )
                    f1_score = macro_f1

                logger.info(f"📊 試行結果: F1スコア={f1_score:.4f}")
                logger.info("-" * 50)
                return f1_score

            except Exception as e:
                logger.warning(f"目的関数評価中にエラーが発生しました: {e}")
                # エラーの場合は低いスコアを返す
                return 0.0

        return objective_function


# グローバルインスタンス（デフォルトはLightGBM）
ml_training_service = MLTrainingService(trainer_type="lightgbm")
