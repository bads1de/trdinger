"""
ML学習サービス

MLモデルの学習・評価・保存を取り扱うサービス層です。
内部実装の詳細や特定の最適化手法の説明はDocstringに含めず、
サービスの役割（学習ワークフローの調整と結果の提供）に限定して記述します。
"""

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from ...utils.data_processing import data_processor as data_preprocessor
from ...utils.error_handler import safe_ml_operation
from ..optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace
from .base_ml_trainer import BaseMLTrainer
from .common.base_resource_manager import BaseResourceManager, CleanupLevel
from .config import ml_config
from .model_manager import model_manager
from .single_model.single_model_trainer import SingleModelTrainer
from .ensemble.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class OptimizationSettings:
    """最適化設定クラス（簡素化版）"""

    def __init__(
        self,
        enabled: bool = False,
        n_calls: int = 50,
        parameter_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enabled = enabled
        self.n_calls = n_calls
        self.parameter_space = parameter_space or {}


class MLTrainingService(BaseResourceManager):
    """
    ML学習サービス

    BaseMLTrainerを使用してMLモデルの学習、評価、保存を専門的に行うサービス。
    コードの重複を解消し、保守性を向上させます。
    """

    def __init__(
        self,
        trainer_type: str = "ensemble",
        automl_config: Optional[Dict[str, Any]] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            trainer_type: 使用するトレーナーのタイプ（'ensemble' または 'single'）
            automl_config: AutoML設定（辞書形式）
            ensemble_config: アンサンブル設定（辞書形式）
            single_model_config: 単一モデル設定（辞書形式）
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.config = ml_config
        self.automl_config = automl_config
        self.ensemble_config = ensemble_config
        self.single_model_config = single_model_config

        # 統合されたトレーナー設定を作成
        trainer_config = self._create_trainer_config(
            trainer_type, ensemble_config, single_model_config
        )

        # トレーナーを選択して初期化
        if trainer_type.lower() == "single":
            model_type = trainer_config.get("model_type", "lightgbm")
            # 明示的に SingleModelTrainer を使用（テスト期待と一致）
            self.trainer = SingleModelTrainer(
                model_type=model_type, automl_config=automl_config
            )
        else:
            # デフォルトは統合 BaseMLTrainer
            self.trainer = BaseMLTrainer(
                automl_config=automl_config, trainer_config=trainer_config
            )

        self.trainer_type = trainer_type

        if trainer_type == "single" and single_model_config:
            logger.info(f"単一モデル設定: {single_model_config}")

    def _create_trainer_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        統合されたトレーナー設定を作成

        Args:
            trainer_type: トレーナータイプ
            ensemble_config: アンサンブル設定
            single_model_config: 単一モデル設定

        Returns:
            トレーナー設定辞書
        """
        if trainer_type.lower() == "ensemble":
            # アンサンブル設定のデフォルト値
            default_ensemble_config = {
                "method": "bagging",
                "bagging_params": {
                    "n_estimators": 5,
                    "bootstrap_fraction": 0.8,
                    "base_model_type": "lightgbm",
                },
            }

            # 設定をマージ
            final_ensemble_config = default_ensemble_config.copy()
            if ensemble_config:
                final_ensemble_config.update(ensemble_config)

            return {
                "type": "ensemble",
                "model_type": final_ensemble_config.get("method", "bagging"),
                "ensemble_config": final_ensemble_config,
            }

        elif trainer_type.lower() == "single":
            # 単一モデル設定のデフォルト値
            model_type = "lightgbm"
            if single_model_config and "model_type" in single_model_config:
                model_type = single_model_config["model_type"]

            return {
                "type": "single",
                "model_type": model_type,
                "model_params": single_model_config,
            }

        else:
            raise ValueError(
                f"サポートされていないトレーナータイプ: {trainer_type}。"
                f"サポートされているタイプ: 'ensemble', 'single'"
            )

    @staticmethod
    def get_available_single_models() -> list:
        """利用可能な単一モデルのリストを取得"""
        return SingleModelTrainer.get_available_models()

    @staticmethod
    def determine_trainer_type(ensemble_config: Optional[Dict[str, Any]]) -> str:
        """
        アンサンブル設定に基づいてトレーナータイプを決定

        Args:
            ensemble_config: アンサンブル設定

        Returns:
            トレーナータイプ（'ensemble' または 'single'）
        """
        if ensemble_config and ensemble_config.get("enabled", True) is False:
            return "single"
        return "ensemble"

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        optimization_settings: Optional[OptimizationSettings] = None,
        automl_config: Optional[Dict[str, Any]] = None,
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
            automl_config: AutoML特徴量エンジニアリング設定（オプション）
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書

        Raises:
            MLDataError: データが無効な場合
            MLModelError: 学習に失敗した場合
        """
        # AutoML設定の処理
        effective_automl_config = automl_config or self.automl_config
        if effective_automl_config:
            # AutoML設定が提供された場合、新しいアンサンブルトレーナーインスタンスを作成
            ensemble_config = self.ensemble_config or {
                "method": "bagging",
                "bagging_params": {
                    "n_estimators": 5,
                    "bootstrap_fraction": 0.8,
                    "base_model_type": "lightgbm",
                },
            }
            trainer = EnsembleTrainer(
                ensemble_config=ensemble_config, automl_config=effective_automl_config
            )
            logger.info(
                "🤖 AutoML特徴量エンジニアリングを使用してアンサンブル学習を実行します"
            )
        else:
            trainer = self.trainer
            logger.info(
                "📊 基本特徴量エンジニアリングを使用してアンサンブル学習を実行します"
            )

        # 最適化が有効な場合は最適化ワークフローを実行
        if optimization_settings and optimization_settings.enabled:
            return self._train_with_optimization(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                optimization_settings=optimization_settings,
                trainer=trainer,  # 適切なトレーナーを渡す
                **training_params,
            )
        else:
            # 通常のトレーニング
            return trainer.train_model(
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
        # トレーナーから基本情報を取得
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



    def generate_signals(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        予測信号を生成（MLSignalGeneratorのpredictメソッドから移植）

        Args:
            features: 特徴量DataFrame

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        try:

            if not self.trainer.is_trained or self.trainer.model is None:
                # モデル未学習時は警告レベルでログ出力
                logger.warning("モデルが学習されていません。デフォルト値を返します。")
                default_predictions = self.config.prediction.get_default_predictions()
                return default_predictions

            if self.trainer.feature_columns is None:
                # 特徴量カラムが設定されていない場合、利用可能な全カラムを使用
                logger.warning(
                    "特徴量カラムが設定されていません。利用可能な全カラムを使用します。"
                )
                # 統計的手法で欠損値を補完
                features_selected = data_preprocessor.transform_missing_values(
                    features, strategy="median"
                )
            else:
                # 特徴量を選択・整形
                available_columns = [
                    col
                    for col in self.trainer.feature_columns
                    if col in features.columns
                ]
                missing_columns = [
                    col
                    for col in self.trainer.feature_columns
                    if col not in features.columns
                ]

                if len(missing_columns) > 0:
                    logger.warning(f"欠損している特徴量カラム: {missing_columns}")

                if not available_columns:
                    logger.warning(
                        "指定された特徴量カラムが見つかりません。デフォルト値を返します。"
                    )
                    return self.config.prediction.get_default_predictions()
                else:
                    # 利用可能な特徴量のみを使用し、統計的手法で欠損値を補完
                    features_subset = features[available_columns]
                    features_selected = data_preprocessor.transform_missing_values(
                        features_subset, strategy="median"
                    )

                    # 不足している特徴量を一度にまとめて追加（DataFrame断片化を防ぐ）
                    if missing_columns:
                        # 不足特徴量のDataFrameを作成
                        missing_features_df = pd.DataFrame(
                            0.0, index=features_selected.index, columns=missing_columns
                        )
                        # pd.concatで一度に結合（断片化を防ぐ）
                        features_selected = pd.concat(
                            [features_selected, missing_features_df], axis=1
                        )

                    # 学習時と同じ順序で並び替え
                    features_selected = features_selected[self.trainer.feature_columns]

            # 標準化
            if self.trainer.scaler is not None:
                features_scaled = self.trainer.scaler.transform(features_selected)
            else:
                logger.warning(
                    "スケーラーが設定されていません。標準化をスキップします。"
                )
                features_scaled = features_selected.values

            # 予測（LightGBMモデルの場合）
            # best_iteration属性の存在を確認してから使用
            if hasattr(self.trainer.model, "best_iteration"):
                predictions = np.array(
                    self.trainer.model.predict(
                        features_scaled, num_iteration=self.trainer.model.best_iteration
                    )
                )
            else:
                predictions = np.array(self.trainer.model.predict(features_scaled))

            # 最新の予測結果を取得
            if predictions.ndim == 2:
                latest_pred = predictions[-1]  # 最後の行
            else:
                latest_pred = predictions

            # 予測結果を3クラス（down, range, up）の確率に変換
            if latest_pred.shape[0] == 3:
                predictions = {
                    "down": float(latest_pred[0]),
                    "range": float(latest_pred[1]),
                    "up": float(latest_pred[2]),
                }
                return predictions
            else:
                # 3クラス以外の場合はエラー
                logger.error(
                    f"予期しない予測結果の形式: {latest_pred.shape}. 3クラス分類が期待されます。"
                )
                default_predictions = self.config.prediction.get_default_predictions()
                return default_predictions

        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            default_predictions = self.config.prediction.get_default_predictions()
            return default_predictions

    def _train_with_optimization(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        optimization_settings: OptimizationSettings = None,
        trainer: Optional[Any] = None,  # カスタムトレーナーを受け取る
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
            trainer: 使用するトレーナー（オプション、指定されない場合はself.trainerを使用）
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書（最適化情報を含む）
        """
        optimizer = None
        try:
            # 使用するトレーナーを決定
            effective_trainer = trainer if trainer is not None else self.trainer

            logger.info("🚀 Optuna最適化を開始")
            logger.info(f"🎯 目標試行回数: {optimization_settings.n_calls}")
            logger.info(f"🤖 使用トレーナー: {type(effective_trainer).__name__}")

            # Optunaオプティマイザーを作成
            optimizer = OptunaOptimizer()
            logger.info("✅ OptunaOptimizer を作成しました")

            # パラメータ空間を準備
            if not optimization_settings.parameter_space:
                # アンサンブルトレーナーの場合は専用のパラメータ空間を使用
                if hasattr(effective_trainer, "ensemble_config"):
                    ensemble_method = effective_trainer.ensemble_config.get(
                        "method", "bagging"
                    )
                    enabled_models = effective_trainer.ensemble_config.get(
                        "models", ["lightgbm", "xgboost", "randomforest"]
                    )
                    parameter_space = optimizer.get_ensemble_parameter_space(
                        ensemble_method, enabled_models
                    )
                    logger.info(
                        f"📊 アンサンブル用パラメータ空間を使用: {ensemble_method}, モデル: {enabled_models}"
                    )
                else:
                    # デフォルトのLightGBMパラメータ空間を使用
                    parameter_space = optimizer.get_default_parameter_space()
                    logger.info("📊 デフォルトのLightGBMパラメータ空間を使用")
            else:
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
                trainer=effective_trainer,  # カスタムトレーナーを渡す
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

            # Optunaリソースをクリーンアップ（メモリーリーク防止）
            try:
                optimizer.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"OptunaOptimizer クリーンアップ警告: {cleanup_error}")

            # 最適化されたパラメータで最終モデルを学習
            final_training_params = {
                **training_params,
                **optimization_result.best_params,
            }
            final_result = effective_trainer.train_model(  # カスタムトレーナーを使用
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                **final_training_params,
            )

            # 最適化情報を結果に追加
            final_result["optimization_result"] = {
                "method": "optuna",
                "best_params": optimization_result.best_params,
                "best_score": optimization_result.best_score,
                "total_evaluations": optimization_result.total_evaluations,
                "optimization_time": optimization_result.optimization_time,
            }

            return final_result

        except Exception as e:
            logger.error(f"最適化学習中にエラーが発生しました: {e}")
            raise
        finally:
            # 例外が発生した場合でもOptunaリソースをクリーンアップ
            if optimizer is not None:
                try:
                    optimizer.cleanup()
                except Exception as cleanup_error:
                    logger.warning(
                        f"例外処理でのOptunaOptimizer クリーンアップ警告: {cleanup_error}"
                    )

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
        trainer: Optional[Any] = None,  # カスタムトレーナーを受け取る
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
        # 使用するトレーナーを決定
        effective_trainer = trainer if trainer is not None else self.trainer

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

                # 一時的なアンサンブルトレーナーを作成（元のトレーナーに影響しないように）
                # AutoML設定がある場合はそれを引き継ぐ
                temp_ensemble_config = {
                    "method": "bagging",
                    "bagging_params": {
                        "n_estimators": 3,  # 最適化中は高速化のため少なめ
                        "bootstrap_fraction": 0.8,
                        "base_model_type": "lightgbm",
                    },
                }

                if hasattr(effective_trainer, "automl_config"):
                    temp_trainer = EnsembleTrainer(
                        ensemble_config=temp_ensemble_config,
                        automl_config=effective_trainer.automl_config,
                    )
                else:
                    temp_trainer = EnsembleTrainer(ensemble_config=temp_ensemble_config)

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

    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ"""
        # MLTrainingServiceでは特に一時ファイルは作成しないため、パス
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ"""
        # MLTrainingServiceでは特にキャッシュは管理しないため、パス
        pass

    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ"""
        try:
            # トレーナーのクリーンアップ
            if hasattr(self, "trainer") and self.trainer:
                if hasattr(self.trainer, "cleanup_resources"):
                    self.trainer.cleanup_resources(level)
                    logger.debug("トレーナーをクリーンアップしました")

            # 最適化器のクリーンアップ
            if hasattr(self, "optimizer") and self.optimizer:
                if hasattr(self.optimizer, "cleanup"):
                    self.optimizer.cleanup()
                    logger.debug("最適化器をクリーンアップしました")

        except Exception as e:
            logger.warning(f"MLTrainingServiceモデルクリーンアップエラー: {e}")


# グローバルインスタンス（デフォルトはアンサンブル、AutoML設定なし）
ml_training_service = MLTrainingService(trainer_type="ensemble", automl_config=None)
