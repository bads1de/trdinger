"""
ML学習サービス

MLモデルのトレーニングフローを制御し、
データの準備、特徴量エンジニアリング、学習、評価、保存を一括管理します。
アンサンブル学習と単一モデル学習の両方に対応し、
ハイパーパラメータ最適化（Optuna）との連携も行います。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from app.services.ml.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)

from ....config.unified_config import unified_config
from ....utils.error_handler import safe_ml_operation
from ..common.base_resource_manager import BaseResourceManager, CleanupLevel
from ..ensemble.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class MLTrainingService(BaseResourceManager):
    """
    MLモデルのトレーニングフローを管理するサービス

    主な責務:
    1. トレーナーの初期化（アンサンブル or 単一モデル）
    2. 学習データの準備と検証
    3. 特徴量エンジニアリングの実行（トレーナー内）
    4. モデル学習と評価の実行
    5. モデルの永続化
    6. ハイパーパラメータ最適化の連携
    """

    def __init__(
        self,
        trainer_type: str = "ensemble",  # "ensemble" or "single"
        ensemble_config: Optional[Dict[str, Any]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            trainer_type: トレーナータイプ ("ensemble" または "single")
                - "ensemble": アンサンブル学習（スタッキング等）を使用
                - "single": 単一モデル（LightGBM等）を使用
            ensemble_config: アンサンブル設定（trainer_type="ensemble"の場合に使用）
            single_model_config: 単一モデル設定（trainer_type="single"の場合に使用）
        """
        super().__init__()
        self.trainer_type = trainer_type
        # 統一設定の作成
        config = self._create_unified_config(
            trainer_type, ensemble_config, single_model_config
        )
        self.trainer = EnsembleTrainer(ensemble_config=config)
        self.optimization_service = OptimizationService()

    def _create_unified_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        設定に基づいて統一設定を作成

        Args:
            trainer_type: トレーナータイプ
            ensemble_config: アンサンブル設定
            single_model_config: 単一モデル設定

        Returns:
            統一設定辞書
        """
        if trainer_type == "ensemble":
            if ensemble_config is None:
                # 辞書形式で取得
                ensemble_config = unified_config.ml.ensemble.model_dump()
                # キー名の変換 (UnifiedConfig -> EnsembleTrainer内部期待形式)
                if "default_method" in ensemble_config:
                    ensemble_config["method"] = ensemble_config.pop("default_method")
                if "algorithms" in ensemble_config:
                    ensemble_config["models"] = ensemble_config.pop("algorithms")
            return ensemble_config

        elif trainer_type == "single":
            if single_model_config is None:
                single_model_config = {"model_type": "lightgbm"}

            model_type = single_model_config.get("model_type", "lightgbm")
            # EnsembleTrainerが期待する形式に変換
            return {
                "model_type": model_type,
                "models": [model_type],
                "method": "stacking",  # デフォルト
                **single_model_config,
            }
        else:
            raise ValueError(f"サポートされていないトレーナータイプ: {trainer_type}")

    @safe_ml_operation(context="モデル学習")
    def train_model(
        self,
        training_data: Any,  # Dict[str, pd.DataFrame] または pd.DataFrame
        save_model: bool = True,
        optimization_settings: Optional[OptimizationSettings] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        モデルの学習を実行

        ハイパーパラメータ最適化が有効な場合は、最適化を実行した後に
        最適なパラメータで再学習を行います。

        Args:
            training_data: 学習データ（ohlcv, funding_rate, open_interestなどを含む辞書、またはOHLCVのDataFrame）
            save_model: 学習後にモデルを保存するかどうか
            optimization_settings: 最適化設定（Noneの場合は最適化なし）
            test_size: テストデータの割合
            random_state: 乱数シード

        Returns:
            学習結果（メトリクス、特徴量重要度、モデル情報など）
        """
        # データの正規化
        if isinstance(training_data, pd.DataFrame):
            data_dict = {"ohlcv": training_data}
        else:
            data_dict = training_data

        ohlcv = data_dict.get("ohlcv")
        funding_rate = data_dict.get("funding_rate")
        open_interest = data_dict.get("open_interest")

        # 1. ハイパーパラメータ最適化（有効な場合）
        best_params = {}
        is_optimized = False
        optimization_result = None

        if optimization_settings and optimization_settings.enabled:
            logger.info("ハイパーパラメータ最適化を開始します")
            try:
                optimization_result = self.optimization_service.optimize_parameters(
                    trainer=self.trainer,
                    training_data=ohlcv,
                    optimization_settings=optimization_settings,
                    funding_rate_data=funding_rate,
                    open_interest_data=open_interest,
                )
                best_params = optimization_result.get("best_params", {})
                is_optimized = True
                logger.info("ハイパーパラメータ最適化が完了しました")
            except Exception as e:
                logger.error(f"ハイパーパラメータ最適化に失敗しました: {e}")
                logger.warning(
                    "最適化をスキップし、デフォルトパラメータで学習を続行します"
                )

        # 2. モデル学習（最適化されたパラメータを使用）
        # 学習パラメータの構築
        training_params = {
            "test_size": test_size,
            "random_state": random_state,
            "optimize_hyperparameters": False,
            **best_params,
        }

        logger.info("モデル学習を開始します")
        # BaseMLTrainer.train_modelを呼び出す（trainではなく）
        result = self.trainer.train_model(
            training_data=ohlcv,
            funding_rate_data=funding_rate,
            open_interest_data=open_interest,
            save_model=save_model,
            **training_params,
        )

        # テストでのMock対応: Mockオブジェクトは項目代入をサポートしていない場合があるため
        # 辞書であることを確認してから代入する
        if isinstance(result, dict):
            result["is_optimized"] = is_optimized
            if optimization_result:
                result["optimization_result"] = optimization_result

        return result

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果を含む辞書
        """
        predictions = self.trainer.predict(features_df)
        return {
            "predictions": predictions,
            "model_type": self.trainer_type,
            "feature_count": len(features_df.columns),
        }

    def generate_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        シグナル生成

        Args:
            features_df: 特徴量DataFrame

        Returns:
            シグナル辞書
        """
        return self.trainer.predict_signal(features_df)

    def evaluate_model(
        self,
        test_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        モデル評価

        Args:
            test_data: テストデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉データ

        Returns:
            評価結果
        """
        return self.trainer.evaluate_model(
            test_data, funding_rate_data, open_interest_data
        )

    def get_training_status(self) -> Dict[str, Any]:
        """
        学習状態取得

        Returns:
            学習状態辞書
        """
        return {
            "is_trained": self.trainer.is_trained,
            "trainer_type": self.trainer_type,
            "feature_count": len(self.trainer.feature_columns)
            if self.trainer.feature_columns
            else 0,
            "model_type": self.trainer.model_type,
        }

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度取得

        Args:
            top_n: 上位N件

        Returns:
            特徴量重要度辞書
        """
        return self.trainer.get_feature_importance(top_n)

    @classmethod
    def get_available_single_models(cls) -> list[str]:
        """
        利用可能な単一モデル

        Returns:
            モデル名のリスト
        """
        return ["lightgbm", "xgboost", "catboost"]

    def load_model(self, model_path: str) -> bool:
        """
        指定されたパスからモデルを読み込む

        Args:
            model_path: モデルファイルのパス

        Returns:
            読み込み成功時はTrue
        """
        return self.trainer.load_model(model_path)

    def get_current_model_path(self) -> Optional[str]:
        """現在メモリに読み込まれているモデルのパスを取得"""
        # BaseMLTrainerのmetadataからパスを取得できる場合がある
        if hasattr(self.trainer, "metadata") and self.trainer.metadata:
            return self.trainer.metadata.get("model_path")
        return None

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """現在メモリに読み込まれているモデルのメタデータを取得"""
        if hasattr(self.trainer, "metadata"):
            return self.trainer.metadata
        return None

    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ"""
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ"""
        pass

    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ"""
        if self.trainer and hasattr(self.trainer, "cleanup_resources"):
            self.trainer.cleanup_resources(level)


# グローバルインスタンス（デフォルトはアンサンブル）
ml_training_service = MLTrainingService(trainer_type="ensemble")
