"""
MLトレーニングサービス

MLモデルのトレーニングフローを制御し、
データの準備、特徴量エンジニアリング、学習、評価、保存を一括管理します。
バックグラウンドタスクとしての実行管理や進捗管理機能も備えています。
"""

import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union, cast

import pandas as pd
from sqlalchemy.orm import Session

from app.services.ml.common.config import ml_config_manager
from app.services.ml.models.model_manager import model_manager
from app.services.ml.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)
from app.services.ml.orchestration.bg_task_orchestration_service import (
    background_task_manager,
)
from app.services.ml.orchestration.training_config_validator import (
    validate_training_config,
)
from app.utils.error_handler import safe_ml_operation
from app.utils.datetime_utils import parse_datetime_range_optional
from app.utils.response import api_response, ensure_response_dict
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from ..common.base_resource_manager import BaseResourceManager, CleanupLevel
from ..common.config import get_default_single_model_config
from ..common.training_utils import resolve_holdout_test_size
from ..ensemble.ensemble_trainer import EnsembleConfig, EnsembleTrainer
from ..trainers.volatility_regression_trainer import VolatilityRegressionTrainer

logger = logging.getLogger(__name__)

# --- モデル情報取得ユーティリティ (旧orchestration_utilsより統合) ---


def load_model_metadata_safely(model_path: str) -> Optional[Dict[str, Any]]:
    """モデルファイルからメタデータを安全に読み込む"""
    try:
        metadata_data = model_manager.load_metadata_only(model_path)
        if not metadata_data or "metadata" not in metadata_data:
            return None
        return metadata_data
    except Exception as e:
        logger.warning(f"モデルメタデータ読み込みエラー {model_path}: {e}")
        return None


def get_latest_model_with_info(
    model_name_pattern: str = "*",
) -> Optional[Dict[str, Any]]:
    """最新モデルの情報とメトリクスを取得"""
    try:
        latest_model = model_manager.get_latest_model(
            model_name_pattern,
            metadata_filters={
                "task_type": "volatility_regression",
                "target_kind": "log_realized_vol",
            },
        )
        if not latest_model or not os.path.exists(latest_model):
            return None

        model_data = load_model_metadata_safely(latest_model)
        if not model_data:
            return None

        metadata = model_data["metadata"]
        metrics = model_manager.extract_model_performance_metrics(
            latest_model, metadata=metadata
        )
        file_info = {
            "size_mb": os.path.getsize(latest_model) / (1024 * 1024),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(latest_model)),
        }
        return {
            "path": latest_model,
            "metadata": metadata,
            "metrics": metrics,
            "file_info": file_info,
        }
    except Exception as e:
        logger.warning(f"最新モデル情報取得エラー: {e}")
        return None


def get_model_info_with_defaults(
    model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """モデル情報にデフォルト値を適用"""
    from ..evaluation.metrics import get_default_metrics

    if not model_info:
        return {
            **get_default_metrics(),
            "model_type": "No Model",
            "feature_count": 0,
            "training_samples": 0,
            "last_updated": "未学習",
            "file_size_mb": 0.0,
        }

    m, meta, f = (
        model_info.get("metrics", get_default_metrics()),
        model_info["metadata"],
        model_info["file_info"],
    )
    return {
        **m,
        "model_type": meta.get("model_type", "Unknown"),
        "task_type": meta.get("task_type", "volatility_regression"),
        "target_kind": meta.get("target_kind", "log_realized_vol"),
        "feature_count": meta.get("feature_count", 0),
        "training_samples": meta.get("training_samples", 0),
        "test_samples": meta.get("test_samples", 0),
        "last_updated": f["modified_at"].isoformat(),
        "file_size_mb": f["size_mb"],
        "num_classes": meta.get("num_classes", 1),
        "best_iteration": meta.get("best_iteration", 0),
        "train_test_split": meta.get("train_test_split", 0.8),
        "random_state": meta.get("random_state", 42),
        "feature_importance": meta.get("feature_importance", {}),
        "classification_report": meta.get("classification_report", {}),
    }


# グローバルトレーニング状態管理
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "待機中",
    "start_time": None,
    "end_time": None,
    "model_info": None,
    "error": None,
    "training_id": None,
    "task_id": None,
}
training_status_lock = threading.Lock()


class MLTrainingService(BaseResourceManager):
    """
    MLモデルのトレーニングフローを管理するサービス

    主な責務:
    1. トレーニングのバックグラウンド実行管理と進捗管理
    2. トレーナーの初期化（アンサンブル or 単一モデル）
    3. 学習データの準備と検証
    4. モデル学習と評価の実行（最適化含む）
    5. モデルの永続化
    """

    def __init__(
        self,
        trainer_type: str = "ensemble",
        ensemble_config: Optional[Dict[str, Any]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        MLTrainingServiceを初期化

        機械学習モデルのトレーニングフローを管理するサービスを初期化します。
        アンサンブル学習または単一モデル学習のいずれかを選択できます。

        Args:
            trainer_type: トレーナータイプ（"ensemble"または"single"、デフォルト: "ensemble"）
            ensemble_config: アンサンブル学習の設定辞書（オプション）
            single_model_config: 単一モデル学習の設定辞書（オプション）

        Note:
            trainer_typeが"ensemble"の場合、EnsembleTrainerが使用されます。
            trainer_typeが"single"の場合、VolatilityRegressionTrainerが使用されます。
        """
        super().__init__()
        self.trainer_type = trainer_type
        self.trainer: Union[VolatilityRegressionTrainer, EnsembleTrainer]
        if trainer_type == "single":
            model_type = (single_model_config or {}).get("model_type", "lightgbm")
            self.trainer = VolatilityRegressionTrainer(
                model_type=model_type,
                model_params=single_model_config or {},
            )
        else:
            config = self._create_trainer_config(
                trainer_type, ensemble_config, single_model_config
            )
            self.trainer = EnsembleTrainer(ensemble_config=cast(EnsembleConfig, config))
        self.optimization_service = OptimizationService()

    def _create_trainer_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Union[EnsembleConfig, Dict[str, Any]]:
        """
        設定に基づいてトレーナー設定を作成

        指定されたトレーナータイプと設定辞書から、
        トレーナーに渡すための統合設定辞書を作成します。

        Args:
            trainer_type: トレーナータイプ（"ensemble"または"single"）
            ensemble_config: アンサンブル学習の設定辞書（オプション）
            single_model_config: 単一モデル学習の設定辞書（オプション）

        Returns:
            Dict[str, Any]: トレーナー設定辞書

        Raises:
            ValueError: サポートされていないトレーナータイプが指定された場合

        Note:
            trainer_typeが"ensemble"でensemble_configがNoneの場合、
            デフォルトのアンサンブル設定が使用されます。
            trainer_typeが"single"でsingle_model_configがNoneの場合、
            デフォルトの単一モデル設定が使用されます。
        """
        if trainer_type == "ensemble":
            if ensemble_config is None:
                ensemble_config = ml_config_manager.config.ensemble.model_dump()
                if "default_method" in ensemble_config:
                    ensemble_config["method"] = ensemble_config.pop("default_method")
                if "algorithms" in ensemble_config:
                    ensemble_config["models"] = ensemble_config.pop("algorithms")
            return cast(EnsembleConfig, ensemble_config)

        elif trainer_type == "single":
            if single_model_config is None:
                single_model_config = {"model_type": "lightgbm"}
            model_type = single_model_config.get("model_type", "lightgbm")
            return {
                "model_type": model_type,
                "models": [model_type],
                "method": "stacking",
                **single_model_config,
            }
        else:
            raise ValueError(f"サポートされていないトレーナータイプ: {trainer_type}")

    # --- オーケストレーション（API向け）機能 ---

    async def start_training(
        self, config, background_tasks, db: Session
    ) -> Dict[str, Any]:
        """
        MLトレーニングをバックグラウンドタスクとして開始します。

        このメソッドはAPIリクエストを受け取り、以下の手順を実行します：
        1. 入力されたトレーニング設定のバリデーション（`_validate_training_config`）。
        2. ユニークな `training_id` を生成。
        3. グローバルトレーニング状態（`training_status`）を更新し、二重起動を防止。
        4. FastAPIの `BackgroundTasks` に `_train_in_background` を登録。

        Args:
            config (MLTrainingConfig): トレーニング設定（期間、シンボル、モデル設定等）。
            background_tasks (BackgroundTasks): FastAPIのバックグラウンドタスク管理オブジェクト。
            db (Session): データベースセッション。

        Returns:
            Dict[str, Any]: 実行結果のレスポンス辞書。`training_id` を含みます。

        Raises:
            ValueError: 既にトレーニングが実行中の場合、または設定が不正な場合。
            Exception: タスクの登録中に予期しないエラーが発生した場合。
        """
        try:
            # 設定の検証
            self._validate_training_config(config)

            training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # 実行開始前に状態を確保して、連打による二重起動を防ぐ
            with training_status_lock:
                if training_status["is_training"]:
                    raise ValueError("既にトレーニングが実行中です")

                training_status.update(
                    {
                        "is_training": True,
                        "progress": 0,
                        "status": "starting",
                        "message": "トレーニングを開始しています...",
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "error": None,
                        "training_id": training_id,
                        "task_id": None,
                    }
                )

            try:
                # バックグラウンドタスクを追加
                background_tasks.add_task(
                    self._train_in_background, config, training_id
                )
            except Exception:
                # タスク登録に失敗した場合は状態を元に戻す
                with training_status_lock:
                    training_status.update(
                        {
                            "is_training": False,
                            "progress": 0,
                            "status": "idle",
                            "message": "待機中",
                            "start_time": None,
                            "end_time": None,
                            "error": None,
                            "training_id": None,
                            "task_id": None,
                        }
                    )
                raise

            return api_response(
                success=True,
                message="MLトレーニングを開始しました",
                data={"training_id": training_id},
            )
        except Exception as e:
            logger.error(f"MLトレーニング開始エラー: {e}")
            raise

    def _validate_training_config(self, config) -> None:
        """
        トレーニング設定の検証（委譲メソッド）

        実際のバリデーションロジックは training_config_validator モジュールに
        移行済み。このメソッドは後方互換性のため残している。

        Args:
            config: MLTrainingConfigオブジェクト

        Raises:
            ValueError: 設定値が不正な場合
        """
        validate_training_config(config)

    async def get_training_status(self) -> Dict[str, Any]:
        """
        トレーニング状態を取得

        現在のMLトレーニングの状態情報を返します。
        進捗率、ステータス、開始・終了時刻、エラー情報などを含みます。

        Returns:
            Dict[str, Any]: トレーニング状態辞書。以下のキーを含みます：
                - is_training: トレーニング中かどうか
                - progress: 進捗率（0-100）
                - status: ステータス文字列（idle, starting, loading_data, training, completed, error, stopped）
                - message: ステータスメッセージ
                - start_time: 開始時刻（ISO8601形式）
                - end_time: 終了時刻（ISO8601形式、完了時のみ）
                - model_info: モデル情報（完了時のみ）
                - error: エラーメッセージ（エラー時のみ）
                - training_id: トレーニングID
                - task_id: タスクID
        """
        with training_status_lock:
            return dict(training_status)

    async def get_model_info(self) -> Dict[str, Any]:
        """
        現在のモデル情報を取得

        最新のトレーニング済みMLモデルの詳細情報を返します。
        モデルタイプ、パフォーマンス指標、特徴量重要度、
        ハイパーパラメータなどを含みます。

        Returns:
            Dict[str, Any]: モデル情報を含むAPIレスポンス辞書。
                以下のキーを含みます：
                - success: 成功フラグ
                - message: メッセージ
                - data: モデル情報辞書（model_status, last_trainingを含む）

        Note:
            モデルが存在しない場合は、デフォルト値を含む情報が返されます。
        """
        try:
            model_info_data = get_latest_model_with_info()
            model_status_base = get_model_info_with_defaults(model_info_data)

            model_status = {
                "is_loaded": model_info_data is not None,
                "model_path": model_info_data["path"] if model_info_data else None,
                **model_status_base,
            }
            return api_response(
                success=True,
                message="MLモデル情報を取得しました",
                data={
                    "model_status": model_status,
                    "last_training": training_status.get("model_info"),
                },
            )
        except Exception as e:
            logger.error(f"MLモデル情報取得エラー: {e}")
            default_status = get_model_info_with_defaults(None)
            default_status.update({"is_loaded": False, "model_path": None})
            return api_response(success=True, data={"model_status": default_status})

    async def stop_training(self) -> Dict[str, Any]:
        """
        トレーニングを停止

        現在実行中のMLトレーニングプロセスを安全に停止します。
        停止処理中は現在のエポックまでの結果が保存されます。

        Returns:
            Dict[str, Any]: 停止処理結果を含むAPIレスポンス辞書。
                以下のキーを含みます：
                - success: 成功フラグ
                - message: メッセージ

        Raises:
            ValueError: 実行中のトレーニングがない場合

        Note:
            停止後、バックグラウンドタスクもクリーンアップされます。
        """
        with training_status_lock:
            if not training_status["is_training"]:
                raise ValueError("実行中のトレーニングがありません")

            training_status.update(
                {
                    "is_training": False,
                    "status": "stopped",
                    "message": "トレーニングが停止されました",
                    "end_time": datetime.now().isoformat(),
                    "training_id": None,
                    "task_id": None,
                }
            )

        background_task_manager.cleanup_all_tasks()
        return api_response(success=True, message="MLトレーニングを停止しました")

    async def _train_in_background(self, config, training_id: str):
        """
        実際のトレーニング処理をバックグラウンドスレッドで実行します。

        このメソッドは以下のトレーニングパイプラインを実行します：
        1. データベースからOHLCV、OI、資金調達率データを取得。
        2. `_determine_trainer_config` を使用してトレーナー設定を確定。
        3. `_execute_actual_training` を呼び出して学習プロセスを開始。
        4. 進捗状況（`training_status`）を随時更新。
        5. 完了または失敗時に状態をリセットし、結果を記録。

        Args:
            config (MLTrainingConfig): トレーニング設定。
            training_id (str): 管理用のトレーニングID。

        Side Effects:
            - グローバルな `training_status` を頻繁に更新します。
            - 学習済みのモデルファイルをローカルストレージに保存します。
            - `background_task_manager` を通じてタスクのライフサイクルを管理します。
        """
        from database.connection import get_session

        db = get_session()
        try:
            with background_task_manager.managed_task(
                task_name=f"MLトレーニング_{training_id}_{config.symbol}_{config.timeframe}",
            ) as task_id:
                try:
                    with training_status_lock:
                        if not training_status["is_training"]:
                            logger.info(
                                "トレーニング停止済みのため、バックグラウンド処理を中断します"
                            )
                            return

                        training_status.update(
                            {
                                "progress": 0,
                                "status": "starting",
                                "message": "トレーニングを開始しています...",
                                "end_time": None,
                                "error": None,
                                "training_id": training_id,
                                "task_id": task_id,
                            }
                        )

                    # データ準備
                    from app.services.backtest.services.backtest_data_service import (
                        BacktestDataService,
                    )

                    data_service = BacktestDataService(
                        OHLCVRepository(db),
                        OpenInterestRepository(db),
                        FundingRateRepository(db),
                    )

                    with training_status_lock:
                        if not training_status["is_training"]:
                            logger.info(
                                "トレーニング停止済みのため、データ読み込み後の更新を中断します"
                            )
                            return

                        training_status.update(
                            {
                                "progress": 10,
                                "status": "loading_data",
                                "message": "データを読み込み中...",
                            }
                        )
                    date_range = parse_datetime_range_optional(
                        config.start_date, config.end_date
                    )
                    if date_range is None:
                        raise ValueError("開始日は終了日より前である必要があります")

                    training_data = data_service.get_ml_training_data(
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        start_date=date_range[0],
                        end_date=date_range[1],
                    )

                    # トレーナー設定の決定
                    trainer_type, ensemble_cfg, single_cfg = (
                        self._determine_trainer_config(config)
                    )

                    # 自分自身のインスタンスまたは新しいインスタンスで学習実行
                    # ここでは設定が異なる可能性があるため、新しいパラメータで自分を再初期化するか、
                    # 別のメソッドで実行する。
                    training_params = self._build_training_params(config)
                    self._execute_actual_training(
                        trainer_type,
                        ensemble_cfg,
                        single_cfg,
                        config,
                        training_data,
                        training_params,
                    )

                except Exception as e:
                    logger.error(
                        f"バックグラウンドトレーニングエラー: {e}", exc_info=True
                    )
                    with training_status_lock:
                        if training_status["status"] == "stopped":
                            logger.info(
                                "停止済みのため、エラー状態への更新をスキップします"
                            )
                            return

                        training_status.update(
                            {
                                "is_training": False,
                                "status": "error",
                                "message": f"エラー: {e}",
                                "end_time": datetime.now().isoformat(),
                                "error": str(e),
                                "training_id": training_id,
                            }
                        )
        finally:
            try:
                db.close()
            except Exception as close_error:
                logger.warning(
                    f"バックグラウンド用DBセッションのクローズに失敗しました: {close_error}"
                )

    def _determine_trainer_config(
        self, config: Any
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        設定からトレーナータイプ等を決定

        APIリクエストの設定から、使用するトレーナータイプ、
        アンサンブル設定、単一モデル設定を決定します。

        Args:
            config: MLTrainingConfigオブジェクト

        Returns:
            Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
                - trainer_type: トレーナータイプ（"single"）
                - ensemble_cfg: アンサンブル設定辞書（現在は常にNone）
                - single_cfg: 単一モデル設定辞書

        Note:
            現在の実装では、常に単一モデルトレーナーが使用されます。
            single_model_configが指定されていない場合は、
            デフォルト設定が使用されます。
        """
        trainer_type = "single"
        ensemble_cfg = None
        single_cfg = (
            config.single_model_config.model_dump()
            if config.single_model_config
            else None
        )
        if not single_cfg:
            single_cfg = get_default_single_model_config()

        return trainer_type, ensemble_cfg, single_cfg

    def _execute_actual_training(
        self,
        trainer_type,
        ensemble_cfg,
        single_cfg,
        config,
        training_data,
        training_params,
    ):
        """
        トレーナーを初期化し、実際の学習・評価プロセスを実行します。

        このメソッドは以下の手順を実行します：
        1. ターゲットとなる `task_type` に基づいて適切なトレーナー（`VolatilityRegressionTrainer` または `EnsembleTrainer`）を初期化。
        2. ハイパーパラメータ最適化設定（`optimization_settings`）を確認。
        3. `train_model` を呼び出して、データの分割、学習、評価、保存を実行。
        4. 学習結果（成功/失敗）に基づいてグローバルトレーニング状態を最終更新。

        Args:
            trainer_type (str): トレーナーの種類（"single" または "ensemble"）。
            ensemble_cfg (Optional[Dict]): アンサンブル学習の設定。
            single_cfg (Optional[Dict]): 単一モデル学習の設定。
            config (MLTrainingConfig): トレーニング全体の実行設定。
            training_data (Any): 学習に使用する生データ（DataFrame または辞書）。
            training_params (Dict): トレーナーに渡す追加の学習パラメータ。

        Returns:
            Dict[str, Any]: 学習結果のサマリー（メトリクス、モデルパス等）。
        """
        # 現在のインスタンスを更新
        self.trainer_type = trainer_type
        self.trainer: Union[VolatilityRegressionTrainer, EnsembleTrainer]
        if config.task_type == "volatility_regression":
            model_type = (single_cfg or {}).get("model_type", "lightgbm")
            self.trainer = VolatilityRegressionTrainer(
                model_type=model_type,
                model_params=single_cfg or {},
            )
        else:
            cfg = self._create_trainer_config(trainer_type, ensemble_cfg, single_cfg)
            self.trainer = EnsembleTrainer(ensemble_config=cast(EnsembleConfig, cfg))

        opt_settings = (
            config.optimization_settings
            if config.optimization_settings and config.optimization_settings.enabled
            else None
        )

        with training_status_lock:
            if not training_status["is_training"]:
                logger.info("トレーニング停止済みのため学習を開始しません")
                return

            training_status.update(
                {
                    "progress": 50,
                    "status": "training",
                    "message": "モデルをトレーニング中...",
                }
            )

        result = self.train_model(
            training_data=training_data,
            save_model=config.save_model,
            optimization_settings=opt_settings,
            **training_params,
        )

        result = ensure_response_dict(result)
        if not result:
            result = {
                "success": False,
                "message": "学習結果の取得に失敗しました",
            }

        if not result.get("success", True):
            error_message = (
                result.get("message")
                or result.get("error")
                or "モデル学習に失敗しました"
            )
            with training_status_lock:
                if (
                    not training_status["is_training"]
                    and training_status["status"] == "stopped"
                ):
                    logger.info(
                        "トレーニング停止済みのため失敗状態への更新をスキップします"
                    )
                    return

                training_status.update(
                    {
                        "is_training": False,
                        "progress": 100,
                        "status": "error",
                        "message": error_message,
                        "end_time": datetime.now().isoformat(),
                        "error": error_message,
                        "model_info": result,
                    }
                )
            logger.error(f"MLトレーニング失敗: {error_message}")
            return

        with training_status_lock:
            if (
                not training_status["is_training"]
                and training_status["status"] == "stopped"
            ):
                logger.info(
                    "トレーニング停止済みのため完了状態への更新をスキップします"
                )
                return

            training_status.update(
                {
                    "is_training": False,
                    "progress": 100,
                    "status": "completed",
                    "message": "トレーニングが完了しました",
                    "end_time": datetime.now().isoformat(),
                    "model_info": result,
                }
            )

    def _build_training_params(self, config: Any) -> Dict[str, Any]:
        """
        APIリクエスト設定を学習用パラメータに変換する

        APIリクエストから受け取った設定を、
        トレーナーのtrain_modelメソッドに渡すためのパラメータ辞書に変換します。

        Args:
            config: MLTrainingConfigオブジェクト

        Returns:
            Dict[str, Any]: トレーナー用パラメータ辞書。
                以下のキーを含みます：
                - test_size: テストデータ分割比率
                - symbol: 取引ペアシンボル
                - timeframe: 時間軸
                - train_test_split: トレーニング/テスト分割比率
                - validation_split: 検証データ分割比率
                - prediction_horizon: 予測期間
                - horizon_n: 予測期間（別名）
                - task_type: タスク種別
                - target_kind: 目的変数種別
                - gate_quantile: ゲート分位点
                - use_cross_validation: クロスバリデーション使用フラグ
                - cv_splits: クロスバリデーション分割数
                - cross_validation_folds: クロスバリデーション分割数（別名）
                - random_state: ランダムシード
                - early_stopping_rounds: 早期停止ラウンド数
                - max_depth: 最大深度
                - n_estimators: 推定器数
                - learning_rate: 学習率

        Note:
            test_sizeはtrain_test_splitとvalidation_splitから計算されます。
        """
        test_size = resolve_holdout_test_size(
            train_test_split=getattr(config, "train_test_split", None),
            validation_split=getattr(config, "validation_split", None),
        )
        cross_validation_folds = config.cross_validation_folds
        gate_quantile = getattr(config, "gate_quantile", 0.67)

        return {
            "test_size": test_size,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "train_test_split": config.train_test_split,
            "validation_split": config.validation_split,
            "prediction_horizon": config.prediction_horizon,
            "horizon_n": config.prediction_horizon,
            "task_type": config.task_type,
            "target_kind": config.target_kind,
            "gate_quantile": gate_quantile,
            "use_cross_validation": cross_validation_folds > 1,
            "cv_splits": cross_validation_folds,
            "cross_validation_folds": cross_validation_folds,
            "random_state": config.random_state,
            "early_stopping_rounds": config.early_stopping_rounds,
            "max_depth": config.max_depth,
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
        }

    # --- コア学習機能 ---

    @safe_ml_operation(context="モデル学習")
    def train_model(
        self,
        training_data: Any,
        save_model: bool = True,
        optimization_settings: Optional[OptimizationSettings] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        モデルの学習、ハイパーパラメータ最適化（オプション）、評価を実行します。

        Args:
            training_data (Any): 学習データ。DataFrame（OHLCVのみ）または
                辞書（"ohlcv", "funding_rate", "open_interest" を含む）。
            save_model (bool): 学習後にモデルファイルを保存するかどうか。デフォルトはTrue。
            optimization_settings (Optional[OptimizationSettings]): ハイパーパラメータ最適化の設定。
                有効な場合、`OptimizationService` を使用して最適なパラメータを探索します。
            test_size (float): ホールドアウト検証用のテストデータ割合。デフォルトは0.2。
            random_state (int): 再現性のための乱数シード。デフォルトは42。
            **kwargs: トレーナーの `train_model` メソッドに渡される追加のパラメータ。

        Returns:
            Dict[str, Any]: 学習結果の辞書。以下のキーを含みます：
                - "metrics": 学習・評価のパフォーマンスメトリクス。
                - "model_path": 保存されたモデルのパス（`save_model=True` の場合）。
                - "best_params": 最適化されたハイパーパラメータ（最適化実行時）。
                - "feature_importance": 特徴量重要度の情報。

        Raises:
            ValueError: サポートされていないターゲット設定や最適化設定が指定された場合。
            Exception: 学習プロセス中に致命的なエラーが発生した場合。
        """
        data_dict = (
            {"ohlcv": training_data}
            if isinstance(training_data, pd.DataFrame)
            else training_data
        )
        ohlcv = data_dict.get("ohlcv")
        fr, oi = data_dict.get("funding_rate"), data_dict.get("open_interest")

        best_params, is_optimized, opt_result = {}, False, None
        if optimization_settings and optimization_settings.enabled:
            task_type = kwargs.get("task_type", self.trainer.config.training.task_type)
            if task_type == "volatility_regression":
                raise ValueError(
                    "volatility_regression では optimization_settings をサポートしていません"
                )
            logger.info("ハイパーパラメータ最適化を開始")
            try:
                opt_result = self.optimization_service.optimize_parameters(
                    trainer=self.trainer,
                    training_data=ohlcv,
                    optimization_settings=optimization_settings,
                    funding_rate_data=fr,
                    open_interest_data=oi,
                )
                best_params = opt_result.get("best_params", {})
                is_optimized = True
            except Exception as e:
                logger.error(f"最適化失敗: {e}")

        training_params = {
            "test_size": test_size,
            "random_state": random_state,
            "optimize_hyperparameters": False,
            **best_params,
            **kwargs,
        }
        result = self.trainer.train_model(
            ohlcv,
            funding_rate_data=fr,
            open_interest_data=oi,
            save_model=save_model,
            **training_params,
        )

        if isinstance(result, dict):
            result["is_optimized"] = is_optimized
            if opt_result:
                result["optimization_result"] = opt_result
        return result

    def generate_forecast(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ予測を生成"""
        return self.trainer.predict_volatility(features_df)

    def load_model(self, model_path: str) -> bool:
        """モデルを読み込む"""
        return self.trainer.load_model(model_path)

    def get_current_model_path(self) -> Optional[str]:
        """読み込まれているモデルのパスを取得"""
        metadata = getattr(self.trainer, "metadata", None)
        if not metadata:
            return None

        model_path = metadata.get("model_path")
        return model_path if isinstance(model_path, str) else None

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """読み込まれているモデルのメタデータを取得"""
        return getattr(self.trainer, "metadata", None)

    @staticmethod
    def get_available_single_models() -> list[str]:
        """利用可能な単一モデルのリストを取得"""
        return ["lightgbm", "xgboost"]

    # --- リソース管理 ---
    def _cleanup_temporary_files(self, level: CleanupLevel):
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        pass

    def _cleanup_models(self, level: CleanupLevel):
        if self.trainer and hasattr(self.trainer, "cleanup_resources"):
            self.trainer.cleanup_resources(level)


# グローバルインスタンス
ml_training_service = MLTrainingService(trainer_type="ensemble")
