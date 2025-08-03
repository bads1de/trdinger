"""
ML オーケストレーター

責任を分離した軽量なMLサービス統合クラス。
特徴量計算、モデル予測、結果の統合を行います。

MLPredictionInterfaceを実装し、統一されたML予測APIを提供します。
学習機能は削除され、予測機能に特化しています。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from app.services.ml.config import ml_config
from app.utils.unified_error_handler import (
    UnifiedErrorHandler,
    MLDataError,
    MLValidationError,
    unified_timeout_decorator,
    unified_operation_context,
)
from app.utils.data_preprocessing import data_preprocessor
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
)
from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.model_manager import model_manager

from app.services.ml.interfaces import MLPredictionInterface
from app.services.backtest.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.connection import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class MLOrchestrator(MLPredictionInterface):
    """
    ML オーケストレーター

    責任を分離した軽量なMLサービス統合クラス。
    特徴量計算、モデル予測、結果の統合を行います。

    MLPredictionInterfaceを実装し、統一されたML予測APIを提供します。
    学習機能は削除され、予測機能に特化しています。
    """

    def __init__(
        self,
        ml_training_service: Optional[MLTrainingService] = None,
        enable_automl: bool = True,
        automl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            ml_training_service: MLTrainingServiceインスタンス（オプション）
            enable_automl: AutoML機能を有効にするか
            automl_config: AutoML設定（辞書形式）
        """
        self.config = ml_config
        self.enable_automl = enable_automl
        self.automl_config = automl_config

        # AutoML機能の有効/無効に応じて特徴量サービスを選択
        if enable_automl:
            # AutoML設定を作成
            if automl_config:
                automl_config_obj = self._create_automl_config_from_dict(automl_config)
            else:
                automl_config_obj = AutoMLConfig.get_financial_optimized_config()

            self.feature_service = EnhancedFeatureEngineeringService(automl_config_obj)
            logger.info("🤖 AutoML特徴量エンジニアリングを有効化しました")
        else:
            self.feature_service = FeatureEngineeringService()
            logger.info("📊 基本特徴量エンジニアリングを使用します")

        self.ml_training_service = (
            ml_training_service
            if ml_training_service
            else MLTrainingService(trainer_type="ensemble")
        )

        # BacktestDataServiceを初期化（完全なデータセット取得用）
        self._backtest_data_service = None
        # モデル状態の統合管理（ml_training_serviceのtrainerを使用）
        self.is_model_loaded = getattr(
            self.ml_training_service.trainer, "is_trained", False
        )
        self._last_predictions = self.config.prediction.get_default_predictions()

        # 既存の学習済みモデルを自動読み込み
        self._try_load_latest_model()

    def _create_automl_config_from_dict(
        self, config_dict: Dict[str, Any]
    ) -> AutoMLConfig:
        """辞書からAutoMLConfigオブジェクトを作成"""
        from app.services.ml.feature_engineering.automl_features.automl_config import (
            TSFreshConfig,
            AutoFeatConfig,
        )

        # TSFresh設定
        tsfresh_dict = config_dict.get("tsfresh", {})
        tsfresh_config = TSFreshConfig(
            enabled=tsfresh_dict.get("enabled", True),
            feature_selection=tsfresh_dict.get("feature_selection", True),
            fdr_level=tsfresh_dict.get("fdr_level", 0.05),
            feature_count_limit=tsfresh_dict.get("feature_count_limit", 100),
            parallel_jobs=tsfresh_dict.get("parallel_jobs", 2),
            performance_mode=tsfresh_dict.get("performance_mode", "balanced"),
        )

        # AutoFeat設定
        autofeat_dict = config_dict.get("autofeat", {})
        autofeat_config = AutoFeatConfig(
            enabled=autofeat_dict.get("enabled", True),
            max_features=autofeat_dict.get("max_features", 50),
            feateng_steps=autofeat_dict.get("feateng_steps", 2),
            max_gb=autofeat_dict.get("max_gb", 1.0),
            generations=autofeat_dict.get("generations", 20),
            population_size=autofeat_dict.get("population_size", 50),
            tournament_size=autofeat_dict.get("tournament_size", 3),
        )

        return AutoMLConfig(
            tsfresh_config=tsfresh_config,
            autofeat_config=autofeat_config,
        )

    def get_backtest_data_service(self, db: Session) -> BacktestDataService:
        """BacktestDataServiceのインスタンスを取得（依存性注入対応）"""
        if self._backtest_data_service is None:
            ohlcv_repo = OHLCVRepository(db)
            fr_repo = FundingRateRepository(db)
            oi_repo = OpenInterestRepository(db)
            self._backtest_data_service = BacktestDataService(
                ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
            )
        return self._backtest_data_service

    @unified_timeout_decorator(
        timeout_seconds=ml_config.data_processing.FEATURE_CALCULATION_TIMEOUT
    )
    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        ML予測確率指標を計算

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            ML指標の辞書 {"ML_UP_PROB": array, "ML_DOWN_PROB": array, "ML_RANGE_PROB": array}
        """
        with unified_operation_context("ML指標計算"):
            try:
                # データサイズ制限
                df = self._limit_data_size(df)

                # カラム名の正規化（検証前に実行）
                df = self._normalize_column_names(df)

                # 入力データの検証（正規化後に実行）
                self._validate_input_data(df)

                # ファンディングレートと建玉残高データが提供されていない場合は自動取得
                if funding_rate_data is None or open_interest_data is None:
                    logger.info("ファンディングレートと建玉残高データを自動取得します")
                    enhanced_df = self._get_enhanced_data_with_fr_oi(df)
                    if enhanced_df is not None:
                        df = enhanced_df
                        # 拡張データから個別のデータフレームを抽出
                        funding_rate_data, open_interest_data = (
                            self._extract_fr_oi_data(df)
                        )

                # 特徴量計算
                features_df = self._calculate_features(
                    df, funding_rate_data, open_interest_data
                )

                # 特徴量計算が失敗した場合はエラーを発生させる
                if features_df is None or features_df.empty:
                    error_msg = (
                        "特徴量計算が失敗しました。MLモデルの予測を実行できません。"
                    )
                    logger.error(error_msg)
                    raise MLDataError(error_msg)

                # ML予測の実行
                predictions = self._safe_ml_prediction(features_df)

                # 予測確率を全データ長に拡張
                ml_indicators = self._expand_predictions_to_data_length(
                    predictions, len(df)
                )

                # 結果の妥当性チェック
                self._validate_ml_indicators(ml_indicators)

                return ml_indicators

            except (MLDataError, MLValidationError) as e:
                logger.error(f"ML指標計算で検証エラー: {e}")
                raise  # エラーを再発生させて処理を停止
            except Exception as e:
                logger.error(f"ML指標計算で予期しないエラー: {e}")
                raise MLDataError(
                    f"ML指標計算で予期しないエラーが発生しました: {e}"
                ) from e

    def calculate_single_ml_indicator(
        self,
        indicator_type: str,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        単一のML指標を計算

        Args:
            indicator_type: 指標タイプ（ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB）
            df: OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            指標値の配列
        """
        # データフレームの基本検証
        if df is None or df.empty:
            error_msg = f"空のデータフレームが提供されました: {indicator_type}"
            logger.error(error_msg)
            raise MLDataError(error_msg)

        # ML指標を計算（エラー時は例外が発生）
        ml_indicators = self.calculate_ml_indicators(
            df, funding_rate_data, open_interest_data
        )

        if indicator_type in ml_indicators:
            return ml_indicators[indicator_type]
        else:
            error_msg = f"未知のML指標タイプ: {indicator_type}"
            logger.error(error_msg)
            raise MLValidationError(error_msg)

    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            success = self.ml_training_service.load_model(model_path)
            if success:
                self.is_model_loaded = True
                logger.info(f"MLモデル読み込み成功: {model_path}")
            else:
                logger.warning(f"MLモデル読み込み失敗: {model_path}")
            return success

        except Exception as e:
            logger.error(f"MLモデル読み込みエラー: {e}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """
        モデルの状態を取得

        Returns:
            モデル状態の辞書
        """

        status = {
            "is_model_loaded": self.is_model_loaded,
            "is_trained": getattr(
                self.ml_training_service.trainer, "is_trained", False
            ),
            "last_predictions": self._last_predictions,
            "feature_count": (
                len(self.ml_training_service.trainer.feature_columns)
                if self.ml_training_service.trainer.feature_columns
                else 0
            ),
        }

        # 最新のモデルファイルから性能指標を取得
        try:
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                # ModelManagerから直接メタデータを取得
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    # 新しい形式の性能指標を抽出（全ての評価指標を含む）
                    performance_metrics = {
                        # 基本指標
                        "accuracy": metadata.get("accuracy", 0.0),
                        "precision": metadata.get("precision", 0.0),
                        "recall": metadata.get("recall", 0.0),
                        "f1_score": metadata.get("f1_score", 0.0),
                        # AUC指標
                        "auc_roc": metadata.get("auc_roc", 0.0),
                        "auc_pr": metadata.get("auc_pr", 0.0),
                        # 高度な指標
                        "balanced_accuracy": metadata.get("balanced_accuracy", 0.0),
                        "matthews_corrcoef": metadata.get("matthews_corrcoef", 0.0),
                        "cohen_kappa": metadata.get("cohen_kappa", 0.0),
                        # 専門指標
                        "specificity": metadata.get("specificity", 0.0),
                        "sensitivity": metadata.get("sensitivity", 0.0),
                        "npv": metadata.get("npv", 0.0),
                        "ppv": metadata.get("ppv", 0.0),
                        # 確率指標
                        "log_loss": metadata.get("log_loss", 0.0),
                        "brier_score": metadata.get("brier_score", 0.0),
                        # その他
                        "loss": metadata.get("loss", 0.0),
                        "val_accuracy": metadata.get("val_accuracy", 0.0),
                        "val_loss": metadata.get("val_loss", 0.0),
                        "training_time": metadata.get("training_time", 0.0),
                    }
                    status["performance_metrics"] = performance_metrics
                else:
                    pass
            else:
                pass
        except Exception as e:
            logger.error(f"性能指標取得エラー: {e}")

        return status

    def update_predictions(self, predictions: Dict[str, float]):
        """
        予測値を更新（外部から設定する場合）

        Args:
            predictions: 予測確率の辞書
        """
        try:
            UnifiedErrorHandler.validate_predictions(predictions)
            self._last_predictions = predictions
        except MLValidationError as e:
            logger.warning(f"無効な予測値形式: {e}")

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        try:
            logger.info(f"特徴量重要度取得開始: top_n={top_n}")

            # 1. 現在読み込まれているモデルから取得を試行
            if self.is_model_loaded and getattr(
                self.ml_training_service.trainer, "is_trained", False
            ):
                logger.info(
                    f"現在読み込まれているモデルから特徴量重要度を取得: trainer_type={type(self.ml_training_service.trainer).__name__}"
                )
                feature_importance = self.ml_training_service.get_feature_importance()
                if feature_importance:
                    logger.info(
                        f"現在のモデルから特徴量重要度を取得: {len(feature_importance)}個"
                    )
                    # 上位N個を取得
                    sorted_importance = sorted(
                        feature_importance.items(), key=lambda x: x[1], reverse=True
                    )[:top_n]
                    return dict(sorted_importance)
                else:
                    logger.warning("現在のモデルから特徴量重要度を取得できませんでした")

            # 2. 最新のモデルファイルから特徴量重要度を取得
            from ...ml.model_manager import model_manager

            logger.info("最新のモデルファイルから特徴量重要度を取得を試行")
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                logger.info(f"最新モデルファイル: {latest_model}")
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    feature_importance = metadata.get("feature_importance", {})
                    logger.info(
                        f"メタデータから特徴量重要度を確認: {len(feature_importance)}個"
                    )

                    if feature_importance:
                        # 上位N個を取得
                        sorted_importance = sorted(
                            feature_importance.items(), key=lambda x: x[1], reverse=True
                        )[:top_n]
                        logger.info(
                            f"メタデータから特徴量重要度を取得: {len(sorted_importance)}個"
                        )
                        return dict(sorted_importance)
                    else:
                        logger.warning("メタデータに特徴量重要度が含まれていません")
                else:
                    logger.warning("モデルデータまたはメタデータが見つかりません")
            else:
                logger.warning("最新のモデルファイルが見つかりません")

            logger.warning("特徴量重要度データが見つかりません")
            return {}

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}

    def predict_probabilities(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        特徴量から予測確率を計算（MLServiceInterface実装）

        Args:
            features: 特徴量データ

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        return self._safe_ml_prediction(features)

    def _validate_input_data(self, df: pd.DataFrame):
        """入力データの検証"""
        required_columns = ["open", "high", "low", "close", "volume"]
        UnifiedErrorHandler.validate_dataframe(
            df, required_columns=required_columns, min_rows=1
        )

    def _limit_data_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """データサイズ制限を無効化（制限なしでデータをそのまま返す）"""
        # 制限を外したため、データをそのまま返す
        logger.info(f"データサイズ: {len(df)}行（制限なし）")
        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名を正規化"""
        # 必要なカラムの存在確認（大文字・小文字両方に対応）
        required_columns_lower = ["open", "high", "low", "close", "volume"]
        required_columns_upper = ["Open", "High", "Low", "Close", "Volume"]

        # 小文字のカラムが存在するかチェック
        missing_lower = [col for col in required_columns_lower if col not in df.columns]
        # 大文字のカラムが存在するかチェック
        missing_upper = [col for col in required_columns_upper if col not in df.columns]

        # どちらかのセットが完全に存在すればOK
        if len(missing_lower) == 0:
            # 小文字のカラムが揃っている
            return df.copy()
        elif len(missing_upper) == 0:
            # 大文字のカラムが揃っている場合、小文字に変換
            df_normalized = df.copy()
            df_normalized.columns = [
                col.lower() if col in required_columns_upper else col
                for col in df_normalized.columns
            ]
            return df_normalized
        else:
            raise MLDataError(
                f"必要なカラムが不足: {missing_lower} (小文字) または {missing_upper} (大文字)"
            )

    def _calculate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """特徴量を計算（AutoML統合版）"""
        try:

            # 特徴量計算用にカラム名を大文字に変換
            df_for_features = df.copy()
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }

            # カラム名を大文字に変換
            df_for_features.columns = [
                column_mapping.get(col, col) for col in df_for_features.columns
            ]

            # AutoML機能が有効な場合は拡張特徴量計算を実行
            if self.enable_automl and isinstance(
                self.feature_service, EnhancedFeatureEngineeringService
            ):
                logger.info("🤖 AutoML拡張特徴量計算を実行中...")

                # ターゲット変数を計算（AutoML特徴量生成用）
                target = self._calculate_target_for_automl(df_for_features)

                features_df = self.feature_service.calculate_enhanced_features(
                    ohlcv_data=df_for_features,
                    funding_rate_data=funding_rate_data,
                    open_interest_data=open_interest_data,
                    automl_config=self.automl_config,
                    target=target,
                )
            else:
                logger.info("📊 基本特徴量計算を実行中...")
                features_df = self.feature_service.calculate_advanced_features(
                    df_for_features, funding_rate_data, open_interest_data
                )
            if features_df is not None:
                logger.info(f"特徴量計算結果: {len(features_df)}行")
            else:
                logger.warning("特徴量計算結果がNone")

            return features_df
        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            return None

    def _calculate_target_for_automl(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """AutoML特徴量生成用のターゲット変数を計算"""
        try:
            if df is None or df.empty or "Close" not in df.columns:
                logger.warning("ターゲット変数計算用のデータが不足しています")
                return None

            # 価格変化率をターゲット変数として使用
            close_prices = df["Close"]
            price_change = close_prices.pct_change()

            # 将来の価格変化を予測するため、1期間先にシフト
            target = price_change.shift(-1)

            # 統計的手法で欠損値を補完
            target_df = pd.DataFrame({"target": target})
            target_df = data_preprocessor.transform_missing_values(
                target_df, strategy="median"
            )
            target = target_df["target"]

            return target

        except Exception as e:
            logger.error(f"ターゲット変数計算エラー: {e}")
            return None

    def set_automl_enabled(
        self, enabled: bool, automl_config: Optional[Dict[str, Any]] = None
    ):
        """AutoML機能の有効/無効を設定"""
        try:
            self.enable_automl = enabled
            self.automl_config = automl_config

            if enabled:
                # AutoML設定を作成
                if automl_config:
                    automl_config_obj = self._create_automl_config_from_dict(
                        automl_config
                    )
                else:
                    automl_config_obj = AutoMLConfig.get_financial_optimized_config()

                # 既存のサービスをクリーンアップ
                if hasattr(self.feature_service, "cleanup_resources"):
                    self.feature_service.cleanup_resources()

                self.feature_service = EnhancedFeatureEngineeringService(
                    automl_config_obj
                )
                logger.info("🤖 AutoML特徴量エンジニアリングを有効化しました")
            else:
                # 既存のサービスをクリーンアップ
                if hasattr(self.feature_service, "cleanup_resources"):
                    self.feature_service.cleanup_resources()

                self.feature_service = FeatureEngineeringService()
                logger.info("📊 基本特徴量エンジニアリングに切り替えました")

        except Exception as e:
            logger.error(f"AutoML設定変更エラー: {e}")
            raise

    def get_automl_status(self) -> Dict[str, Any]:
        """AutoML機能の状態を取得"""
        return {
            "enabled": self.enable_automl,
            "service_type": type(self.feature_service).__name__,
            "config": self.automl_config,
            "available_features": (
                self.feature_service.get_available_automl_features()
                if hasattr(self.feature_service, "get_available_automl_features")
                else {}
            ),
        }

    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """厳格なML予測実行（エラー時はデフォルト値を返さない）"""
        try:
            # 予測を実行
            predictions = self.ml_training_service.generate_signals(features_df)

            # 予測値の妥当性チェック
            UnifiedErrorHandler.validate_predictions(predictions)
            self._last_predictions = predictions
            return predictions
        except Exception as e:
            error_msg = f"ML予測でエラーが発生しました: {e}"
            logger.error(error_msg)
            raise MLDataError(error_msg) from e

    def _expand_predictions_to_data_length(
        self, predictions: Dict[str, float], data_length: int
    ) -> Dict[str, np.ndarray]:
        """予測値をデータ長に拡張"""
        try:
            result = {
                "ML_UP_PROB": np.full(data_length, predictions["up"]),
                "ML_DOWN_PROB": np.full(data_length, predictions["down"]),
                "ML_RANGE_PROB": np.full(data_length, predictions["range"]),
            }
            return result
        except Exception as e:
            logger.error(f"予測値拡張エラー: {e}")
            default_result = self._get_default_indicators(data_length)
            return default_result

    def _validate_ml_indicators(self, ml_indicators: Dict[str, np.ndarray]):
        """ML指標の妥当性を検証"""
        required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        # 必要な指標が存在するか
        missing_indicators = [
            ind for ind in required_indicators if ind not in ml_indicators
        ]
        if missing_indicators:
            raise MLValidationError(f"必要なML指標が不足: {missing_indicators}")

        # 各指標の妥当性をチェック
        for indicator, values in ml_indicators.items():
            if not isinstance(values, np.ndarray):
                raise MLValidationError(f"ML指標が配列ではありません: {indicator}")
            if len(values) == 0:
                raise MLValidationError(f"ML指標が空です: {indicator}")
            if not np.all((values >= 0) & (values <= 1)):
                raise MLValidationError(f"ML指標の値が範囲外です: {indicator}")
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                raise MLValidationError(
                    f"ML指標に無効な値が含まれています: {indicator}"
                )

    def _get_default_indicators(self, data_length: int) -> Dict[str, np.ndarray]:
        """
        デフォルトのML指標を取得（非推奨）

        注意: エラーハンドリングの厳格化により、このメソッドは使用されなくなりました。
        エラー時はデフォルト値を返すのではなく、例外を発生させるべきです。
        """
        logger.warning(
            "_get_default_indicators は非推奨です。エラー時は例外を発生させてください。"
        )
        config = self.config.prediction
        return {
            "ML_UP_PROB": np.full(data_length, config.DEFAULT_UP_PROB),
            "ML_DOWN_PROB": np.full(data_length, config.DEFAULT_DOWN_PROB),
            "ML_RANGE_PROB": np.full(data_length, config.DEFAULT_RANGE_PROB),
        }

    def _try_load_latest_model(self) -> bool:
        """最新の学習済みモデルを自動読み込み"""
        try:
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                success = self.ml_training_service.load_model(latest_model)
                if success:
                    self.is_model_loaded = True
                    logger.info(f"最新のMLモデルを自動読み込み: {latest_model}")
                    return True
                else:
                    logger.warning(f"MLモデルの読み込みに失敗: {latest_model}")
            else:
                logger.info(
                    "学習済みMLモデルが見つかりません。ML機能はデフォルト値で動作します。"
                )
            return False

        except Exception as e:
            logger.warning(f"MLモデルの自動読み込み中にエラー: {e}")
            return False

    def _get_enhanced_data_with_fr_oi(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        ファンディングレートと建玉残高データを含む拡張データを取得

        Args:
            df: 元のOHLCVデータ

        Returns:
            拡張されたデータフレーム（FR/OIデータを含む）
        """
        try:
            # データフレームのインデックスから時間範囲を取得
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("データフレームが空か、DatetimeIndexではありません")
                return None

            start_date = df.index.min()
            end_date = df.index.max()

            # シンボルとタイムフレームを動的に推定
            symbol = self._infer_symbol_from_data(df)
            timeframe = self._infer_timeframe_from_data(df)

            logger.info(f"推定されたシンボル: {symbol}, タイムフレーム: {timeframe}")

            # BacktestDataServiceを使用して拡張データを取得
            # 注意: この部分は依存性注入が必要ですが、現在のアーキテクチャでは
            # セッションを直接取得する必要があります
            db = next(get_db())
            try:
                backtest_service = self.get_backtest_data_service(db)
                enhanced_df = backtest_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
            finally:
                db.close()

            if enhanced_df is not None and not enhanced_df.empty:
                logger.info(
                    f"拡張データ取得成功: {len(enhanced_df)}行, カラム: {list(enhanced_df.columns)}"
                )
                return enhanced_df
            else:
                logger.warning("拡張データの取得に失敗しました")
                return None

        except Exception as e:
            logger.error(f"拡張データ取得エラー: {e}")
            return None

    def _extract_fr_oi_data(
        self, enhanced_df: pd.DataFrame
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        拡張データフレームからファンディングレートと建玉残高データを抽出

        Args:
            enhanced_df: 拡張されたデータフレーム

        Returns:
            (funding_rate_data, open_interest_data)のタプル
        """
        try:
            funding_rate_data = None
            open_interest_data = None

            # ファンディングレートデータを抽出
            if "funding_rate" in enhanced_df.columns:
                funding_rate_data = enhanced_df[["funding_rate"]].copy()

            # 建玉残高データを抽出
            if "open_interest" in enhanced_df.columns:
                open_interest_data = enhanced_df[["open_interest"]].copy()

            return funding_rate_data, open_interest_data

        except Exception as e:
            logger.error(f"FR/OIデータ抽出エラー: {e}")
            return None, None

    def _infer_symbol_from_data(self, df: pd.DataFrame) -> str:
        """
        データフレームからシンボルを推定

        Args:
            df: OHLCVデータフレーム

        Returns:
            推定されたシンボル
        """
        try:
            # データフレームのメタデータからシンボルを取得を試行
            if hasattr(df, "attrs") and "symbol" in df.attrs:
                return df.attrs["symbol"]

            # カラム名からシンボルを推定
            if hasattr(df, "columns"):
                for col in df.columns:
                    if isinstance(col, str) and (
                        "BTC" in col.upper() or "ETH" in col.upper()
                    ):
                        if "BTC" in col.upper():
                            return "BTC/USDT:USDT"
                        elif "ETH" in col.upper():
                            return "ETH/USDT:USDT"

            # 価格レンジからシンボルを推定（BTCは通常高価格）
            if "Close" in df.columns and not df["Close"].empty:
                avg_price = df["Close"].mean()
                if avg_price > 10000:  # BTCの価格レンジ
                    return "BTC/USDT:USDT"
                elif avg_price > 1000:  # ETHの価格レンジ
                    return "ETH/USDT:USDT"

            # デフォルトはBTC
            logger.info(
                "シンボルを推定できませんでした。デフォルトのBTC/USDT:USDTを使用します"
            )
            return "BTC/USDT:USDT"

        except Exception as e:
            logger.warning(f"シンボル推定エラー: {e}")
            return "BTC/USDT:USDT"

    def _infer_timeframe_from_data(self, df: pd.DataFrame) -> str:
        """
        データフレームからタイムフレームを推定

        Args:
            df: OHLCVデータフレーム

        Returns:
            推定されたタイムフレーム
        """
        try:
            # データフレームのメタデータからタイムフレームを取得を試行
            if hasattr(df, "attrs") and "timeframe" in df.attrs:
                return df.attrs["timeframe"]

            # インデックスの時間間隔から推定
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                # 最初の数個の時間差を計算
                time_diffs = []
                for i in range(1, min(6, len(df.index))):
                    diff = df.index[i] - df.index[i - 1]
                    time_diffs.append(diff.total_seconds() / 60)  # 分単位

                if time_diffs:
                    avg_diff_minutes = sum(time_diffs) / len(time_diffs)

                    # 時間間隔に基づいてタイムフレームを判定
                    if abs(avg_diff_minutes - 1) < 0.5:
                        return "1m"
                    elif abs(avg_diff_minutes - 5) < 2:
                        return "5m"
                    elif abs(avg_diff_minutes - 15) < 5:
                        return "15m"
                    elif abs(avg_diff_minutes - 30) < 10:
                        return "30m"
                    elif abs(avg_diff_minutes - 60) < 15:
                        return "1h"
                    elif abs(avg_diff_minutes - 240) < 30:
                        return "4h"
                    elif abs(avg_diff_minutes - 1440) < 60:
                        return "1d"

            # デフォルトは1時間
            logger.info(
                "タイムフレームを推定できませんでした。デフォルトの1hを使用します"
            )
            return "1h"

        except Exception as e:
            logger.warning(f"タイムフレーム推定エラー: {e}")
            return "1h"


# グローバルインスタンス（デフォルトでAutoML有効）
ml_orchestrator = MLOrchestrator(enable_automl=True)
