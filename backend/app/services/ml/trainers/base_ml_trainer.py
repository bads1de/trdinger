"""
ML学習基盤クラス

学習・評価・前処理・保存に関わる共通ロジックを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ....config.unified_config import unified_config
from ....utils.error_handler import (
    DataError,
    ml_operation_context,
    safe_ml_operation,
)
from ..common.base_resource_manager import BaseResourceManager, CleanupLevel
from ..common.utils import (
    get_feature_importance_unified,
    prepare_data_for_prediction,
    get_t1_series,
)
from ..cross_validation import PurgedKFold
from ..common.exceptions import MLModelError
from ..feature_engineering.feature_engineering_service import FeatureEngineeringService
from ..label_generation.label_generation_service import LabelGenerationService
from ..common.registry import ModelMetadata
from ..models.model_manager import model_manager

logger = logging.getLogger(__name__)


class BaseMLTrainer(BaseResourceManager, ABC):
    """
    ML学習基盤クラス

    共通の学習ロジックを提供し、具体的な実装は継承クラスで行います。
    単一責任原則に従い、学習に関する責任のみを持ちます。
    """

    def __init__(
        self,
        trainer_config: Optional[Dict[str, Any]] = None,
        trainer_type: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """
        初期化

        Args:
            trainer_config: トレーナー設定
            trainer_type: トレーナータイプ（互換性のため維持）
            model_type: モデルタイプ（互換性のため維持）
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.feature_service = FeatureEngineeringService()
        self.label_service = LabelGenerationService()
        logger.debug(
            "特徴量エンジニアリングサービスとラベル生成サービスを初期化しました"
        )

        self.trainer_config = trainer_config or {}

        # 以下のプロパティはサブクラスで使用される可能性があるため保持（互換性）
        self.trainer_type = trainer_type or self.trainer_config.get("type", "single")
        self.model_type = model_type or self.trainer_config.get(
            "model_type", "lightgbm"
        )

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self._model = None
        self.current_model_path = None
        self.current_model_metadata = None

    @property
    def config(self):
        """現在の統一ML設定を取得"""
        return unified_config.ml

    @property
    def model(self) -> Any:
        """学習済みモデルを取得"""
        return self._model

    @safe_ml_operation(
        default_return={"success": False}, context="MLモデル学習でエラーが発生しました"
    )
    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習（テンプレートメソッド）

        データ準備、特徴量計算、クロスバリデーションまたはホールドアウト分割、
        モデル学習、およびオプションでのモデル保存を一連のフローとして実行します。

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: 学習後にモデルを永続化するかどうか
            model_name: 保存時のモデル名（Noneの場合はデフォルト名）
            **training_params:
                - use_cross_validation (bool): CVを実行するか
                - test_size (float): ホールドアウト分割比率
                - cv_splits (int): CV分割数
                - random_state (int): 乱数シード

        Returns:
            学習結果、評価メトリクス、モデルパス等を含む辞書
        """
        with ml_operation_context("MLモデル学習"):
            if training_data is None or len(training_data) < 100:
                raise DataError("学習データが不足しています")

            # 1. 特徴量計算とデータ準備
            X, y = self._prepare_training_data(
                self._calculate_features(
                    training_data, funding_rate_data, open_interest_data
                ),
                training_data,
                **training_params,
            )
            if X is None or X.empty:
                raise DataError("学習データが空です")

            # 2. 学習実行 (CV or Single)
            if training_params.get("use_cross_validation", False):
                cv_res = self._time_series_cross_validate(X, y, **training_params)
                # 全データで最終学習
                X_s = self._preprocess_data(X, X)[0]
                idx = int(len(X) * (1 - training_params.get("test_size", 0.2)))
                res = self._train_model_impl(
                    X_s.iloc[:idx],
                    X_s.iloc[idx:],
                    y.iloc[:idx],
                    y.iloc[idx:],
                    **training_params,
                )
                res.update(cv_res)
            else:
                X_tr, X_te, y_tr, y_te = self._split_data(X, y, **training_params)
                X_tr_s, X_te_s = self._preprocess_data(X_tr, X_te)
                res = self._train_model_impl(
                    X_tr_s, X_te_s, y_tr, y_te, **training_params
                )

            self.is_trained = True

            # 3. モデル保存
            if save_model:
                meta = ModelMetadata.from_training_result(
                    res,
                    training_params,
                    self.__class__.__name__,
                    len(self.feature_columns or []),
                )
                if not meta.validate()["is_valid"]:
                    logger.warning(f"メタデータ警告: {meta.validate()['warnings']}")

                path = self.save_model(
                    model_name or self.config.model.auto_strategy_model_name,
                    meta.to_dict(),
                )
                res["model_path"] = self.current_model_path = path
                self.current_model_metadata = meta.to_dict()

            return self._format_training_result(res, X, y)

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        予測を実行（抽象メソッド）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """

    def predict_signal(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        最新の特徴量データからシグナル（有効確率）を予測

        入力データの前処理、期待される形式への変換、モデル推論を行い、
        最終的な「エントリーが有効である確率」を返します。

        Args:
            features_df: 特徴量DataFrame（生データ）

        Returns:
            予測確率の辞書:
            - {"is_valid": float}: エントリーが有効である期待確率 (0.0〜1.0)
        """
        if not self.is_trained:
            logger.warning("学習済みモデルがありません")
            return self.config.prediction.get_default_predictions()

        try:
            # 1. 前処理（カラム調整、スケーリング）- 共通ユーティリティを直接使用
            processed_features = prepare_data_for_prediction(
                features_df,
                expected_columns=self.feature_columns,
                scaler=self.scaler,
            )

            # 2. 予測実行（サブクラスのpredictを呼び出し）
            # predictは確率配列を返すことを期待
            predictions = self.predict(processed_features)

            # 3. 最新の予測結果を取得（時系列データの場合は最後の行）
            if predictions.ndim == 2:
                latest_pred = predictions[-1]
            else:
                latest_pred = predictions

            # 4. 結果の整形（二値分類専用）
            if latest_pred.shape[0] >= 2:
                # 2クラス分類
                return {
                    "is_valid": float(latest_pred[1]),
                }
            else:
                logger.error(f"予期しないクラス数: {latest_pred.shape[0]}")
                return self.config.prediction.get_default_predictions()

        except Exception as e:
            logger.error(f"シグナル予測エラー: {e}")
            return self.config.prediction.get_default_predictions()

    @abstractmethod
    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        モデル学習の実装（抽象メソッド）
        """

    def _calculate_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """入力データから特徴量集合を計算"""
        try:
            if ohlcv_data is None or ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            profile = unified_config.ml.feature_engineering.profile
            logger.info(f"📊 特徴量計算を実行中（profile: {profile}）...")

            basic_features = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                profile=profile,
            )

            logger.info(f"✅ 特徴量生成完了: {len(basic_features.columns)}個の特徴量")
            return basic_features

        except Exception as e:
            logger.warning(f"特徴量計算でエラー、基本特徴量のみ使用: {e}")
            return ohlcv_data.copy()

    def _prepare_training_data(
        self, features_df: pd.DataFrame, ohlcv_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """学習用データを準備"""
        try:
            features_clean, labels_numeric = self.label_service.prepare_labels(
                features_df, ohlcv_df, **training_params
            )

            self.feature_columns = features_clean.columns.tolist()
            return features_clean, labels_numeric

        except Exception as e:
            logger.error(f"学習データ準備エラー: {e}")
            raise DataError(f"学習データの準備に失敗しました: {e}")

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データを分割（常に時系列分割）"""
        test_size = training_params.get("test_size", 0.2)

        logger.info("🕒 時系列分割を使用")
        n_samples = len(X)
        train_size = int(n_samples * (1 - test_size))

        X_train = X.iloc[:train_size].copy()
        X_test = X.iloc[train_size:].copy()
        y_train = y.iloc[:train_size].copy()
        y_test = y.iloc[train_size:].copy()

        return X_train, X_test, y_train, y_test

    def _time_series_cross_validate(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Dict[str, Any]:
        """時間軸を考慮したパージング・エンバーゴ付きクロスバリデーションを実行"""
        n_splits = training_params.get("cv_splits", self.config.training.cv_folds)
        logger.info(f"🔄 時系列クロスバリデーション開始（{n_splits}分割）")

        t1_horizon_n = self.config.training.prediction_horizon

        t1 = get_t1_series(
            X.index,
            t1_horizon_n,
            timeframe=self.config.training.label_generation.timeframe,
        )

        pct_embargo = getattr(self.config.training, "pct_embargo", 0.01)
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_cv),
                columns=X_train_cv.columns,
                index=X_train_cv.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_cv),
                columns=X_test_cv.columns,
                index=X_test_cv.index,
            )

            fold_result = self._train_fold_with_error_handling(
                fold + 1,
                X_train_scaled,
                X_test_scaled,
                y_train_cv,
                y_test_cv,
                X_train_cv,
                X_test_cv,
                training_params,
            )

            fold_results.append(fold_result)

            score = fold_result.get(
                "balanced_accuracy", fold_result.get("accuracy", 0.0)
            )
            cv_scores.append(score)

        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        std_score = np.std(cv_scores) if cv_scores else 0.0

        logger.info(
            f"✅ クロスバリデーション完了: 平均スコア={mean_score:.4f} (+/- {std_score:.4f})"
        )

        return {
            "cv_scores": cv_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_results": fold_results,
        }

    def _preprocess_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データを前処理（スケーリング）"""
        if self.scaler is None:
            self.scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled

    def _get_model_to_save(self) -> Any:
        """保存対象のモデルオブジェクトを取得"""
        return self._model

    def _get_model_specific_metadata(self, model_name: str) -> Dict[str, Any]:
        """モデル固有のメタデータを取得"""
        return {}

    def save_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """学習済みモデルを永続化"""
        if not self.is_trained:
            raise MLModelError("学習済みモデルがありません")

        final_metadata = {
            "model_type": self.__class__.__name__,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "is_trained": self.is_trained,
        }
        if metadata:
            final_metadata.update(metadata)

        final_metadata.update(self._get_model_specific_metadata(model_name))

        try:
            feature_importance = self.get_feature_importance(top_n=100)
            if feature_importance:
                final_metadata["feature_importance"] = feature_importance
        except Exception as e:
            logger.warning(f"特徴量重要度の取得に失敗: {e}")

        model_to_save = self._get_model_to_save()
        if model_to_save is None:
            logger.warning("保存対象モデルがNoneです。トレーナー自体を保存します。")
            model_to_save = self

        model_path = model_manager.save_model(
            model=model_to_save,
            model_name=model_name,
            metadata=final_metadata,
            scaler=self.scaler,
            feature_columns=self.feature_columns,
        )

        logger.info(f"モデル保存完了: {model_path}")
        return model_path

    def _format_training_result(
        self, training_result: Dict[str, Any], X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """学習結果を整形"""
        result = {
            "success": True,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "total_samples": len(X),
            **training_result,
        }
        return result

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained:
            logger.warning("学習済みモデルがありません")
            return {}

        return get_feature_importance_unified(
            self._model, self.feature_columns, top_n=top_n
        )

    @safe_ml_operation(
        default_return=False, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> bool:
        """モデルを読み込み"""
        model_data = model_manager.load_model(model_path)

        if model_data is None:
            return False

        self._model = model_data.get("model")
        self.scaler = model_data.get("scaler")
        self.feature_columns = model_data.get("feature_columns")

        if self._model is None:
            raise MLModelError("モデルデータにモデルが含まれていません")

        self.is_trained = True
        self.current_model_path = model_path
        self.current_model_metadata = model_data.get("metadata", {})
        return True

    def _cleanup_temporary_files(self, level: CleanupLevel):
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        pass

    def _cleanup_models(self, level: CleanupLevel):
        try:
            if self.feature_service is not None:
                if hasattr(self.feature_service, "cleanup_resources"):
                    self.feature_service.cleanup_resources()

            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
            self.current_model_path = None
            self.current_model_metadata = None
        except Exception as e:
            logger.warning(f"モデルクリーンアップ警告: {e}")
            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
            self.current_model_path = None
            self.current_model_metadata = None

    @safe_ml_operation(
        default_return={
            "fold": 0,
            "error": "フォールド学習でエラーが発生しました",
            "accuracy": 0.0,
        },
        context="フォールド学習",
    )
    def _train_fold_with_error_handling(
        self,
        fold: int,
        X_train_scaled: pd.DataFrame,
        X_test_scaled: pd.DataFrame,
        y_train_cv: pd.Series,
        y_test_cv: pd.Series,
        X_train_cv: pd.DataFrame,
        X_test_cv: pd.DataFrame,
        training_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """エラーハンドリング付きフォールド学習"""
        fold_result = self._train_model_impl(
            X_train_scaled,
            X_test_scaled,
            y_train_cv,
            y_test_cv,
            **training_params,
        )

        fold_result.update(
            {
                "fold": fold,
                "train_samples": len(X_train_cv),
                "test_samples": len(X_test_cv),
                "train_period": f"{X_train_cv.index[0]} ～ {X_train_cv.index[-1]}",
                "test_period": f"{X_test_cv.index[0]} ～ {X_test_cv.index[-1]}",
            }
        )
        return fold_result