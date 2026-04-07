"""
ML学習基盤クラス

学習・評価・前処理・保存に関わる共通ロジックを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ....utils.error_handler import (
    DataError,
    ml_operation_context,
    safe_ml_operation,
)
from .. import cross_validation as cross_validation_module
from ..common.base_resource_manager import BaseResourceManager, CleanupLevel
from ..common.config import ml_config_manager
from ..common.exceptions import MLModelError
from ..common.registry import ModelMetadata
from ..common.training_utils import resolve_holdout_test_size
from ..common.utils import (
    get_feature_importance_unified,
    prepare_data_for_prediction,
)
from ..cross_validation import PurgedKFold
from ..feature_engineering.feature_engineering_service import FeatureEngineeringService
from ..feature_selection.feature_selector import FeatureSelector
from ..label_generation.label_generation_service import LabelGenerationService
from ..models.model_manager import model_manager
from ..targets.volatility_target_service import VolatilityTargetService

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
    ):
        """
        初期化

        Args:
            trainer_config: トレーナー設定
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.feature_service = FeatureEngineeringService()
        self.label_service = LabelGenerationService()
        self.target_service = VolatilityTargetService()

        # 特徴量選択器の初期化（動的ノイズ除去設定）
        # 新しい sklearn 互換 API を使用
        self.feature_selector = FeatureSelector(
            method="staged",  # 段階的選択（Filter → Wrapper → Embedded）
            target_k=None,  # 数による制限を廃止
            cumulative_importance=0.95,  # 予測力の95%を維持
            min_relative_importance=0.02,  # 寄与が低すぎるノイズをカット
            correlation_threshold=0.85,  # 冗長性排除
            min_features=10,  # 最低限確保する特徴量数
            cv_folds=3,  # クロスバリデーションフォールド数
            n_jobs=1,  # 並列処理（メモリ安全のため1）
        )

        logger.debug(
            "特徴量エンジニアリング、ラベル生成、特徴量選択サービスを初期化しました"
        )

        self.trainer_config = trainer_config or {}

        self.scaler: Optional[StandardScaler] = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.is_trained: bool = False
        self._model: Any = None
        self.current_model_path: Optional[str] = None
        self.current_model_metadata: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None

    @property
    def config(self):
        """現在のML設定を取得"""
        return ml_config_manager.config

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

        データ準備、特徴量計算、特徴量選択、クロスバリデーションまたはホールドアウト分割、
        モデル学習、およびオプションでのモデル保存を一連のフローとして実行します。
        """
        with ml_operation_context("MLモデル学習"):
            if training_data is None or len(training_data) < 100:
                raise DataError("学習データが不足しています")

            # 1. 特徴量計算とデータ準備
            X_all, y = self._prepare_training_data(
                self._calculate_features(
                    training_data, funding_rate_data, open_interest_data
                ),
                training_data,
                **training_params,
            )
            if X_all is None or X_all.empty:
                raise DataError("学習データが空です")

            # 2. データ分割（時系列ホールドアウト）
            X_tr, X_te, y_tr, y_te = self._split_data(X_all, y, **training_params)

            gate_quantile = float(
                training_params.get(
                    "gate_quantile",
                    self.config.training.gate_quantile,
                )
            )
            gate_cutoff_log_rv = float(y_tr.quantile(gate_quantile))
            gate_cutoff_vol = float(np.exp(gate_cutoff_log_rv))
            training_params = {
                **training_params,
                "task_type": training_params.get(
                    "task_type",
                    self.config.training.task_type,
                ),
                "target_kind": training_params.get(
                    "target_kind",
                    self.config.training.target_kind,
                ),
                "gate_quantile": gate_quantile,
                "gate_cutoff_log_rv": gate_cutoff_log_rv,
                "gate_cutoff_vol": gate_cutoff_vol,
            }

            # 3. クロスバリデーション（特徴量選択の前に実行してリークを防ぐ）
            cv_res = None
            if training_params.get("use_cross_validation", False):
                cv_res = self._time_series_cross_validate(X_tr, y_tr, **training_params)

            # 4. 動的な特徴量選択（学習データのみを使用）
            X_tr, X_te = self._apply_feature_selection(X_tr, X_te, y_tr)

            # 5. 学習実行
            X_tr_s, X_te_s = self._preprocess_data(X_tr, X_te)
            res = self._train_model_impl(X_tr_s, X_te_s, y_tr, y_te, **training_params)
            if cv_res is not None:
                res.update(cv_res)

            self.is_trained = True

            # 5. モデル保存
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
                self.metadata = self.current_model_metadata

            # 元のX, yを返す必要がある場合は、選択後の特徴量を持つ全データを再構築
            # （レポート出力用など）
            X_final, y_final = self._recombine_split_data(X_tr, X_te, y_tr, y_te)

            return self._format_training_result(res, X_final, y_final)

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        予測を実行（抽象メソッド）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """

    def predict_volatility(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        最新の特徴量データから将来ボラティリティを予測

        入力データの前処理、期待される形式への変換、モデル推論を行い、
        最終的な `forecast_log_rv`, `forecast_vol`, `gate_open` を返します。

        Args:
            features_df: 特徴量DataFrame（生データ）

        Returns:
            ボラティリティ予測結果
        """
        if not self.is_trained:
            logger.warning("学習済みモデルがありません")
            return self.config.prediction.get_default_predictions()

        try:
            if self.feature_columns is None:
                logger.warning("特徴量カラムが設定されていません")
                return self.config.prediction.get_default_predictions()

            # 1. 前処理（カラム調整、スケーリング）- 共通ユーティリティを直接使用
            processed_features = prepare_data_for_prediction(
                features_df,
                expected_columns=self.feature_columns,
                scaler=self.scaler,
            )

            predictions = np.asarray(self.predict(processed_features))

            # 3. 最新の予測結果を取得（時系列データの場合は最後の行）
            if predictions.ndim == 0:
                latest_pred = predictions.item()
            elif predictions.ndim == 1:
                latest_pred = predictions[-1]
            else:
                latest_pred = predictions[-1]

            latest_pred_array = np.asarray(latest_pred)

            if latest_pred_array.ndim == 0:
                forecast_log_rv = float(latest_pred_array)
            else:
                forecast_log_rv = float(latest_pred_array.reshape(-1)[-1])

            metadata = self.current_model_metadata or self.metadata or {}
            gate_cutoff_log_rv = float(metadata.get("gate_cutoff_log_rv", 0.0))
            forecast_vol = float(np.exp(forecast_log_rv))

            return {
                "forecast_log_rv": forecast_log_rv,
                "forecast_vol": forecast_vol,
                "gate_open": bool(forecast_log_rv >= gate_cutoff_log_rv),
            }

        except Exception as e:
            logger.error(f"ボラティリティ予測エラー: {e}")
            return self.config.prediction.get_default_predictions()

    def predict_signal(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """後方互換の薄いラッパー。"""
        return self.predict_volatility(features_df)

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

            logger.info("📊 特徴量計算を実行中...")
            basic_features = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
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
            task_type = training_params.get(
                "task_type",
                self.config.training.task_type,
            )
            if task_type == "volatility_regression":
                features_clean, labels_numeric = self.target_service.prepare_targets(
                    features_df, ohlcv_df, **training_params
                )
            else:
                features_clean, labels_numeric = self.label_service.prepare_labels(
                    features_df, ohlcv_df, **training_params
                )

            self.feature_columns = features_clean.columns.tolist()
            return features_clean, labels_numeric

        except Exception as e:
            logger.error(f"学習データ準備エラー: {e}")
            raise DataError(f"学習データの準備に失敗しました: {e}")

    def _apply_feature_selection(
        self, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """学習用データのみに対して特徴量選択を適用する"""
        logger.info(
            f"🎯 動的な特徴量選択を実行中... (学習データ: {len(X_tr)}サンプル, 候補数: {len(X_tr.columns)})"
        )
        try:
            # fitは学習データのみで行う（これが重要）
            self.feature_selector.fit(X_tr, y_tr)

            selected_columns = list(self.feature_selector.get_feature_names_out())

            # transformは学習・テスト両方に適用
            X_tr_selected = pd.DataFrame(
                self.feature_selector.transform(X_tr),
                columns=selected_columns,
                index=X_tr.index,
            )
            X_te_selected = pd.DataFrame(
                self.feature_selector.transform(X_te),
                columns=selected_columns,
                index=X_te.index,
            )

            self.feature_columns = selected_columns
            logger.info(f"✅ 特徴量選択完了: {len(self.feature_columns)}個を採用")
            return X_tr_selected, X_te_selected
        except Exception as e:
            logger.warning(
                f"特徴量選択中にエラーが発生しました。全特徴量を使用します: {e}"
            )
            self.feature_columns = X_tr.columns.tolist()
            return X_tr, X_te

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データを分割（常に時系列分割）"""
        test_size = resolve_holdout_test_size(
            test_size=training_params.get("test_size"),
            train_test_split=training_params.get("train_test_split"),
            validation_split=training_params.get("validation_split"),
        )

        logger.info("🕒 時系列分割を使用")
        n_samples = len(X)
        train_size = int(n_samples * (1 - test_size))

        X_train = X.iloc[:train_size].copy()
        X_test = X.iloc[train_size:].copy()
        y_train = y.iloc[:train_size].copy()
        y_test = y.iloc[train_size:].copy()

        return X_train, X_test, y_train, y_test

    def _recombine_split_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """学習・テスト分割後のデータをレポート用に再結合する。"""
        try:
            X_final = pd.concat([X_train, X_test]).sort_index()
            y_final = pd.concat([y_train, y_test]).sort_index()
            return X_final, y_final
        except Exception as exc:
            logger.warning(
                "学習結果の全データ再構築に失敗したため学習データのみ返します: %s",
                exc,
                exc_info=True,
            )
            return X_train, y_train

    def _time_series_cross_validate(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Dict[str, Any]:
        """時間軸を考慮したパージング・エンバーゴ付きクロスバリデーションを実行"""
        n_splits = training_params.get("cv_splits", self.config.training.cv_folds)
        logger.info(f"🔄 時系列クロスバリデーション開始（{n_splits}分割）")

        t1_horizon_n = training_params.get(
            "horizon_n", self.config.training.label_generation.horizon_n
        )
        timeframe = training_params.get(
            "timeframe", self.config.training.label_generation.timeframe
        )

        t1 = cross_validation_module.get_t1_series(
            pd.to_datetime(X.index),
            t1_horizon_n,
            timeframe=timeframe,
        )

        pct_embargo = getattr(self.config.training, "pct_embargo", 0.01)
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

        cv_scores = []
        fold_results = []
        task_type = training_params.get("task_type", self.config.training.task_type)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                np.asarray(scaler.fit_transform(X_train_cv)),
                columns=X_train_cv.columns,
                index=X_train_cv.index,
            )
            X_test_scaled = pd.DataFrame(
                np.asarray(scaler.transform(X_test_cv)),
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

            if task_type == "volatility_regression":
                score = fold_result.get("qlike", 0.0)
            else:
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
            np.asarray(self.scaler.fit_transform(X_train)),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            np.asarray(self.scaler.transform(X_test)),
            columns=X_test.columns,
            index=X_test.index,
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

        if self._model is None or self.feature_columns is None:
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

        loaded_model = model_data.get("model")
        loaded_scaler = model_data.get("scaler")
        loaded_feature_columns = model_data.get("feature_columns")
        metadata = model_data.get("metadata", {})
        task_type = metadata.get("task_type")
        target_kind = metadata.get("target_kind")

        if loaded_model is None:
            raise MLModelError("モデルデータにモデルが含まれていません")
        if task_type != self.config.training.task_type:
            logger.warning(
                "期待する task_type と異なるモデルのため読み込みを拒否します: %s",
                task_type,
            )
            return False
        if target_kind != self.config.training.target_kind:
            logger.warning(
                "期待する target_kind と異なるモデルのため読み込みを拒否します: %s",
                target_kind,
            )
            return False

        self._model = loaded_model
        self.scaler = loaded_scaler
        self.feature_columns = loaded_feature_columns
        self.is_trained = True
        self.current_model_path = model_path
        self.current_model_metadata = metadata
        self.metadata = self.current_model_metadata
        return True

    def _cleanup_temporary_files(self, level: CleanupLevel):
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        pass

    def _cleanup_models(self, level: CleanupLevel):
        try:
            if self.feature_service is not None:
                if hasattr(self.feature_service, "cleanup_resources"):
                    self.feature_service.cleanup_resources()  # type: ignore[reportAttributeAccessIssue]

            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
            self.current_model_path = None
            self.current_model_metadata = None
            self.metadata = None
        except Exception as e:
            logger.warning(f"モデルクリーンアップ警告: {e}")
            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
            self.current_model_path = None
            self.current_model_metadata = None
            self.metadata = None

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
