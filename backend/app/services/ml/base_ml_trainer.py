"""
ML学習基盤クラス

学習・評価・前処理・保存に関わる共通ロジックを提供する抽象基盤クラスです。
具体的なアルゴリズムや最適化手法の詳細説明はDocstringに含めません。
継承クラスがモデル固有の学習処理を実装します。
"""

import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...config.unified_config import unified_config
from ...utils.error_handler import (
    DataError,
    ml_operation_context,
    safe_ml_operation,
)
from .cross_validation import PurgedKFold


from .common.base_resource_manager import BaseResourceManager, CleanupLevel
from .common.evaluation_utils import evaluate_model_predictions
from .common.ml_utils import get_feature_importance_unified, prepare_data_for_prediction
from .config import ml_config
from .exceptions import MLModelError
from .feature_engineering.feature_engineering_service import FeatureEngineeringService
from .ml_metadata import ModelMetadata
from .model_manager import model_manager
from .label_generation.label_generation_service import LabelGenerationService

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

        self.config = ml_config

        self.feature_service = FeatureEngineeringService()
        self.label_service = LabelGenerationService()
        logger.debug(
            "特徴量エンジニアリングサービスとラベル生成サービスを初期化しました"
        )

        self.trainer_config = trainer_config or {}

        # プロパティの設定（サブクラスで使用される可能性があるため維持）
        self.trainer_type = trainer_type or self.trainer_config.get("type", "single")
        self.model_type = model_type or self.trainer_config.get(
            "model_type", "lightgbm"
        )
        self.ensemble_config = self.trainer_config.get("ensemble_config", {})

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self._model = None
        self.last_training_results = None

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
        MLモデルを学習（テンプレートメソッドパターン）

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            model_name: モデル名（オプション）
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書
        """
        with ml_operation_context("MLモデル学習"):
            # 1. 入力データの検証
            self._validate_training_data(training_data)

            # 2. 特徴量を計算
            features_df = self._calculate_features(
                training_data, funding_rate_data, open_interest_data
            )

            # 3. 学習用データを準備
            X, y = self._prepare_training_data(features_df, **training_params)

            # 学習データが空でないことを確認
            if X is None or X.empty or y is None or y.empty:
                raise DataError("前処理後の学習データが空です")

            # 4. クロスバリデーションを実行するかチェック
            use_cross_validation = training_params.get("use_cross_validation", False)

            if use_cross_validation:
                # 時系列クロスバリデーションを実行
                cv_result = self._time_series_cross_validate(X, y, **training_params)

                # 最終モデルは全データで学習
                logger.info("🎯 最終モデルを全データで学習中...")
                X_scaled = self._preprocess_data(X, X)[0]  # 全データをスケーリング

                # ダミーのテストデータ（最後の20%）を作成
                test_size = training_params.get("test_size", 0.2)
                n_samples = len(X)
                train_size = int(n_samples * (1 - test_size))

                X_train_final = X_scaled.iloc[:train_size]
                X_test_final = X_scaled.iloc[train_size:]
                y_train_final = y.iloc[:train_size]
                y_test_final = y.iloc[train_size:]

                training_result = self._train_model_impl(
                    X_train_final,
                    X_test_final,
                    y_train_final,
                    y_test_final,
                    **training_params,
                )

                # クロスバリデーション結果を追加
                training_result.update(cv_result)

            else:
                # 通常の単一分割学習
                # 4. データを分割
                X_train, X_test, y_train, y_test = self._split_data(
                    X, y, **training_params
                )

                # 5. データを前処理
                X_train_scaled, X_test_scaled = self._preprocess_data(X_train, X_test)

                # 6. モデルを学習（継承クラスで実装）
                training_result = self._train_model_impl(
                    X_train_scaled, X_test_scaled, y_train, y_test, **training_params
                )

            # 7. 学習完了フラグを設定
            self.is_trained = True

            # 8. モデルを保存
            should_save_model = bool(save_model) if save_model is not None else True
            if should_save_model:
                model_metadata = ModelMetadata.from_training_result(
                    training_result=training_result,
                    training_params=training_params,
                    model_type=self.__class__.__name__,
                    feature_count=(
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                )

                model_metadata.log_summary()

                validation_result = model_metadata.validate()
                if not validation_result["is_valid"]:
                    logger.warning(
                        f"モデルメタデータの問題: {validation_result['errors']}"
                    )

                model_path = self.save_model(
                    model_name or self.config.model.AUTO_STRATEGY_MODEL_NAME,
                    model_metadata.to_dict(),
                )
                training_result["model_path"] = model_path

            # 9. 学習結果を整形
            result = self._format_training_result(training_result, X, y)

            logger.info("MLモデル学習完了")
            return result

    @safe_ml_operation(default_return={}, context="モデル評価でエラーが発生しました")
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
        if not self.is_trained:
            raise MLModelError("評価対象の学習済みモデルがありません")

        # 特徴量を計算
        features_df = self._calculate_features(
            test_data, funding_rate_data, open_interest_data
        )

        # 予測を実行（全データ）
        # predictは確率を返す（SingleModelTrainer, StackingEnsembleともに）
        predictions_proba = self.predict(features_df)

        # クラス予測（確率最大）
        if predictions_proba.ndim == 2:
            predictions_class = np.argmax(predictions_proba, axis=1)
        else:
            predictions_class = (predictions_proba > 0.5).astype(int)

        # 評価結果を作成
        evaluation_result = {
            "predictions_proba": predictions_proba,
            "predictions_class": predictions_class,
            "test_samples": len(features_df),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "model_status": "trained" if self.is_trained else "not_trained",
        }

        # ラベル生成とメトリクス計算
        try:
            # ラベル生成（NaNは削除される）
            # configからパラメータを取得してラベルを生成
            features_clean, labels_numeric = self.label_service.prepare_labels(
                features_df,
                prediction_horizon=self.config.training.PREDICTION_HORIZON,
            )

            if len(labels_numeric) > 0:
                # インデックスを使用して予測値をアラインメント
                # features_cleanのインデックスに対応する予測値を抽出
                valid_indices = features_df.index.get_indexer(features_clean.index)

                # 予測値をフィルタリング
                y_pred_class_aligned = predictions_class[valid_indices]
                y_pred_proba_aligned = (
                    predictions_proba[valid_indices]
                    if predictions_proba.ndim == 2
                    else predictions_proba[valid_indices]
                )

                # メトリクス計算（共通ユーティリティを使用）
                metrics = evaluate_model_predictions(
                    labels_numeric, y_pred_class_aligned, y_pred_proba_aligned
                )

                evaluation_result.update(metrics)
                logger.info("✅ モデル評価メトリクス計算完了")
            else:
                logger.warning(
                    "評価用ラベルが生成できませんでした（データ不足の可能性）"
                )

        except Exception as e:
            logger.warning(f"評価用ラベル生成またはメトリクス計算失敗: {e}")

        return evaluation_result

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        予測を実行（抽象メソッド）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        pass

    def predict_signal(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        シグナル予測（クラス確率）を実行
        前処理から予測、フォーマットまでを一貫して行う

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        if not self.is_trained:
            logger.warning("学習済みモデルがありません")
            return self.config.prediction.get_default_predictions()

        try:
            # 1. 前処理（カラム調整、スケーリング）
            processed_features = self._preprocess_features_for_prediction(features_df)

            # 2. 予測実行（サブクラスのpredictを呼び出し）
            # predictは確率配列を返すことを期待
            predictions = self.predict(processed_features)

            # 3. 最新の予測結果を取得（時系列データの場合は最後の行）
            if predictions.ndim == 2:
                latest_pred = predictions[-1]
            else:
                latest_pred = predictions

            # 4. 結果の整形
            if latest_pred.shape[0] == 3:
                # 3クラス分類 (down, range, up)
                return {
                    "down": float(latest_pred[0]),
                    "range": float(latest_pred[1]),
                    "up": float(latest_pred[2]),
                }
            elif latest_pred.shape[0] == 2:
                # 2クラス分類 (range, trend) または (class0, class1)
                # 既存のロジックに合わせて range, trend とする
                # ただし、(down, up)の可能性もあるので注意が必要だが、
                # プロジェクトの慣習として (range, trend) が多いと仮定
                return {
                    "range": float(latest_pred[0]),
                    "trend": float(latest_pred[1]),
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

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ラベル
            y_test: テスト用ラベル
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        pass

    def _preprocess_features_for_prediction(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        予測用の特徴量前処理
        - 必要なカラムの抽出
        - 欠損カラムの補完（0埋め）
        - スケーリング

        Args:
            features_df: 特徴量DataFrame

        Returns:
            前処理済み特徴量
        """
        try:
            return prepare_data_for_prediction(
                features_df,
                expected_columns=self.feature_columns,
                scaler=self.scaler,
            )
        except Exception as e:
            logger.error(f"特徴量前処理エラー: {e}")
            return features_df

    def _validate_training_data(self, training_data: pd.DataFrame) -> None:
        """入力データの検証"""
        if training_data is None or training_data.empty:
            raise DataError("学習データが空です")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in training_data.columns
        ]
        if missing_columns:
            raise DataError(f"必要なカラムが不足しています: {missing_columns}")

        if len(training_data) < 100:
            raise DataError("学習データが不足しています（最低100行必要）")

    def _calculate_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """特徴量を計算"""
        try:
            # 入力データの検証
            if ohlcv_data is None or ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # 設定からprofileを取得
            profile = unified_config.ml.feature_engineering.profile
            logger.info(f"📊 特徴量計算を実行中（profile: {profile}）...")

            # 基本特徴量計算
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
            # フォールバック処理（簡略化）
            return ohlcv_data.copy()

    def _prepare_training_data(
        self, features_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """学習用データを準備"""
        try:
            features_clean, labels_numeric = self.label_service.prepare_labels(
                features_df, **training_params
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
        """時系列クロスバリデーション"""
        n_splits = training_params.get(
            "cv_splits", self.config.training.CROSS_VALIDATION_FOLDS
        )
        logger.info(f"🔄 時系列クロスバリデーション開始（{n_splits}分割）")

        # ラベル生成設定からパラメータを取得
        t1_horizon_n = self.config.training.PREDICTION_HORIZON

        # 共通ユーティリティを使用してt1を計算（時間足は自動推定）
        from .common.time_series_utils import get_t1_series

        t1 = get_t1_series(X.index, t1_horizon_n)

        pct_embargo = getattr(self.config.training, "PCT_EMBARGO", 0.01)
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            # インデックスを使用してデータを分割
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

            # スケーリング
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

            # 各フォールドで学習・評価
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

            # スコアを記録（accuracyを基本とするが、利用可能ならbalanced_accuracyを使用）
            score = fold_result.get(
                "balanced_accuracy", fold_result.get("accuracy", 0.0)
            )
            cv_scores.append(score)

        # 平均スコアと標準偏差
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
        """
        保存対象のモデルオブジェクトを取得
        サブクラスでオーバーライド可能
        """
        return self._model

    def _get_model_specific_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        モデル固有のメタデータを取得
        サブクラスでオーバーライド可能
        """
        return {}

    def save_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """モデルを保存"""
        if not self.is_trained:
            raise MLModelError("学習済みモデルがありません")

        final_metadata = {
            "model_type": self.__class__.__name__,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "is_trained": self.is_trained,
        }
        if metadata:
            final_metadata.update(metadata)

        # モデル固有のメタデータを追加
        final_metadata.update(self._get_model_specific_metadata(model_name))

        try:
            feature_importance = self.get_feature_importance(top_n=100)
            if feature_importance:
                final_metadata["feature_importance"] = feature_importance
        except Exception as e:
            logger.warning(f"特徴量重要度の取得に失敗: {e}")

        # 保存対象のモデルを取得
        model_to_save = self._get_model_to_save()
        if model_to_save is None:
            # フォールバック: selfを保存（ただし推奨されない）
            logger.warning("保存対象モデルがNoneです。トレーナー自体を保存します。")
            model_to_save = self

        # 統一されたモデル保存を使用
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
        """
        特徴量重要度を取得（デフォルト実装）
        サブクラスでオーバーライド可能
        """
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
        except Exception as e:
            logger.warning(f"モデルクリーンアップ警告: {e}")
            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False

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

    def _prepare_combined_training_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """学習データの統合準備（互換性維持）"""
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])
        training_data = X_combined.copy()
        training_data["target"] = y_combined
        return training_data
