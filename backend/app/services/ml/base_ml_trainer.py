"""
ML学習基盤クラス

学習・評価・前処理・保存に関わる共通ロジックを提供する抽象基盤クラスです。
具体的なアルゴリズムや最適化手法の詳細説明はDocstringに含めません。
継承クラスがモデル固有の学習処理を実装します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from ...config.unified_config import unified_config
from ...utils.data_processing import data_processor as data_preprocessor
from ...utils.error_handler import (
    DataError,
    ml_operation_context,
    safe_ml_operation,
)
from ...utils.label_generation.presets import (
    apply_preset_by_name,
    forward_classification_preset,
    get_common_presets,
)
from ...utils.cross_validation import PurgedKFold

from .data_processing.sampling import ImbalanceSampler
from .common.base_resource_manager import BaseResourceManager, CleanupLevel
from .config import ml_config
from .exceptions import MLModelError
from .feature_engineering.feature_engineering_service import FeatureEngineeringService
from .ml_metadata import ModelMetadata
from .model_manager import model_manager

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
        logger.debug("特徴量エンジニアリングサービスを初期化しました")

        self.trainer_config = trainer_config or {}
        
        # プロパティの設定（サブクラスで使用される可能性があるため維持）
        self.trainer_type = trainer_type or self.trainer_config.get("type", "single")
        self.model_type = model_type or self.trainer_config.get("model_type", "lightgbm")
        self.ensemble_config = self.trainer_config.get("ensemble_config", {})

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self._model = None
        self.last_training_results = None

    @safe_ml_operation(default_return={}, context="MLモデル学習でエラーが発生しました")
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
                    logger.warning(f"モデルメタデータの問題: {validation_result['errors']}")

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

        # 予測を実行
        predictions = self.predict(features_df)

        # 評価結果を作成
        evaluation_result = {
            "predictions": predictions,
            "test_samples": len(test_data),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "model_status": "trained" if self.is_trained else "not_trained",
        }

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

        Args:
            features_df: 特徴量DataFrame

        Returns:
            前処理済み特徴量
        """
        try:
            # 特徴量カラムの選択
            if self.feature_columns is not None:
                available_columns = [
                    col for col in self.feature_columns if col in features_df.columns
                ]
                processed_features = features_df[available_columns].copy()
            else:
                processed_features = features_df.copy()

            # スケーリング
            if hasattr(self, "scaler") and self.scaler is not None:
                try:
                    processed_features = pd.DataFrame(
                        self.scaler.transform(processed_features),
                        columns=processed_features.columns,
                        index=processed_features.index,
                    )
                except Exception as e:
                    logger.warning(f"スケーリングをスキップ: {e}")

            return processed_features

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
        # ラベル生成設定を取得
        label_config = unified_config.ml.training.label_generation

        # target_columnが明示的に指定されている場合は既存ロジックを使用（後方互換性）
        target_column = training_params.get("target_column")
        if target_column is not None and target_column in features_df.columns:
            logger.info(f"📌 後方互換性モード: target_column='{target_column}' を使用")
            
            from ...utils.label_generation import LabelGenerator
            label_generator = LabelGenerator()

            features_clean, labels_clean, threshold_info = (
                data_preprocessor.prepare_training_data(
                    features_df, label_generator, **training_params
                )
            )
            
            self.feature_columns = features_clean.columns.tolist()
            return features_clean, labels_clean

        # デフォルトのプリセット/カスタム設定ロジック
        try:
            logger.info("🎯 新しいラベル生成設定を使用")
            
            if label_config.use_preset:
                try:
                    labels, preset_info = apply_preset_by_name(
                        features_df, label_config.default_preset
                    )
                except ValueError:
                    logger.warning("⚠️ フォールバック: カスタム設定を使用します")
                    labels = forward_classification_preset(
                        df=features_df,
                        timeframe=label_config.timeframe,
                        horizon_n=label_config.horizon_n,
                        threshold=label_config.threshold,
                        price_column=label_config.price_column,
                        threshold_method=label_config.get_threshold_method_enum(),
                    )
            else:
                labels = forward_classification_preset(
                    df=features_df,
                    timeframe=label_config.timeframe,
                    horizon_n=label_config.horizon_n,
                    threshold=label_config.threshold,
                    price_column=label_config.price_column,
                    threshold_method=label_config.get_threshold_method_enum(),
                )

            # NaNを削除してクリーンなデータを作成
            valid_idx = labels.notna()
            features_clean = features_df[valid_idx].copy()
            labels_clean = labels[valid_idx].copy()

            # 文字列ラベルを数値に変換
            unique_labels = set(labels_clean.unique())
            if "TREND" in unique_labels:
                label_mapping = {"RANGE": 0, "TREND": 1}
            else:
                label_mapping = {"DOWN": 0, "RANGE": 1, "UP": 2}

            labels_numeric = labels_clean.map(label_mapping)

            if labels_numeric.isna().any():
                valid_numeric_idx = labels_numeric.notna()
                features_clean = features_clean[valid_numeric_idx]
                labels_numeric = labels_numeric[valid_numeric_idx]

            self.feature_columns = features_clean.columns.tolist()
            logger.info(f"✅ ラベル生成完了: {len(features_clean)}サンプル")

            return features_clean, labels_numeric

        except Exception as e:
            logger.error(f"❌ 新しいラベル生成設定でエラー発生: {e}")
            # フォールバック処理（省略）
            raise DataError(f"ラベル生成に失敗しました: {e}")

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データを分割"""
        test_size = training_params.get("test_size", 0.2)
        random_state = training_params.get("random_state", 42)
        use_random_split = training_params.get("use_random_split", False)
        use_time_series_split = training_params.get(
            "use_time_series_split",
            (self.config.training.USE_TIME_SERIES_SPLIT if not use_random_split else False),
        )

        if use_time_series_split and not use_random_split:
            logger.info("🕒 時系列分割を使用")
            n_samples = len(X)
            train_size = int(n_samples * (1 - test_size))
            
            X_train = X.iloc[:train_size].copy()
            X_test = X.iloc[train_size:].copy()
            y_train = y.iloc[:train_size].copy()
            y_test = y.iloc[train_size:].copy()
        else:
            logger.info("🔀 ランダム分割を使用")
            stratify_param = y if y.nunique() > 1 else None
            splits = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
            X_train = cast(pd.DataFrame, splits[0])
            X_test = cast(pd.DataFrame, splits[1])
            y_train = cast(pd.Series, splits[2])
            y_test = cast(pd.Series, splits[3])

        return X_train, X_test, y_train, y_test

    def _time_series_cross_validate(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Dict[str, Any]:
        """時系列クロスバリデーション"""
        n_splits = training_params.get("cv_splits", self.config.training.CROSS_VALIDATION_FOLDS)
        max_train_size = training_params.get("max_train_size", self.config.training.MAX_TRAIN_SIZE)

        logger.info(f"🔄 時系列クロスバリデーション開始（{n_splits}分割）")

        if self.config.training.USE_PURGED_KFOLD:
            # ラベル生成設定からパラメータを取得
            label_config = unified_config.ml.training.label_generation
            # 簡易的にデフォルト値を使用（詳細は省略）
            t1_horizon_n = self.config.training.PREDICTION_HORIZON
            t1_timeframe = "1h"
            
            t1 = self._get_t1_series(X.index, t1_horizon_n, t1_timeframe)
            splitter = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=0.01)
        else:
            splitter = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), 1):
            logger.info(f"フォールド {fold}/{n_splits} を実行中...")
            X_train_cv = X.iloc[train_idx]
            X_test_cv = X.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]

            # SMOTE処理（省略）

            X_train_scaled, X_test_scaled = self._preprocess_data(X_train_cv, X_test_cv)

            fold_result = self._train_fold_with_error_handling(
                fold, X_train_scaled, X_test_scaled, y_train_cv, y_test_cv,
                X_train_cv, X_test_cv, training_params
            )

            cv_scores.append(fold_result.get("accuracy", 0.0))
            fold_results.append(fold_result)

        cv_result = {
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "fold_results": fold_results,
        }
        return cv_result

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

        try:
            feature_importance = self.get_feature_importance(top_n=100)
            if feature_importance:
                final_metadata["feature_importance"] = feature_importance
        except Exception as e:
            logger.warning(f"特徴量重要度の取得に失敗: {e}")

        # 統一されたモデル保存を使用
        model_path = model_manager.save_model(
            model=self,
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

        # LightGBM/XGBoostモデルの場合の一般的な処理
        if hasattr(self._model, "feature_importance") and self.feature_columns:
            try:
                importance_scores = self._model.feature_importance(importance_type="gain")
                feature_importance = dict(zip(self.feature_columns, importance_scores))
                sorted_importance = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:top_n]
                return dict(sorted_importance)
            except Exception:
                pass
        
        # get_feature_importanceメソッドを持つモデルの場合
        if hasattr(self._model, "get_feature_importance"):
            try:
                return self._model.get_feature_importance(top_n)
            except Exception:
                pass

        return {}

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

    @staticmethod
    def _get_t1_series(
        indices: pd.DatetimeIndex, horizon_n: int, timeframe: str
    ) -> pd.Series:
        """PurgedKFold用t1計算"""
        if timeframe == "1h":
            delta = pd.Timedelta(hours=horizon_n)
        elif timeframe == "4h":
            delta = pd.Timedelta(hours=4 * horizon_n)
        elif timeframe == "1d":
            delta = pd.Timedelta(days=horizon_n)
        elif timeframe == "15m":
            delta = pd.Timedelta(minutes=15 * horizon_n)
        else:
            delta = pd.Timedelta(hours=horizon_n)

        t1 = pd.Series(indices + delta, index=indices)
        return t1