"""
ML学習基盤クラス

学習・評価・前処理・保存に関わる共通ロジックを提供する抽象基盤クラスです。
具体的なアルゴリズムや最適化手法の詳細説明はDocstringに含めません。
継承クラスがモデル固有の学習処理を実装します。
"""

import logging
from abc import ABC
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
            trainer_config: トレーナー設定（単一モデル/アンサンブル設定）
            trainer_type: トレーナータイプ（脆弱性修正）
            model_type: モデルタイプ（脆弱性修正）
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.config = ml_config

        self.feature_service = FeatureEngineeringService()
        logger.debug("特徴量エンジニアリングサービスを初期化しました")

        # トレーナー設定の処理（脆弱性修正）
        self.trainer_config = trainer_config or {}

        # パラメーターの優先順位: 直接指定 > trainer_config > デフォルト
        self.trainer_type = trainer_type or self.trainer_config.get(
            "type", "single"
        )  # "single" or "ensemble"

        self.model_type = model_type or self.trainer_config.get(
            "model_type", "lightgbm"
        )
        self.ensemble_config = self.trainer_config.get("ensemble_config", {})

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self._model = None  # プライベート属性として定義（子クラスのプロパティと競合を回避）
        self.models = {}  # アンサンブル用の複数モデル格納
        self.last_training_results = None  # 最後の学習結果を保持

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

        Raises:
            DataError: データが無効な場合
            ModelError: 学習に失敗した場合
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

            # 7. 学習完了フラグを設定（保存前に設定）
            self.is_trained = True

            # 8. モデルを保存
            # save_modelパラメータの安全な処理
            should_save_model = bool(save_model) if save_model is not None else True
            if should_save_model:
                # training_resultからメタデータを構築
                # ModelMetadata dataclassを使用してメタデータを構築
                model_metadata = ModelMetadata.from_training_result(
                    training_result=training_result,
                    training_params=training_params,
                    model_type=self.__class__.__name__,
                    feature_count=(
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                )

                # メタデータのサマリーをログ出力
                model_metadata.log_summary()

                # メタデータの妥当性を検証
                validation_result = model_metadata.validate()
                if not validation_result["is_valid"]:
                    logger.warning("モデルメタデータに問題があります:")
                    for error in validation_result["errors"]:
                        logger.warning(f"  エラー: {error}")
                for warning in validation_result["warnings"]:
                    logger.warning(f"  警告: {warning}")

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

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        統合された予測実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")

        if self.trainer_type == "ensemble":
            return self._predict_ensemble(features_df)
        else:
            return self._predict_single(features_df)

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        統合されたモデル学習実装

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ラベル
            y_test: テスト用ラベル
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        if self.trainer_type == "ensemble":
            return self._train_ensemble_model(
                X_train, X_test, y_train, y_test, **training_params
            )
        else:
            return self._train_single_model(
                X_train, X_test, y_train, y_test, **training_params
            )

    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        単一モデルの学習（統合実装）

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ラベル
            y_test: テスト用ラベル
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        try:
            logger.info(f"🤖 単一モデル学習開始: {self.model_type}")

            # 学習データを結合（旧実装との互換性維持）
            training_data = self._prepare_combined_training_data(
                X_train, X_test, y_train, y_test
            )

            # 統合されたモデル学習実行
            result = self._execute_single_model_training(
                training_data, **training_params
            )

            # 結果の後処理
            self.is_trained = True

            logger.info(f"✅ 単一モデル学習完了: {self.model_type}")
            return result

        except Exception as e:
            logger.error(f"❌ 単一モデル学習エラー: {e}")
            raise

    def _train_ensemble_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        アンサンブルモデルの学習（統合実装）

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徃
            y_train: 学習用ラベル
            y_test: テスト用ラベル
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        try:
            logger.info(
                f"🎯 アンサンブル学習開始: {self.ensemble_config.get('method', 'stacking')}"
            )

            # 学習データを結合（旧実装との互換性維持）
            training_data = self._prepare_combined_training_data(
                X_train, X_test, y_train, y_test
            )

            # 統合されたアンサンブル学習実行
            result = self._execute_ensemble_model_training(
                training_data, **training_params
            )

            # 結果の後処理
            self.is_trained = True

            logger.info(
                f"✅ アンサンブル学習完了: {self.ensemble_config.get('method', 'stacking')}"
            )
            return result

        except Exception as e:
            logger.error(f"❌ アンサンブル学習エラー: {e}")
            raise

    def _predict_single(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        単一モデルの予測

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        if self._model is None:
            raise ValueError("単一モデルが学習されていません")

        try:
            # 特徴量の前処理
            processed_features = self._preprocess_features_for_prediction(features_df)

            # 予測実行
            if hasattr(self._model, "predict"):
                predictions = self._model.predict(processed_features)
            else:
                # SingleModelTrainerの場合
                predictions = self._model.predict(features_df)

            return predictions

        except Exception as e:
            logger.error(f"❌ 単一モデル予測エラー: {e}")
            raise

    def _predict_ensemble(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        アンサンブルモデルの予測

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        if self._model is None:
            raise ValueError("アンサンブルモデルが学習されていません")

        try:
            # EnsembleTrainerの予測メソッドを使用
            predictions = self._model.predict(features_df)
            return predictions

        except Exception as e:
            logger.error(f"❌ アンサンブル予測エラー: {e}")
            raise

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
                # 学習時の特徴量カラムのみを使用
                available_columns = [
                    col for col in self.feature_columns if col in features_df.columns
                ]
                processed_features = features_df[available_columns].copy()
            else:
                processed_features = features_df.copy()

            # スケーリング（必要に応じて）
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
        """
        特徴量を計算（FeatureEngineeringServiceに完全委譲）

        責務分割により、具体的な特徴量計算ロジックは
        FeatureEngineeringServiceに移譲されました。

        設定からfeature profileを読み込み、FeatureEngineeringServiceに渡します。
        """
        try:
            # 入力データの検証
            if ohlcv_data is None or ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # 必要な列の存在確認
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in ohlcv_data.columns
            ]
            if missing_columns:
                raise ValueError(f"必要な列が不足しています: {missing_columns}")

            # 設定からprofileを取得
            profile = unified_config.ml.feature_engineering.profile
            logger.info(f"📊 特徴量計算を実行中（profile: {profile}）...")

            # 基本特徴量計算（autofeat機能は削除済み）
            basic_features = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                profile=profile,
            )

            # 生成された特徴量数をログ出力
            logger.info(f"✅ 特徴量生成完了: {len(basic_features.columns)}個の特徴量")

            # 基本特徴量計算後の検証
            if basic_features is not None and not basic_features.empty:
                return basic_features
            else:
                raise ValueError("基本特徴量計算も失敗しました")

        except Exception as e:
            logger.warning(f"拡張特徴量計算でエラー、基本特徴量のみ使用: {e}")

            try:
                # フォールバック：基本特徴量のみ（エラーハンドリング強化）
                logger.info("🔄 フォールバック: 基本特徴量計算を実行")
                basic_features = self.feature_service.calculate_advanced_features(
                    ohlcv_data,
                    funding_rate_data,
                    open_interest_data,
                )

                # フォールバック後の検証
                if basic_features is not None and not basic_features.empty:
                    # 最低限必要な特徴量の検証
                    required_features = ["open", "high", "low", "close", "volume"]
                    available_features = [
                        col
                        for col in required_features
                        if col in basic_features.columns
                    ]

                    if len(available_features) >= 3:  # 最低3つの価格列が利用可能
                        logger.info(
                            f"✅ フォールバック成功: {len(basic_features.columns)}個の特徴量"
                        )
                        return basic_features
                    else:
                        raise ValueError(
                            f"基本特徴量も不足しています: {available_features}"
                        )
                else:
                    raise ValueError("基本特徴量計算結果が空です")

            except Exception as fallback_error:
                logger.error(f"フォールバックも失敗: {fallback_error}")

                # 最後の手段：元データに最低の特徴量のみ追加
                logger.warning("🆘 最終手段: 元データに基礎特徴量のみ追加")
                result_df = ohlcv_data.copy()

                try:
                    # 最低価格変動率特徴量のみ計算
                    if "close" in result_df.columns:
                        result_df["returns"] = result_df["close"].pct_change()
                        result_df["returns"] = result_df["returns"].fillna(0.0)

                    if "volume" in result_df.columns:
                        result_df["volume_change"] = result_df["volume"].pct_change()
                        result_df["volume_change"] = result_df["volume_change"].fillna(
                            0.0
                        )

                    logger.info(f"✅ 最終手段成功: {len(result_df.columns)}個の基本列")
                    return result_df

                except Exception as final_error:
                    logger.error(f"最終手段も失敗: {final_error}")
                    error_msg = (
                        f"特徴量計算に完全に失敗しました: "
                        f"{str(e)} -> {str(fallback_error)} -> {str(final_error)}"
                    )
                    raise DataError(error_msg)

    def _prepare_training_data(
        self, features_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備

        新しいラベル生成設定を使用し、プリセットまたはカスタム設定でラベルを生成します。
        既存のtarget_columnパラメータが指定されている場合は、後方互換性のため
        既存のロジックを使用します。

        Args:
            features_df: 特徴量DataFrame（OHLCVデータを含む）
            **training_params: 学習パラメータ
                - target_column: ターゲットカラム名（後方互換性用）
                - その他のパラメータはdata_preprocessorに渡される

        Returns:
            Tuple[pd.DataFrame, pd.Series]: クリーンな特徴量とラベルのタプル

        Raises:
            DataError: ラベル生成に失敗した場合
            ValueError: プリセット名が存在しない場合

        Note:
            - プリセット使用時: unified_config.ml.training.label_generation.use_preset=True
            - カスタム設定使用時: use_preset=False（個別パラメータを使用）
            - 後方互換性: target_columnが指定されている場合は既存ロジックを使用
        """
        # ラベル生成設定を取得
        label_config = unified_config.ml.training.label_generation

        # target_columnが明示的に指定されている場合は既存ロジックを使用（後方互換性）
        target_column = training_params.get("target_column")
        if target_column is not None and target_column in features_df.columns:
            logger.info(f"📌 後方互換性モード: target_column='{target_column}' を使用")

            # 既存のLabelGeneratorを使用
            from ...utils.label_generation import LabelGenerator

            label_generator = LabelGenerator()

            # データ前処理を委譲
            features_clean, labels_clean, threshold_info = (
                data_preprocessor.prepare_training_data(
                    features_df, label_generator, **training_params
                )
            )

            # 特徴量カラムを保存
            self.feature_columns = features_clean.columns.tolist()

            return features_clean, labels_clean

        # 新しいプリセット/カスタム設定を使用
        try:
            logger.info("🎯 新しいラベル生成設定を使用")

            # ラベル生成
            if label_config.use_preset:
                # プリセットを使用
                try:
                    labels, preset_info = apply_preset_by_name(
                        features_df, label_config.default_preset
                    )
                    logger.info(f"✅ プリセット使用: {label_config.default_preset}")
                    logger.info(f"   設定: {preset_info.get('description', 'N/A')}")
                except ValueError:
                    # プリセットが見つからない場合
                    available_presets = list(get_common_presets().keys())
                    logger.error(
                        f"❌ プリセット '{label_config.default_preset}' が見つかりません。"
                        f"利用可能なプリセット: {', '.join(sorted(available_presets[:5]))}..."
                    )
                    # フォールバック: カスタム設定を使用
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
                # カスタム設定を使用
                logger.info("🔧 カスタムラベル生成設定を使用")
                labels = forward_classification_preset(
                    df=features_df,
                    timeframe=label_config.timeframe,
                    horizon_n=label_config.horizon_n,
                    threshold=label_config.threshold,
                    price_column=label_config.price_column,
                    threshold_method=label_config.get_threshold_method_enum(),
                )
                logger.info(
                    f"   設定: {label_config.timeframe}, "
                    f"horizon={label_config.horizon_n}, "
                    f"threshold={label_config.threshold}, "
                    f"method={label_config.threshold_method}"
                )

            # ラベル分布をログ出力
            label_counts = labels.value_counts()
            total_labels = len(labels.dropna())
            if total_labels > 0:
                logger.info("📊 ラベル分布:")
                for label_value in ["UP", "RANGE", "DOWN"]:
                    if label_value in label_counts.index:
                        count = label_counts[label_value]
                        pct = (count / total_labels) * 100
                        logger.info(f"   {label_value}: {count}個 ({pct:.1f}%)")

            # NaNを削除してクリーンなデータを作成
            valid_idx = labels.notna()
            features_clean = features_df[valid_idx].copy()
            labels_clean = labels[valid_idx].copy()

            # 文字列ラベルを数値に変換（既存のロジックとの互換性のため）
            # "DOWN" -> 0, "RANGE" -> 1, "UP" -> 2
            label_mapping = {"DOWN": 0, "RANGE": 1, "UP": 2}
            labels_numeric = labels_clean.map(label_mapping)

            # 欠損値がないことを確認
            if labels_numeric.isna().any():
                logger.warning(
                    f"⚠️ {labels_numeric.isna().sum()}個の不明なラベルを除外します"
                )
                valid_numeric_idx = labels_numeric.notna()
                features_clean = features_clean[valid_numeric_idx]
                labels_numeric = labels_numeric[valid_numeric_idx]

            # 特徴量カラムを保存
            self.feature_columns = features_clean.columns.tolist()

            logger.info(
                f"✅ ラベル生成完了: {len(features_clean)}サンプル "
                f"({len(features_df) - len(features_clean)}個を除外)"
            )

            return features_clean, labels_numeric

        except Exception as e:
            logger.error(f"❌ 新しいラベル生成設定でエラー発生: {e}")
            logger.warning("⚠️ フォールバック: 既存のLabelGeneratorを使用します")

            # フォールバック: 既存のLabelGeneratorを使用
            try:
                from ...utils.label_generation import LabelGenerator

                label_generator = LabelGenerator()

                # データ前処理を委譲
                features_clean, labels_clean, threshold_info = (
                    data_preprocessor.prepare_training_data(
                        features_df, label_generator, **training_params
                    )
                )

                # 特徴量カラムを保存
                self.feature_columns = features_clean.columns.tolist()

                logger.info(
                    "✅ フォールバック成功: 既存のLabelGeneratorでラベル生成完了"
                )
                return features_clean, labels_clean

            except Exception as fallback_error:
                logger.error(f"❌ フォールバックも失敗: {fallback_error}")
                raise DataError(
                    f"ラベル生成に完全に失敗しました: {str(e)} -> {str(fallback_error)}"
                )

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        データを分割（デフォルトで時系列分割）

        Args:
            X: 特徴量DataFrame
            y: ラベルSeries
            **training_params: 学習パラメータ
                - use_time_series_split: 時系列分割を使用（デフォルト: 設定値またはTrue）
                - use_random_split: ランダム分割を使用（下位互換性、デフォルト: False）
                - test_size: テストデータの割合（デフォルト: 0.2）
                - random_state: 乱数シード（デフォルト: 42）

        Returns:
            分割されたデータ (X_train, X_test, y_train, y_test)

        Note:
            時系列データでは、将来のデータが学習データに含まれることを防ぐため、
            時間順序を保持した分割を行います。デフォルトでTimeSeriesSplitを使用します。
        """
        test_size = training_params.get("test_size", 0.2)
        random_state = training_params.get("random_state", 42)

        # 下位互換性: use_random_split=Trueが指定された場合はランダム分割
        use_random_split = training_params.get("use_random_split", False)

        # デフォルトで時系列分割を使用（設定から読み込み）
        use_time_series_split = training_params.get(
            "use_time_series_split",
            (
                self.config.training.USE_TIME_SERIES_SPLIT
                if not use_random_split
                else False
            ),
        )

        if use_time_series_split and not use_random_split:
            # 時系列分割：時間順序を保持して分割
            logger.info("🕒 時系列分割を使用（データリーク防止）")

            # データの長さを取得
            n_samples = len(X)
            train_size = int(n_samples * (1 - test_size))

            # 時間順序を保持して分割
            X_train = X.iloc[:train_size].copy()
            X_test = X.iloc[train_size:].copy()
            y_train = y.iloc[:train_size].copy()
            y_test = y.iloc[train_size:].copy()

            logger.info(
                f"時系列分割結果: 学習={len(X_train)}サンプル, テスト={len(X_test)}サンプル"
            )
            logger.info(f"学習期間: {X_train.index[0]} ～ {X_train.index[-1]}")
            logger.info(f"テスト期間: {X_test.index[0]} ～ {X_test.index[-1]}")

        else:
            # ランダム分割（下位互換性維持）
            logger.info(
                "🔀 ランダム分割を使用（use_random_split=True または use_time_series_split=False）"
            )

            # 層化抽出は、ラベルが2種類以上ある場合にのみ有効
            stratify_param = y if y.nunique() > 1 else None
            if stratify_param is None:
                logger.warning(
                    "ラベルが1種類以下のため、層化抽出なしでデータを分割します。"
                )

            # train_test_splitはリストを返すため、一度変数に受けてからキャストする
            splits = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_param,
            )

            # 型チェッカーのために明示的にキャスト
            X_train = cast(pd.DataFrame, splits[0])
            X_test = cast(pd.DataFrame, splits[1])
            y_train = cast(pd.Series, splits[2])
            y_test = cast(pd.Series, splits[3])

        # 分割後のラベル分布を確認
        logger.info("学習データのラベル分布:")
        for label_value in sorted(y_train.unique()):
            count = (y_train == label_value).sum()
            percentage = count / len(y_train) * 100
            logger.info(f"  ラベル {label_value}: {count}サンプル ({percentage:.1f}%)")

        logger.info("テストデータのラベル分布:")
        for label_value in sorted(y_test.unique()):
            count = (y_test == label_value).sum()
            percentage = count / len(y_test) * 100
            logger.info(f"  ラベル {label_value}: {count}サンプル ({percentage:.1f}%)")

        return X_train, X_test, y_train, y_test

    def _time_series_cross_validate(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Dict[str, Any]:
        """
        時系列クロスバリデーション

        ウォークフォワード検証を行い、より堅牢なモデル評価を提供します。

        Args:
            X: 特徴量DataFrame
            y: ラベルSeries
            **training_params: 学習パラメータ
                - cv_splits: 分割数（デフォルト: ml_config.training.CROSS_VALIDATION_FOLDS）
                - max_train_size: 最大学習サイズ（デフォルト: ml_config.training.MAX_TRAIN_SIZE）

        Returns:
            クロスバリデーション結果の辞書
        """
        # ml_configからデフォルト値を読み込み
        n_splits = training_params.get(
            "cv_splits", self.config.training.CROSS_VALIDATION_FOLDS
        )
        max_train_size = training_params.get(
            "max_train_size", self.config.training.MAX_TRAIN_SIZE
        )

        logger.info(f"🔄 時系列クロスバリデーション開始（{n_splits}分割）")

        # TimeSeriesSplitを初期化
        tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"フォールド {fold}/{n_splits} を実行中...")

            # データを分割
            X_train_cv = X.iloc[train_idx]
            X_test_cv = X.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]

            # データを前処理
            X_train_scaled, X_test_scaled = self._preprocess_data(X_train_cv, X_test_cv)

            # フォールド学習を実行
            fold_result = self._train_fold_with_error_handling(
                fold,
                X_train_scaled,
                X_test_scaled,
                y_train_cv,
                y_test_cv,
                X_train_cv,
                X_test_cv,
                training_params,
            )

            cv_scores.append(fold_result.get("accuracy", 0.0))
            fold_results.append(fold_result)

        # クロスバリデーション結果を集計
        cv_result = {
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_min": np.min(cv_scores),
            "cv_max": np.max(cv_scores),
            "fold_results": fold_results,
            "n_splits": n_splits,
        }

        logger.info("時系列クロスバリデーション完了:")
        logger.info(
            f"  平均精度: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}"
        )
        logger.info(f"  最小精度: {cv_result['cv_min']:.4f}")
        logger.info(f"  最大精度: {cv_result['cv_max']:.4f}")

        return cv_result

    def _preprocess_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データを前処理（スケーリング）"""
        # LightGBMベースのモデルはスケーリング不要
        if hasattr(self, "model_type") and "LightGBM" in str(self.model_type):
            return X_train, X_test

        # その他のモデルはスケーリングを実行
        assert self.scaler is not None, "Scalerが初期化されていません"
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

        # アンサンブルトレーナーが存在する場合は、そちらに委譲
        if hasattr(self, "_ensemble_trainer") and self._ensemble_trainer:
            logger.info("アンサンブルトレーナーに保存を委譲します")
            return self._ensemble_trainer.save_model(model_name, metadata)

        # 基本的なメタデータを準備
        final_metadata = {
            "model_type": self.__class__.__name__,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "is_trained": self.is_trained,
        }
        # 提供されたメタデータで更新
        if metadata:
            final_metadata.update(metadata)

        # 特徴量重要度をメタデータに追加
        try:
            feature_importance = self.get_feature_importance(top_n=100)
            if feature_importance:
                final_metadata["feature_importance"] = feature_importance
                logger.info(
                    f"特徴量重要度をメタデータに追加: {len(feature_importance)}個"
                )
        except Exception as e:
            logger.warning(f"特徴量重要度の取得に失敗: {e}")

        # 統一されたモデル保存を使用
        model_path = model_manager.save_model(
            model=self,  # トレーナー全体を保存
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

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        return {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "trainer_type": getattr(self, "trainer_type", "unknown"),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
        }

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained:
            logger.warning("学習済みモデルがありません")
            return {}

        # アンサンブルトレーナーが存在する場合は、そちらに委譲
        if hasattr(self, "_ensemble_trainer") and self._ensemble_trainer:
            if hasattr(self._ensemble_trainer, "get_feature_importance"):
                try:
                    feature_importance = self._ensemble_trainer.get_feature_importance()
                    if feature_importance:
                        # 上位N個を取得
                        sorted_importance = sorted(
                            feature_importance.items(), key=lambda x: x[1], reverse=True
                        )[:top_n]
                        logger.info(
                            f"アンサンブルトレーナーから特徴量重要度を取得: {len(sorted_importance)}個"
                        )
                        return dict(sorted_importance)
                except Exception as e:
                    logger.error(
                        f"アンサンブルトレーナーからの特徴量重要度取得エラー: {e}"
                    )

        # モデルが特徴量重要度を提供する場合
        if hasattr(self._model, "get_feature_importance"):
            try:
                feature_importance = self._model.get_feature_importance(top_n)
                if feature_importance:
                    logger.info(
                        f"モデルから特徴量重要度を取得: {len(feature_importance)}個"
                    )
                    return feature_importance
            except Exception as e:
                logger.error(f"モデルからの特徴量重要度取得エラー: {e}")

        # LightGBMモデルの場合
        if hasattr(self._model, "feature_importance") and self.feature_columns:
            try:
                importance_scores = self._model.feature_importance(
                    importance_type="gain"
                )
                feature_importance = dict(zip(self.feature_columns, importance_scores))

                # 重要度でソートして上位N個を取得
                sorted_importance = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:top_n]

                logger.info(
                    f"LightGBMから特徴量重要度を取得: {len(sorted_importance)}個"
                )
                return dict(sorted_importance)
            except Exception as e:
                logger.error(f"LightGBM特徴量重要度取得エラー: {e}")
                return {}

        logger.warning("このモデルは特徴量重要度をサポートしていません")
        return {}

    @safe_ml_operation(
        default_return=False, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> bool:
        """
        モデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        model_data = model_manager.load_model(model_path)

        if model_data is None:
            return False

        # モデルデータから各要素を取得
        self._model = model_data.get("model")
        self.scaler = model_data.get("scaler")
        self.feature_columns = model_data.get("feature_columns")

        if self._model is None:
            raise MLModelError("モデルデータにモデルが含まれていません")

        self.is_trained = True
        return True

    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ"""
        # BaseMLTrainerでは特に一時ファイルは作成しないため、パス
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ"""
        try:
            # 特徴量サービスのキャッシュクリーンアップ
            # AutoML機能は削除済みのため、特にクリーンアップ処理なし
            if self.feature_service is not None:
                logger.debug("特徴量サービスキャッシュのクリーンアップをスキップ")
        except Exception as e:
            logger.warning(f"キャッシュクリーンアップ警告: {e}")

    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ"""
        try:
            # 特徴量サービスのクリーンアップ
            if self.feature_service is not None:
                if hasattr(self.feature_service, "cleanup_resources"):
                    self.feature_service.cleanup_resources()
                    logger.debug("特徴量サービスをクリーンアップしました")

            # モデルとスケーラーをクリア
            self._model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False

        except Exception as e:
            logger.warning(f"モデルクリーンアップ警告: {e}")
            # エラーが発生してもクリーンアップは続行
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
        """
        エラーハンドリング付きフォールド学習

        Args:
            fold: フォールド番号
            X_train_scaled: スケーリング済み学習用特徴量
            X_test_scaled: スケーリング済みテスト用特徴量
            y_train_cv: 学習用ラベル
            y_test_cv: テスト用ラベル
            X_train_cv: 元の学習用特徴量（期間情報用）
            X_test_cv: 元のテスト用特徴量（期間情報用）
            training_params: 学習パラメータ

        Returns:
            フォールド学習結果の辞書
        """
        # モデルを学習（継承クラスで実装）
        fold_result = self._train_model_impl(
            X_train_scaled,
            X_test_scaled,
            y_train_cv,
            y_test_cv,
            **training_params,
        )

        # フォールド情報を追加
        fold_result.update(
            {
                "fold": fold,
                "train_samples": len(X_train_cv),
                "test_samples": len(X_test_cv),
                "train_period": f"{X_train_cv.index[0]} ～ {X_train_cv.index[-1]}",
                "test_period": f"{X_test_cv.index[0]} ～ {X_test_cv.index[-1]}",
            }
        )

        logger.info(
            f"フォールド {fold} 完了: 精度={fold_result.get('accuracy', 0.0):.4f}"
        )

        return fold_result

    def _prepare_combined_training_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        学習データの統合準備（旧実装との互換性維持）

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ラベル
            y_test: テスト用ラベル

        Returns:
            統合された学習データ
        """
        # 学習データを結合（旧実装との互換性維持）
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])
        training_data = X_combined.copy()
        training_data["target"] = y_combined
        return training_data

    def _execute_single_model_training(
        self, training_data: pd.DataFrame, **training_params
    ) -> Dict[str, Any]:
        """
        単一モデル学習の実行

        Args:
            training_data: 統合済み学習データ
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        # 旧実装との互換性を維持しつつ、テンプレートメソッドパターンを適用
        from .single_model.single_model_trainer import SingleModelTrainer

        trainer = SingleModelTrainer(model_type=self.model_type)

        result = trainer.train_model(training_data, **training_params)

        # モデルを保存
        self._model = trainer.model

        return result

    def _execute_ensemble_model_training(
        self, training_data: pd.DataFrame, **training_params
    ) -> Dict[str, Any]:
        """
        アンサンブルモデル学習の実行

        Args:
            training_data: 統合済み学習データ
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        # 旧実装との互換性を維持しつつ、テンプレートメソッドパターンを適用
        from .ensemble.ensemble_trainer import EnsembleTrainer

        trainer = EnsembleTrainer(ensemble_config=self.ensemble_config)

        result = trainer.train_model(training_data, **training_params)

        # モデルを保存
        self.models = trainer.models
        self._model = trainer  # アンサンブルトレーナー自体を保存
        self._ensemble_trainer = trainer  # 参照を保持

        return result
