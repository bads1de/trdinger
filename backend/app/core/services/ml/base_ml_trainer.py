"""
ML学習基盤クラス

MLTrainingServiceとMLIndicatorServiceで重複していた学習ロジックを統合し、
共通の学習基盤を提供します。SOLID原則に従い、責任を明確化します。
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, cast
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    log_loss,
    brier_score_loss,
    cohen_kappa_score,
)

from .config import ml_config
from ...utils.unified_error_handler import (
    UnifiedDataError,
    UnifiedModelError,
    safe_ml_operation,
    ml_operation_context,
)
from .feature_engineering.feature_engineering_service import FeatureEngineeringService
from .model_manager import model_manager
from ...utils.label_generation import LabelGenerator, ThresholdMethod
from database.connection import SessionLocal
from database.repositories.external_market_repository import (
    ExternalMarketRepository,
)
from database.repositories.fear_greed_repository import FearGreedIndexRepository

logger = logging.getLogger(__name__)


class BaseMLTrainer(ABC):
    """
    ML学習基盤クラス

    共通の学習ロジックを提供し、具体的な実装は継承クラスで行います。
    単一責任原則に従い、学習に関する責任のみを持ちます。
    """

    def __init__(self):
        """初期化"""
        self.config = ml_config
        self.feature_service = FeatureEngineeringService()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.model = None

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
            UnifiedDataError: データが無効な場合
            UnifiedModelError: 学習に失敗した場合
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

            # 4. データを分割
            X_train, X_test, y_train, y_test = self._split_data(X, y, **training_params)

            # 5. データを前処理
            X_train_scaled, X_test_scaled = self._preprocess_data(X_train, X_test)

            # 6. モデルを学習（継承クラスで実装）
            training_result = self._train_model_impl(
                X_train_scaled, X_test_scaled, y_train, y_test, **training_params
            )

            # 7. 学習完了フラグを設定（保存前に設定）
            self.is_trained = True

            # 8. モデルを保存
            if save_model:
                # training_resultからメタデータを構築
                model_metadata = {
                    "accuracy": training_result.get("accuracy", 0.0),
                    "precision": training_result.get("precision", 0.0),
                    "recall": training_result.get("recall", 0.0),
                    "f1_score": training_result.get("f1_score", 0.0),
                    "auc_score": training_result.get("auc_score", 0.0),
                    "training_samples": training_result.get("train_samples", 0),
                    "test_samples": training_result.get("test_samples", 0),
                    "feature_importance": training_result.get("feature_importance", {}),
                    "classification_report": training_result.get(
                        "classification_report", {}
                    ),
                    "best_iteration": training_result.get("best_iteration", 0),
                }
                model_path = self.save_model(
                    model_name or self.config.model.AUTO_STRATEGY_MODEL_NAME,
                    model_metadata,
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
            raise UnifiedModelError("評価対象の学習済みモデルがありません")

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

    def calculate_detailed_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        詳細な評価指標を計算

        Args:
            y_true: 実際のラベル
            y_pred: 予測ラベル
            y_pred_proba: 予測確率（オプション）

        Returns:
            評価指標の辞書
        """
        metrics = {}

        try:
            # 基本的な評価指標
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1_score"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )

            # 新しい評価指標
            metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, y_pred)
            )
            metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
            metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

            # 混同行列から特異度を計算
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):  # 二値分類の場合
                tn, fp, fn, tp = cm.ravel()
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                metrics["npv"] = (
                    float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                )  # Negative Predictive Value
                metrics["ppv"] = (
                    float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                )  # Positive Predictive Value

            # 確率ベースの指標（予測確率が利用可能な場合）
            if y_pred_proba is not None:
                try:
                    # AUC-ROC
                    if len(np.unique(y_true)) > 2:
                        # 多クラス分類
                        metrics["auc_roc"] = float(
                            roc_auc_score(
                                y_true,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                    else:
                        # 二値分類
                        metrics["auc_roc"] = float(
                            roc_auc_score(
                                y_true,
                                (
                                    y_pred_proba[:, 1]
                                    if y_pred_proba.ndim > 1
                                    else y_pred_proba
                                ),
                            )
                        )

                    # PR-AUC (Precision-Recall AUC)
                    if len(np.unique(y_true)) == 2:
                        metrics["auc_pr"] = float(
                            average_precision_score(
                                y_true,
                                (
                                    y_pred_proba[:, 1]
                                    if y_pred_proba.ndim > 1
                                    else y_pred_proba
                                ),
                            )
                        )

                    # Log Loss
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))

                    # Brier Score (二値分類のみ)
                    if len(np.unique(y_true)) == 2:
                        y_prob_positive = (
                            y_pred_proba[:, 1]
                            if y_pred_proba.ndim > 1
                            else y_pred_proba
                        )
                        metrics["brier_score"] = float(
                            brier_score_loss(y_true, y_prob_positive)
                        )

                except Exception as e:
                    logger.warning(f"確率ベース指標計算エラー: {e}")

        except Exception as e:
            logger.error(f"評価指標計算エラー: {e}")

        return metrics

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        予測を実行（継承クラスで実装）

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
        モデル学習の具体的な実装（継承クラスで実装）

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

    def _validate_training_data(self, training_data: pd.DataFrame) -> None:
        """入力データの検証"""
        if training_data is None or training_data.empty:
            raise UnifiedDataError("学習データが空です")

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in training_data.columns
        ]
        if missing_columns:
            raise UnifiedDataError(f"必要なカラムが不足しています: {missing_columns}")

        if len(training_data) < 100:
            raise UnifiedDataError("学習データが不足しています（最低100行必要）")

    def _calculate_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """特徴量を計算（外部市場データとFear & Greed Indexデータを含む）"""
        try:
            # 外部市場データを取得
            external_market_data = self._get_external_market_data(ohlcv_data)

            # Fear & Greed Indexデータを取得
            fear_greed_data = self._get_fear_greed_data(ohlcv_data)

            # 拡張された特徴量計算
            return self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                external_market_data=external_market_data,
                fear_greed_data=fear_greed_data,
            )

        except Exception as e:
            logger.warning(f"拡張特徴量計算でエラー、基本特徴量のみ使用: {e}")
            # フォールバック：基本特徴量のみ
            return self.feature_service.calculate_advanced_features(
                ohlcv_data, funding_rate_data, open_interest_data
            )

    def _get_external_market_data(
        self, ohlcv_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        外部市場データを取得

        Args:
            ohlcv_data: OHLCVデータ（期間の参考用）

        Returns:
            外部市場データのDataFrame（取得できない場合はNone）
        """
        try:
            if ohlcv_data.empty:
                return None

            # データの期間を取得
            if "timestamp" in ohlcv_data.columns:
                start_date_val = ohlcv_data["timestamp"].min()
                end_date_val = ohlcv_data["timestamp"].max()
            else:
                start_date_val = ohlcv_data.index.min()
                end_date_val = ohlcv_data.index.max()

            # datetime型に変換
            start_date = cast(datetime, pd.to_datetime(start_date_val).to_pydatetime())
            end_date = cast(datetime, pd.to_datetime(end_date_val).to_pydatetime())

            with SessionLocal() as db:
                repository = ExternalMarketRepository(db)

                # 外部市場データを取得
                external_data = repository.get_external_market_data(
                    start_time=start_date, end_time=end_date
                )

                if not external_data:
                    logger.info("外部市場データが見つかりませんでした")
                    return None

                # DataFrameに変換
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": data.data_timestamp,
                            "symbol": data.symbol,
                            "close": data.close,
                            "volume": data.volume,
                            "high": data.high,
                            "low": data.low,
                            "open": data.open,
                        }
                        for data in external_data
                    ]
                )

                if df.empty:
                    return None

                # タイムスタンプをインデックスに設定
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

                logger.info(
                    f"外部市場データを取得: {len(df)}行, シンボル: {df['symbol'].unique()}"
                )
                return df

        except Exception as e:
            logger.warning(f"外部市場データ取得エラー: {e}")
            return None

    def _get_fear_greed_data(self, ohlcv_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fear & Greed Indexデータを取得

        Args:
            ohlcv_data: OHLCVデータ（期間の参考用）

        Returns:
            Fear & Greed IndexデータのDataFrame（取得できない場合はNone）
        """
        try:
            if ohlcv_data.empty:
                return None

            # データの期間を取得
            if "timestamp" in ohlcv_data.columns:
                start_date_val = ohlcv_data["timestamp"].min()
                end_date_val = ohlcv_data["timestamp"].max()
            else:
                start_date_val = ohlcv_data.index.min()
                end_date_val = ohlcv_data.index.max()

            # datetime型に変換
            start_date = cast(datetime, pd.to_datetime(start_date_val).to_pydatetime())
            end_date = cast(datetime, pd.to_datetime(end_date_val).to_pydatetime())

            with SessionLocal() as db:
                repository = FearGreedIndexRepository(db)

                # Fear & Greed Indexデータを取得
                fear_greed_data = repository.get_fear_greed_data(
                    start_time=start_date, end_time=end_date
                )

                if not fear_greed_data:
                    logger.info("Fear & Greed Indexデータが見つかりませんでした")
                    return None

                # DataFrameに変換
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": data.data_timestamp,
                            "value": data.value,
                            "value_classification": data.value_classification,
                        }
                        for data in fear_greed_data
                    ]
                )

                if df.empty:
                    return None

                # タイムスタンプをインデックスに設定
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

                logger.info(f"Fear & Greed Indexデータを取得: {len(df)}行")
                return df

        except Exception as e:
            logger.warning(f"Fear & Greed Indexデータ取得エラー: {e}")
            return None

    def _generate_dynamic_labels(
        self, price_data: pd.Series, **training_params
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        動的ラベル生成

        Args:
            price_data: 価格データ（Close価格）
            **training_params: 学習パラメータ

        Returns:
            ラベルSeries, 閾値情報の辞書
        """
        try:
            # ラベル生成器を初期化
            label_generator = LabelGenerator()

            # 閾値計算方法を決定
            threshold_method_str = training_params.get(
                "threshold_method", "std_deviation"
            )

            # 文字列からEnumに変換
            method_mapping = {
                "fixed": ThresholdMethod.FIXED,
                "quantile": ThresholdMethod.QUANTILE,
                "std_deviation": ThresholdMethod.STD_DEVIATION,
                "adaptive": ThresholdMethod.ADAPTIVE,
            }

            threshold_method = method_mapping.get(
                threshold_method_str, ThresholdMethod.STD_DEVIATION
            )

            # 目標分布を設定
            target_distribution = training_params.get(
                "target_distribution", {"up": 0.33, "down": 0.33, "range": 0.34}
            )

            # 方法固有のパラメータを準備
            method_params = {}

            if threshold_method == ThresholdMethod.FIXED:
                method_params["threshold"] = training_params.get("threshold_up", 0.02)
            elif threshold_method == ThresholdMethod.STD_DEVIATION:
                method_params["std_multiplier"] = training_params.get(
                    "std_multiplier", 0.25
                )
            elif threshold_method in [
                ThresholdMethod.QUANTILE,
                ThresholdMethod.ADAPTIVE,
            ]:
                method_params["target_distribution"] = target_distribution

            # ラベルを生成
            labels, threshold_info = label_generator.generate_labels(
                price_data,
                method=threshold_method,
                target_distribution=target_distribution,
                **method_params,
            )

            # ラベル分布を検証
            validation_result = LabelGenerator.validate_label_distribution(labels)

            if not validation_result["is_valid"]:
                logger.warning("ラベル分布に問題があります:")
                for error in validation_result["errors"]:
                    logger.warning(f"  エラー: {error}")
                for warning in validation_result["warnings"]:
                    logger.warning(f"  警告: {warning}")

                # 1クラスしかない場合は適応的方法にフォールバック
                if labels.nunique() <= 1:
                    logger.info("適応的閾値計算にフォールバック")
                    labels, threshold_info = label_generator.generate_labels(
                        price_data,
                        method=ThresholdMethod.ADAPTIVE,
                        target_distribution=target_distribution,
                    )

            return labels, threshold_info

        except Exception as e:
            logger.error(f"動的ラベル生成エラー: {e}")
            # フォールバック：従来の固定閾値
            logger.info("従来の固定閾値にフォールバック")
            price_change = price_data.pct_change().shift(-1)
            threshold_up = training_params.get("threshold_up", 0.02)
            threshold_down = training_params.get("threshold_down", -0.02)

            labels = pd.Series(1, index=price_change.index, dtype=int)
            labels[price_change > threshold_up] = 2
            labels[price_change < threshold_down] = 0
            labels = labels.iloc[:-1]

            threshold_info = {
                "method": "fixed_fallback",
                "threshold_up": threshold_up,
                "threshold_down": threshold_down,
                "description": f"フォールバック固定閾値±{threshold_up*100:.2f}%",
            }

            return labels, threshold_info

    def _prepare_training_data(
        self, features_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """学習用データを準備（継承クラスでオーバーライド可能）"""
        # デフォルト実装：最後の列をラベルとして使用
        if features_df.empty:
            raise UnifiedDataError("特徴量データが空です")

        # 数値列のみを選択
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df_numeric = features_df[numeric_columns]

        # NaNを0で埋める
        features_df_clean = features_df_numeric.fillna(0)

        # 特徴量とラベルを分離（改善されたラベル生成ロジック）
        if "Close" in features_df_clean.columns:
            # 動的ラベル生成を使用
            labels, threshold_info = self._generate_dynamic_labels(
                features_df_clean["Close"], **training_params
            )

            # 閾値情報をログ出力
            logger.info(f"ラベル生成方法: {threshold_info['description']}")
            logger.info(
                f"使用閾値: {threshold_info['threshold_down']:.6f} ～ {threshold_info['threshold_up']:.6f}"
            )

            # 最後の行は予測できないので除外
            features_df_clean = features_df_clean.iloc[:-1]
        else:
            raise UnifiedDataError("価格データ（Close）が見つかりません")

        # 無効なデータを除外
        valid_mask = ~(features_df_clean.isnull().any(axis=1) | labels.isnull())
        features_clean = features_df_clean[valid_mask]
        labels_clean = labels[valid_mask]

        if len(features_clean) == 0:
            raise UnifiedDataError("有効な学習データがありません")

        self.feature_columns = features_clean.columns.tolist()

        logger.info(
            f"学習データ準備完了: {len(features_clean)}サンプル, {len(self.feature_columns)}特徴量"
        )
        logger.info(
            f"ラベル分布: 下落={sum(labels_clean==0)}, レンジ={sum(labels_clean==1)}, 上昇={sum(labels_clean==2)}"
        )

        return features_clean, labels_clean

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, **training_params
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データを分割"""
        test_size = training_params.get("test_size", 0.2)
        random_state = training_params.get("random_state", 42)

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

        return X_train, X_test, y_train, y_test

    def _preprocess_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データを前処理（スケーリング）"""
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
            raise UnifiedModelError("学習済みモデルがありません")

        # 基本的なメタデータを準備
        final_metadata = {
            "model_type": self.__class__.__name__,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "is_trained": self.is_trained,
        }
        # 提供されたメタデータで更新
        if metadata:
            final_metadata.update(metadata)

        model_path = model_manager.save_model(
            model=self.model,
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
        self.model = model_data.get("model")
        self.scaler = model_data.get("scaler")
        self.feature_columns = model_data.get("feature_columns")

        if self.model is None:
            raise UnifiedModelError("モデルデータにモデルが含まれていません")

        self.is_trained = True
        logger.info(f"モデル読み込み完了: {model_path}")
        return True
