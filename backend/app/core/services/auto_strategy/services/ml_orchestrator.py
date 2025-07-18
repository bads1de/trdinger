"""
ML オーケストレーター

MLIndicatorServiceの責任を分離し、軽量化したオーケストレーター。
各ML関連サービスを統合し、一貫性のあるML指標を提供します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from app.core.services.ml.config import ml_config
from app.core.utils.unified_error_handler import (
    UnifiedErrorHandler,
    MLDataError,
    MLModelError,
    MLValidationError,
    unified_timeout_decorator,
    unified_safe_operation,
    unified_operation_context,
)
from app.core.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.core.services.ml.signal_generator import MLSignalGenerator
from app.core.services.ml.model_manager import model_manager
from app.core.services.ml.performance_extractor import performance_extractor
from app.core.services.ml.lightgbm_trainer import LightGBMTrainer
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.connection import get_db

logger = logging.getLogger(__name__)


class MLOrchestrator:
    """
    ML オーケストレーター

    責任を分離した軽量なMLサービス統合クラス。
    特徴量計算、モデル予測、結果の統合を行います。

    MLServiceInterfaceを実装し、統一されたML予測・学習APIを提供します。
    """

    def __init__(
        self,
        ml_signal_generator: Optional[MLSignalGenerator] = None,
        trainer: Optional[LightGBMTrainer] = None,
    ):
        """
        初期化

        Args:
            ml_signal_generator: MLSignalGeneratorインスタンス（オプション）
            trainer: LightGBMTrainerインスタンス（オプション、学習機能用）
        """
        self.config = ml_config
        self.feature_service = FeatureEngineeringService()
        self.ml_generator = (
            ml_signal_generator if ml_signal_generator else MLSignalGenerator()
        )

        # BacktestDataServiceを初期化（完全なデータセット取得用）
        self._backtest_data_service = None
        # 学習機能のためのトレーナー（MLIndicatorServiceとの統合）
        self.trainer = trainer if trainer else LightGBMTrainer()
        # モデル状態の統合管理
        self.is_model_loaded = self.trainer.is_trained or getattr(
            self.ml_generator, "is_trained", False
        )
        self._last_predictions = self.config.prediction.get_default_predictions()

        # 既存の学習済みモデルを自動読み込み
        self._try_load_latest_model()

    @property
    def backtest_data_service(self) -> BacktestDataService:
        """BacktestDataServiceのインスタンスを取得（遅延初期化）"""
        if self._backtest_data_service is None:
            # データベース接続を取得してリポジトリを初期化
            db = next(get_db())
            try:
                ohlcv_repo = OHLCVRepository(db)
                fr_repo = FundingRateRepository(db)
                oi_repo = OpenInterestRepository(db)
                self._backtest_data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
            finally:
                db.close()
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

                # 特徴量計算が失敗した場合はデフォルト値を返す
                if features_df is None or features_df.empty:
                    logger.warning(
                        "特徴量計算が失敗しました。デフォルト値を使用します。"
                    )
                    return self._get_default_indicators(len(df))

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
                logger.warning(f"ML指標計算で検証エラー: {e}")
                return self._get_default_indicators(len(df))
            except Exception as e:
                logger.error(f"ML指標計算で予期しないエラー: {e}")
                return self._get_default_indicators(len(df))

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
        try:
            # データフレームの基本検証
            if df is None or df.empty:
                logger.warning(f"空のデータフレームが提供されました: {indicator_type}")
                return np.full(100, self.config.prediction.DEFAULT_UP_PROB)

            ml_indicators = self.calculate_ml_indicators(
                df, funding_rate_data, open_interest_data
            )

            if indicator_type in ml_indicators:
                return ml_indicators[indicator_type]
            else:
                logger.warning(f"未知のML指標タイプ: {indicator_type}")
                return np.full(len(df), self.config.prediction.DEFAULT_UP_PROB)

        except Exception as e:
            logger.error(f"単一ML指標計算エラー {indicator_type}: {e}")
            default_length = len(df) if df is not None and not df.empty else 100
            return np.full(default_length, self.config.prediction.DEFAULT_UP_PROB)

    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            success = self.ml_generator.load_model(model_path)
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
        print("=== DEBUG: MLOrchestrator.get_model_status呼び出し ===")

        status = {
            "is_model_loaded": self.is_model_loaded,
            "is_trained": getattr(self.ml_generator, "is_trained", False),
            "last_predictions": self._last_predictions,
            "feature_count": (
                len(self.ml_generator.feature_columns)
                if self.ml_generator.feature_columns
                else 0
            ),
        }

        # 最新のモデルファイルから性能指標を取得
        try:
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                print(f"=== DEBUG: 最新モデルファイル: {latest_model} ===")
                performance_metrics = performance_extractor.extract_performance_metrics(
                    latest_model
                )
                status["performance_metrics"] = performance_metrics
                print(f"=== DEBUG: 性能指標を追加: {performance_metrics} ===")
            else:
                print("=== DEBUG: 最新モデルファイルが見つかりません ===")
        except Exception as e:
            print(f"=== DEBUG: 性能指標取得エラー: {e} ===")
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
            logger.debug(f"ML予測値を更新: {predictions}")
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
        if self.is_model_loaded and getattr(self.ml_generator, "is_trained", False):
            return self.ml_generator.get_feature_importance(top_n)
        else:
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

    # MLIndicatorServiceから統合された学習機能
    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        train_test_split: float = 0.8,
        cross_validation_folds: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 100,
        max_depth: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習（MLIndicatorServiceから統合）

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            train_test_split: トレーニング/テスト分割比率
            cross_validation_folds: クロスバリデーション分割数
            random_state: ランダムシード
            early_stopping_rounds: 早期停止ラウンド数
            max_depth: 最大深度
            n_estimators: 推定器数
            learning_rate: 学習率

        Returns:
            学習結果
        """
        try:
            # BaseMLTrainerに委譲
            result = self.trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name="auto_strategy_ml_model",
                test_size=1 - train_test_split,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )

            self.is_model_loaded = True
            logger.info("MLモデル学習完了")

            return result

        except Exception as e:
            logger.error(f"MLモデル学習エラー: {e}")
            raise

    def save_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        学習済みモデルを保存（MLServiceInterface実装）

        Args:
            model_name: モデル名
            metadata: メタデータ（オプション）

        Returns:
            保存されたモデルのパス

        Raises:
            MLValidationError: モデルが学習されていない場合
            MLModelError: モデルの保存に失敗した場合
        """
        if not self.trainer.is_trained:
            raise MLValidationError("学習されていないモデルは保存できません")

        model_path = self.trainer.save_model(model_name, metadata)
        if model_path is None:
            raise MLModelError("モデルの保存に失敗しました。")

        return model_path

    def _validate_input_data(self, df: pd.DataFrame):
        """入力データの検証"""
        required_columns = ["open", "high", "low", "close", "volume"]
        UnifiedErrorHandler.validate_dataframe(
            df, required_columns=required_columns, min_rows=1
        )

    def _limit_data_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """データサイズを制限"""
        max_rows = self.config.data_processing.MAX_OHLCV_ROWS
        if len(df) > max_rows:
            logger.warning(f"大量のデータ（{len(df)}行）、最新{max_rows}行に制限")
            return df.tail(max_rows)
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
        """特徴量を計算"""
        try:
            logger.debug(f"特徴量計算開始 - データ形状: {df.shape}")
            logger.debug(f"カラム名: {list(df.columns)}")

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
            logger.debug(f"変換後カラム名: {list(df_for_features.columns)}")

            features_df = self.feature_service.calculate_advanced_features(
                df_for_features, funding_rate_data, open_interest_data
            )
            if features_df is not None:
                logger.debug(f"特徴量計算完了 - 特徴量形状: {features_df.shape}")
            else:
                logger.warning("特徴量計算結果がNone")
            return features_df
        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            return None

    @unified_safe_operation(
        default_return=ml_config.prediction.get_default_predictions(),
        context="ML予測でエラーが発生しました",
    )
    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """安全なML予測実行"""
        # 予測を実行
        predictions = self.ml_generator.predict(features_df)

        # 予測値の妥当性チェック
        UnifiedErrorHandler.validate_predictions(predictions)
        self._last_predictions = predictions
        return predictions

    def _expand_predictions_to_data_length(
        self, predictions: Dict[str, float], data_length: int
    ) -> Dict[str, np.ndarray]:
        """予測値をデータ長に拡張"""
        try:
            logger.debug(f"予測値拡張: {predictions} -> データ長: {data_length}")
            result = {
                "ML_UP_PROB": np.full(data_length, predictions["up"]),
                "ML_DOWN_PROB": np.full(data_length, predictions["down"]),
                "ML_RANGE_PROB": np.full(data_length, predictions["range"]),
            }
            logger.debug(
                f"拡張結果サンプル: UP={result['ML_UP_PROB'][0]}, DOWN={result['ML_DOWN_PROB'][0]}, RANGE={result['ML_RANGE_PROB'][0]}"
            )
            return result
        except Exception as e:
            logger.error(f"予測値拡張エラー: {e}")
            default_result = self._get_default_indicators(data_length)
            logger.debug(f"デフォルト値使用: {default_result}")
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
        """デフォルトのML指標を取得"""
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
                success = self.ml_generator.load_model(latest_model)
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

            # シンボルを推定（デフォルトはBTC/USDT）
            symbol = "BTC/USDT:USDT"  # TODO: 実際のシンボルを取得する方法を実装

            # BacktestDataServiceを使用して拡張データを取得
            enhanced_df = self.backtest_data_service.get_data_for_backtest(
                symbol=symbol,
                timeframe="1h",  # TODO: 実際のタイムフレームを取得
                start_date=start_date,
                end_date=end_date,
            )

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
                logger.debug(
                    f"ファンディングレートデータ抽出: {len(funding_rate_data)}行"
                )

            # 建玉残高データを抽出
            if "open_interest" in enhanced_df.columns:
                open_interest_data = enhanced_df[["open_interest"]].copy()
                logger.debug(f"建玉残高データ抽出: {len(open_interest_data)}行")

            return funding_rate_data, open_interest_data

        except Exception as e:
            logger.error(f"FR/OIデータ抽出エラー: {e}")
            return None, None
