"""
AutoML特徴量生成サービス

AutoML特徴量生成のビジネスロジックを統合管理するサービスクラス。
APIエンドポイントからビジネスロジックを分離し、責務を明確化します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)

from .feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class AutoMLFeatureGenerationService:
    """
    AutoML特徴量生成サービス

    OHLCVデータ取得、ターゲット変数生成、特徴量生成の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。
    """

    def __init__(self, db_session: Session):
        """
        初期化

        Args:
            db_session: データベースセッション
        """
        self.db_session = db_session
        self.market_data_service = MarketDataOrchestrationService(db_session)
        # AutoML機能を有効にしてFeatureEngineeringServiceを初期化
        from .automl_features.automl_config import AutoMLConfig

        automl_config = AutoMLConfig.get_financial_optimized_config()
        self.feature_service = FeatureEngineeringService(automl_config=automl_config)

    async def generate_features(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        automl_config: Optional[Dict[str, Any]] = None,
        include_target: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        AutoML特徴量を生成

        Args:
            symbol: 取引シンボル
            timeframe: 時間枠
            limit: データ数
            automl_config: AutoML設定
            include_target: ターゲット変数を含むか

        Returns:
            特徴量DataFrame, 統計情報の辞書

        Raises:
            Exception: データ取得または特徴量生成に失敗した場合
        """
        logger.info(f"AutoML特徴量生成開始: {symbol}, {timeframe}, limit={limit}")

        # OHLCVデータを取得
        ohlcv_data = await self._get_ohlcv_data(symbol, timeframe, limit)

        # ターゲット変数を生成（必要な場合）
        target = None
        if include_target:
            target = self._generate_target_variable(ohlcv_data)

        # AutoML設定を適用
        if automl_config:
            self.feature_service._update_automl_config(automl_config)

        # 特徴量生成を実行
        result_df = self.feature_service.calculate_enhanced_features(
            ohlcv_data=ohlcv_data, target=target
        )

        # 統計情報を取得
        stats = self.feature_service.get_enhancement_stats()

        logger.info(f"AutoML特徴量生成完了: {len(result_df.columns)}個の特徴量")

        return result_df, stats

    async def _get_ohlcv_data(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得

        Args:
            symbol: 取引シンボル
            timeframe: 時間枠
            limit: データ数

        Returns:
            OHLCVデータのDataFrame

        Raises:
            Exception: データ取得に失敗した場合
        """
        try:

            # MarketDataOrchestrationServiceを使用してデータを取得
            data_response = await self.market_data_service.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )

            if not data_response.get("success", False):
                raise Exception(
                    f"OHLCVデータ取得失敗: {data_response.get('message', 'Unknown error')}"
                )

            ohlcv_records = data_response.get("data", [])
            if not ohlcv_records:
                raise Exception(
                    f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした"
                )

            # DataFrameに変換
            ohlcv_data = self._convert_to_dataframe(ohlcv_records)

            return ohlcv_data

        except Exception as e:
            logger.error(f"OHLCVデータ取得エラー: {e}")
            raise Exception(f"OHLCVデータの取得に失敗しました: {e}")

    def _convert_to_dataframe(self, ohlcv_records: list) -> pd.DataFrame:
        """
        OHLCVレコードをDataFrameに変換

        Args:
            ohlcv_records: OHLCVレコードのリスト

        Returns:
            OHLCVデータのDataFrame
        """
        try:
            # レコードの形式を確認し、適切にDataFrameに変換
            if not ohlcv_records:
                raise ValueError("空のOHLCVレコードが提供されました")

            # 最初のレコードの形式を確認
            first_record = ohlcv_records[0]

            if hasattr(first_record, "__dict__"):
                # SQLAlchemyモデルの場合
                data = []
                for record in ohlcv_records:
                    data.append(
                        {
                            "timestamp": record.timestamp,
                            "Open": (
                                float(record.open)
                                if hasattr(record, "open") and record.open is not None
                                else 0.0
                            ),
                            "High": (
                                float(record.high)
                                if hasattr(record, "high") and record.high is not None
                                else 0.0
                            ),
                            "Low": (
                                float(record.low)
                                if hasattr(record, "low") and record.low is not None
                                else 0.0
                            ),
                            "Close": (
                                float(record.close)
                                if hasattr(record, "close") and record.close is not None
                                else 0.0
                            ),
                            "Volume": (
                                float(record.volume)
                                if hasattr(record, "volume")
                                and record.volume is not None
                                else 0.0
                            ),
                        }
                    )
                df = pd.DataFrame(data)
            elif isinstance(first_record, dict):
                # 辞書の場合
                df = pd.DataFrame(ohlcv_records)
                # カラム名を正規化
                column_mapping = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                df = df.rename(columns=column_mapping)
            else:
                raise ValueError(f"未対応のレコード形式: {type(first_record)}")

            # timestampをインデックスに設定
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

            # データ型を確保
            numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # NaNを除去
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"DataFrame変換エラー: {e}")
            raise Exception(f"OHLCVデータのDataFrame変換に失敗しました: {e}")

    def _generate_target_variable(
        self, ohlcv_data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """
        ターゲット変数を生成

        Args:
            ohlcv_data: OHLCVデータ

        Returns:
            ターゲット変数のSeries（計算できない場合はNone）
        """
        try:
            if ohlcv_data.empty or "Close" not in ohlcv_data.columns:
                logger.warning("ターゲット変数計算用のデータが不足しています")
                return None

            # 価格変化率を計算（次の期間の価格変化）
            close_prices = ohlcv_data["Close"].copy()

            # 将来の価格変化率を計算（24時間後の変化率）
            prediction_horizon = 24  # デフォルト値
            future_returns = close_prices.pct_change(periods=prediction_horizon).shift(
                -prediction_horizon
            )

            # 閾値を使用してクラス分類
            threshold_up = 0.02  # 2%上昇
            threshold_down = -0.02  # 2%下落

            # 3クラス分類：0=下落、1=横ばい、2=上昇
            target = pd.Series(1, index=future_returns.index)  # デフォルトは横ばい
            target[future_returns > threshold_up] = 2  # 上昇
            target[future_returns < threshold_down] = 0  # 下落

            # NaNを除去
            target = target.dropna()

            logger.info(f"ターゲット変数生成完了: {len(target)}サンプル")
            logger.info(
                f"クラス分布 - 下落: {(target == 0).sum()}, 横ばい: {(target == 1).sum()}, 上昇: {(target == 2).sum()}"
            )

            return target

        except Exception as e:
            logger.warning(f"ターゲット変数生成エラー: {e}")
            return None

    def get_feature_names(self, result_df: pd.DataFrame) -> list:
        """
        特徴量名のリストを取得

        Args:
            result_df: 特徴量DataFrame

        Returns:
            特徴量名のリスト
        """
        return list(result_df.columns)

    def get_processing_time(self, stats: Dict[str, Any]) -> float:
        """
        処理時間を取得

        Args:
            stats: 統計情報

        Returns:
            処理時間（秒）
        """
        return stats.get("total_time", 0.0)

    def validate_automl_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        AutoML設定を検証

        Args:
            config_dict: AutoML設定辞書

        Returns:
            Dict[str, Any]: 検証結果
                - valid: bool - 設定が有効かどうか
                - errors: List[str] - エラーメッセージのリスト
                - warnings: List[str] - 警告メッセージのリスト
        """
        return self.feature_service.validate_automl_config(config_dict)

    def clear_automl_cache(self):
        """
        AutoMLキャッシュをクリア

        Returns:
            None
        """
        self.feature_service.clear_automl_cache()
