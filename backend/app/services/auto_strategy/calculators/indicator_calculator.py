"""
指標計算器

テクニカル指標の計算を担当します。
"""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

from app.services.indicators import TechnicalIndicatorService

from ..models.gene_strategy import IndicatorGene
from ..services.ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算器

    テクニカル指標の計算を担当します。
    """

    def __init__(self, ml_orchestrator=None, technical_indicator_service=None):
        """初期化"""
        self.technical_indicator_service = (
            technical_indicator_service or TechnicalIndicatorService()
        )
        self.ml_orchestrator = ml_orchestrator or MLOrchestrator()

    def calculate_indicator(
        self, data, indicator_type: str, parameters: Dict[str, Any]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...], None]:
        """
        指標計算

        Args:
            data: backtesting.pyのデータオブジェクト
            indicator_type: 指標タイプ
            parameters: パラメータ

        Returns:
            計算結果（numpy配列）
        """
        try:
            # backtesting.pyのデータオブジェクトをDataFrameに変換
            df = data.df

            # データの基本検証
            if df.empty:
                raise ValueError(f"データが空です: {indicator_type}")

            # ML指標の場合は専用サービスを使用
            if indicator_type.startswith("ML_"):
                # TODO: 将来的にはファンディングレートと建玉残高データも取得して渡す
                # 現在は不足特徴量を0で埋める処理で対応済み
                result = self.ml_orchestrator.calculate_single_ml_indicator(
                    indicator_type, df, funding_rate_data=None, open_interest_data=None
                )
                return result

            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"不足しているカラム: {missing_columns}")

            # パラメータマッピング（period -> length）
            mapped_parameters = parameters.copy()
            if "period" in mapped_parameters and "length" not in mapped_parameters:
                mapped_parameters["length"] = mapped_parameters["period"]

            logger.warning(
                f"指標計算開始: {indicator_type}, 元パラメータ: {parameters}, マップ後: {mapped_parameters}"
            )

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, mapped_parameters
            )

            logger.warning(
                f"指標計算成功: {indicator_type}, 結果タイプ: {type(result)}"
            )
            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}", exc_info=True)

            # 基本的な移動平均指標の場合はフォールバック処理を試行
            if indicator_type in ["SMA", "EMA", "WMA"] and "period" in parameters:
                try:
                    logger.warning(
                        f"フォールバック処理を試行: {indicator_type}, パラメータ: {parameters}"
                    )
                    period = parameters["period"]

                    # 期間の妥当性チェック
                    if not isinstance(period, (int, float)) or period <= 0:
                        logger.error(f"無効な期間: {period}")
                        raise ValueError(f"無効な期間: {period}")

                    period = int(period)
                    close_data = df["Close"].values

                    # データ長の妥当性チェック
                    if len(close_data) < period:
                        logger.error(
                            f"データ長({len(close_data)})が期間({period})より短い"
                        )
                        raise ValueError(
                            f"データ長({len(close_data)})が期間({period})より短い"
                        )

                    if indicator_type == "SMA":
                        # 簡易SMA計算
                        result = np.convolve(
                            close_data, np.ones(period) / period, mode="valid"
                        )
                        # 元の長さに合わせてNaNでパディング
                        padded_result = np.full(len(close_data), np.nan)
                        padded_result[period - 1 :] = result
                        logger.info(
                            f"フォールバックSMA計算成功: 期間={period}, 結果長={len(padded_result)}"
                        )
                        return padded_result
                    elif indicator_type == "EMA":
                        # 簡易EMA計算
                        alpha = 2.0 / (period + 1)
                        result = np.full(len(close_data), np.nan)
                        result[0] = close_data[0]
                        for i in range(1, len(close_data)):
                            if not np.isnan(result[i - 1]):
                                result[i] = (
                                    alpha * close_data[i] + (1 - alpha) * result[i - 1]
                                )
                        logger.info(
                            f"フォールバックEMA計算成功: 期間={period}, 結果長={len(result)}"
                        )
                        return result
                    elif indicator_type == "WMA":
                        # 簡易WMA計算（重み付き移動平均）
                        weights = np.arange(1, period + 1)
                        weights = weights / weights.sum()
                        result = np.convolve(close_data, weights[::-1], mode="valid")
                        padded_result = np.full(len(close_data), np.nan)
                        padded_result[period - 1 :] = result
                        logger.info(
                            f"フォールバックWMA計算成功: 期間={period}, 結果長={len(padded_result)}"
                        )
                        return padded_result

                except Exception as fallback_error:
                    logger.error(
                        f"フォールバック処理も失敗 {indicator_type}: {fallback_error}"
                    )

            # エラーを再発生させて上位で適切に処理
            raise

    def init_indicator(self, indicator_gene: IndicatorGene, strategy_instance):
        """
        単一指標の初期化

        Args:
            indicator_gene: 指標遺伝子
            strategy_instance: 戦略インスタンス
        """
        try:
            logger.warning(
                f"指標初期化開始: {indicator_gene.type}, パラメータ: {indicator_gene.parameters}"
            )

            # 指標計算を直接実行
            result = self.calculate_indicator(
                strategy_instance.data, indicator_gene.type, indicator_gene.parameters
            )

            if result is not None:
                logger.warning(
                    f"指標計算結果取得: {indicator_gene.type}, タイプ: {type(result)}"
                )

                # 指標をstrategy.I()で登録（クロージャ問題を完全に回避）
                if isinstance(result, tuple):
                    # 複数の出力がある指標（MACD等）
                    for i, output in enumerate(result):
                        indicator_name = f"{indicator_gene.type}_{i}"

                        # 複数の方法で指標を登録（backtesting.py対応）
                        # 1. 直接属性として設定
                        setattr(strategy_instance, indicator_name, output)

                        # 2. indicators辞書にも保存（永続化）
                        if not hasattr(strategy_instance, "indicators"):
                            strategy_instance.indicators = {}
                        strategy_instance.indicators[indicator_name] = output

                        # 3. クラス変数としても設定（backtesting.py対応）
                        setattr(strategy_instance.__class__, indicator_name, output)

                        logger.warning(f"複数出力指標登録: {indicator_name}")
                else:
                    # 単一出力の指標
                    # 複数の方法で指標を登録（backtesting.py対応）
                    # 1. 直接属性として設定
                    setattr(strategy_instance, indicator_gene.type, result)

                    # 2. indicators辞書にも保存（永続化）
                    if not hasattr(strategy_instance, "indicators"):
                        strategy_instance.indicators = {}
                    strategy_instance.indicators[indicator_gene.type] = result

                    # 3. クラス変数としても設定（backtesting.py対応）
                    setattr(strategy_instance.__class__, indicator_gene.type, result)

                    logger.warning(f"単一出力指標登録: {indicator_gene.type}")

                # 登録確認（複数出力指標対応）
                if isinstance(result, tuple):
                    # 複数出力指標の場合、各出力の登録を確認
                    all_registered = True
                    for i in range(len(result)):
                        indicator_name = f"{indicator_gene.type}_{i}"
                        if not hasattr(strategy_instance, indicator_name):
                            all_registered = False
                            break

                    if all_registered:
                        logger.warning(
                            f"複数出力指標登録確認成功: {indicator_gene.type}"
                        )
                    else:
                        logger.error(f"複数出力指標登録確認失敗: {indicator_gene.type}")
                else:
                    # 単一出力指標の場合
                    if hasattr(strategy_instance, indicator_gene.type):
                        logger.warning(f"指標登録確認成功: {indicator_gene.type}")
                    else:
                        logger.error(f"指標登録確認失敗: {indicator_gene.type}")
            else:
                logger.error(f"指標計算結果がNullです: {indicator_gene.type}")
                raise ValueError(f"指標計算に失敗しました: {indicator_gene.type}")

        except Exception as e:
            logger.error(f"指標初期化エラー {indicator_gene.type}: {e}", exc_info=True)
            # エラーを再発生させて上位で適切に処理
            raise
