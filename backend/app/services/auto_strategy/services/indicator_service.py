"""
指標計算器

テクニカル指標の計算を担当します。
マルチタイムフレーム（MTF）対応。
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.services.indicators import TechnicalIndicatorService

from ..genes import IndicatorGene
from .mtf_data_provider import MultiTimeframeDataProvider

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算器

    テクニカル指標の計算を担当します。
    マルチタイムフレーム（MTF）対応。
    """

    def __init__(
        self,
        technical_indicator_service: Optional[TechnicalIndicatorService] = None,
        mtf_data_provider: Optional[MultiTimeframeDataProvider] = None,
    ):
        """
        初期化

        Args:
            technical_indicator_service: テクニカル指標サービス
            mtf_data_provider: MTFデータプロバイダー（オプション）
        """
        self.technical_indicator_service = (
            technical_indicator_service or TechnicalIndicatorService()
        )
        self.mtf_data_provider = mtf_data_provider

    def calculate_indicator(
        self, data, indicator_type: str, parameters: Dict[str, Any]
    ) -> Union[
        np.ndarray, pd.Series, Tuple[np.ndarray, ...], Tuple[pd.Series, ...], None
    ]:
        """
        指標計算

        Args:
            data: backtesting.pyのデータオブジェクト
            indicator_type: 指標タイプ
            parameters: パラメータ

        Returns:
            計算結果（numpy配列またはpandas Series、またはそのtuple）
            pandasオンリー移行対応により、pd.SeriesとTuple[pd.Series, ...]も返却可能。
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context=f"指標計算 ({indicator_type})", is_api_call=False)
        def _calculate_indicator():
            # backtesting.pyのデータオブジェクトをDataFrameに変換
            if data is None:
                raise ValueError(f"データオブジェクトがNoneです: {indicator_type}")
            df = data.df

            # データの基本検証
            if df.empty:
                raise ValueError(f"データが空です: {indicator_type}")

            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                # logger.warning(f"不足しているカラム: {missing_columns}")
                pass

            # logger.warning(f"指標計算開始: {indicator_type}, パラメータ: {parameters}")

            # パラメータマッピング
            # ここで mapped_parameters を確実に定義する
            mapped_parameters = parameters.copy()

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, mapped_parameters
            )

            # logger.warning(
            #     f"指標計算成功: {indicator_type}, 結果タイプ: {type(result)}"
            # )
            return result

        return _calculate_indicator()

    def _calculate_indicator_from_df(
        self, df: pd.DataFrame, indicator_type: str, parameters: Dict[str, Any]
    ) -> Union[
        np.ndarray, pd.Series, Tuple[np.ndarray, ...], Tuple[pd.Series, ...], None
    ]:
        """
        OHLCVのDataFrameを直接指定して指標を計算

        MTFデータプロバイダーから取得した、異なるタイムフレームの
        DataFrameに対して指標計算を行う際に使用されます。

        Args:
            df: OHLCVデータ（Open, High, Low, Close, Volume）を含むDataFrame
            indicator_type: 指標の名称
            parameters: 指標の計算パラメータ

        Returns:
            計算された指標値（単一または複数のSeries/NDArray）
        """
        try:
            if df is None or df.empty:
                raise ValueError(f"データが空です: {indicator_type}")

            # logger.debug(f"MTF指標計算開始: {indicator_type}, パラメータ: {parameters}")

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, parameters.copy()
            )

            # logger.debug(
            #     f"MTF指標計算成功: {indicator_type}, 結果タイプ: {type(result)}"
            # )
            return result

        except Exception as e:
            logger.error(f"MTF指標計算エラー ({indicator_type}): {e}", exc_info=True)
            return None

    def init_indicator(self, indicator_gene: IndicatorGene, strategy_instance):
        """
        単一指標の初期化

        Args:
            indicator_gene: 指標遺伝子
            strategy_instance: 戦略インスタンス

        MTF対応:
            indicator_gene.timeframe が設定されている場合、
            MTFデータプロバイダーから適切なタイムフレームのデータを取得して
            指標を計算します。
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=f"指標初期化 ({indicator_gene.type})", is_api_call=False
        )
        def _init_indicator():
            indicator_timeframe = getattr(indicator_gene, "timeframe", None)
            timeframe_suffix = f"_{indicator_timeframe}" if indicator_timeframe else ""

            # logger.warning(
            #     f"指標初期化開始: {indicator_gene.type}, "
            #     f"パラメータ: {indicator_gene.parameters}, "
            #     f"タイムフレーム: {indicator_timeframe or 'デフォルト'}"
            # )

            if strategy_instance is None:
                raise ValueError(f"戦略インスタンスがNoneです: {indicator_gene.type}")

            # MTF対応: タイムフレームに応じたデータを取得
            if indicator_timeframe and self.mtf_data_provider:
                # MTFデータプロバイダーからタイムフレームに応じたデータを取得
                mtf_df = self.mtf_data_provider.get_data(indicator_timeframe)
                # logger.debug(
                #     f"MTFデータ取得: {indicator_timeframe}, rows={len(mtf_df)}"
                # )
                # DataFrameを使用して指標計算
                raw_result = self._calculate_indicator_from_df(
                    mtf_df, indicator_gene.type, indicator_gene.parameters
                )

                # ベースデータのインデックスに合わせてリインデックス（ffill）
                # 1つ前の（確定済みの）足を参考にするため、上位足の時点で1つシフトする
                # これにより未来予知（Look-ahead bias）を防止する
                if raw_result is not None:
                    base_index = strategy_instance.data.df.index
                    
                    def _align_to_base(series: pd.Series) -> pd.Series:
                        # 上位足の時点で1つシフト（未来予知防止）
                        shifted = series.shift(1)
                        
                        # タイムゾーン情報を合わせる（必要な場合）
                        if shifted.index.tz != base_index.tz:
                            shifted.index = shifted.index.tz_convert(base_index.tz)
                        
                        return shifted.reindex(base_index, method="ffill")

                    if isinstance(raw_result, tuple):
                        result = tuple(_align_to_base(s) for s in raw_result)
                    else:
                        result = _align_to_base(raw_result)
                else:
                    result = None
            else:
                # デフォルト: strategy_instance.data を使用
                result = self.calculate_indicator(
                    strategy_instance.data,
                    indicator_gene.type,
                    indicator_gene.parameters,
                )

            if result is not None:
                # MTF指標名のベースを決定（タイムフレームサフィックス付き）
                indicator_id_suffix = (
                    f"_{indicator_gene.id[:8]}"
                    if hasattr(indicator_gene, "id") and indicator_gene.id
                    else ""
                )
                base_indicator_name = (
                    f"{indicator_gene.type}{timeframe_suffix}{indicator_id_suffix}"
                )

                # indicators辞書を確実に作成
                if not hasattr(strategy_instance, "indicators"):
                    strategy_instance.indicators = {}

                # 指標を戦略インスタンスに登録する共通ヘルパー
                def _register(name, val):
                    try:
                        # 1. indicators辞書（独自の管理用）
                        strategy_instance.indicators[name] = val
                        # 2. インスタンス属性（backtesting.pyアクセス用）
                        setattr(strategy_instance, name, val)
                    except Exception as e:
                        logger.error(f"指標登録エラー {name}: {e}")

                if isinstance(result, tuple):
                    # 複数の出力がある指標（MACD等）
                    for i, output in enumerate(result):
                        _register(f"{base_indicator_name}_{i}", output)
                else:
                    # 単一出力の指標
                    _register(base_indicator_name, result)

                # logger.debug(f"指標登録完了: {base_indicator_name}")
            else:
                logger.error(f"指標計算結果がNullです: {indicator_gene.type}")
                raise ValueError(f"指標計算に失敗しました: {indicator_gene.type}")

        _init_indicator()