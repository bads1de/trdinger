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
                logger.warning(f"不足しているカラム: {missing_columns}")

            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"不足しているカラム: {missing_columns}")

            logger.warning(f"指標計算開始: {indicator_type}, パラメータ: {parameters}")

            # パラメータマッピングをTechnicalIndicatorServiceに任せる
            mapped_parameters = parameters.copy()

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, mapped_parameters
            )

            logger.warning(
                f"指標計算成功: {indicator_type}, 結果タイプ: {type(result)}"
            )
            return result

        return _calculate_indicator()

    def _calculate_indicator_from_df(
        self, df: pd.DataFrame, indicator_type: str, parameters: Dict[str, Any]
    ) -> Union[
        np.ndarray, pd.Series, Tuple[np.ndarray, ...], Tuple[pd.Series, ...], None
    ]:
        """
        DataFrameから指標を計算

        MTFデータプロバイダーから取得したDataFrameに対して指標を計算します。

        Args:
            df: OHLCVデータを含むDataFrame
            indicator_type: 指標タイプ
            parameters: パラメータ

        Returns:
            計算結果
        """
        try:
            if df is None or df.empty:
                raise ValueError(f"データが空です: {indicator_type}")

            logger.debug(f"MTF指標計算開始: {indicator_type}, パラメータ: {parameters}")

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, parameters.copy()
            )

            logger.debug(
                f"MTF指標計算成功: {indicator_type}, 結果タイプ: {type(result)}"
            )
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

            logger.warning(
                f"指標初期化開始: {indicator_gene.type}, "
                f"パラメータ: {indicator_gene.parameters}, "
                f"タイムフレーム: {indicator_timeframe or 'デフォルト'}"
            )

            if strategy_instance is None:
                raise ValueError(f"戦略インスタンスがNoneです: {indicator_gene.type}")

            # MTF対応: タイムフレームに応じたデータを取得
            if indicator_timeframe and self.mtf_data_provider:
                # MTFデータプロバイダーからタイムフレームに応じたデータを取得
                mtf_df = self.mtf_data_provider.get_data(indicator_timeframe)
                logger.debug(
                    f"MTFデータ取得: {indicator_timeframe}, rows={len(mtf_df)}"
                )
                # DataFrameを使用して指標計算
                result = self._calculate_indicator_from_df(
                    mtf_df, indicator_gene.type, indicator_gene.parameters
                )
            else:
                # デフォルト: strategy_instance.data を使用
                result = self.calculate_indicator(
                    strategy_instance.data,
                    indicator_gene.type,
                    indicator_gene.parameters,
                )

            if result is not None:
                logger.warning(
                    f"指標計算結果取得: {indicator_gene.type}, タイプ: {type(result)}"
                )

                # MTF指標名のベースを決定（タイムフレームサフィックス付き）
                # IDを使って一意にする（SMA_10, SMA_50のように区別するため）
                indicator_id_suffix = (
                    f"_{indicator_gene.id[:8]}"
                    if hasattr(indicator_gene, "id") and indicator_gene.id
                    else ""
                )
                base_indicator_name = (
                    f"{indicator_gene.type}{timeframe_suffix}{indicator_id_suffix}"
                )

                # 指標をbacktesting.pyと互換性のある方法で確実に登録
                logger.warning(
                    f"指標登録開始: {base_indicator_name}, 結果タイプ: {type(result)}"
                )

                # indicators辞書を確実に作成
                if not hasattr(strategy_instance, "indicators"):
                    strategy_instance.indicators = {}
                    logger.warning("indicators辞書を新規作成")

                if isinstance(result, tuple):
                    # 複数の出力がある指標（MACD等）
                    logger.warning(
                        f"複数出力指標処理: {base_indicator_name}, 出力数: {len(result)}"
                    )
                    for i, output in enumerate(result):
                        indicator_name = f"{base_indicator_name}_{i}"

                        # 複数の方法で指標を確実に登録
                        try:
                            # 1. 戦略インスタンスの__dict__に直接追加（最も確実）
                            strategy_instance.__dict__[indicator_name] = output

                            # 2. setattr でも設定
                            setattr(strategy_instance, indicator_name, output)

                            # 3. indicators辞書にも保存
                            strategy_instance.indicators[indicator_name] = output

                            # 4. クラス変数としても設定
                            setattr(strategy_instance.__class__, indicator_name, output)

                            logger.warning(f"複数出力指標登録完了: {indicator_name}")

                        except Exception as e:
                            logger.error(
                                f"複数出力指標登録エラー: {indicator_name}, エラー: {e}"
                            )
                else:
                    # 単一出力の指標
                    logger.warning(f"単一出力指標処理: {base_indicator_name}")
                    try:
                        # 複数の方法で指標を確実に登録
                        # 1. 戦略インスタンスの__dict__に直接追加（最も確実）
                        strategy_instance.__dict__[base_indicator_name] = result

                        # 2. setattr でも設定
                        setattr(strategy_instance, base_indicator_name, result)

                        # 3. indicators辞書にも保存
                        strategy_instance.indicators[base_indicator_name] = result

                        # 4. クラス変数としても設定
                        setattr(
                            strategy_instance.__class__, base_indicator_name, result
                        )

                        logger.warning(f"単一出力指標登録完了: {base_indicator_name}")

                    except Exception as e:
                        logger.error(
                            f"単一出力指標登録エラー: {base_indicator_name}, エラー: {e}"
                        )

                # 詳細な登録確認
                logger.warning(f"指標登録確認開始: {base_indicator_name}")

                if isinstance(result, tuple):
                    # 複数出力指標の登録確認
                    all_registered = True
                    for i in range(len(result)):
                        indicator_name = f"{base_indicator_name}_{i}"

                        # 複数の方法で確認
                        in_dict = indicator_name in strategy_instance.__dict__
                        has_attr = hasattr(strategy_instance, indicator_name)
                        in_indicators = indicator_name in strategy_instance.indicators

                        logger.warning(
                            f"{indicator_name}: __dict__={in_dict}, hasattr={has_attr}, indicators={in_indicators}"
                        )

                        if not (in_dict and has_attr and in_indicators):
                            all_registered = False

                    if all_registered:
                        logger.warning(
                            f"✅ 複数出力指標登録確認成功: {base_indicator_name}"
                        )
                    else:
                        logger.error(
                            f"❌ 複数出力指標登録確認失敗: {base_indicator_name}"
                        )
                else:
                    # 単一出力指標の登録確認
                    in_dict = base_indicator_name in strategy_instance.__dict__
                    has_attr = hasattr(strategy_instance, base_indicator_name)
                    in_indicators = base_indicator_name in strategy_instance.indicators

                    logger.warning(
                        f"{base_indicator_name}: __dict__={in_dict}, hasattr={has_attr}, indicators={in_indicators}"
                    )

                    if in_dict and has_attr and in_indicators:
                        logger.warning(
                            f"✅ 単一出力指標登録確認成功: {base_indicator_name}"
                        )
                    else:
                        logger.error(
                            f"❌ 単一出力指標登録確認失敗: {base_indicator_name}"
                        )

                # 戦略インスタンスの全属性をログ出力（デバッグ用）
                all_attrs = [
                    attr for attr in dir(strategy_instance) if not attr.startswith("_")
                ]
                logger.warning(f"戦略インスタンス属性: {all_attrs}")
                logger.warning(
                    f"indicators辞書キー: {list(strategy_instance.indicators.keys()) if hasattr(strategy_instance, 'indicators') else 'なし'}"
                )
            else:
                logger.error(f"指標計算結果がNullです: {indicator_gene.type}")
                raise ValueError(f"指標計算に失敗しました: {indicator_gene.type}")

        _init_indicator()





