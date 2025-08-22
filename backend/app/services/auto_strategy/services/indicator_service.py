"""
指標計算器（サービス統合版）

テクニカル指標の計算を担当します。
"""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

from app.services.indicators import TechnicalIndicatorService

from ..models.gene_strategy import IndicatorGene
from .ml_orchestrator import MLOrchestrator

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context=f"指標計算 ({indicator_type})", is_api_call=False)
        def _calculate_indicator():
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
            if "period" in mapped_parameters:
                if "length" not in mapped_parameters:
                    mapped_parameters["length"] = mapped_parameters["period"]
                # tclengthはSTCインジケータ専用パラメータなのでSTCのみに追加
                if indicator_type == "STC" and "tclength" not in mapped_parameters:
                    mapped_parameters["tclength"] = mapped_parameters["period"]

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

        return _calculate_indicator()

    def init_indicator(self, indicator_gene: IndicatorGene, strategy_instance):
        """
        単一指標の初期化

        Args:
            indicator_gene: 指標遺伝子
            strategy_instance: 戦略インスタンス
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=f"指標初期化 ({indicator_gene.type})", is_api_call=False
        )
        def _init_indicator():
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

                # 指標をbacktesting.pyと互換性のある方法で確実に登録
                logger.warning(
                    f"指標登録開始: {indicator_gene.type}, 結果タイプ: {type(result)}"
                )

                # indicators辞書を確実に作成
                if not hasattr(strategy_instance, "indicators"):
                    strategy_instance.indicators = {}
                    logger.warning("indicators辞書を新規作成")

                if isinstance(result, tuple):
                    # 複数の出力がある指標（MACD等）
                    logger.warning(
                        f"複数出力指標処理: {indicator_gene.type}, 出力数: {len(result)}"
                    )
                    for i, output in enumerate(result):
                        indicator_name = f"{indicator_gene.type}_{i}"

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
                    logger.warning(f"単一出力指標処理: {indicator_gene.type}")
                    try:
                        # 複数の方法で指標を確実に登録
                        # 1. 戦略インスタンスの__dict__に直接追加（最も確実）
                        strategy_instance.__dict__[indicator_gene.type] = result

                        # 2. setattr でも設定
                        setattr(strategy_instance, indicator_gene.type, result)

                        # 3. indicators辞書にも保存
                        strategy_instance.indicators[indicator_gene.type] = result

                        # 4. クラス変数としても設定
                        setattr(
                            strategy_instance.__class__, indicator_gene.type, result
                        )

                        logger.warning(f"単一出力指標登録完了: {indicator_gene.type}")

                    except Exception as e:
                        logger.error(
                            f"単一出力指標登録エラー: {indicator_gene.type}, エラー: {e}"
                        )

                # 詳細な登録確認
                logger.warning(f"指標登録確認開始: {indicator_gene.type}")

                if isinstance(result, tuple):
                    # 複数出力指標の登録確認
                    all_registered = True
                    for i in range(len(result)):
                        indicator_name = f"{indicator_gene.type}_{i}"

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
                            f"✅ 複数出力指標登録確認成功: {indicator_gene.type}"
                        )
                    else:
                        logger.error(
                            f"❌ 複数出力指標登録確認失敗: {indicator_gene.type}"
                        )
                else:
                    # 単一出力指標の登録確認
                    in_dict = indicator_gene.type in strategy_instance.__dict__
                    has_attr = hasattr(strategy_instance, indicator_gene.type)
                    in_indicators = indicator_gene.type in strategy_instance.indicators

                    logger.warning(
                        f"{indicator_gene.type}: __dict__={in_dict}, hasattr={has_attr}, indicators={in_indicators}"
                    )

                    if in_dict and has_attr and in_indicators:
                        logger.warning(
                            f"✅ 単一出力指標登録確認成功: {indicator_gene.type}"
                        )
                    else:
                        logger.error(
                            f"❌ 単一出力指標登録確認失敗: {indicator_gene.type}"
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
