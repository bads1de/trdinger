"""
指標計算器

テクニカル指標の計算を担当します。
"""

import logging
import numpy as np
from typing import Dict, Any, Union, Tuple

from app.core.services.indicators import TechnicalIndicatorService
from ..models.gene_strategy import IndicatorGene

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算器

    テクニカル指標の計算を担当します。
    """

    def __init__(self):
        """初期化"""
        self.technical_indicator_service = TechnicalIndicatorService()

    def calculate_indicator(
        self, indicator_type: str, parameters: Dict[str, Any], data
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...], None]:
        """
        指標計算

        Args:
            indicator_type: 指標タイプ
            parameters: パラメータ
            data: backtesting.pyのデータオブジェクト

        Returns:
            計算結果（numpy配列）
        """
        try:
            # backtesting.pyのデータオブジェクトをDataFrameに変換
            df = data.df

            # データの基本検証
            if df.empty:
                raise ValueError(f"データが空です: {indicator_type}")

            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"不足しているカラム: {missing_columns}")

            # TechnicalIndicatorServiceを使用して計算
            result = self.technical_indicator_service.calculate_indicator(
                df, indicator_type, parameters
            )

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}", exc_info=True)
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
            # 指標計算を直接実行
            result = self.calculate_indicator(
                indicator_gene.type, indicator_gene.parameters, strategy_instance.data
            )

            if result is not None:
                # 指標をstrategy.I()で登録
                if isinstance(result, tuple):
                    # 複数の出力がある指標（MACD等）
                    for i, output in enumerate(result):
                        indicator_name = f"{indicator_gene.type}_{i}"

                        # クロージャ問題を回避するため、デフォルト引数を使用
                        def create_indicator_func(data=output):
                            return data

                        setattr(
                            strategy_instance,
                            indicator_name,
                            strategy_instance.I(create_indicator_func),
                        )
                else:
                    # 単一出力の指標

                    def create_indicator_func(data=result):
                        return data

                    setattr(
                        strategy_instance,
                        indicator_gene.type,
                        strategy_instance.I(create_indicator_func),
                    )
            else:
                logger.error(f"指標計算結果がNullです: {indicator_gene.type}")
                raise ValueError(f"指標計算に失敗しました: {indicator_gene.type}")

        except Exception as e:
            logger.error(f"指標初期化エラー {indicator_gene.type}: {e}", exc_info=True)
            # エラーを再発生させて上位で適切に処理
            raise
