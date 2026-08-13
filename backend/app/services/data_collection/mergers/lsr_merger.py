"""
Long/Short Ratio データマージャー

Long/Short Ratio データのマージロジックを提供します。
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from app.utils.data_conversion import normalize_market_symbol
from app.utils.error_handler import ErrorHandler
from database.models import LongShortRatioData
from database.repositories.long_short_ratio_repository import (
    LongShortRatioRepository,
)

logger = logging.getLogger(__name__)


class LSROMerger:
    """Long/Short Ratio データマージャー"""

    def __init__(self, lsr_repo: LongShortRatioRepository):
        """
        初期化

        Args:
            lsr_repo: Long/Short Ratio リポジトリ
        """
        self.lsr_repo = lsr_repo

    def merge_lsr_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Long/Short Ratio データをマージ

        Args:
            df: ベースとなるDataFrame
            symbol: 取引ペア
            timeframe: 時間軸（LSRのperiodとして使用）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            LSRデータがマージされたDataFrame
        """
        try:
            # LSRテーブルには「BTC/USDT」と「BTC/USDT:USDT」の両方が混在するため
            # まずコロン付き（標準形式）で検索し、見つからなければベース形式で再検索する
            normalized_symbol = normalize_market_symbol(symbol)
            lsr_data = self.lsr_repo.get_long_short_ratio_data(
                symbol=normalized_symbol,
                period=timeframe,
                start_time=start_date,
                end_time=end_date,
            )
            if not lsr_data:
                db_symbol = normalized_symbol.split(":")[0]
                lsr_data = self.lsr_repo.get_long_short_ratio_data(
                    symbol=db_symbol,
                    period=timeframe,
                    start_time=start_date,
                    end_time=end_date,
                )

            if lsr_data:
                lsr_df = self._convert_lsr_to_dataframe(lsr_data)

                # toleranceはタイムフレームの3期間分（例: 4hなら12h）に設定する
                tolerance = _period_to_timedelta(timeframe) * 3
                df = pd.merge_asof(
                    df.sort_index(),
                    lsr_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                    tolerance=tolerance,
                )

            else:
                logger.warning(
                    f"シンボル {symbol} / {timeframe} のLong/Short Ratioデータが"
                    f"見つかりませんでした。デフォルト値1.0を使用します。"
                )
                df["long_short_ratio"] = 1.0

        except Exception as e:
            ErrorHandler.handle_model_error(e, context="merge_lsr_data")
            df["long_short_ratio"] = 1.0

        return df

    def _convert_lsr_to_dataframe(
        self, lsr_data: list[LongShortRatioData]
    ) -> pd.DataFrame:
        """
        LongShortRatioDataリストをpandas.DataFrameに変換

        買い比率 / 売り比率をロング/ショート比率として算出します。

        Args:
            lsr_data: LongShortRatioDataオブジェクトのリスト

        Returns:
            Long/Short RatioのDataFrame
        """
        data = {
            "long_short_ratio": [
                _safe_ratio(float(r.buy_ratio), float(r.sell_ratio)) for r in lsr_data
            ]
        }
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.timestamp for r in lsr_data])
        # 重複timestampを防御（merge_asofは厳密増加インデックスを要求する）
        df = df[~df.index.duplicated(keep="last")]
        return df


def _period_to_timedelta(period: str) -> timedelta:
    """期間文字列（'4h', '1d', '15min' 等）を timedelta に変換（未知形式は1日）"""
    period = period.strip().lower()
    try:
        if period.endswith("min"):
            return timedelta(minutes=int(period[:-3]))
        if period.endswith("h"):
            return timedelta(hours=int(period[:-1]))
        if period.endswith("d"):
            return timedelta(days=int(period[:-1]))
        if period.endswith("w"):
            return timedelta(weeks=int(period[:-1]))
    except ValueError:
        pass
    return timedelta(days=1)


def _safe_ratio(buy_ratio: float, sell_ratio: float) -> float:
    """買い/売り比率からロング/ショート比率を算出（0除算を安全に処理）"""
    if sell_ratio and sell_ratio > 0:
        return float(buy_ratio) / float(sell_ratio)
    return 1.0
