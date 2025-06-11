"""
その他のテクニカル指標

PSAR（Parabolic SAR）の実装を提供します。
"""

import pandas as pd

from .abstract_indicator import BaseIndicator


class PSARIndicator(BaseIndicator):
    """PSAR（Parabolic SAR）指標"""

    def __init__(self):
        super().__init__(
            indicator_type="PSAR", supported_periods=[1]  # PSARは期間に依存しない
        )

    def calculate(self, df: pd.DataFrame, period: int, **kwargs) -> pd.Series:
        """
        PSAR（Parabolic SAR）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間（通常1、PSARは期間に依存しない）

        Returns:
            PSAR値のSeries
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # PSARの初期設定
        af = 0.02  # 加速因子の初期値
        max_af = 0.20  # 加速因子の最大値

        # 結果を格納するSeries
        psar = pd.Series(index=df.index, dtype=float)

        if len(df) < 2:
            return psar

        # 初期値設定
        psar.iloc[0] = low.iloc[0]
        trend = 1  # 1: 上昇トレンド, -1: 下降トレンド
        ep = high.iloc[0]  # Extreme Point
        current_af = af

        for i in range(1, len(df)):
            if trend == 1:  # 上昇トレンド
                psar.iloc[i] = psar.iloc[i - 1] + current_af * (ep - psar.iloc[i - 1])

                # PSARが前日または当日の安値を上回った場合、トレンド転換
                if psar.iloc[i] > low.iloc[i] or psar.iloc[i] > low.iloc[i - 1]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    current_af = af
                else:
                    # 新しい高値更新
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        current_af = min(current_af + af, max_af)
            else:  # 下降トレンド
                psar.iloc[i] = psar.iloc[i - 1] + current_af * (ep - psar.iloc[i - 1])

                # PSARが前日または当日の高値を下回った場合、トレンド転換
                if psar.iloc[i] < high.iloc[i] or psar.iloc[i] < high.iloc[i - 1]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    current_af = af
                else:
                    # 新しい安値更新
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        current_af = min(current_af + af, max_af)

        return psar

    def get_description(self) -> str:
        """指標の説明を取得"""
        return "PSAR - パラボリックSAR、トレンド転換点を示す"


# 指標インスタンスのファクトリー関数
def get_other_indicator(indicator_type: str) -> BaseIndicator:
    """
    その他の指標のインスタンスを取得

    Args:
        indicator_type: 指標タイプ（'PSAR'）

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    indicators = {
        "PSAR": PSARIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(
            f"サポートされていないその他の指標です: {indicator_type}. "
            f"サポート対象: {list(indicators.keys())}"
        )

    return indicators[indicator_type]()


# サポートされている指標の情報
OTHER_INDICATORS_INFO = {
    "PSAR": {
        "periods": [1],
        "description": "PSAR - パラボリックSAR、トレンド転換点を示す",
        "category": "other",
    },
}
