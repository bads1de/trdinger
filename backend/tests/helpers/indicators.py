"""テクニカル指標のテストヘルパー

テストで使用するインジケーター計算の共通実装を提供します。
"""

import pandas as pd


def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """RSIを計算するテストヘルパー。

    Args:
        prices: 価格系列
        period: RSIの期間

    Returns:
        RSIの系列

    Note:
        本番実装 (MomentumIndicators.rsi) との結合を避けるため、
        意図的に簡易実装をスタンドアロンで保持しています。
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
