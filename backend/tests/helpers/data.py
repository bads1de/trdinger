"""テストデータ生成ヘルパー

テストで使用するサンプルデータの共通生成関数を提供します。
"""

import numpy as np
import pandas as pd


def make_ohlcv_df(
    periods: int = 100,
    start: str = "2024-01-01",
    freq: str = "h",
    seed: int = 42,
    base_price: float = 100.0,
    close_vol: float = 0.5,
) -> pd.DataFrame:
    """ランダムウォーク型の標準 OHLCV DataFrame を生成する。

    Args:
        periods: 生成する行数
        start: 開始日時
        freq: 時間間隔
        seed: 乱数シード
        base_price: 基準価格
        close_vol: close のランダムウォークのボラティリティ

    Returns:
        open/high/low/close/volume を持つ DataFrame (DatetimeIndex)

    Note:
        high >= max(open, close) かつ low <= min(open, close) を保証します。
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq=freq)

    close = pd.Series(
        base_price + np.cumsum(rng.normal(0, close_vol, periods)),
        index=index,
    )
    open_ = close + rng.normal(0, close_vol * 0.2, periods)
    high = pd.Series(
        np.maximum(open_, close) + np.abs(rng.normal(0, close_vol, periods)),
        index=index,
    )
    low = pd.Series(
        np.minimum(open_, close) - np.abs(rng.normal(0, close_vol, periods)),
        index=index,
    )
    volume = pd.Series(
        rng.uniform(1000, 1500, periods),
        index=index,
    )

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
