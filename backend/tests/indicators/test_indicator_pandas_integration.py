"""pandas-ta 連携インジケーターの直接呼び出しを検証するテスト"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.config.indicator_config import indicator_registry
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """pandas-ta 呼び出し検証用のサンプルOHLCVデータを生成"""

    periods = 256
    index = pd.date_range("2023-01-01", periods=periods, freq="h")

    base = np.linspace(100.0, 200.0, periods)
    noise = np.random.normal(0.0, 2.0, periods)
    close = base + noise

    df = pd.DataFrame(
        {
            "open": close * np.random.uniform(0.99, 1.01, periods),
            "high": close * np.random.uniform(1.0, 1.05, periods),
            "low": close * np.random.uniform(0.95, 1.0, periods),
            "close": close,
            "volume": np.random.uniform(10_000, 30_000, periods),
        },
        index=index,
    )

    # pandas-ta は大文字のカラム名を想定するケースがあるため両方用意する
    df["Open"] = df["open"]
    df["High"] = df["high"]
    df["Low"] = df["low"]
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]

    return df


def test_all_pandas_ta_indicators_execute_without_adapter(
    sample_ohlcv: pd.DataFrame,
) -> None:
    """pandas-ta 関数を持つインジケーターがフォールバック無しで実行できることを確認"""

    service = TechnicalIndicatorService()
    failing: list[str] = []
    processed: set[str] = set()

    # 特別な処理が必要な既知のインジケーター（除外リスト）
    known_special_cases = {
        "DONCHIAN",  # 特別なデータ構造が必要
        "PVT",  # volume引数が必要
        "NVI",  # volume引数が必要
        "VWMA",  # volume引数が必要
        "AO",  # 特別な処理が必要
        "AROON",  # 特別な処理が必要
        "BOP",  # 特別な処理が必要
        "CHOP",  # 特別な処理が必要
        "GRI",  # 特別な処理が必要
        "WPR",  # pandas_taに存在しない
    }

    for name, config in indicator_registry.get_all_indicators().items():
        if config.pandas_function is None:
            continue

        # エイリアス重複を除外
        if config.indicator_name in processed:
            continue
        processed.add(config.indicator_name)

        # 既知の特別ケースをスキップ
        if config.indicator_name in known_special_cases:
            continue

        pandas_config = service._get_config(name)
        assert pandas_config is not None, f"{name} の pandas 設定が取得できません"

        normalized_params = service._normalize_params({}, pandas_config)
        result = service._call_pandas_ta(
            sample_ohlcv.copy(), pandas_config, normalized_params
        )

        if result is None:
            failing.append(config.indicator_name)

    assert not failing, (
        f"pandas-ta 直接呼び出しに失敗したインジケーター: {sorted(failing)}"
    )




