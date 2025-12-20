import numpy as np
import pandas as pd
import pytest

from app.services.indicators.config.indicator_config import indicator_registry
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用のサンプルOHLCVデータを作成"""
    size = 256
    dates = pd.date_range("2023-01-01", periods=size, freq="1H")
    data = {
        "open": np.random.uniform(100, 200, size),
        "high": np.random.uniform(200, 300, size),
        "low": np.random.uniform(50, 100, size),
        "close": np.random.uniform(100, 200, size),
        "volume": np.random.uniform(1000, 50000, size),
    }
    # 追加のカラム
    data["Open"] = data["open"]
    data["High"] = data["high"]
    data["Low"] = data["low"]
    data["Close"] = data["close"]
    data["Volume"] = data["volume"]

    return pd.DataFrame(data, index=dates)


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
        "LONG_RUN",  # 必須引数が必要
        "SHORT_RUN",  # 必須引数が必要
        "XSIGNALS",  # 必須引数が必要
        "MA",  # 汎用的すぎて単体テストには不向き
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

        pandas_config = service._get_pandas_ta_config(name)
        assert pandas_config is not None, f"{name} の pandas 設定が取得できません"

        normalized_params = service._normalize_params({}, pandas_config)
        result = service._call_pandas_ta(
            sample_ohlcv.copy(), pandas_config, normalized_params
        )

        if result is None:
            failing.append(config.indicator_name)

    assert (
        not failing
    ), f"pandas-ta 直接呼び出しに失敗したインジケーター: {sorted(failing)}"