import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry


def make_df(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    # 緩やかなトレンドと妥当な範囲のダミーデータ
    open_ = np.linspace(100, 120, n)
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.2
    volume = np.full(n, 1000, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def _default_params(config):
    params = {}
    for name, pconf in config.parameters.items():
        params[name] = pconf.default_value
    return params


# adapter_function が存在する（=TechnicalIndicatorServiceで直接計算可能な）指標のみを対象
SUPPORTED_NAMES = [
    name
    for name in indicator_registry.get_supported_indicator_names()
    if (
        indicator_registry.get_indicator_config(name)
        and indicator_registry.get_indicator_config(name).adapter_function
    )
]


@pytest.mark.parametrize("indicator_name", SUPPORTED_NAMES)
def test_indicator_initialization_no_exception(indicator_name):
    df = make_df()
    svc = TechnicalIndicatorService()
    config = indicator_registry.get_indicator_config(indicator_name)
    assert config is not None and config.adapter_function is not None

    params = _default_params(config)

    # 実行: 例外が出ないこと
    result = svc.calculate_indicator(df, indicator_name, params)

    # 結果長さの基本検証
    # 注: 一部の指標（STOCHなど）は計算の性質上、入力より短い結果を返すことがある
    if isinstance(result, tuple):
        for arr in result:
            assert hasattr(
                arr, "__len__"
            ), f"{indicator_name} result element must be array-like"
            assert len(arr) > 0, f"{indicator_name} result element must not be empty"
            assert len(arr) <= len(df), f"{indicator_name} result element length ({len(arr)}) should not exceed input length ({len(df)})"
    else:
        assert hasattr(result, "__len__"), f"{indicator_name} result must be array-like"
        assert len(result) > 0, f"{indicator_name} result must not be empty"
        assert len(result) <= len(df), f"{indicator_name} result length ({len(result)}) should not exceed input length ({len(df)})"
