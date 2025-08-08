import numpy as np
import pandas as pd
from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry


def test_bb_works_with_int_columns():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    # 整数型のカラムを意図的に作成
    close = np.arange(100, 100 + n, dtype=np.int64)
    df = pd.DataFrame({"Close": close}, index=idx)

    svc = TechnicalIndicatorService()
    cfg = indicator_registry.get_indicator_config("BB")
    params = {p.name: p.default_value for p in cfg.parameters.values()}

    upper, middle, lower = svc.calculate_indicator(df, "BB", params)

    assert len(upper) == n and len(middle) == n and len(lower) == n
    assert upper.dtype == np.float64
    assert middle.dtype == np.float64
    assert lower.dtype == np.float64

