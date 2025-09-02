import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.trend import TrendIndicators


def test_hilo_result_type_undefined_fix():
    """HILO関数の'result_type'未定義エラーテスト"""
    # サンプルデータ作成
    high = pd.Series([105, 106, 107, 108, 109, 108, 110, 111, 112, 113,
                      114, 115, 116, 117, 116, 118, 119, 120, 119, 121],
                     name='high')
    low = pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                     105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                    name='low')
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
                      name='close')

    # HILO関数呼び出しで'result_type'未定義エラーが発生しないことを確認
    result = TrendIndicators.hilo(high, low, close, length=14)

    # pd.Seriesが正しく返されることを確認
    assert isinstance(result, pd.Series)
    assert len(result) == len(high)
    assert not result.isna().all()