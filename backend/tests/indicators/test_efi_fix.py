import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.volume import VolumeIndicators


def test_efi_length_undefined_fix():
    """EFI関数の'length'未定義エラーテスト"""
    # サンプルデータ作成
    close = pd.Series([100, 101, 102, 103, 102, 104, 105, 106, 107, 106,
                       108, 109, 110, 109, 111, 112, 113, 112, 114, 115],
                      name='close')
    volume = pd.Series([1000, 1500, 1200, 1800, 1100, 1600, 1400, 1700, 1300, 1900,
                        1600, 1200, 1800, 1400, 1700, 1300, 2000, 1500, 1800, 1600],
                       name='volume')

    # EFI関数呼び出しで'length'未定義エラーが発生しないことを確認
    result = VolumeIndicators.efi(close, volume, period=13, mamode='ema', drift=1)

    # pd.Seriesが正しく返されることを確認
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    assert not result.isna().all()