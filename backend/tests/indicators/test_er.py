import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestER:
    def test_er_basic_calculation(self):
        """ERの基本計算テスト"""
        data = pd.Series([10, 11, 12, 13, 14, 13.5, 13, 12.5, 13, 13.5], dtype=float)
        length = 5
        result = MomentumIndicators.er(data, length=length)
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])