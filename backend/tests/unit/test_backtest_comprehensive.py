"""
注意: このテストは独自実装のStrategyExecutorに依存していましたが、
backtesting.pyライブラリへの統一により無効化されました。

新しいテストは以下を参照してください:
- backend/tests/unit/test_backtest_service.py
- backend/tests/integration/test_unified_backtest_system.py
"""

import pytest

# 独自実装が削除されたため、このテストファイルは無効化
pytestmark = pytest.mark.skip(
    reason="StrategyExecutor was removed in favor of backtesting.py library"
)


# 以下は無効化されたコード（参考用）
# import pytest
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from unittest.mock import Mock, patch

# from backtest.engine.strategy_executor import StrategyExecutor, Trade, Position  # 削除済み
# from backtest.engine.indicators import TechnicalIndicators  # 削除済み


class TestStrategyExecutorValidation:
    """StrategyExecutorのバリデーション機能テスト"""

    def test_valid_parameters(self):
        """正常なパラメータでの初期化"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
        assert executor.initial_capital == 100000
        assert executor.commission_rate == 0.001

    def test_negative_initial_capital(self):
        """負の初期資金でエラー"""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            StrategyExecutor(initial_capital=-1000)

    def test_zero_initial_capital(self):
        """ゼロの初期資金でエラー"""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            StrategyExecutor(initial_capital=0)

    def test_invalid_commission_rate_negative(self):
        """負の手数料率でエラー"""
        with pytest.raises(ValueError, match="Commission rate must be between 0 and 1"):
            StrategyExecutor(commission_rate=-0.1)

    def test_invalid_commission_rate_over_one(self):
        """1を超える手数料率でエラー"""
        with pytest.raises(ValueError, match="Commission rate must be between 0 and 1"):
            StrategyExecutor(commission_rate=1.1)

    def test_invalid_initial_capital_type(self):
        """初期資金の型エラー"""
        with pytest.raises(TypeError, match="Initial capital must be a number"):
            StrategyExecutor(initial_capital="invalid")

    def test_invalid_commission_rate_type(self):
        """手数料率の型エラー"""
        with pytest.raises(TypeError, match="Commission rate must be a number"):
            StrategyExecutor(commission_rate="invalid")


class TestTechnicalIndicatorsColumnNames:
    """TechnicalIndicatorsの列名対応テスト"""

    def create_test_data_lowercase(self):
        """小文字列名のテストデータ"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 2000, 50)
        }, index=dates)

    def create_test_data_uppercase(self):
        """大文字列名のテストデータ"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        return pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        }, index=dates)

    def test_sma_with_lowercase_columns(self):
        """小文字列名でのSMA計算"""
        data = self.create_test_data_lowercase()
        result = TechnicalIndicators.calculate_indicator(data, 'SMA', {'period': 20})
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_sma_with_uppercase_columns(self):
        """大文字列名でのSMA計算"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'SMA', {'period': 20})
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_ema_with_mixed_columns(self):
        """混在列名でのEMA計算"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
        }, index=dates)
        result = TechnicalIndicators.calculate_indicator(data, 'EMA', {'period': 12})
        assert isinstance(result, pd.Series)

    def test_rsi_calculation(self):
        """RSI計算テスト"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'RSI', {'period': 14})
        assert isinstance(result, pd.Series)
        # RSIは0-100の範囲
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_macd_calculation(self):
        """MACD計算テスト"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'MACD', {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        })
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result

    def test_bollinger_bands_calculation(self):
        """ボリンジャーバンド計算テスト"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'BB', {
            'period': 20,
            'std_dev': 2.0
        })
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result

    def test_stochastic_calculation(self):
        """ストキャスティクス計算テスト"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'STOCH', {
            'k_period': 14,
            'd_period': 3
        })
        assert isinstance(result, dict)
        assert 'k' in result
        assert 'd' in result

    def test_atr_calculation(self):
        """ATR計算テスト"""
        data = self.create_test_data_uppercase()
        result = TechnicalIndicators.calculate_indicator(data, 'ATR', {'period': 14})
        assert isinstance(result, pd.Series)
        # ATRは正の値
        valid_values = result.dropna()
        assert all(val >= 0 for val in valid_values)

    def test_unsupported_indicator(self):
        """サポートされていない指標でエラー"""
        data = self.create_test_data_uppercase()
        with pytest.raises(ValueError, match="Unsupported indicator"):
            TechnicalIndicators.calculate_indicator(data, 'INVALID', {})

    def test_missing_column_error(self):
        """必要な列が存在しない場合のエラー"""
        data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        with pytest.raises(KeyError, match="Column 'Close' not found"):
            TechnicalIndicators.calculate_indicator(data, 'SMA', {'period': 20})


class TestStrategyExecutorPriceDataHandling:
    """StrategyExecutorの価格データ処理テスト"""

    def test_get_price_from_data_lowercase(self):
        """小文字列名からの価格取得"""
        executor = StrategyExecutor()
        data = pd.Series({'open': 100, 'high': 110, 'low': 90, 'close': 105})
        price = executor._get_price_from_data(data, 'close')
        assert price == 105

    def test_get_price_from_data_uppercase(self):
        """大文字列名からの価格取得"""
        executor = StrategyExecutor()
        data = pd.Series({'Open': 100, 'High': 110, 'Low': 90, 'Close': 105})
        price = executor._get_price_from_data(data, 'close')
        assert price == 105

    def test_get_price_from_data_mixed_case(self):
        """混在列名からの価格取得"""
        executor = StrategyExecutor()
        data = pd.Series({'open': 100, 'High': 110, 'low': 90, 'Close': 105})
        price = executor._get_price_from_data(data, 'high')
        assert price == 110


class TestStrategyExecutorTradeExecution:
    """StrategyExecutorの取引実行テスト"""

    def test_buy_trade_execution(self):
        """買い取引の実行"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
        timestamp = datetime.now()
        
        trade = executor.execute_trade('buy', 100.0, timestamp)
        
        assert trade is not None
        assert trade.type == 'buy'
        assert trade.price == 100.0
        assert executor.position.quantity > 0
        assert executor.capital < executor.initial_capital

    def test_sell_trade_execution(self):
        """売り取引の実行"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
        timestamp = datetime.now()
        
        # まず買いポジションを作成
        executor.execute_trade('buy', 100.0, timestamp)
        initial_quantity = executor.position.quantity
        
        # 売り取引を実行
        trade = executor.execute_trade('sell', 110.0, timestamp)
        
        assert trade is not None
        assert trade.type == 'sell'
        assert trade.price == 110.0
        assert trade.pnl > 0  # 利益が出ているはず
        assert executor.position.quantity == 0

    def test_sell_without_position(self):
        """ポジションなしでの売り取引"""
        executor = StrategyExecutor()
        timestamp = datetime.now()
        
        trade = executor.execute_trade('sell', 100.0, timestamp)
        assert trade is None

    def test_insufficient_capital(self):
        """資金不足での買い取引"""
        executor = StrategyExecutor(initial_capital=100, commission_rate=0.001)
        timestamp = datetime.now()
        
        # 高額な取引を試行
        trade = executor.execute_trade('buy', 1000.0, timestamp, quantity=1000)
        assert trade is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
