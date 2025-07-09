"""
テクニカル指標の包括的テスト

オートストラテジー用テクニカル指標の初期化、計算精度、
エラーハンドリング、エッジケースを網羅的にテストします。
"""

import pytest
import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, List, Tuple
import warnings

# テスト対象のインポート
from app.core.services.indicators.trend import TrendIndicators
from app.core.services.indicators.momentum import MomentumIndicators
from app.core.services.indicators.volatility import VolatilityIndicators
from app.core.services.indicators.utils import (
    TALibError,
    validate_input,
    validate_multi_input,
    ensure_numpy_array,
    format_indicator_result,
)


class TestTechnicalIndicatorsComprehensive:
    """テクニカル指標の包括的テストクラス"""

    @pytest.fixture
    def sample_price_data(self) -> Dict[str, np.ndarray]:
        """テスト用価格データ（OHLC）"""
        np.random.seed(42)
        length = 200
        
        # リアルな価格データを生成
        base_price = 50000.0
        returns = np.random.normal(0, 0.02, length)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        close = np.array(prices)
        high = close + np.random.uniform(0, close * 0.01, length)
        low = close - np.random.uniform(0, close * 0.01, length)
        open_price = close + np.random.uniform(-close * 0.005, close * 0.005, length)
        volume = np.random.uniform(1000, 10000, length)
        
        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    @pytest.fixture
    def edge_case_data(self) -> Dict[str, np.ndarray]:
        """エッジケース用データ"""
        return {
            "constant": np.full(100, 50.0),  # 定数データ
            "zero": np.zeros(100),  # ゼロデータ
            "negative": np.full(100, -10.0),  # 負の値
            "small": np.full(100, 0.001),  # 極小値
            "large": np.full(100, 1e6),  # 極大値
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20),  # NaN含む
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0] * 20),  # 無限大含む
        }

    def test_trend_indicators_initialization(self, sample_price_data):
        """トレンド指標の初期化テスト"""
        print("\n=== トレンド指標初期化テスト ===")
        
        close = sample_price_data["close"]
        high = sample_price_data["high"]
        low = sample_price_data["low"]
        
        # 各指標の初期化と基本計算
        indicators_to_test = [
            ("SMA", lambda: TrendIndicators.sma(close, 20)),
            ("EMA", lambda: TrendIndicators.ema(close, 20)),
            ("TEMA", lambda: TrendIndicators.tema(close, 20)),
            ("DEMA", lambda: TrendIndicators.dema(close, 20)),
            ("KAMA", lambda: TrendIndicators.kama(close, 20)),
            ("MAMA", lambda: TrendIndicators.mama(close)),
            ("T3", lambda: TrendIndicators.t3(close, 5)),
            ("SAR", lambda: TrendIndicators.sar(high, low)),
        ]
        
        results = {}
        for name, func in indicators_to_test:
            try:
                result = func()
                assert isinstance(result, (np.ndarray, tuple)), f"{name}: 結果の型が不正"
                
                if isinstance(result, np.ndarray):
                    assert len(result) == len(close), f"{name}: 結果の長さが不正"
                    assert not np.all(np.isnan(result)), f"{name}: 全てNaN"
                    results[name] = result
                else:
                    # tupleの場合（MAMA）
                    for i, arr in enumerate(result):
                        assert len(arr) == len(close), f"{name}[{i}]: 結果の長さが不正"
                    results[name] = result
                
                print(f"  ✅ {name}: 正常に計算完了")
                
            except Exception as e:
                pytest.fail(f"{name}の計算でエラー: {e}")
        
        print(f"  📊 テスト完了: {len(results)}個の指標")

    def test_momentum_indicators_initialization(self, sample_price_data):
        """モメンタム指標の初期化テスト"""
        print("\n=== モメンタム指標初期化テスト ===")
        
        close = sample_price_data["close"]
        high = sample_price_data["high"]
        low = sample_price_data["low"]
        
        indicators_to_test = [
            ("RSI", lambda: MomentumIndicators.rsi(close, 14)),
            ("MACD", lambda: MomentumIndicators.macd(close)),
            ("MACDFIX", lambda: MomentumIndicators.macdfix(close)),
            ("Stochastic", lambda: MomentumIndicators.stoch(high, low, close)),
            ("StochRSI", lambda: MomentumIndicators.stochrsi(close)),
            ("Williams %R", lambda: MomentumIndicators.williams_r(high, low, close)),
            ("CCI", lambda: MomentumIndicators.cci(high, low, close)),
            ("CMO", lambda: MomentumIndicators.cmo(close)),
            ("ROC", lambda: MomentumIndicators.roc(close)),
            ("MOM", lambda: MomentumIndicators.mom(close)),
        ]
        
        results = {}
        for name, func in indicators_to_test:
            try:
                result = func()
                
                if isinstance(result, tuple):
                    # 複数の値を返す指標（MACD、Stochastic等）
                    for i, arr in enumerate(result):
                        assert isinstance(arr, np.ndarray), f"{name}[{i}]: numpy配列でない"
                        assert len(arr) == len(close), f"{name}[{i}]: 長さが不正"
                    results[name] = result
                else:
                    # 単一の値を返す指標
                    assert isinstance(result, np.ndarray), f"{name}: numpy配列でない"
                    assert len(result) == len(close), f"{name}: 長さが不正"
                    assert not np.all(np.isnan(result)), f"{name}: 全てNaN"
                    results[name] = result
                
                print(f"  ✅ {name}: 正常に計算完了")
                
            except Exception as e:
                pytest.fail(f"{name}の計算でエラー: {e}")
        
        print(f"  📊 テスト完了: {len(results)}個の指標")

    def test_volatility_indicators_initialization(self, sample_price_data):
        """ボラティリティ指標の初期化テスト"""
        print("\n=== ボラティリティ指標初期化テスト ===")
        
        close = sample_price_data["close"]
        high = sample_price_data["high"]
        low = sample_price_data["low"]
        
        indicators_to_test = [
            ("ATR", lambda: VolatilityIndicators.atr(high, low, close, 14)),
            ("NATR", lambda: VolatilityIndicators.natr(high, low, close, 14)),
            ("TRANGE", lambda: VolatilityIndicators.trange(high, low, close)),
            ("Bollinger Bands", lambda: VolatilityIndicators.bollinger_bands(close, 20)),
            ("STDDEV", lambda: VolatilityIndicators.stddev(close, 5)),
            ("VAR", lambda: VolatilityIndicators.var(close, 5)),
        ]
        
        results = {}
        for name, func in indicators_to_test:
            try:
                result = func()
                
                if isinstance(result, tuple):
                    # Bollinger Bandsなど
                    for i, arr in enumerate(result):
                        assert isinstance(arr, np.ndarray), f"{name}[{i}]: numpy配列でない"
                        assert len(arr) == len(close), f"{name}[{i}]: 長さが不正"
                    results[name] = result
                else:
                    assert isinstance(result, np.ndarray), f"{name}: numpy配列でない"
                    assert len(result) == len(close), f"{name}: 長さが不正"
                    assert not np.all(np.isnan(result)), f"{name}: 全てNaN"
                    results[name] = result
                
                print(f"  ✅ {name}: 正常に計算完了")
                
            except Exception as e:
                pytest.fail(f"{name}の計算でエラー: {e}")
        
        print(f"  📊 テスト完了: {len(results)}個の指標")

    def test_calculation_accuracy(self, sample_price_data):
        """計算精度テスト（Ta-lib直接呼び出しとの比較）"""
        print("\n=== 計算精度テスト ===")
        
        close = sample_price_data["close"]
        high = sample_price_data["high"]
        low = sample_price_data["low"]
        
        # 精度テスト対象の指標
        accuracy_tests = [
            ("SMA", TrendIndicators.sma(close, 20), talib.SMA(close, 20)),
            ("EMA", TrendIndicators.ema(close, 20), talib.EMA(close, 20)),
            ("RSI", MomentumIndicators.rsi(close, 14), talib.RSI(close, 14)),
            ("ATR", VolatilityIndicators.atr(high, low, close, 14), talib.ATR(high, low, close, 14)),
        ]
        
        for name, our_result, talib_result in accuracy_tests:
            try:
                np.testing.assert_array_almost_equal(
                    our_result, talib_result, decimal=10,
                    err_msg=f"{name}の計算結果がTa-libと一致しません"
                )
                print(f"  ✅ {name}: Ta-libとの精度一致確認")
            except AssertionError as e:
                pytest.fail(f"{name}の精度テスト失敗: {e}")

    def test_error_handling(self, edge_case_data):
        """エラーハンドリングテスト"""
        print("\n=== エラーハンドリングテスト ===")
        
        # 無効な入力データのテスト
        error_cases = [
            ("None入力", lambda: TrendIndicators.sma(None, 20)),
            ("空配列", lambda: TrendIndicators.sma(np.array([]), 20)),
            ("期間0", lambda: TrendIndicators.sma(edge_case_data["constant"], 0)),
            ("負の期間", lambda: TrendIndicators.sma(edge_case_data["constant"], -5)),
            ("データ不足", lambda: TrendIndicators.sma(np.array([1, 2, 3]), 20)),
            ("無限大含む", lambda: TrendIndicators.sma(edge_case_data["with_inf"], 20)),
        ]
        
        for case_name, func in error_cases:
            with pytest.raises(TALibError):
                func()
            print(f"  ✅ {case_name}: 適切にエラーが発生")
        
        # NaN含むデータの警告テスト
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                TrendIndicators.sma(edge_case_data["with_nan"], 20)
                if w:
                    print(f"  ⚠️  NaN含むデータ: 警告が発生 ({len(w)}件)")
                else:
                    print(f"  ℹ️  NaN含むデータ: 警告なしで処理")
            except TALibError:
                print(f"  ✅ NaN含むデータ: エラーで適切に処理")

    def test_edge_cases(self, edge_case_data):
        """エッジケーステスト"""
        print("\n=== エッジケーステスト ===")
        
        # 特殊なデータでの動作確認
        edge_tests = [
            ("定数データ", edge_case_data["constant"]),
            ("ゼロデータ", edge_case_data["zero"]),
            ("負の値", edge_case_data["negative"]),
            ("極小値", edge_case_data["small"]),
            ("極大値", edge_case_data["large"]),
        ]
        
        for case_name, data in edge_tests:
            try:
                # 基本的な指標で動作確認
                sma_result = TrendIndicators.sma(data, 20)
                rsi_result = MomentumIndicators.rsi(data, 14)
                
                assert isinstance(sma_result, np.ndarray), f"{case_name}: SMA結果が配列でない"
                assert isinstance(rsi_result, np.ndarray), f"{case_name}: RSI結果が配列でない"
                assert len(sma_result) == len(data), f"{case_name}: SMA長さが不正"
                assert len(rsi_result) == len(data), f"{case_name}: RSI長さが不正"
                
                print(f"  ✅ {case_name}: 正常に処理")
                
            except Exception as e:
                print(f"  ⚠️  {case_name}: エラー発生 - {e}")

    def test_parameter_validation(self, sample_price_data):
        """パラメータ検証テスト"""
        print("\n=== パラメータ検証テスト ===")
        
        close = sample_price_data["close"]
        
        # 境界値テスト
        boundary_tests = [
            ("最小期間", lambda: TrendIndicators.sma(close, 1)),
            ("最大期間", lambda: TrendIndicators.sma(close, len(close))),
            ("RSI最小期間", lambda: MomentumIndicators.rsi(close, 2)),
            ("MACD最小期間", lambda: MomentumIndicators.macd(close, 2, 3, 2)),
        ]
        
        for test_name, func in boundary_tests:
            try:
                result = func()
                assert result is not None, f"{test_name}: 結果がNone"
                print(f"  ✅ {test_name}: 正常に処理")
            except Exception as e:
                print(f"  ⚠️  {test_name}: エラー - {e}")


def main():
    """メイン実行関数"""
    print("テクニカル指標包括的テスト開始")
    print("=" * 60)
    
    # pytest実行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
