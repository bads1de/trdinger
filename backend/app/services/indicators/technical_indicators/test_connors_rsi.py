"""Connors RSIのテスト用モジュール

Note: 旧 OriginalIndicators 実装に依存しており、現行コードベースでは無効のためスキップ。
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Depends on removed backend.app.services.original. Skipped as legacy test."
)


class TestConnorsRSI:
    """Connors RSIのテストクラス"""

    def test_connors_rsi_basic_functionality(self):
        """Connors RSIの基本機能テスト"""
        # テスト用データを生成
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # トレンドのある価格データを生成
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + noise

        close = pd.Series(prices, index=dates, name="close")

        # Connors RSIを計算
        result = OriginalIndicators.connors_rsi(close, 3, 2, 100)

        # 結果がNoneでないことを確認
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

        # Connors RSIは0-100の範囲内にあるべき
        valid_values = [val for val in result if not np.isnan(val)]
        assert all(0 <= val <= 100 for val in valid_values)

    def test_connors_rsi_with_different_parameters(self):
        """異なるパラメータでのテスト"""
        # 単純な価格データ
        close = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])

        # 異なるパラメータでテスト
        result1 = OriginalIndicators.connors_rsi(close, 3, 2, 100)
        result2 = OriginalIndicators.connors_rsi(close, 5, 3, 50)

        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, pd.Series)
        assert isinstance(result2, pd.Series)
        assert len(result1) == len(close)
        assert len(result2) == len(close)

    def test_connors_rsi_edge_cases(self):
        """エッジケースのテスト"""
        close = pd.Series([100, 101, 102])

        # 最小データ長でテスト
        result = OriginalIndicators.connors_rsi(close, 3, 2, 100)
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_connors_rsi_calculation(self):
        """計算結果の基本的な整合性テスト"""
        # 単調増加の価格データ
        close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = OriginalIndicators.connors_rsi(close, 3, 2, 5)

        # 結果が適切な範囲にあることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

        # 有効な値が含まれているべき
        valid_values = [val for val in result if not np.isnan(val)]
        assert len(valid_values) > 0
        assert all(0 <= val <= 100 for val in valid_values)

    def test_connors_rsi_error_handling(self):
        """エラーハンドリングのテスト"""
        close = pd.Series([100, 101, 102])

        # 無効なパラメータのテスト
        with pytest.raises(ValueError, match="rsi_periods must be >= 2"):
            OriginalIndicators.connors_rsi(close, 1, 2, 100)

        with pytest.raises(ValueError, match="streak_periods must be >= 1"):
            OriginalIndicators.connors_rsi(close, 3, 0, 100)

        with pytest.raises(ValueError, match="rank_periods must be >= 2"):
            OriginalIndicators.connors_rsi(close, 3, 2, 1)

    def test_connors_rsi_empty_data(self):
        """空データのテスト"""
        close = pd.Series([])

        with pytest.raises(ValueError, match="rsi_periods must be >= 2"):
            OriginalIndicators.connors_rsi(close, 3, 2, 100)

    def test_calculate_connors_rsi_dataframe(self):
        """DataFrameラッパーメソッドのテスト"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = np.random.normal(100, 5, 50)

        data = pd.DataFrame(
            {
                "close": prices,
                "open": prices - 1,
                "high": prices + 2,
                "low": prices - 2,
                "volume": np.random.randint(100, 1000, 50),
            },
            index=dates,
        )

        result = OriginalIndicators.calculate_connors_rsi(data, 3, 2, 20)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "CONNORS_RSI_3_2_20" in result.columns

        # Connors RSIの値が適切な範囲にあることを確認
        valid_values = [
            val for val in result["CONNORS_RSI_3_2_20"] if not np.isnan(val)
        ]
        assert all(0 <= val <= 100 for val in valid_values)

    def test_connors_rsi_trend_identification(self):
        """トレンド識別能力のテスト"""
        # 明確な上昇トレンド
        uptrend = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        uptrend_result = OriginalIndicators.connors_rsi(uptrend, 3, 2, 5)

        # 明確な下降トレンド
        downtrend = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        downtrend_result = OriginalIndicators.connors_rsi(downtrend, 3, 2, 5)

        # トレンド終了付近の値を比較
        assert isinstance(uptrend_result, pd.Series)
        assert isinstance(downtrend_result, pd.Series)
