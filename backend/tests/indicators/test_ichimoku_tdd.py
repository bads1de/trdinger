"""
ICHIMOKUインジケーターのTDD(テスト駆動開発)テストファイル

このテストは失敗することを検証してから、実装を修正して成功することを確認する。
"""
import pytest
import numpy as np
import pandas as pd
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestICHIMOKUTDD:
    """ICHIMOKUインジケーターのTDDテスト"""

    @pytest.fixture
    def service(self):
        """TechnicalIndicatorServiceフィクスチャ"""
        return TechnicalIndicatorService()

    @pytest.fixture
    def sample_data_short(self):
        """データ長が不足しているサンプルデータ (50行)"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        high = 100 + np.random.normal(0, 5, 50)
        low = 90 + np.random.normal(0, 5, 50)
        close = 95 + np.random.normal(0, 5, 50)
        df = pd.DataFrame({
            'timestamp': dates,
            'high': high,
            'low': low,
            'close': close
        })
        return df

    @pytest.fixture
    def sample_data_sufficient(self):
        """データ長が十分なサンプルデータ (100行)"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        high = 100 + np.random.normal(0, 5, 100)
        low = 90 + np.random.normal(0, 5, 100)
        close = 95 + np.random.normal(0, 5, 100)
        df = pd.DataFrame({
            'timestamp': dates,
            'high': high,
            'low': low,
            'close': close
        })
        return df

    def test_ichimoku_with_insufficient_data_should_fail(self, service, sample_data_short):
        """データ長が不十分な場合、ICHIMOKUはNaNの結果を返すべき"""
        # kijun=26, senkou=52なので最低52行が必要
        result = service.calculate_indicator(sample_data_short, "ICHIMOKU", {})

        # 結果が存在し、全てNaNであることを確認
        assert result is not None, "ICHIMOKUはデータ長不足でもNoneを返すべきではない"
        assert len(result) == 5, "ICHIMOKUは5つの結果($1, base, span_a, span_b, lag)を返す"

        # 全てのコンポーネントが全てNaNであることを確認
        for component in result:
            assert isinstance(component, np.ndarray)
            assert component.shape == (50,), f"コンポーネントの長さが正しくない: {component.shape}"
            assert np.all(np.isnan(component)), "データ長不足時はすべての値がNaNになるべき"

    def test_ichimoku_with_sufficient_data_should_succeed(self, service, sample_data_sufficient):
        """データ長が十分な場合、ICHIMOKUは正しく計算されるべき"""
        result = service.calculate_indicator(sample_data_sufficient, "ICHIMOKU", {})

        # 基本的な結果確認
        assert result is not None, "ICHIMOKUがNoneを返してはいけない"
        assert len(result) == 5, "ICHIMOKUは5つの結果を返すべき"
        assert all(isinstance(comp, np.ndarray) for comp in result), "全ての結果がnumpy配列であるべき"

        # 各コンポーネントを確認
        conv, base, span_a, span_b, lag = result

        # conversion line: tenkan=9 の移動平均なので、最初8個がNaN
        assert np.isnan(conv[:9]).all(), "conversion lineの最初の9値はNaNになるべき"
        assert not np.isnan(conv[9:]).all(), "変換線は10番目以降に値を持つべき"

        # base line: kijun=26の移動平均なので、最初25個がNaN
        assert np.isnan(base[:26]).all(), "base lineの最初の26値はNaNになるべき"
        assert not np.isnan(base[26:]).all(), "基準線は27番目以降に値を持つべき"

        # span_a, span_b: span_aは(conv + base)/2 を kijun=26シフト
        # span_bはsenkou=52の移動平均をkijun=26シフト
        # シフトにより、さらに遅れる
        assert np.isnan(span_a[:52]).all(), "span_aの最初の52値はNaNになるべき"
        assert np.isnan(span_b[:78]).all(), "span_bの最初の78値はNaNになるべき"

        # lag: kijun=26シフトなので最後26個がNaN
        assert np.isnan(lag[-26:]).all(), "lagging spanの最後26値はNaNになるべき"

    def test_ichimoku_with_custom_parameters(self, service, sample_data_sufficient):
        """カスタムパラメータでICHIMOKU計算をテスト"""
        custom_params = {
            'tenkan': 5,
            'kijun': 15,
            'senkou': 30
        }

        result = service.calculate_indicator(sample_data_sufficient, "ICHIMOKU", custom_params)

        assert result is not None
        assert len(result) == 5

        conv, base, span_a, span_b, lag = result

        # カスタムパラメータが適用されていることを確認
        # conversion: tenkan=5なので最初の5値がNaN
        assert np.isnan(conv[:5]).all(), "カスタムtenkan=5のconversion lineの最初の5値はNaN"
        assert not np.isnan(conv[5]).any(), "変換線は6番目以降に値を持つべき"

        # base: kijun=15なので最初の15値がNaN
        assert np.isnan(base[:15]).all(), "カスタムkijun=15のbase lineの最初の15値はNaN"

        # span_a: (conv + base)/2 を kijun=15シフトなので、最初の15+15=30?
        # span_b: senkou=30, kijun=15なので最初の30+15=45?

        # lag: 最後15個がNaN (kijun=15シフト)
        assert np.isnan(lag[-15:]).all(), "最後15値がNaNになるべき"

    def test_ichimoku_output_format_validation(self, service, sample_data_sufficient):
        """ICHIMOKUの出力形式が正しいことを確認"""
        result = service.calculate_indicator(sample_data_sufficient, "ICHIMOKU", {})

        assert isinstance(result, tuple), "ICHIMOKUはタプルを返す"
        assert len(result) == 5, "ICHIMOKUは5つのコンポーネントを返す"

        # 各コンポーネントがnumpy配列で、同じ長さ
        lengths = [len(comp) for comp in result]
        assert all(length == 100 for length in lengths), f"全てのコンポーネントが同じ長さであるべき: {lengths}"

        # 各コンポーネントがnumpy配列であることを確認
        for i, comp in enumerate(result):
            assert isinstance(comp, np.ndarray), f"コンポーネント{i}はnumpy配列であるべき"
            assert comp.shape == (100,), f"コンポーネント{i}のshapeが正しくない"

    def test_ichimoku_all_nan_output_for_insufficient_data(self, service):
        """極端に小さいデータでのICHIMOKU挙動確認"""
        # kijun=26, senkou=52なので、最低52行必要
        small_df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [98, 99, 100],
            'close': [99, 100, 101]
        })

        result = service.calculate_indicator(small_df, "ICHIMOKU", {})

        # データ長不足でもNoneを返してはならない
        assert result is not None
        assert len(result) == 5

        # 全てのコンポーネントが全てNaNであることを確認
        for i, comp in enumerate(result):
            assert isinstance(comp, np.ndarray), f"コンポーネント{i}はnumpy配列であるべき"
            assert comp.shape == (3,), f"コンポーネント{i}の長さが一致すべき"
            assert np.all(np.isnan(comp)), f"データ長不足時はコンポーネント{i}がすべてNaNであるべき"

    # これらのテストは実装変更後に有効になる
    # def test_ichimoku_uses_pandas_ta_backend(self, service, sample_data_sufficient):
    #     # pandas-ta.backendを使用していることを検証
    #     pass

    # def test_ichimoku_nan_handling_like_pandas_ta(self, service, sample_data_sufficient):
    #     # pandas-taのNaN処理挙動に近いことを確認
    #     pass