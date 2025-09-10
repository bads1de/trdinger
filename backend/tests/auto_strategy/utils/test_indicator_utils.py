"""indicator_utils.py のテストモジュール"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.auto_strategy.utils.indicator_utils import (
    _load_indicator_registry,
    indicators_by_category,
    get_volume_indicators,
    get_momentum_indicators,
    get_trend_indicators,
    get_volatility_indicators,
    get_all_indicators,
    validate_symbol,
    validate_timeframe,
    get_all_indicator_ids,
    get_id_to_indicator_mapping,
    get_valid_indicator_types,
)


class MockIndicatorConfig:
    """モック指標設定クラス"""
    def __init__(self, indicator_name, category):
        self.indicator_name = indicator_name
        self.category = category


class MockRegistry:
    """モックレジストリクラス"""
    def __init__(self):
        self._configs = {
            'RSI': MockIndicatorConfig('RSI', 'momentum'),
            'SMA': MockIndicatorConfig('SMA', 'trend'),
            'EMA': MockIndicatorConfig('EMA', 'trend'),
            'MACD': MockIndicatorConfig('MACD', 'momentum'),
            'ATR': MockIndicatorConfig('ATR', 'volatility'),
            'AD': MockIndicatorConfig('AD', 'volume'),
            'OBV': MockIndicatorConfig('OBV', 'volume'),
            'STOCH': MockIndicatorConfig('STOCH', 'momentum'),
            'CCI': MockIndicatorConfig('CCI', 'momentum'),
            'ADX': MockIndicatorConfig('ADX', 'trend'),
            'AO': MockIndicatorConfig('AO', 'oscillator'),
            'APO': MockIndicatorConfig('APO', 'momentum'),
        }


class TestIndicatorUtils:
    """IndicatorUtilsクラスのテスト"""

    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_volume(self, mock_load_registry):
        """ボリューム指標取得テスト"""
        mock_load_registry.return_value = MockRegistry()

        result = indicators_by_category("volume")
        expected = sorted(['AD', 'OBV'])  # モックデータのvolume指標

        assert isinstance(result, list)
        assert sorted(result) == expected  # sortは関数内でされてる

    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_momentum(self, mock_load_registry):
        """モメンタム指標取得テスト"""
        mock_load_registry.return_value = MockRegistry()

        result = indicators_by_category("momentum")
        expected = sorted(['RSI', 'MACD', 'STOCH', 'CCI', 'APO'])  # モックデータのmomentum指標

        assert isinstance(result, list)
        assert sorted(result) == expected

    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_trend(self, mock_load_registry):
        """トレンド指標取得テスト"""
        mock_load_registry.return_value = MockRegistry()

        result = indicators_by_category("trend")
        expected = sorted(['SMA', 'EMA', 'ADX'])  # モックデータのtrend指標

        assert isinstance(result, list)
        assert sorted(result) == expected

    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_volatility(self, mock_load_registry):
        """ボラティリティ指標取得テスト"""
        mock_load_registry.return_value = MockRegistry()

        result = indicators_by_category("volatility")
        expected = sorted(['ATR'])  # モックデータのvolatility指標

        assert isinstance(result, list)
        assert sorted(result) == expected

    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_invalid(self, mock_load_registry):
        """無効なカテゴリのテスト"""
        mock_load_registry.return_value = MockRegistry()

        result = indicators_by_category("invalid_category")

        assert isinstance(result, list)
        assert result == []

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    def test_get_volume_indicators(self, mock_indicators_by_category):
        """get_volume_indicatorsテスト"""
        mock_indicators_by_category.return_value = ['AD', 'OBV']

        result = get_volume_indicators()

        assert isinstance(result, list)
        assert result == ['AD', 'OBV']
        mock_indicators_by_category.assert_called_once_with("volume")

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    def test_get_momentum_indicators(self, mock_indicators_by_category):
        """get_momentum_indicatorsテスト"""
        mock_indicators_by_category.return_value = ['RSI', 'MACD']

        result = get_momentum_indicators()

        assert isinstance(result, list)
        assert result == ['RSI', 'MACD']
        mock_indicators_by_category.assert_called_once_with("momentum")

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    def test_get_trend_indicators(self, mock_indicators_by_category):
        """get_trend_indicatorsテスト"""
        mock_indicators_by_category.return_value = ['SMA', 'EMA']

        result = get_trend_indicators()

        assert isinstance(result, list)
        assert result == ['SMA', 'EMA']
        mock_indicators_by_category.assert_called_once_with("trend")

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    def test_get_volatility_indicators(self, mock_indicators_by_category):
        """get_volatility_indicatorsテスト"""
        mock_indicators_by_category.return_value = ['ATR']

        result = get_volatility_indicators()

        assert isinstance(result, list)
        assert result == ['ATR']
        mock_indicators_by_category.assert_called_once_with("volatility")

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    @patch('app.services.auto_strategy.utils.indicator_utils.ML_INDICATOR_TYPES', ['ML_RENV_PROB', 'ML_TREND_PROB'])
    @patch('app.services.auto_strategy.utils.indicator_utils.COMPOSITE_INDICATORS', ['ICHIMOKU'])
    def test_get_all_indicators(self, mock_composite, mock_ml, mock_indicators_by_category):
        """get_all_indicatorsテスト"""
        # 各カテゴリの返り値を設定
        def mock_side_effect(category):
            mocks = {
                "volume": ["AD", "OBV"],
                "momentum": ["RSI", "MACD"],
                "trend": ["SMA", "EMA"],
                "volatility": ["ATR"]
            }
            return mocks.get(category, [])

        mock_indicators_by_category.side_effect = mock_side_effect

        result = get_all_indicators()

        # 重複を避けた全指標が含まれているか確認
        expected_in_result = set(["AD", "OBV", "RSI", "MACD", "SMA", "EMA", "ATR", "ICHIMOKU", "ML_RENV_PROB", "ML_TREND_PROB"])
        assert isinstance(result, list)
        assert set(result) == expected_in_result

    @patch('app.services.auto_strategy.utils.indicator_utils.SUPPORTED_SYMBOLS', ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
    def test_validate_symbol_valid(self, mock_supported_symbols):
        """有効なシンボルの検証テスト"""
        assert validate_symbol('BTC/USDT:USDT') == True
        assert validate_symbol('ETH/USDT:USDT') == True
        assert validate_symbol('ADA/USDT:USDT') == False  # 無効なシンボル

    @patch('app.services.auto_strategy.utils.indicator_utils.SUPPORTED_SYMBOLS', ['BTC/USDT:USDT'])
    def test_validate_symbol_invalid(self, mock_supported_symbols):
        """無効なシンボルの検証テスト"""
        assert validate_symbol('ETH/USDT:USDT') == False

    @patch('app.services.auto_strategy.utils.indicator_utils.SUPPORTED_TIMEFRAMES', ['15m', '30m', '1h'])
    def test_validate_timeframe_valid(self, mock_supported_timeframes):
        """有効な時間軸の検証テスト"""
        assert validate_timeframe('15m') == True
        assert validate_timeframe('30m') == True
        assert validate_timeframe('1h') == True

    @patch('app.services.auto_strategy.utils.indicator_utils.SUPPORTED_TIMEFRAMES', ['15m', '30m', '1h'])
    def test_validate_timeframe_invalid(self, mock_supported_timeframes):
        """無効な時間軸の検証テスト"""
        assert validate_timeframe('5m') == False
        assert validate_timeframe('10m') == False

    @patch('app.services.indicators.TechnicalIndicatorService.get_supported_indicators', return_value={'RSI': {'type': 'oscillator'}, 'SMA': {'type': 'moving_avg'}})
    @patch('app.services.indicators.TechnicalIndicatorService.__init__', return_value=None)
    def test_get_all_indicator_ids(self, mock_init, mock_get_supported_indicators):
        """全指標IDマッピング取得テスト"""
        result = get_all_indicator_ids()

        # 空文字列は0、それ以外は1から開始
        assert isinstance(result, dict)
        assert '' in result
        assert result[''] == 0
        assert 'RSI' in result
        assert 'SMA' in result

    def test_get_id_to_indicator_mapping(self):
        """ID→指標逆引きマッピングテスト"""
        indicator_ids = {'': 0, 'RSI': 1, 'SMA': 2, 'EMA': 3}
        result = get_id_to_indicator_mapping(indicator_ids)

        expected = {0: '', 1: 'RSI', 2: 'SMA', 3: 'EMA'}
        assert result == expected

    @patch('app.services.auto_strategy.utils.indicator_utils.indicators_by_category')
    @patch('app.services.auto_strategy.utils.indicator_utils.ML_INDICATOR_TYPES', ['ML_RENV_PROB'])
    @patch('app.services.auto_strategy.utils.indicator_utils.COMPOSITE_INDICATORS', ['ICHIMOKU'])
    def test_get_valid_indicator_types(self, mock_composite, mock_ml, mock_indicators_by_category):
        """有効な指標タイプ取得テスト"""
        def mock_side_effect(category):
            mocks = {
                "volume": ["AD"],
                "momentum": ["RSI"],
                "trend": ["SMA"],
                "volatility": ["ATR"]
            }
            return mocks.get(category, [])

        mock_indicators_by_category.side_effect = mock_side_effect

        result = get_valid_indicator_types()

        expected_in_result = set(["AD", "RSI", "SMA", "ATR", "ICHIMOKU", "ML_RENV_PROB"])
        assert isinstance(result, list)
        assert set(result) == expected_in_result

    # エラー処理テスト
    @patch('app.services.auto_strategy.utils.indicator_utils._load_indicator_registry')
    def test_indicators_by_category_with_exception_handling(self, mock_load_registry):
        """例外処理テスト：レジストリアクセス時に例外が発生"""

        class FaultyRegistry:
            """例外を投げるモックレジストリ"""
            @property
            def _configs(self):
                raise AttributeError("'_configs' attribute not accessible")

        mock_load_registry.return_value = FaultyRegistry()

        # 例外が発生しても関数は上手く処理すべき
        try:
            result = indicators_by_category("volume")
            # 例外が適切に処理されたら空リストが返されるはず
            assert isinstance(result, list)
            assert result == []
        except Exception:
            # 実装によっては例外が伝播する可能性あり - バグ発見のヒントになる
            pytest.fail("indicators_by_categoryが例外処理を適切に行わない可能性")

    @patch('app.services.auto_strategy.utils.indicator_utils.ML_INDICATOR_TYPES', ['ML_RENV_PROB', 'ML_RENV_PROB'])  # 重複
    @patch('app.services.auto_strategy.utils.indicator_utils.COMPOSITE_INDICATORS', ['ICHIMOKU'])
    @pytest.mark.skip(reason="バグ発見テストのサンプル")
    def test_get_all_indicators_with_duplicate_handling(self, mock_composite, mock_ml):
        """重複除去テスト：重複オプションがある場合のバグを探す"""

        # 重複を含むデータを渡して重複除去が正しく機能するかテスト

        # 実際の関数呼び出し
        try:
            result = get_all_indicators()
            # 重複が除去されているか確認
            assert len(result) == len(set(result))
        except Exception as e:
            # もし実装にバグがあればここでエラーが出る
            pytest.fail(f"get_all_indicatorsが重複除去を適切に行っていない: {e}")

    @patch('app.services.indicators.TechnicalIndicatorService')
    def test_get_all_indicator_ids_with_technical_indicator_service_exception(self, mock_service):
        """TechnicalIndicatorServiceが例外を投げる場合のテスト"""

        # 例外をシミュレート
        mock_service.side_effect = Exception("Service error")

        try:
            result = get_all_indicator_ids()
            # 実装によってはエラーログを出して{}を返すはず
            assert isinstance(result, dict)
            # エラーログが適切に記録されているか手動確認
        except Exception:
            # 実装によっては例外が伝播するかもしれない
            pytest.fail("get_all_indicator_idsがTechnicalIndicatorService例外を適切に処理していない")
